import asyncio
from copy import deepcopy

import pytest

from reward_engine.adapters import scheduling
from reward_engine.adapters.workbuddy.io import read, write


def test_batch_override_preserves_group_limit_deadline_and_frozen_config(tmp_path):
    config = {"output": str(tmp_path), "batch": {
        "concurrency": 3, "group_concurrency": 2, "deadline_s": 1800},
        "actor_timeout_s": 1800}
    before = deepcopy(config)
    original = scheduling.batch_policy(config)
    assert original.concurrency == 3
    write(tmp_path / "scheduling.json", {"task_limit": 128, "adaptive": True, "batch_limit": 8})
    updated = scheduling.batch_policy(config)
    assert updated.concurrency == 8
    assert updated.group_concurrency == original.group_concurrency == 2
    assert updated.deadline_s == original.deadline_s == 1800
    assert config == before and original.concurrency == 3


@pytest.mark.parametrize("value", [False, 0, 129, 3.5, "8", None])
def test_invalid_batch_override_rejected(tmp_path, value):
    write(tmp_path / "scheduling.json", {"batch_limit": value})
    with pytest.raises(ValueError, match="Batch limit"):
        scheduling.control({"output": str(tmp_path)})


@pytest.mark.asyncio
async def test_batch_override_fills_available_shared_slots_without_bypassing_them(tmp_path):
    from reward_engine.core.batch import collect_batch

    config = {"output": str(tmp_path), "batch": {"concurrency": 3, "deadline_s": 2}}
    write(tmp_path / "scheduling.json", {"batch_limit": 8})
    slots, release, full = asyncio.Semaphore(5), asyncio.Event(), asyncio.Event()
    active, peak, entered = 0, 0, []

    async def check(index):
        nonlocal active, peak
        async with slots:
            entered.append(index)
            active += 1
            peak = max(peak, active)
            if active == 5:
                full.set()
            try:
                await release.wait()
                return index
            finally:
                active -= 1

    job = asyncio.create_task(collect_batch(
        {str(i): lambda i=i: check(i) for i in range(8)}, scheduling.batch_policy(config)))
    try:
        await asyncio.wait_for(full.wait(), 1)
        assert peak == 5 and len(entered) == 5 and not job.done()
    finally:
        release.set()
        result = await job
    assert result.require_all() == tuple(range(8))
    assert len(entered) == 8 and peak == 5 and active == 0


@pytest.mark.asyncio
async def test_live_resize_and_drain_preserve_active_work(tmp_path):
    config = {"output": str(tmp_path), "task_concurrency": 1}
    started, release = [], asyncio.Event()

    async def one(name):
        started.append(name)
        await release.wait()
        return name

    job = asyncio.create_task(scheduling.run_bounded(config, range(6), one, poll_s=0.01))
    async with asyncio.timeout(2):
        while len(started) != 1:
            await asyncio.sleep(0.01)
        write(tmp_path / "scheduling.json", {"task_limit": 3})
        while len(started) != 3:
            await asyncio.sleep(0.01)
        write(tmp_path / "scheduling.json", {"task_limit": 1, "drain": True})
        while not read(tmp_path / "scheduling_state.json")["drain"]:
            await asyncio.sleep(0.01)
        assert not job.done()
        release.set()
        with pytest.raises(scheduling.DispatchDrained):
            await job
    assert started == [0, 1, 2]


@pytest.mark.asyncio
async def test_resource_limit_caps_requested_parallelism(tmp_path, monkeypatch):
    write(tmp_path / "scheduling.json", {"task_limit": 128, "adaptive": True})
    monkeypatch.setattr(scheduling, "resource_limit", lambda _: 2)
    active, peak = 0, 0

    async def one(name):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return name

    results = await scheduling.run_bounded({"output": str(tmp_path)}, range(7), one)
    assert sorted(results) == list(range(7)) and peak == 2


@pytest.mark.asyncio
async def test_isolated_failures_refill_capacity_while_slow_task_keeps_running(tmp_path, monkeypatch):
    config = {"output": str(tmp_path), "implementation_revision": str(tmp_path / "revision"),
              "identity": "synthetic", "failure_policy": "isolate_tasks_v1"}
    write(tmp_path / "scheduling.json", {"task_limit": 128, "adaptive": True})
    monkeypatch.setattr(scheduling, "resource_limit", lambda _: 6)
    from reward_engine.adapters import task_admission
    monkeypatch.setattr(task_admission, "check_space", lambda _: None)
    started, release = [], asyncio.Event()
    initial_full = asyncio.Event()

    async def one(name):
        started.append(name)
        if len(started) == 6:
            initial_full.set()
        if name == 0:
            await release.wait()
            return {"status": "completed"}
        if name < 6:
            await initial_full.wait()
            return {"status": "failed", "stage": "construction", "error_type": "ResponseContractError"}
        await release.wait()
        return {"status": "completed"}

    job = asyncio.create_task(scheduling.run_bounded(config, range(11), one, poll_s=0.01))
    try:
        async with asyncio.timeout(2):
            while len(started) < 11 and not job.done():
                await asyncio.sleep(0.01)
        assert started == list(range(11))
        assert not job.done()  # Failures did not require draining the slow task.
        state = read(tmp_path / "scheduling_state.json")
        assert state["effective_limit"] == 6 and len(state["active_tasks"]) == 6
        assert state["failure_paused"] is False
    finally:
        release.set()
        results = await job
    assert len(results) == 11
    state = read(tmp_path / "scheduling_state.json")
    assert state["status"] == "failed" and state["pending_tasks"] == 0
    assert len(state["task_failures"]) == 5


@pytest.mark.asyncio
async def test_isolated_dispatch_waits_for_disk_recovery_without_consuming_tasks(tmp_path, monkeypatch):
    from reward_engine.adapters import task_admission
    config = {"output": str(tmp_path), "implementation_revision": str(tmp_path / "revision"),
              "identity": "synthetic", "failure_policy": "isolate_tasks_v1", "task_concurrency": 3}
    available = False
    started = []

    def check(_):
        if not available:
            raise OSError("synthetic disk pressure")

    async def one(name):
        started.append(name)
        return {"status": "completed"}

    monkeypatch.setattr(task_admission, "check_space", check)
    job = asyncio.create_task(scheduling.run_bounded(config, range(3), one, poll_s=0.01))
    try:
        async with asyncio.timeout(2):
            while not job.done():
                path = tmp_path / "scheduling_state.json"
                if path.exists() and read(path).get("resource_paused"):
                    break
                await asyncio.sleep(0.01)
        assert not started and not job.done()
        assert read(tmp_path / "scheduling_state.json")["pending_tasks"] == 3
        available = True
        await asyncio.wait_for(job, 2)
    finally:
        if not job.done():
            job.cancel()
            await asyncio.gather(job, return_exceptions=True)
    assert started == [0, 1, 2]
    assert read(tmp_path / "scheduling_state.json")["status"] == "completed"


@pytest.mark.asyncio
async def test_invalid_live_control_drains_then_fails(tmp_path):
    started = asyncio.Event()
    release = asyncio.Event()
    completed = []

    async def one(name):
        started.set()
        await release.wait()
        completed.append(name)

    job = asyncio.create_task(scheduling.run_bounded(
        {"output": str(tmp_path)}, [1, 2], one, poll_s=0.01))
    await started.wait()
    write(tmp_path / "scheduling.json", {"task_limit": True})
    await asyncio.sleep(0.03)
    assert not job.done()
    release.set()
    with pytest.raises(ValueError, match="Task limit"):
        await job
    assert completed == [1]


@pytest.mark.parametrize("value", [{"task_limit": 129}, {"adaptive": 1}, {"unknown": 1}])
def test_invalid_controls_rejected(tmp_path, value):
    write(tmp_path / "scheduling.json", value)
    with pytest.raises(ValueError):
        scheduling.control({"output": str(tmp_path)})


@pytest.mark.asyncio
@pytest.mark.parametrize("three_b", [True, False])
async def test_office_expanded_limit_keeps_legacy_limit(three_b):
    from experiments.workbuddy.schedule import run_tasks

    config = {"task_concurrency": 8, "rounds": 1}
    if three_b:
        config["generation_protocol"] = "three_b_fixed_baselines_v1"
    state = {"tasks": {}}

    async def one(config, name):
        return {"status": "completed", "scored": 6}

    call = run_tasks(config, state, list("abcde"), one_task=one,
                     check_space=lambda _: None, record=lambda _: None)
    if three_b:
        await call
        assert len(state["tasks"]) == 5
    else:
        with pytest.raises(ValueError, match="1 to 3"):
            await call
