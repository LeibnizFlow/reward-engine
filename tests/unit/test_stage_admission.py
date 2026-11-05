"""Operator drains finish active work and preserve exact resumable stages."""

import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest

from reward_engine.adapters import scheduling
from reward_engine.adapters.workbuddy.io import read, write
from reward_engine.core.outcome.three_b import Stages


@pytest.mark.asyncio
@pytest.mark.parametrize("actor", [False, True])
async def test_drain_finishes_live_stage_and_resumes_without_reexecution(tmp_path, actor):
    config = {"output": str(tmp_path), "task_concurrency": 1}
    records, calls = {}, []
    entered, release = asyncio.Event(), asyncio.Event()
    first = "round_1/B1" if actor else "round_1/construction"
    following = "round_1/B2"
    stages = Stages({}, lambda n, v: records.__setitem__(n, deepcopy(v)), records.get,
                    admit_stage=lambda n: scheduling.check_stage_admission(config, "task", n))

    async def current():
        calls.append(first)
        entered.set()
        await release.wait()
        return {"evidence": "completed current stage"}

    async def next_actor():
        calls.append(following)
        return {"evidence": "completed following actor"}

    async def task(_):
        await stages.run(first, {}, current, actor=actor)
        await stages.run(following, {}, next_actor, actor=True)

    job = asyncio.create_task(scheduling.run_bounded(config, ["task", "queued"], task, poll_s=.01))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        write(tmp_path / "scheduling.json", {"drain": True})
        assert not job.done()
        release.set()
        with pytest.raises(scheduling.DispatchDrained):
            await asyncio.wait_for(job, 2)
    finally:
        release.set()
        if not job.done():
            job.cancel()
        await asyncio.gather(job, return_exceptions=True)
    assert calls == [first]
    checkpoint = deepcopy(records[first + "/stage"])
    assert checkpoint["status"] == "completed"
    assert not any(n.startswith(following) for n in records)
    state = read(tmp_path / "scheduling_state.json")
    assert state["active_tasks"] == [] and state["pending_tasks"] == 1
    assert state["task_failures"][0]["error_type"] == "DispatchDrained"
    assert state["task_failures"][0]["stage"] == following
    # Completed stages still verify and load while draining.
    assert await stages.run(first, {}, current, actor=actor) == checkpoint["value"]
    write(tmp_path / "scheduling.json", {"drain": False})
    await task("task")
    assert calls == [first, following]
    assert records[first + "/stage"] == checkpoint
    assert records[following + "/stage"]["attempt"] == 1


@pytest.mark.asyncio
async def test_drain_does_not_consume_actor_recovery_authorization(tmp_path):
    config = {"output": str(tmp_path)}
    records, claimed = {}, []
    stages = Stages({}, lambda n, v: records.__setitem__(n, deepcopy(v)), records.get,
                    recover_actor=lambda n, s: claimed.append(n) or True,
                    admit_stage=lambda n: scheduling.check_stage_admission(config, "task", n))

    async def failure():
        raise TimeoutError("synthetic interrupted actor")

    async def completed():
        return {"evidence": "synthetic recovered actor"}

    with pytest.raises(TimeoutError):
        await stages.run("round_1/B1", {}, failure, actor=True)
    before = deepcopy(records)
    write(tmp_path / "scheduling.json", {"drain": True})
    with pytest.raises(scheduling.DispatchDrained):
        await stages.run("round_1/B1", {}, completed, actor=True)
    assert not claimed and records == before
    write(tmp_path / "scheduling.json", {"drain": False})
    await stages.run("round_1/B1", {}, completed, actor=True)
    assert claimed == ["round_1/B1"]
    assert records["round_1/B1/stage"]["attempt"] == 2
    assert records["round_1/B1/failures/1"] == before["round_1/B1/failures/1"]


@pytest.mark.asyncio
async def test_drain_does_not_hide_changed_completed_evidence(tmp_path):
    config = {"output": str(tmp_path)}
    records = {}
    stages = Stages({}, lambda n, v: records.__setitem__(n, deepcopy(v)), records.get,
                    admit_stage=lambda n: scheduling.check_stage_admission(config, "task", n))

    async def completed():
        return {"evidence": "original"}

    await stages.run("round_1/construction", {}, completed)
    records["round_1/construction/stage"]["value"]["evidence"] = "altered"
    write(tmp_path / "scheduling.json", {"drain": True})
    with pytest.raises(ValueError, match="Checkpoint content changed"):
        await stages.run("round_1/construction", {}, completed)


@pytest.mark.asyncio
async def test_campaign_supplies_live_stage_admission(tmp_path, monkeypatch):
    from experiments.claw_eval.opencode import campaign
    from reward_engine.core.outcome import three_b

    config = {"output": str(tmp_path), "batch": {}, "identity": "synthetic",
              "image": "synthetic", "rounds": 1}
    monkeypatch.setattr(campaign, "public_task", lambda *_: (None, {"task_id": "task"}))
    monkeypatch.setattr(campaign, "registry", lambda *_: None)
    write(tmp_path / "scheduling.json", {"drain": True})

    async def run_task(task, **kwargs):
        kwargs["admit_stage"]("round_1/B1")
        pytest.fail("Drained campaign admitted a new Actor stage")

    monkeypatch.setattr(three_b, "run_task", run_task)
    with pytest.raises(scheduling.DispatchDrained):
        await campaign.three_b_task(config, SimpleNamespace(), "task")
    state = read(tmp_path / "tasks/task/state.json")
    assert state["status"] == "failed" and state["error_type"] == "DispatchDrained"
    assert not (tmp_path / "tasks/task/round_1/B1/stage.json").exists()


@pytest.mark.asyncio
async def test_campaign_batch_override_reaches_measurement_and_actor_review(tmp_path, monkeypatch):
    from experiments.claw_eval.opencode import campaign
    from reward_engine.core.outcome import three_b
    from tests.support.claw_outcome_feedback_arms import PACK, TASK

    config = {"output": str(tmp_path), "batch": {"concurrency": 3, "deadline_s": 1800},
              "identity": "synthetic", "image": "synthetic", "rounds": 1, "actor_timeout_s": 1800}
    before = deepcopy(config)
    write(tmp_path / "scheduling.json", {"batch_limit": 8})
    monkeypatch.setattr(campaign, "public_task", lambda *_: (None, TASK))
    monkeypatch.setattr(campaign, "registry", lambda *_: None)
    reviews = []

    async def sample(task_id, slot, **kwargs):
        reviews.append(kwargs["review"])
        return {"status": "completed"}

    async def run_task(task, **kwargs):
        assert kwargs["policy"].concurrency == 8
        assert kwargs["policy"].deadline_s == 1800
        await kwargs["sample"]("B1", 1, "synthetic guidance", PACK)
        assert reviews[0].policy is kwargs["policy"]
        return {"status": "completed", "completed_rounds": 1}

    monkeypatch.setattr(three_b, "run_task", run_task)
    await campaign.three_b_task(config, SimpleNamespace(sample=sample), "task")
    assert config == before and len(reviews) == 1
