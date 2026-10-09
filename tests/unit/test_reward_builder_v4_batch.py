import asyncio

import pytest

from reward_engine.core.batch import BatchPolicy, collect_batch


@pytest.mark.parametrize("settings", [
    {"concurrency": 0}, {"concurrency": True}, {"concurrency": 1.5},
    {"deadline_s": 0}, {"deadline_s": -1}, {"deadline_s": float("inf")},
    {"deadline_s": float("nan")}, {"deadline_s": True}, {"group_concurrency": 0},
])
def test_bad_batch_settings_rejected_before_work(settings):
    with pytest.raises(ValueError):
        BatchPolicy(**settings)


@pytest.mark.asyncio
async def test_completed_jobs_with_false_or_none_results_are_not_transport_failures():
    async def result(value):
        return value

    values = (False, None, [], {"status": "unknown"})
    batch = await collect_batch({str(index): lambda row=row: result(row)
        for index, row in enumerate(values)}, BatchPolicy(concurrency=2, deadline_s=1))
    assert batch.complete and batch.require_all() == values
    assert batch.summary()["succeeded"] == 4
    # Reduce, not the scheduler, interprets empty proposals or unknown judgments.


@pytest.mark.asyncio
async def test_parallel_work_finishes_once_in_declared_order():
    ready = asyncio.Event()
    started, finished = [], []

    async def work(index):
        started.append(index)
        if len(started) == 3:
            ready.set()
        await ready.wait()
        await asyncio.sleep((2 - index) * 0.005)
        finished.append(index)
        return index

    batch = await collect_batch({str(i): lambda i=i: work(i) for i in range(3)}, BatchPolicy(3, 1))
    assert batch.require_all() == (0, 1, 2)
    assert finished == [2, 1, 0]
    assert batch.summary()["succeeded"] == 3


@pytest.mark.asyncio
async def test_one_failure_does_not_discard_or_cancel_healthy_peer():
    async def fail():
        raise RuntimeError("controlled provider failure")

    async def healthy():
        await asyncio.sleep(0.01)
        return {"observed": True}

    batch = await collect_batch({"bad": fail, "good": healthy})
    assert not batch.complete
    assert batch.items[1].value == {"observed": True}
    assert batch.summary()["items"][0] == {
        "job_id": "bad", "status": "error", "error_type": "RuntimeError"}
    with pytest.raises(RuntimeError, match="controlled"):
        batch.require_all()


@pytest.mark.asyncio
async def test_deadline_includes_queued_jobs_and_preserves_completed_values():
    cancelled = asyncio.Event()

    async def quick():
        return "real evidence"

    async def slow():
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    async def never_started():
        pytest.fail("queued job must not start after deadline")

    batch = await collect_batch({"quick": quick, "slow": slow, "queued": never_started}, BatchPolicy(1, .03))
    assert [item.status for item in batch.items] == ["ok", "timeout", "timeout"]
    assert batch.items[0].value == "real evidence"
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_external_cancel_cancels_workers_and_is_not_an_unknown_vote():
    started, cancelled = asyncio.Event(), asyncio.Event()

    async def work():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    task = asyncio.create_task(collect_batch({"work": work}))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_late_callback_cannot_turn_deadline_into_success():
    released = asyncio.Event()

    async def ignores_cancel():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await released.wait()
            return "late"

    batch = await asyncio.wait_for(
        collect_batch({"late": ignores_cancel}, BatchPolicy(1, .02)), timeout=.5)
    assert batch.items[0].status == "timeout"
    released.set()
    await asyncio.sleep(0)
    assert batch.items[0].value is None


@pytest.mark.asyncio
async def test_empty_batch_and_invalid_job_definition():
    assert (await collect_batch({})).require_all() == ()
    with pytest.raises(ValueError):
        await collect_batch({"": lambda: None})


@pytest.mark.asyncio
async def test_stalled_group_does_not_consume_all_slots_while_waiting_for_itself():
    async def stalled():
        await asyncio.Event().wait()

    async def healthy():
        return "real independent answer"

    jobs = {**{f"bad-{i}": stalled for i in range(4)}, "good-1": healthy, "good-2": healthy}
    groups = {key: "stalled" if key.startswith("bad") else key for key in jobs}
    result = await collect_batch(jobs, BatchPolicy(3, .03, 2), groups=groups)
    assert [item.status for item in result.items] == ["timeout"] * 4 + ["ok", "ok"]
    with pytest.raises(ValueError, match="every scheduled"):
        await collect_batch(jobs, groups={"bad-0": "stalled"})
