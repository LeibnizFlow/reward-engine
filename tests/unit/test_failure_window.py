"""Admission pauses persist by revision and active tasks finish normally."""

import pytest

from reward_engine.adapters.failure_policy import FailureWindow
from reward_engine.adapters.scheduling import DispatchDrained, run_bounded
from reward_engine.adapters.workbuddy.io import read


def config(tmp_path, revision="one"):
    return {
        "output": str(tmp_path),
        "implementation_revision": str(tmp_path / revision),
        "identity": revision,
        "failure_policy": "pause_repeated_v1",
        "task_concurrency": 2,
    }


@pytest.mark.asyncio
async def test_repeated_errors_drain_and_remain_paused_on_restart(tmp_path):
    cfg = config(tmp_path)
    calls = []

    async def execute(name):
        calls.append(name)
        return {"status": "failed", "stage": "map", "error_code": "missing_source"}

    with pytest.raises(DispatchDrained):
        await run_bounded(cfg, range(10), execute, poll_s=0.01)
    assert len(calls) == 4
    state = read(tmp_path / "scheduling_state.json")
    assert state["active_tasks"] == [] and state["pending_tasks"] == 6
    with pytest.raises(DispatchDrained):
        await run_bounded(cfg, range(10), execute, poll_s=0.01)
    assert len(calls) == 4
    assert not FailureWindow(config(tmp_path, "two")).paused


def test_five_distinct_failures_and_ten_terminal_window(tmp_path):
    window = FailureWindow(config(tmp_path))
    for n in range(4):
        window.terminal(n, {"status": "failed", "stage": str(n), "error_code": str(n)})
    assert not window.paused
    window.terminal(4, {"status": "failed", "stage": "new", "error_code": "new"})
    assert window.paused


@pytest.mark.asyncio
async def test_terminal_dispatch_reports_failures_below_pause_threshold(tmp_path):
    async def execute(name):
        return {"status": "failed", "stage": "measurement", "error_type": "MissingClaim", "reason": "Incomplete claim"}

    await run_bounded(config(tmp_path), ["one"], execute, poll_s=0.01)
    state = read(tmp_path / "scheduling_state.json")
    assert state["status"] == "failed"
    assert state["active_tasks"] == [] and state["pending_tasks"] == 0
    assert state["task_failures"][0]["reason"] == "Incomplete claim"
    assert not state["failure_paused"]
