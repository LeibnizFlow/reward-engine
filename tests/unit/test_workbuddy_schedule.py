"""Protocol-aware task admission and completed-stage scheduling."""

import asyncio

import pytest

from experiments.workbuddy.schedule import run_tasks


@pytest.mark.asyncio
async def test_single_round_three_b_runs_all_tasks_concurrently_and_skips_completed():
    config = {
        "generation_protocol": "three_b_fixed_baselines_v1",
        "rounds": 1,
        "task_concurrency": 3,
    }
    state = {"tasks": {}}
    entered = set()
    ready = asyncio.Event()

    async def one_task(config, name):
        entered.add(name)
        if len(entered) == 3:
            ready.set()
        await asyncio.wait_for(ready.wait(), timeout=1)
        return {"status": "completed", "scored": 6}

    async def run():
        await run_tasks(
            config, state, ["a", "b", "c"], one_task=one_task,
            check_space=lambda _: None, record=lambda _: None,
        )

    await run()
    assert entered == {"a", "b", "c"}
    assert all(row["status"] == "completed" for row in state["tasks"].values())
    entered.clear()
    await run()
    assert not entered


@pytest.mark.asyncio
async def test_legacy_failed_canary_still_blocks_following_tasks():
    calls = []
    state = {"tasks": {}}

    async def one_task(config, name):
        calls.append(name)
        return {"status": "failed", "scored": 0}

    await run_tasks(
        {"task_concurrency": 3}, state, ["a", "b"], one_task=one_task,
        check_space=lambda _: None, record=lambda _: None,
    )
    assert calls == ["a"]
