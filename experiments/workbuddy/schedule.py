"""Bounded concurrent tasks with isolated state and an explicit canary gate."""

import asyncio
import os
import time
from contextlib import nullcontext

from reward_engine.adapters.scheduling import run_bounded
from reward_engine.adapters.task_admission import TaskAdmission


async def run_tasks(config, state, names, *, one_task, check_space, record):
    three_b = config.get("generation_protocol") == "three_b_fixed_baselines_v1"
    expected_scores = config.get("rounds", 3) * 6 if three_b else 9
    limit = config.get("task_concurrency", 1)
    maximum = 128 if three_b else 3
    if type(limit) is not int or not 1 <= limit <= maximum:
        raise ValueError(f"Task concurrency must be an integer from 1 to {maximum}")
    gate = nullcontext() if three_b else asyncio.Semaphore(limit)
    admission = TaskAdmission(config, check_space)
    state.pop("dispatch_blocked", None)
    active = set()

    def save():
        state.update(
            current_tasks=sorted(active),
            current_task=next(iter(sorted(active)), None),
            finished=len(state["tasks"]),
            updated_at=time.time(),
        )
        record(state)

    async def one(name):
        if (
            state["tasks"].get(name, {}).get("status") == "completed"
            and state["tasks"][name].get("scored") == expected_scores
        ):
            return
        async with gate:
            try:
                admission.admit()
            except OSError as error:
                state["dispatch_blocked"] = {
                    "error_type": type(error).__name__, "reason": str(error),
                }
                save()
                return
            state.setdefault("started_at", time.time())
            state.update(status="running", pid=os.getpid())
            active.add(name)
            save()
            print(f"Starting {name}", flush=True)
            try:
                state["tasks"][name] = await one_task(config, name)
            except Exception as error:
                state["tasks"][name] = {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "reason": str(error),
                }
            finally:
                active.discard(name)
                save()
            print(f"Finished {name}: {state['tasks'].get(name)}", flush=True)
            return state["tasks"].get(name)

    pending = list(names)
    if pending and not three_b and not config.get("validated_canary"):
        first = pending.pop(0)
        await one(first)
        row = state["tasks"].get(first, {})
        if pending and (
            row.get("status") != "completed" or row.get("scored") != expected_scores
        ):
            return
    if three_b:
        await run_bounded(config, pending, one)
        return
    jobs = [asyncio.create_task(one(name)) for name in pending]
    try:
        await asyncio.gather(*jobs)
    finally:
        for job in jobs:
            if not job.done():
                job.cancel()
        await asyncio.gather(*jobs, return_exceptions=True)
