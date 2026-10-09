"""Bounded concurrent tasks with isolated state and an explicit canary gate."""

import asyncio
import os
import time


async def run_tasks(config, state, names, *, one_task, check_space, record):
    limit = config.get("task_concurrency", 1)
    if type(limit) is not int or not 1 <= limit <= 3:
        raise ValueError("Task concurrency must be an integer from 1 to 3")
    gate = asyncio.Semaphore(limit)
    active = set()

    def save():
        state.update(current_tasks=sorted(active), current_task=next(iter(sorted(active)), None),
                     finished=len(state["tasks"]), updated_at=time.time())
        record(state)

    async def one(name):
        if state["tasks"].get(name, {}).get("status") == "completed" and state["tasks"][name].get("scored") == 9:
            return
        async with gate:
            check_space(config)
            state.setdefault("started_at", time.time())
            state.update(status="running", pid=os.getpid())
            active.add(name)
            save()
            print(f"Starting {name}", flush=True)
            try:
                state["tasks"][name] = await one_task(config, name)
            except Exception as error:
                state["tasks"][name] = {"status": "failed", "error_type": type(error).__name__, "reason": str(error)}
            finally:
                active.discard(name)
                save()
            print(f"Finished {name}: {state['tasks'].get(name)}", flush=True)

    pending = list(names)
    if pending and not config.get("validated_canary"):
        first = pending.pop(0)
        await one(first)
        row = state["tasks"].get(first, {})
        if pending and (row.get("status") != "completed" or row.get("scored") != 9):
            return
    jobs = [asyncio.create_task(one(name)) for name in pending]
    try:
        await asyncio.gather(*jobs)
    finally:
        for job in jobs:
            if not job.done():
                job.cancel()
        await asyncio.gather(*jobs, return_exceptions=True)
