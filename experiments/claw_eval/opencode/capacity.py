"""Measured campaign-wide concurrency tuning; never cancel admitted work."""

import asyncio
import time
from pathlib import Path

import httpx

from reward_engine.adapters.workbuddy.io import auth, read, write


def retryable_task_result(result):
    failure = result.get("construction_failure") or {}
    return (
        result.get("status") in {"failed", "partial"}
        and result.get("completed_rounds") == 0
        and (
            result.get("stop_reason") in {"jury_unavailable", "incomplete_arms"}
            or failure.get("error_type")
            in {"HTTPStatusError", "ReadTimeout", "TimeoutError"}
            or (
                failure.get("error_type") == "ValueError"
                and "Incomplete or mismatched role reply: finish=length"
                in failure.get("error", "")
            )
        )
    )


def retryable_task_error(error):
    if type(error).__name__ in {
        "HTTPStatusError",
        "ReadTimeout",
        "TimeoutError",
    }:
        return True
    if isinstance(error, ValueError) and str(error) == (
        "Cannot reuse a changed or failed Actor slot"
    ):
        return True
    return isinstance(error, RuntimeError) and str(error).startswith("docker exited ")


def transport_window(root, since):
    rows = []
    for section in ("transport", "official.private"):
        for path in (root / section).glob("*/*/receipt.json"):
            row = read(path)
            if (
                row.get("queued_at", 0) >= since
                and row.get("status") in ("completed", "failed")
                # Structured concurrency cancels redundant sibling requests after
                # a task has enough usable results. That is not transport pressure.
                and row.get("error_type") != "CancelledError"
            ):
                rows.append(row)
    rate_limit = any(
        any(e.get("status") == 429 for e in r.get("http_errors", []))
        or any(e.get("reason") == "rate_limit_error" for e in r.get("retries", []))
        for r in rows
    )
    return {
        "requests": len(rows),
        "failed": sum(r["status"] == "failed" for r in rows),
        "rate_limited": rate_limit,
        "completed": sum(r["status"] == "completed" for r in rows),
    }


async def run_remaining(config, runtime, task_run, report):
    root = Path(config["output"])
    pending = []
    for task_id in config["task_ids"]:
        if task_id in config["canaries"]:
            continue
        state_path = root / "tasks" / task_id / "state.json"
        result_path = root / "tasks" / task_id / "result.private.json"
        state = read(state_path) if state_path.exists() else {}
        result = read(result_path) if result_path.exists() else {}
        if state.get("status") == "completed" and result.get("status") == "completed":
            continue
        pending.append(task_id)
    active = {}
    retries = {}
    task_limit = config["task_concurrency"]
    max_task_limit = config.get("max_task_concurrency", task_limit)
    if type(max_task_limit) is not int or max_task_limit < task_limit:
        raise ValueError("max_task_concurrency must be at least task_concurrency")
    limit = config["model_concurrency"]
    stable = limit
    stage_start = time.time()
    previous_rate = None
    flat = 0
    tuning = True
    next_task_growth = time.time() + 60
    async with httpx.AsyncClient(trust_env=False, timeout=20) as client:

        async def resize(value, reason):
            nonlocal limit, stage_start
            response = await client.post(
                config["gateway_local"] + "/control/concurrency",
                headers={"Authorization": "Bearer " + auth(config)},
                json={"global_limit": value},
            )
            response.raise_for_status()
            write(
                root / "capacity_events" / (str(time.time_ns()) + ".json"),
                {
                    "from": limit,
                    "to": value,
                    "reason": reason,
                    "task_limit": task_limit,
                    "at": time.time(),
                },
            )
            limit = value
            stage_start = time.time()

        while pending or active:
            resource_error = None
            try:
                runtime.check_resources()
            except RuntimeError as error:
                resource_error = str(error)
            while pending and len(active) < task_limit and not resource_error:
                tid = pending.pop(0)
                active[asyncio.create_task(task_run(config, runtime, tid))] = tid
            done = [future for future in active if future.done()]
            for future in done:
                tid = active.pop(future)
                try:
                    result = future.result()
                    retryable = retryable_task_result(result)
                except BaseException as error:
                    retryable = retryable_task_error(error)
                    write(
                        root / "task_failures" / (tid + ".json"),
                        {"error_type": type(error).__name__, "reason": str(error)},
                    )
                if retryable and retries.get(tid, 0) < 2:
                    retries[tid] = retries.get(tid, 0) + 1
                    pending.append(tid)
            metrics = transport_window(root, stage_start)
            elapsed = time.time() - stage_start
            failure_rate = metrics["failed"] / max(1, metrics["requests"])
            if metrics["rate_limited"] or (
                metrics["requests"] >= 10 and failure_rate > 0.02
            ):
                stable = min(stable, max(1, limit - 4))
                await resize(stable, "rate_limit_or_transport_failures")
                tuning = False
            elif tuning and elapsed >= 900 and metrics["requests"] >= 50:
                rate = metrics["completed"] / elapsed
                flat = flat + 1 if previous_rate and rate < previous_rate * 1.05 else 0
                if flat >= 2:
                    await resize(max(8, limit - 8), "two_steps_without_throughput_gain")
                    tuning = False
                elif not resource_error and limit < config["max_model_concurrency"]:
                    stable = limit
                    previous_rate = rate
                    await resize(
                        min(limit + 4, config["max_model_concurrency"]),
                        "stable_observation_window",
                    )
                else:
                    tuning = False
            health = (await client.get(config["gateway_local"] + "/health")).json()
            if (
                not resource_error
                and tuning
                and pending
                and len(active) >= task_limit
                and task_limit < max_task_limit
                and health["inflight"] < limit
                and elapsed >= 60
                and time.time() >= next_task_growth
            ):
                # At most one additional task each minute; 3 Actor slots/task must fit RAM.
                available_kib = next(
                    int(line.split()[1])
                    for line in Path("/proc/meminfo").read_text().splitlines()
                    if line.startswith("MemAvailable:")
                )
                if (task_limit + 1) * 12 * 1024**2 < available_kib - 20 * 1024**2:
                    task_limit = min(task_limit + 1, max_task_limit)
                    next_task_growth = time.time() + 60
            write(
                root / "monitor.json",
                {
                    "at": time.time(),
                    "active_tasks": list(active.values()),
                    "pending_tasks": len(pending),
                    "task_limit": task_limit,
                    "model_limit": limit,
                    "tuning": tuning,
                    "transport": metrics,
                    "resource_pause": resource_error,
                    "gateway": health,
                },
            )
            report(config)
            if pending and not active and resource_error:
                raise RuntimeError(resource_error)
            if active or pending:
                await asyncio.sleep(30)
