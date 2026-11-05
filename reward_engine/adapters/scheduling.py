"""Live operational admission limits, without changing candidate identities."""

import asyncio
import os
import time
from pathlib import Path

from reward_engine.adapters.workbuddy.io import digest, read, write


class DispatchDrained(RuntimeError):
    """Unstarted tasks and stages remain untouched after an operator drain."""


def control(config):
    path = Path(config["output"]) / "scheduling.json" if "output" in config else None
    value = read(path) if path and path.exists() else {}
    if not isinstance(value, dict) or set(value) - {"task_limit", "batch_limit", "adaptive", "drain"}:
        raise ValueError("Unknown scheduling control fields")
    result = {"task_limit": config.get("task_concurrency", 1),
              "adaptive": False, "drain": False, **value}
    if type(result["task_limit"]) is not int or not 1 <= result["task_limit"] <= 128:
        raise ValueError("Task limit must be an integer from 1 to 128")
    if "batch_limit" in result and (
        type(result["batch_limit"]) is not int or not 1 <= result["batch_limit"] <= 128
    ):
        raise ValueError("Batch limit must be an integer from 1 to 128")
    if any(type(result[key]) is not bool for key in ("adaptive", "drain")):
        raise ValueError("Scheduling flags must be boolean")
    return result


def batch_policy(config):
    """Snapshot operational concurrency for a new task; retain its other limits."""
    from dataclasses import replace

    from reward_engine.core.batch import BatchPolicy

    policy = BatchPolicy(**config["batch"])
    return replace(policy, concurrency=control(config).get("batch_limit", policy.concurrency))


def check_stage_admission(config, task_id, stage):
    """Finish live stages, then stop before starting new work during a drain."""
    if control(config)["drain"]:
        error = DispatchDrained(f"Stage dispatch drained before {task_id}/{stage}")
        error.stage = stage
        raise error


def resource_limit(requested):
    """Reserve host memory and budget 4 GiB/container plus 2 GiB/task services."""
    available = next(int(line.split()[1]) * 1024
                     for line in Path("/proc/meminfo").read_text().splitlines()
                     if line.startswith("MemAvailable:"))
    limit = min(requested, max(0, (available - 20 * 1024**3) // (6 * 1024**3)),
                max(1, (os.cpu_count() or 2) // 2))
    for name, threshold in (("memory", 10), ("io", 30)):
        path = Path("/proc/pressure") / name
        if path.exists():
            full = next((line for line in path.read_text().splitlines()
                         if line.startswith("full ")), "")
            averages = dict(part.split("=", 1) for part in full.split()[1:])
            if float(averages.get("avg10", 0)) >= threshold:
                limit = min(limit, max(1, limit // 2))
    return limit


async def run_bounded(config, names, execute, *, poll_s=1):
    """Resize between admissions; never cancel live work to lower a limit."""
    from reward_engine.adapters.failure_policy import FailureWindow
    from reward_engine.adapters.task_admission import check_space

    breaker = FailureWindow(config)
    pending, active, results = list(names), {}, []
    root = Path(config["output"]) if "output" in config else None
    observed, backoff, recover_at = None, 128, 0.0
    error, task_failures = None, []

    def record(name, result):
        breaker.terminal(name, result)
        if isinstance(result, BaseException) or (
            isinstance(result, dict) and result.get("status") in {"failed", "partial"}
        ):
            value = result if isinstance(result, dict) else {}
            task_failures.append({
                "task": name, "stage": value.get("stage", getattr(result, "stage", "task")),
                "error_type": value.get("error_type", type(result).__name__),
                "reason": value.get("reason", str(result)),
            })
    try:
        while pending or active:
            settings = control(config)
            identity = digest(settings)
            if root and identity != observed:
                write(root / "scheduling_events" / (str(time.time_ns()) + ".json"),
                      {"settings": settings, "identity": identity, "at": time.time()})
                observed = identity
            limit = settings["task_limit"]
            if settings["adaptive"]:
                if time.monotonic() >= recover_at:
                    backoff = min(128, backoff + 1)
                    recover_at = time.monotonic() + 60
                limit = min(resource_limit(limit), backoff)
                if breaker.enabled and not breaker.isolated:
                    limit = min(limit, breaker.state.get("admission_limit", 4))
            if settings["drain"] or breaker.paused:
                limit = 0
            resource_error = None
            while pending and len(active) < limit:
                if breaker.isolated:
                    try:
                        check_space(config)
                    except OSError as failure:
                        resource_error = str(failure)
                        limit = len(active)
                        break
                name = pending.pop(0)
                active[asyncio.create_task(execute(name))] = name
            if root:
                write(root / "scheduling_state.json", {
                    "active_tasks": list(active.values()), "pending_tasks": len(pending),
                    "requested_limit": settings["task_limit"], "effective_limit": limit,
                    "drain": settings["drain"], "failure_paused": breaker.paused, "at": time.time(),
                    "resource_paused": resource_error is not None, "resource_error": resource_error,
                })
            if not active:
                if (settings["drain"] or breaker.paused) and pending:
                    raise DispatchDrained("Task dispatch drained; queued stages were not started")
                if pending:
                    await asyncio.sleep(poll_s)
                continue
            done, _ = await asyncio.wait(active, timeout=poll_s,
                                         return_when=asyncio.FIRST_COMPLETED)
            for future in done:
                name = active.pop(future)
                try:
                    result = future.result()
                except Exception as failure:
                    result = failure
                record(name, result)
                results.append(result)
                if not breaker.isolated and (isinstance(result, Exception) or (
                    isinstance(result, dict) and result.get("status") == "failed"
                )):
                    backoff = max(1, limit // 2)
                    recover_at = time.monotonic() + 60
    except Exception as failure:
        # Invalid controls also drain admitted work before reporting the error.
        error = failure
        drained = await asyncio.gather(*active, return_exceptions=True)
        for name, result in zip(active.values(), drained, strict=True):
            record(name, result)
        results.extend(drained)
    finally:
        for future in active:
            if not future.done():
                future.cancel()
        await asyncio.gather(*active, return_exceptions=True)
        if root:
            write(root / "scheduling_state.json", {
                "active_tasks": [], "pending_tasks": len(pending),
                "status": "paused" if breaker.paused else "failed" if error or task_failures else "completed",
                "failure_paused": breaker.paused, "failure_reason": breaker.state.get("reason"),
                "task_failures": task_failures,
                "error": str(error) if error else None, "at": time.time(),
            })
    if error:
        raise error
    return results
