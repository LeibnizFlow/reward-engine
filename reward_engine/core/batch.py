"""One bounded Map for every module; the caller owns its domain-specific Reduce.

Jobs are independent async functions, not live Actor mutations. No retries,
winner-based early stopping, provider SDK, or semantic Judge lives here.
"""

from __future__ import annotations

import asyncio
import math
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping


@dataclass(frozen=True)
class BatchPolicy:
    concurrency: int = 3
    deadline_s: float = 240.0
    group_concurrency: int = 2

    def __post_init__(self):
        if any(
            type(value) is not int or value < 1
            for value in (self.concurrency, self.group_concurrency)
        ):
            raise ValueError("batch concurrency limits must be positive integers")
        if (
            isinstance(self.deadline_s, bool)
            or not isinstance(self.deadline_s, (int, float))
            or not math.isfinite(self.deadline_s)
            or self.deadline_s <= 0
        ):
            raise ValueError("batch deadline_s must be finite and positive")


@dataclass(frozen=True)
class BatchItem:
    job_id: str
    status: str  # ok | error | timeout | cancelled
    value: Any = None
    error: BaseException | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class BatchResult:
    items: tuple[BatchItem, ...]

    @property
    def complete(self) -> bool:
        return all(item.status == "ok" for item in self.items)

    def summary(self) -> dict[str, Any]:
        """Small, JSON-safe receipt. No prompts, secrets, or invented votes."""

        def capacity_details(error):
            details = getattr(error, "details", None)
            if not isinstance(details, dict) or details.get("reason") != "request_size":
                return {}
            # Keep only the structured size diagnostic, never arbitrary exception text.
            return {
                "error_details": {
                    k: deepcopy(details[k])
                    for k in (
                        "reason",
                        "stage",
                        "mode",
                        "request_bytes",
                        "limit_bytes",
                        "api_called",
                        "fields_bytes",
                    )
                    if k in details
                }
            }

        return {
            "complete": self.complete,
            "total": len(self.items),
            "succeeded": sum(item.status == "ok" for item in self.items),
            "items": [
                {
                    "job_id": item.job_id,
                    "status": item.status,
                    "error_type": type(item.error).__name__ if item.error else None,
                    **capacity_details(item.error),
                }
                for item in self.items
            ],
        }

    def require_all(self) -> tuple[Any, ...]:
        """Strict boundaries may reject a partial batch without losing its receipt."""
        for item in self.items:
            if item.status != "ok":
                raise item.error or RuntimeError(
                    f"Batch job {item.job_id}: {item.status}"
                )
        return tuple(item.value for item in self.items)


async def collect_batch(
    jobs: Mapping[str, Callable[[], Awaitable[Any]]],
    policy: BatchPolicy | None = None,
    *,
    groups: Mapping[str, str] | None = None,
) -> BatchResult:
    """Collect every scheduled slot once, preserving input order and failures.

    Deadline includes semaphore queueing. Cancellation propagates to all jobs.
    Workers must cooperate with cancellation and must not publish shared state:
    only Reduce may adopt returned values. A cancellation-resistant callback is
    discarded at the boundary; this function cannot terminate a blocking thread.
    """
    policy = policy or BatchPolicy()
    scheduled = tuple(jobs.items())
    if any(
        not isinstance(key, str) or not key.strip() or not callable(job)
        for key, job in scheduled
    ):
        raise ValueError("batch jobs require nonempty IDs and async callables")
    if not scheduled:
        return BatchResult(())
    groups = dict(groups) if groups is not None else None
    if groups is not None and (
        set(groups) != set(jobs)
        or any(
            not isinstance(group, str) or not group.strip() for group in groups.values()
        )
    ):
        raise ValueError("batch groups must name a group for every scheduled job")
    semaphore = asyncio.Semaphore(policy.concurrency)
    group_limits = (
        {
            group: asyncio.Semaphore(policy.group_concurrency)
            for group in set(groups.values())
        }
        if groups is not None
        else {}
    )

    async def invoke(key, job):
        async def run():
            async with semaphore:
                return BatchItem(key, "ok", await job())

        try:
            if groups is None:
                return await run()
            # Wait for a member's slot BEFORE taking a global slot: several
            # queued calls to one stalled backend must not starve healthy peers.
            async with group_limits[groups[key]]:
                return await run()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            return BatchItem(key, "error", error=error)

    tasks = {
        key: asyncio.create_task(invoke(key, job), name=f"batch:{key}")
        for key, job in scheduled
    }

    def consume(task):
        if not task.cancelled():
            task.exception()

    try:
        done, _ = await asyncio.wait(tuple(tasks.values()), timeout=policy.deadline_s)
        # Seal the batch before cancellation: a late return cannot become success.
        result = BatchResult(
            tuple(
                task.result()
                if task in done and not task.cancelled()
                else BatchItem(key, "cancelled", error=RuntimeError("worker cancelled"))
                if task in done
                else BatchItem(
                    key, "timeout", error=TimeoutError(f"batch deadline: {key}")
                )
                for key, task in tasks.items()
            )
        )
        return result
    finally:
        for task in tasks.values():
            if not task.done():
                task.cancel()
            task.add_done_callback(consume)
        # Give cooperative transports a cancellation turn, without an unbounded
        # gather that can defeat the batch deadline if a callback swallows cancel.
        await asyncio.sleep(0)
