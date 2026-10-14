"""Real loop/measure/batch scheduling with offline checkers and shared slots."""

import asyncio
import json
from collections import Counter
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from reward_engine.core.batch import BatchPolicy
from reward_engine.core.outcome import loop
from reward_engine.core.outcome.harness import Registry
from tests.support.claw_outcome_feedback_arms import PACK, TASK


@pytest.fixture
def scenario(monkeypatch):
    real_measure = loop.measure

    def make(*, limit=None, rounds=1, policy=None, check=None, checks=2):
        pack = deepcopy(PACK)
        pack["tests"] = [
            {**pack["tests"][0], "test_id": f"check_{i}", "checker": "offline.json"}
            for i in range(checks)
        ]
        monkeypatch.setattr(
            loop.jury,
            "propose",
            AsyncMock(return_value={"criteria": pack["criteria"], "gaps": []}),
        )
        monkeypatch.setattr(loop.builder, "build", AsyncMock(return_value=pack))
        monkeypatch.setattr(
            loop.hypotheses,
            "propose",
            AsyncMock(return_value={"status": "keep", "method": ""}),
        )
        monkeypatch.setattr(
            loop.jury, "rank", AsyncMock(return_value={"status": "unavailable"})
        )
        state = SimpleNamespace(
            records={}, entered=[], finished=[], checks=[], active=0, peak=0
        )
        policy = policy or BatchPolicy(deadline_s=5)

        async def checker(test, context, evidence):
            ref = test["arguments"]["ref"]
            value = json.loads(evidence[ref])
            name = value["candidate"]
            state.checks.append((name, test["test_id"]))
            if check:
                await check(name, test["test_id"])
            return {"status": "pass", "reason": "Parsed JSON", "evidence_refs": [ref]}

        registry = Registry()
        registry.register(
            "offline.json", checker, "Parse offline JSON after optional queueing"
        )

        async def measured(pack, registry, context, evidence, **kwargs):
            assert kwargs["policy"] is policy
            assert kwargs["test_ids"] is None
            name = json.loads(evidence["product_1:artifact:final_response"])[
                "candidate"
            ]
            state.entered.append(name)
            state.active += 1
            state.peak = max(state.peak, state.active)
            try:
                result = await real_measure(pack, registry, context, evidence, **kwargs)
                state.finished.append((name, result))
                return result
            finally:
                state.active -= 1

        monkeypatch.setattr(loop, "measure", measured)

        def candidate(name):
            return {
                "candidate_id": name,
                "artifact": {"final_response": json.dumps({"candidate": name})},
            }

        async def sample(arm, number, *_):
            return {"status": "completed", "candidate": candidate(f"{number}{arm}")}

        async def forbidden(_):
            pytest.fail("Scheduling controls must not call providers")

        state.run = lambda: loop.run_task(
            TASK,
            [candidate(f"seed_{i}") for i in range(3)],
            roles=lambda *_: forbidden,
            registry=registry,
            sample=sample,
            rounds=rounds,
            full_audit=True,
            policy=policy,
            continuation_policy="fixed_rounds",
            record=lambda name, value: state.records.update({name: deepcopy(value)}),
            **(
                {"measurement_candidate_concurrency": limit}
                if limit is not None
                else {}
            ),
        )
        return state

    return make


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [None, 1])
async def test_shared_queue_exhausts_overlapping_batches_but_admission_preserves_checks(
    scenario, limit
):
    slots = asyncio.Semaphore(2)

    async def check(name, _):
        if name.startswith("seed_"):
            async with slots:
                await asyncio.sleep(0.2)

    # Each candidate needs 0.2s of the same two slots. The third simultaneous
    # batch cannot complete by 0.5s; admitted batches each retain that full clock.
    state = scenario(
        limit=limit, check=check, policy=BatchPolicy(concurrency=2, deadline_s=0.5)
    )
    await state.run()
    rows = state.records["round_1/construction_checks"]
    assert len(rows) == 3
    assert all(row["execution_summary"]["submitted_checks"] == 2 for row in rows)
    if limit is None:
        assert state.peak == 3
        assert any(
            item["status"] == "timeout"
            for row in rows
            for item in row["batch"]["items"]
        )
        assert any(result.get("error") for row in rows for result in row["results"])
        assert "measurement_candidate_concurrency" not in state.records["loop_policy"]
        assert not any("construction_candidates/" in key for key in state.records)
    else:
        assert state.peak == 1
        assert all(row["batch"]["complete"] for row in rows)
        for index, row in enumerate(rows, 1):
            assert state.records[f"round_1/construction_candidates/{index}"] == row
        assert all(
            result["status"] == "pass" for row in rows for result in row["results"]
        )
        assert state.records["loop_policy"]["measurement_candidate_concurrency"] == 1
    assert Counter(state.checks) == Counter(
        {(name, f"check_{i}"): 1 for name in state.entered for i in range(2)}
    )
    assert slots._value == 2


@pytest.mark.asyncio
async def test_three_round_pools_keep_all_18_checks_order_and_single_measurement(
    scenario,
):
    state = scenario(limit=1, rounds=3, checks=18)
    result = await state.run()
    assert result["completed_rounds"] == 3
    pool, expected = [f"seed_{i}" for i in range(3)], []
    for number, size in enumerate((3, 6, 9), 1):
        rows = state.records[f"round_{number}/construction_checks"]
        assert len(rows) == size
        assert all(row["execution_summary"]["submitted_checks"] == 18 for row in rows)
        assert all(row["execution_summary"]["reused_checks"] == 0 for row in rows)
        assert all(row["batch"]["complete"] for row in rows)
        assert all(
            result["status"] == "pass" for row in rows for result in row["results"]
        )
        for index, row in enumerate(rows, 1):
            assert (
                state.records[f"round_{number}/construction_candidates/{index}"] == row
            )
        expected.extend(pool)
        delivered = [f"{number}{arm}" for arm in "ABC"]
        expected.extend(delivered)
        pool.extend(delivered)
    assert state.entered == expected
    assert [name for name, _ in state.finished] == expected
    assert Counter(state.checks) == Counter(
        (name, f"check_{i}") for name in expected for i in range(18)
    )
    assert state.peak == 1


@pytest.mark.asyncio
async def test_explicit_limit_two_and_out_of_order_completion_preserve_receipt_order(
    scenario,
):
    second_finished = asyncio.Event()

    async def check(name, _):
        if name == "seed_0":
            await second_finished.wait()
            await asyncio.sleep(0.01)
        elif name == "seed_1":
            second_finished.set()

    state = scenario(limit=2, check=check)
    await state.run()
    assert state.peak == 2
    assert state.finished[0][0] == "seed_1"
    rows = state.records["round_1/construction_checks"]
    by_name = dict(state.finished)
    assert rows == [by_name[f"seed_{i}"] for i in range(3)]
    assert [
        state.records[f"round_1/construction_candidates/{i}"] for i in range(1, 4)
    ] == rows


@pytest.mark.asyncio
async def test_active_candidate_timeout_is_not_retried_and_next_candidate_retains_policy(
    scenario,
):
    cancelled = []

    async def check(name, test_id):
        if name == "seed_0":
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.append(test_id)
        elif name == "seed_1":
            retained = state.records["round_1/construction_candidates/1"]
            assert not retained["batch"]["complete"]
            assert "round_1/construction_checks" not in state.records

    state = scenario(limit=1, check=check, policy=BatchPolicy(deadline_s=0.1))
    await state.run()
    rows = state.records["round_1/construction_checks"]
    assert [item["status"] for item in rows[0]["batch"]["items"]] == [
        "timeout",
        "timeout",
    ]
    assert all(
        row["status"] == "unknown" and row["error"] for row in rows[0]["results"]
    )
    assert all(row["batch"]["complete"] for row in rows[1:])
    assert all(
        result["status"] == "pass" for row in rows[1:] for result in row["results"]
    )
    assert state.entered.count("seed_0") == 1
    assert state.records["round_1/construction_candidates/1"] == rows[0]
    assert sorted(cancelled) == ["check_0", "check_1"]


@pytest.mark.asyncio
async def test_parent_cancellation_cancels_active_checks_without_admitting_waiters(
    scenario,
):
    entered, cancelled = asyncio.Event(), []

    async def check(name, test_id):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append((name, test_id))

    state = scenario(limit=1, check=check)
    task = asyncio.create_task(state.run())
    await asyncio.wait_for(entered.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert state.entered == ["seed_0"]
    assert state.active == 0
    assert sorted(cancelled) == [("seed_0", "check_0"), ("seed_0", "check_1")]
    assert "round_1/construction_checks" not in state.records
    assert not any("construction_candidates/" in key for key in state.records)


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, "1", 129, [], {}])
async def test_invalid_core_limit_fails_before_measurement(scenario, value):
    state = scenario(limit=value)
    with pytest.raises(ValueError, match="measurement_candidate_concurrency"):
        await state.run()
    assert state.entered == []
