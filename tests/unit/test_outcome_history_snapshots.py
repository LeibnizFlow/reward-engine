"""Round receipts must not follow a shared native review's growing memory."""

import json
from copy import deepcopy
from unittest.mock import AsyncMock

import pytest

from reward_engine.core.outcome import loop
from reward_engine.core.outcome.harness import defaults
from reward_engine.core.outcome.memory import Memory
from reward_engine.core.outcome.runner import DeliveryReview
from tests.support.claw_outcome_feedback_arms import PACK, TASK


def gap(key, number):
    return {
        "gap_id": key,
        "gap": f"JSON requirement {key}",
        "expected": "Valid JSON",
        "verification": "Parse the delivered JSON",
        "status": "observed",
        "observed": f"Observation in round {number}",
        "product_ids": ["product_1"],
        "expected_basis": "task",
        "expected_refs": ["task"],
        "evidence_refs": ["product_1:artifact:final_response"],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_third_round", [False, True])
async def test_controller_history_stays_at_recorded_round_with_live_shared_memory(
    monkeypatch, fail_third_round
):
    memory = Memory()
    records, serialized, native_rows, reviews = {}, {}, {}, {}

    def record(name, value):
        # Exercise both a retaining callback and an immediate JSON writer. The
        # controller owns snapshot boundaries; callbacks need not repair aliases.
        records[name] = value
        serialized[name] = json.loads(json.dumps(value))

    async def forbidden_model(_):
        pytest.fail("This regression must never call a model")

    registry = defaults(forbidden_model)
    round_number = 0

    async def propose(*args, **kwargs):
        nonlocal round_number
        round_number += 1
        if fail_third_round and round_number == 3:
            raise RuntimeError("Scripted Jury unavailable")
        keys = (
            ["shared", "one", "two", "three"]
            if round_number == 1
            else ["shared", f"new-{round_number}-a", f"new-{round_number}-b"]
        )
        return {
            "criteria": deepcopy(PACK["criteria"]),
            "gaps": [gap(key, round_number) for key in keys],
        }

    # Only authored construction/ranking is scripted. The round controller,
    # Memory, DeliveryReview, and deterministic measurement harness are real.
    monkeypatch.setattr(loop.jury, "propose", propose)
    monkeypatch.setattr(loop.builder, "build", AsyncMock(return_value=deepcopy(PACK)))
    monkeypatch.setattr(
        loop.hypotheses,
        "propose",
        AsyncMock(return_value={"status": "keep", "method": ""}),
    )

    async def rank(context, *args, **kwargs):
        return {
            "status": "compared",
            "groups": [[p["product_id"] for p in context["products"]]],
        }

    monkeypatch.setattr(loop.jury, "rank", rank)

    async def sample(arm, number, prompt, pack):
        candidate = {
            "candidate_id": f"round-{number}-{arm}",
            "artifact": {"final_response": "{}"},
        }
        row = {"status": "completed", "candidate": candidate}
        if arm == "B":
            review = DeliveryReview(
                TASK,
                pack,
                registry,
                memory=memory,
                audit_mode="full",
                source_prefix=f"round_{number}/actors/B/checks",
            )
            if number > 1:
                feedback = await review({"artifact": {"final_response": "{bad"}})
                assert feedback and "Invalid JSON" in feedback
            assert await review(candidate) is None
            assert review.status == "satisfied_under_estimator"
            reviews[number] = review
            # Match the native Claw producer: its delivered row exposes the
            # same live memory used by subsequent Jury observations/reviews.
            row.update(
                memory=review.memory.rows,
                final_measurement=review.history[-1],
                review_history=review.history,
                review_status=review.status,
            )
            native_rows[number] = row
        return row

    result = await loop.run_task(
        TASK,
        [{"artifact": {"final_response": "{}"}} for _ in range(3)],
        rounds=3,
        roles=lambda *_: forbidden_model,
        registry=registry,
        sample=sample,
        memory=memory,
        full_audit=True,
        continuation_policy="fixed_rounds",
        record=record,
    )
    completed = 2 if fail_third_round else 3
    assert result["completed_rounds"] == completed
    assert result["status"] == ("partial" if fail_third_round else "completed")
    assert len(memory.rows) == (7 if fail_third_round else 10)
    assert native_rows[1]["memory"] is memory.rows  # Reproduce the real alias.
    assert len(serialized["round_1/B/delivery"]["memory"]) == 4
    # This is the saved-run defect: 4 in round_1/B/delivery.json, but 10 in
    # result.private.json's round-1 B history before the snapshot repair.
    assert len(result["rounds"][0]["arms"]["B"]["memory"]) == 4

    for number in range(1, completed + 1):
        prefix = f"round_{number}"
        expected_count = 4 + 3 * (number - 1)
        summary = result["rounds"][number - 1]
        assert summary == serialized[f"{prefix}/summary.private"]
        assert summary["arms"]["B"] == serialized[f"{prefix}/B/delivery"]
        assert len(summary["arms"]["B"]["memory"]) == expected_count
        assert len(records[f"{prefix}/memory"]) == expected_count
        assert records[f"{prefix}/checkpoint"]["memory"] == records[f"{prefix}/memory"]
        # The existing shared gap gets updated in place, without list growth.
        assert (
            summary["arms"]["B"]["memory"][0]["value"]["source"]
            == f"{prefix}/jury.json"
        )
        assert (
            summary["arms"]["B"]["memory"][0]["value"]["observed"]
            == f"Observation in round {number}"
        )
        if number > 1:
            # Live memory still reaches later rounds and records real fail/pass
            # checks from same-controller revisions; copying must not reset it.
            assert records[f"{prefix}/context"]["memories"]
            change = summary["arms"]["B"]["memory"][-1]["value"]["observed_changes"][0]
            assert (change["before"], change["after"]) == ("fail", "pass")

    if fail_third_round:
        recovery = records["round_3/recovery_state"]
        assert recovery["completed_rounds"] == result["rounds"]
        assert recovery["memory"] == result["memory"]
    # Retained callback values must also remain equal to their on-disk image,
    # including contexts later enriched with gaps and accumulating check lists.
    assert records == serialized
    before = deepcopy(result)
    memory.rows[0]["value"]["evidence_refs"].append("later observation")
    memory.observe_jury(TASK, {"gaps": [gap("later", 4)]}, source="later/jury.json")
    native_rows[1]["candidate"]["artifact"]["final_response"] = "changed later"
    reviews[1].history[0]["results"][0]["reason"] = "changed later"
    assert result == before
    assert records == serialized
    # A caller mutating its returned report must not mutate the recorded image.
    result["rounds"][0]["arms"]["B"]["memory"][0]["value"]["source"] = "caller edit"
    assert records == serialized
