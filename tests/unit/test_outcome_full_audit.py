"""Full audits measure delivered artifacts; targeted Actor evidence stays intact."""

from copy import deepcopy
from unittest.mock import AsyncMock

import pytest

from reward_engine.core.outcome import loop
from reward_engine.core.outcome.context import build_context
from reward_engine.core.outcome.harness import defaults, measure
from tests.support.claw_outcome_feedback_arms import PACK


@pytest.mark.asyncio
@pytest.mark.parametrize("full_audit", [False, True])
@pytest.mark.parametrize("review_tests", [[], ["parse"], ["parse", "payload"]])
@pytest.mark.parametrize("effective_scope", [False, True])
async def test_full_audit_rechecks_all_delivered_tests_and_preserves_actor_review(
    monkeypatch, full_audit, review_tests, effective_scope
):
    task = {"task_id": "json-task", "prompt": "Return JSON response and payload"}
    pack = deepcopy(PACK)
    pack["task_prompt"] = task["prompt"]
    pack["tests"].append(
        {
            "test_id": "payload",
            "criterion_id": "syntax",
            "checker": "json.parse",
            "question": "Does the delivered payload parse as JSON?",
            "arguments": {"ref": "product_1:artifact:payload"},
        }
    )
    current_task, current_pack = deepcopy(task), deepcopy(pack)
    if effective_scope:
        current_task["prompt"] += "; include the user's revised fields"
        current_pack.update(
            pack_id="current-user-scope", task_prompt=current_task["prompt"]
        )

    async def forbidden_model(_):
        pytest.fail("This regression uses only deterministic JSON checks")

    registry = defaults(forbidden_model)
    # Stub model-authored construction, never the controller or measurement harness.
    monkeypatch.setattr(
        loop.jury,
        "propose",
        AsyncMock(return_value={"criteria": deepcopy(pack["criteria"]), "gaps": []}),
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
    draft = {"artifact": {"final_response": "{}", "payload": "[]"}}
    context, evidence = build_context(current_task, [draft])
    review = await measure(
        current_pack, registry, context, evidence, test_ids=review_tests
    )
    assert all(row["status"] == "pass" for row in review["results"])
    review.update(
        review_mode="targeted",
        deferred_tests=[
            test for test in ("parse", "payload") if test not in review_tests
        ],
        review_diagnostics={"reason": "Actor review budget exhausted", "attempts": 1},
        delivery_review={
            "status": "needs_verification",
            "reason": "Retain pending checks",
        },
    )
    retained_review = deepcopy(review)
    delivered = {"artifact": {"final_response": "broken response", "payload": "["}}
    records = {}

    async def sample(arm, number, prompt, supplied_pack):
        assert number == 1
        row = {"status": "completed", "candidate": deepcopy(draft)}
        if arm == "B":
            assert supplied_pack == pack
            row.update(candidate=deepcopy(delivered), final_measurement=review)
            if effective_scope:
                row.update(effective_task=current_task, effective_pack=current_pack)
        return row

    result = await loop.run_task(
        task,
        [deepcopy(draft) for _ in range(3)],
        rounds=1,
        roles=lambda *_: forbidden_model,
        registry=registry,
        sample=sample,
        full_audit=full_audit,
        continuation_policy="fixed_rounds",
        record=lambda name, value: records.update({name: deepcopy(value)}),
    )
    assert result["status"] == "completed" and result["completed_rounds"] == 1
    audit = records["round_1/B/outcome_estimate"]
    assert audit["pack_id"] == current_pack["pack_id"]
    if full_audit:
        assert {row["test_id"]: row["status"] for row in audit["results"]} == {
            "parse": "fail",
            "payload": "fail",
        }
        assert audit["execution_summary"]["selected_checks"] == 2
        assert audit["execution_summary"]["submitted_checks"] == 2
        assert audit["execution_summary"]["reused_checks"] == 0
        assert audit["ready_under_estimator"] is False
        for arm in "AC":
            assert {
                row["test_id"]: row["status"]
                for row in records[f"round_1/{arm}/outcome_estimate"]["results"]
            } == {"parse": "pass", "payload": "pass"}
        # Subsequent selection/learning receives the full delivered-product checks.
        checks = next(
            row
            for row in records["round_1/check_sources"]
            if row["product_id"] == "product_5"
        )
        assert {row["test_id"]: row["status"] for row in checks["checks"]} == {
            "parse": "fail",
            "payload": "fail",
        }
    else:
        assert audit == retained_review

    # The replacement estimate must not erase or overwrite partial review diagnostics.
    assert review == retained_review
    assert records["round_1/B/delivery"]["final_measurement"] == retained_review
    assert result["rounds"][0]["arms"]["B"]["final_measurement"] == retained_review
    assert records["round_1/B/delivery"]["candidate"] == delivered
