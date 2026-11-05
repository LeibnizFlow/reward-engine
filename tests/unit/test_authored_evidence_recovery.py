"""Closed evidence access permits bounded complete correction, never more reads."""

from copy import deepcopy

import pytest

from reward_engine.core.outcome.authored_json import RECOVERY, VERSION
from reward_engine.core.outcome.completed_operations import CompletedOperations
from reward_engine.core.outcome.context import ask, build_context
from reward_engine.core.outcome.jury import validate_ranking


def enable(call):
    call.role_output_protocol, call.role_recovery_policy = VERSION, RECOVERY
    call.bounded_stage = True
    return call


def inputs():
    return build_context({"task_id": "synthetic", "prompt": "Compare the delivered outcomes"}, [
        {"artifact": {"final_response": "First delivery"}, "source_documents": [{"text": "one " * 300}, {"text": "unread " * 300}]},
        {"artifact": {"final_response": "Second delivery"}, "source_documents": [{"text": "two " * 300}]},
    ])


def ranking(first="product_1:source:0", second="product_2:source:0"):
    return {"groups": [["product_1"], ["product_2"]], "unknown": [], "reason": "Compared supplied outcomes",
            "observations": [{"product_id": "product_1", "reason": "First comparison", "evidence_refs": [first]},
                             {"product_id": "product_2", "reason": "Second comparison", "evidence_refs": [second]}]}


@pytest.mark.asyncio
@pytest.mark.parametrize("mistake", ["new_read", "unread_citation"])
async def test_closed_ranking_can_be_fully_corrected_without_another_read(mistake):
    context, evidence = inputs()
    calls = []

    @enable
    async def model(request):
        calls.append(deepcopy(request))
        if len(calls) == 1:
            return {"need_evidence": ["product_1:source:0"]}
        if len(calls) == 2:
            return {"need_evidence": ["product_2:source:0"]}
        assert request["evidence_budget"]["remaining_rounds"] == 0
        if len(calls) == 3:
            return {"need_evidence": ["product_1:source:1"]} if mistake == "new_read" else ranking("product_1:source:1")
        assert request["response_recovery"]["rejected_response"]
        final = ranking()
        final["groups"].reverse()
        return final

    result = await ask(model, "outcome_ranking_map", context, evidence, "Compare outcomes", ranking(),
                       validate=lambda r: validate_ranking(r, context, evidence), verify_used_previews=True)
    assert len(calls) == 4 and result["groups"] == [["product_2"], ["product_1"]]
    assert [r["evidence_budget"]["remaining_rounds"] for r in calls] == [2, 1, 0, 0]
    assert set(calls[-1]["requested_evidence"]) == {"product_1:source:0", "product_2:source:0"}


@pytest.mark.asyncio
async def test_successful_ranking_seat_revalidates_across_explicit_revision(tmp_path):
    context, evidence = inputs()
    calls, validations = [], []

    @enable
    async def model(request):
        calls.append(request)
        assert len(calls) == 1
        return ranking("product_1:artifact:final_response", "product_2:artifact:final_response")

    def validate(row):
        validations.append(row)
        return validate_ranking(row, context, evidence)

    model.completed_operations = CompletedOperations(tmp_path, {"config_identity": "old"})
    first = await ask(model, "outcome_ranking_map", context, evidence, "Compare outcomes", ranking(), validate=validate, verify_used_previews=True)
    model.completed_operations = CompletedOperations(tmp_path, {"config_identity": "new"}, ancestors=["old"])
    second = await ask(model, "outcome_ranking_map", context, evidence, "Compare outcomes", ranking(), validate=validate, verify_used_previews=True)
    assert first == second and len(calls) == 1 and len(validations) == 2
    assert len(list(tmp_path.glob("*.reuse.json"))) == 1
