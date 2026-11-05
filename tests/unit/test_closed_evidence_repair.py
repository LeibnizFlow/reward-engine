"""Closed evidence is a response contract, not permission for another read."""

from copy import deepcopy

import pytest

from reward_engine.core.outcome.context import (
    EvidenceReadLimit,
    ResponseContractError,
    ask,
    build_context,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("corrected", [True, False])
async def test_closed_final_citations_use_existing_contract_repair(corrected):
    context, evidence = build_context({"task_id": "synthetic", "prompt": "Inspect records"}, [
        {"source_documents": [{"source": str(i), "text": "x" * 1000} for i in range(3)]}])
    calls = []
    validated = []

    class Role:
        bounded_stage = True

        async def __call__(self, request):
            calls.append(deepcopy(request))
            if len(calls) <= 3:
                return {"need_evidence": [f"product_1:source:{len(calls) - 1}"]}
            if len(calls) == 4 or not corrected:
                return {"status": "pass", "evidence_refs": ["product_1:source:2"]}
            return {"status": "unknown", "evidence_refs": ["task"]}

    def validate(row):
        validated.append(deepcopy(row))
        return row

    args = (Role(), "outcome_measure", context, evidence, "Inspect records", {"status": "verdict"})
    if corrected:
        result = await ask(*args, verify_used_previews=True, validate=validate)
        assert result["status"] == "unknown"
        assert validated == [result]  # Unsupported final is never admitted by the consumer.
    else:
        with pytest.raises(EvidenceReadLimit):
            await ask(*args, verify_used_previews=True, validate=validate)
        assert not validated
    assert len(calls) == 5
    assert calls[-1]["requested_evidence"] == calls[-2]["requested_evidence"]
    assert "product_1:source:2" not in calls[-1]["requested_evidence"]
    diagnostic = calls[-1]["response_recovery"]["details"]
    assert diagnostic["unread_evidence_refs"] == ["product_1:source:2"]
    assert calls[-1]["evidence_access_closed"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("request_tools", [False, True])
async def test_measurement_closed_correction_cannot_execute_more_tools(request_tools):
    from reward_engine.core.outcome.harness import defaults
    from tests.unit.test_claw_outcome_measurement_tools import case

    _, _, _, test = case()
    context, evidence = build_context({"task_id": "synthetic", "prompt": "Inspect records"}, [
        {"source_documents": [{"source": str(i), "text": "x" * 1000} for i in range(3)]}])
    calls = []

    async def model(request):
        calls.append(deepcopy(request))
        if len(calls) <= 3:
            return {"need_evidence": [f"product_1:source:{len(calls) - 1}"]}
        if len(calls) == 4:
            return {"status": "pass", "evidence_refs": ["product_1:source:2"]}
        assert request["measurement_budget"]["remaining_tool_calls"] == 0
        if request_tools:
            return {"need_tools": [{"tool": "calculate", "arguments": {"expression": "2+3"}}]}
        return {"status": "unknown", "reason": "Necessary record unread", "gap": "Verify source",
                "evidence_refs": ["task"], "claims": []}

    model.bounded_stage = True
    result = await defaults(model).run(test, context, evidence)
    assert len(calls) == 5
    assert result["status"] == "unknown"
    assert bool(result.get("error")) == request_tools
    assert result["execution_usage"]["tool_calls"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("change_supported", [False, True])
async def test_closed_correction_preserves_unaffected_claims(change_supported):
    context, evidence = build_context({"task_id": "synthetic", "prompt": "Inspect records"}, [
        {"source_documents": [{"source": str(i), "text": "x" * 1000} for i in range(3)]}])
    original = {"status": "pass", "claims": [
        {"claim": "First fact", "judgment": "supported", "evidence_refs": ["product_1:source:0"]},
        {"claim": "Second fact", "judgment": "supported", "evidence_refs": ["product_1:source:2"]}]}
    calls = []

    async def model(request):
        calls.append(request)
        if len(calls) <= 3:
            return {"need_evidence": [f"product_1:source:{len(calls) - 1}"]}
        if len(calls) == 4:
            return deepcopy(original)
        result = deepcopy(original)
        result["status"] = "unknown"
        result["claims"][1].update(judgment="unknown", evidence_refs=[])
        if change_supported:
            result["claims"][0]["judgment"] = "unknown"
        return result

    model.bounded_stage = True
    args = (model, "outcome_measure", context, evidence, "Inspect", {"status": "verdict"})
    if change_supported:
        with pytest.raises(ResponseContractError, match="unaffected"):
            await ask(*args, verify_used_previews=True)
    else:
        result = await ask(*args, verify_used_previews=True)
        assert result["status"] == "unknown" and result["claims"][0] == original["claims"][0]
    assert len(calls) == 5
