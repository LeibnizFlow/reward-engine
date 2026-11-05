"""Synthetic alias collisions through the actual bounded repair boundary."""

from copy import deepcopy

import pytest

from reward_engine.core.outcome.context import ResponseContractError, ask
from reward_engine.core.outcome.jury_drafts import (
    apply_part_decisions,
    compact_selection_inputs,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("correct", [True, False])
async def test_only_canonical_choices_are_shown_but_model_must_fix_duplicate(correct):
    inputs = {"independent_expectations": [{"expectation_id": "s", "subrequirements": [
        {"part_id": "s/0", "text": "Preserve the source file"},
        {"part_id": "s/1", "text": "Preserve the source file"}]}], "proposal_expectations": []}
    compact = compact_selection_inputs(inputs)
    assert compact["selection_aliases"] == {"p0": ["p1"]}
    raw = {"criteria": [{"id": "c"}], "selection_protocol": "exact-alias-v1",
           "part_decisions": [{"parts": ["p0", "p1"], "action": "retain", "criterion_id": "c"}]}
    saved = deepcopy((inputs, compact, raw))
    calls = []

    class Role:
        protocol_edits = True

        async def __call__(self, request):
            calls.append(deepcopy(request))
            assert "selection_aliases" not in request
            assert request["selection_catalog"] == ["p0"]
            if len(calls) == 1:
                return deepcopy(raw)
            details = request["response_recovery"]["details"]
            assert "selection_aliases" not in details
            assert details["allowed_selection_ids"] == ["p0"]
            assert request["response_recovery"]["rejected_response"] == raw
            assert request["selection_reference_conflicts"] == [{
                "selection_id": "p0", "occurrences": [
                    {"path": ["part_decisions", 0, "parts", 0], "provided_id": "p0"},
                    {"path": ["part_decisions", 0, "parts", 1], "provided_id": "p1"}]}]
            return {"base_sha256": request["protocol_repair"]["base_sha256"], "edits": [{
                "op": "replace", "path": ["part_decisions", 0, "parts"],
                "value": ["p0"] if correct else ["p0", "p1"],
                "reason": "Model explicitly chooses the canonical group once"}]}

    async def run():
        return await ask(Role(), "outcome_jury_reduce", {}, {"task": "Preserve source"},
                         "Select requirements", {}, extra=compact,
                         validate=lambda value: apply_part_decisions(value, inputs, {"task": "Preserve source"}),
                         max_contract_retries=1)

    if correct:
        result = await run()
        parts = result["coverage"][0]["subrequirements"]
        assert {part["part_id"] for part in parts} == {"s/0", "s/1"}
        assert all(part["status"] == "mapped" for part in parts)
    else:
        with pytest.raises(ResponseContractError, match="duplicate"):
            await run()
    assert len(calls) == 2
    assert (inputs, compact, raw) == saved


@pytest.mark.asyncio
async def test_repair_refreshes_lossless_text_sharing_within_same_byte_limit():
    from reward_engine.core.outcome.construction import fingerprint
    from reward_engine.core.outcome.protocol_repair import ProtocolRepairCall
    from reward_engine.core.outcome.reducer_projection import byte_size, expand_texts, share_texts

    text = "Preserve this exact synthetic requirement. " * 40
    base = {"criteria": [{"id": "c", "standard": text}], "part_decisions": []}
    details = {"field": "part_decisions", "examples": [{"text": text} for _ in range(20)]}
    request = share_texts({"mode": "outcome_jury_reduce", "selection_aliases": {"p0": ["p1"]},
        "independent_expectations": [{"subrequirements": [{"selection_id": "p0", "text": text}]}],
        "response_recovery": {"rejected_response": base, "details": details}})
    saved = deepcopy(request)

    async def model(current):
        assert byte_size(current) <= 16000
        semantic = expand_texts(current)
        assert semantic["response_recovery"]["rejected_response"] == base
        assert semantic["response_recovery"]["details"] == details
        assert current["protocol_repair"]["base_sha256"] == fingerprint(base)
        return {"base_sha256": current["protocol_repair"]["base_sha256"], "edits": [{
            "op": "replace", "path": ["part_decisions"], "value": [],
            "reason": "Synthetic explicit edit; production caller still validates coverage"}]}

    assert await ProtocolRepairCall(model, max_bytes=16000)(request) == base
    assert request == saved
