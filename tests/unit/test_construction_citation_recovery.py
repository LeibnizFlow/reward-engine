"""A citation correction reads its source within the existing operation budget."""

from copy import deepcopy

import pytest

from reward_engine.core.outcome.context import ResponseContractError, ask, build_context
from reward_engine.core.outcome.jury import SCHEMA
from tests.unit.test_authored_json import criterion, enable


@pytest.mark.asyncio
async def test_invalid_quote_opens_its_unread_source_before_the_correction():
    context, evidence = build_context(
        {"task_id": "synthetic", "prompt": "Return JSON"},
        [{"artifact": {"final_response": "Delivery"},
          "source_documents": [{"text": "Context. " * 100 + "Exact supporting passage."}]}],
    )
    ref = "product_1:source:0"
    raw = {"criteria": [criterion()]}
    raw["criteria"][0]["clauses"][0].update(
        use="quality_judgment", evidence_refs=["task", ref],
        source_quotes=[{"ref": ref, "quote": "Invented supporting passage."}],
    )
    original = deepcopy((context, evidence, raw))
    requests = []

    @enable
    async def model(request):
        requests.append(deepcopy(request))
        if len(requests) == 1:
            return deepcopy(raw)
        assert request.get("requested_evidence", {}).get(ref) == evidence[ref]
        assert request["evidence_budget"] == {
            "used_rounds": 1, "remaining_rounds": 1,
            "remaining_new_refs": 5, "final_answer_required": False,
        }
        assert request["response_recovery"]["rejected_response"] == raw
        corrected = deepcopy(raw)
        corrected["criteria"][0]["clauses"][0]["source_quotes"][0]["quote"] = "Exact supporting passage."
        return corrected

    model.role_input_projection = "canonical_sources_v1"
    result = await ask(model, "outcome_expectations", context, evidence, "", SCHEMA,
                       verify_used_previews=True, validate=lambda row: row)
    assert len(requests) == 2
    assert result["criteria"][0]["clauses"][0]["source_quotes"][0]["quote"] == "Exact supporting passage."
    assert (context, evidence, raw) == original


@pytest.mark.asyncio
@pytest.mark.parametrize("prior_reads", [1, 2])
async def test_citation_recovery_respects_remaining_and_closed_read_budgets(prior_reads):
    context, evidence = build_context(
        {"task_id": "synthetic", "prompt": "Return JSON"},
        [{"artifact": {"final_response": "Delivery"},
          "source_documents": [{"text": f"Source {i}. " * 100} for i in range(3)]}],
    )
    ref = "product_1:source:2"
    raw = {"criteria": [criterion()]}
    raw["criteria"][0]["clauses"][0].update(
        use="quality_judgment", evidence_refs=[ref], source_quotes=[{"ref": ref, "quote": "Invented quotation"}],
    )
    calls = []

    @enable
    async def model(request):
        calls.append(deepcopy(request))
        if len(calls) <= prior_reads:
            return {"need_evidence": [f"product_1:source:{len(calls) - 1}"]}
        if len(calls) == prior_reads + 1:
            return deepcopy(raw)
        budget = request["evidence_budget"]
        assert budget["used_rounds"] == 2 and budget["remaining_rounds"] == 0
        assert budget["final_answer_required"] is True
        assert (ref in request["requested_evidence"]) == (prior_reads == 1)
        return {"criteria": [criterion()]}

    model.role_input_projection = "canonical_sources_v1"
    await ask(model, "outcome_expectations", context, evidence, "", SCHEMA,
              verify_used_previews=True, validate=lambda row: row)
    assert len(calls) == prior_reads + 2


@pytest.mark.asyncio
async def test_unfixed_quote_still_fails_and_does_not_reopen_the_same_source():
    context, evidence = build_context(
        {"task_id": "synthetic", "prompt": "Return JSON"},
        [{"artifact": {"final_response": "Delivery"},
          "source_documents": [{"text": "Source context. " * 100}]}],
    )
    ref = "product_1:source:0"
    calls = []

    @enable
    async def model(request):
        calls.append(deepcopy(request))
        if len(calls) > 1:
            assert request["requested_evidence"] == {ref: evidence[ref]}
            assert request["evidence_budget"]["used_rounds"] == 1
        raw = {"criteria": [criterion()]}
        raw["criteria"][0]["clauses"][0].update(
            use="quality_judgment", evidence_refs=[ref],
            source_quotes=[{"ref": ref, "quote": f"Invented quotation {len(calls)}"}],
        )
        return raw

    model.role_input_projection = "canonical_sources_v1"
    with pytest.raises(ResponseContractError) as error:
        await ask(model, "outcome_expectations", context, evidence, "", SCHEMA,
                  verify_used_previews=True, validate=lambda row: row)
    assert len(calls) == 4 and error.value.operation_budget["contract_repairs"] == 3
    assert any(row["code"] == "QUOTE_EXACT" for row in error.value.details["issues"])


@pytest.mark.asyncio
@pytest.mark.parametrize("source_count", [1, 2])
async def test_automatic_citation_read_keeps_attachment_and_distinct_source_caps(source_count):
    context, evidence = build_context(
        {"task_id": "synthetic", "prompt": "Return JSON"},
        [{"artifact": {"final_response": "Delivery"},
          "source_documents": [{"text": f"Source {i}. " * 6000} for i in range(source_count)]}],
    )
    refs = [f"product_1:source:{i}" for i in range(source_count)]
    raw = {"criteria": [criterion()]}
    raw["criteria"][0]["clauses"][0].update(
        use="quality_judgment", evidence_refs=refs,
        source_quotes=[{"ref": ref, "quote": "Invented quotation"} for ref in refs],
    )
    calls = []

    @enable
    async def model(request):
        calls.append(deepcopy(request))
        if len(calls) == 1:
            return deepcopy(raw)
        assert list(request["requested_evidence"]) == refs[:1]
        attached = request["requested_evidence"][refs[0]]
        assert attached["excerpt_only"] is True
        assert len(attached["preview_json"]) == 24 * 1024
        assert request["evidence_budget"] == {
            "used_rounds": 1, "remaining_rounds": 0,
            "remaining_new_refs": 0, "final_answer_required": True,
        }
        if source_count == 2:
            assert request["evidence_read_selection"]["deferred"] == refs[1:]
        return {"criteria": [criterion()]}

    model.role_input_projection = "canonical_sources_v1"
    model.evidence_expansion_limit = 1
    await ask(model, "outcome_expectations", context, evidence, "", SCHEMA,
              verify_used_previews=True, validate=lambda row: row)
    assert len(calls) == 2
