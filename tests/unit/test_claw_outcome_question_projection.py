"""Reduce prompt overhead without deleting questions, authority, or source access."""

import json
from copy import deepcopy

import pytest

from reward_engine.core.outcome.context import ask, build_context
from reward_engine.core.outcome.hypotheses import question_context


def inputs():
    context, evidence = build_context({"task_id": "generic", "prompt": "Explain the data"}, [
        {"artifact": {"final_response": "Observed answer", "public_consultation": {
            "earlier_exchanges": [{"speaker": "user", "text": "Use the corrected date"}]}}}])
    jury = {"criteria": [{"id": "accuracy"}], "gaps": [{"gap": "Observed discrepancy"}],
        "open_questions": [{"questions": [f"Question {i}: what is missing?"], "origin": "domain",
            "status": "unverified_proposal", "why": "Detailed rationale " * 50,
            "verification": "Concrete method " * 50, "evidence_refs": ["task"],
            "source_expectation_ids": [str(i)]} for i in range(40)]}
    return context, evidence, jury


def test_all_questions_preserved_with_lossless_details_and_no_input_mutation():
    context, evidence, jury = inputs()
    original = deepcopy((context, evidence, jury))
    view, refs, thin = question_context(context, evidence, jury)
    assert (context, evidence, jury) == original
    assert thin["criteria"] == jury["criteria"] and thin["gaps"] == jury["gaps"]
    assert view["task"] == context["task"] and view["products"] == context["products"]
    assert len(thin["open_questions"]) == 40
    for source, row in zip(jury["open_questions"], thin["open_questions"], strict=True):
        assert refs[row["details_ref"]] == source
        assert {k: v for k, v in row.items() if k != "details_ref"} == {
            k: v for k, v in source.items() if k not in {"why", "verification"}}
        assert row["details_ref"] in view["available_evidence_refs"]
    before = len(json.dumps({"context": context, "jury": jury}))
    after = len(json.dumps({"context": view, "jury": thin}))
    assert after < before / 2
    assert question_context(view, refs, thin) == (view, refs, thin)


@pytest.mark.asyncio
async def test_requested_detail_is_exact_and_uses_existing_followup():
    context, evidence, jury = question_context(*inputs())
    ref = jury["open_questions"][-1]["details_ref"]
    calls = []

    async def model(request):
        calls.append(request)
        if len(calls) == 1:
            assert len(request["jury"]["open_questions"]) == 40
            assert "Detailed rationale" not in str(request)
            return {"need_evidence": [ref]}
        assert request["requested_evidence"] == {ref: evidence[ref]}
        return {"evidence_refs": [ref], "status": "unknown"}

    result = await ask(model, "probe", context, evidence, "Inspect", {}, extra={"jury": jury},
                       verify_used_previews=True)
    assert len(calls) == 2 and result["status"] == "unknown"


def test_small_questions_stay_inline_and_conflicts_are_not_overwritten():
    context, evidence, jury = inputs()
    jury["open_questions"] = jury["open_questions"][:1]
    assert question_context(context, evidence, jury) == (context, evidence, jury)
    evidence["reference:outcome_question:0"] = {"existing": "different"}
    with pytest.raises(ValueError, match="conflicts"):
        question_context(context, evidence, jury, inline_bytes=1)
