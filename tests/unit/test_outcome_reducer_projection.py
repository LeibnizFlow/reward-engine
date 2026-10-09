"""Compact requests preserve canonical requirements and bounded evidence access."""

from copy import deepcopy

import pytest

from reward_engine.core.outcome.context import RequestBudgetExceeded, ask, build_context
from reward_engine.core.outcome.reducer_projection import (
    TEXT_KEY,
    byte_size,
    context_view,
    expand_texts,
    share_texts,
)


def test_shared_text_is_exact_but_identity_and_scope_stay_separate():
    text = "Only apply this threshold after the exact effective date. " * 50
    request = {
        "context": {"task": {"prompt": "Original Task"}, "products": []},
        "independent_expectations": [{"scope": "site A", "text": text}],
        "proposal_expectations": [{"scope": "site B", "text": text}],
        "requested_evidence": {"ref": text},
        "response": {"text": "literal"},
    }
    saved = deepcopy(request)
    compact = share_texts(request)
    assert byte_size(compact) < byte_size(request)
    assert expand_texts(compact) == saved and request == saved
    assert compact["independent_expectations"][0]["scope"] == "site A"
    assert compact["proposal_expectations"][0]["scope"] == "site B"


def test_small_or_reserved_literal_inputs_do_not_grow_or_change():
    for row in [
        {"x": "short"},
        {"x": {TEXT_KEY: "literal input"}, "y": "long" * 500},
        {
            "response": {TEXT_KEY: "literal schema"},
            "a": "long" * 500,
            "b": "long" * 500,
        },
    ]:
        assert share_texts(row) == row


def test_existing_memory_metadata_is_not_overwritten():
    memory = {
        "details_ref": "existing-ref",
        "value": {"method": "body" * 2000, "method_record": "existing metadata"},
    }
    context, evidence = build_context(
        {"task_id": "x", "prompt": "Task"}, memories=[memory]
    )
    view, _ = context_view(context, evidence)
    assert view["memories"] == context["memories"]


def test_adapter_revision_advice_remains_literal_after_sizing():
    advice = {"refresh_reason": "Reconsider a missing conditional obligation. " * 500}
    request = {
        "reward_revision_advice": advice,
        "copied_advice": advice,
        "context": {"a": "Duplicate fact " * 400, "b": "Duplicate fact " * 400},
    }
    compact = share_texts(request)
    assert byte_size(compact) < byte_size(request)
    assert compact["reward_revision_advice"] == advice
    assert expand_texts(compact) == request


def test_memory_method_is_readable_while_observations_and_limits_stay_inline():
    task = {"task_id": "x", "prompt": "Task " * 400}
    memory = {
        "memory_id": "m1",
        "key": task["prompt"] + "\nUnresolved condition",
        "relevance": 0.9,
        "value": {
            "method": "Historical method " * 3000,
            "observed_changes": [{"before": "pass", "after": "fail"}],
            "status": "observed_not_causal_or_official",
            "limitation": "Not causal",
            "applicability": "Scope A",
            "unknown_future_field": "Preserve me",
        },
    }
    context, evidence = build_context(task, memories=[memory])
    saved = deepcopy((context, evidence))
    view, refs = context_view(context, evidence)
    thin = view["memories"][0]
    assert refs[thin["details_ref"]] == memory
    assert thin["value"]["observed_changes"] == memory["value"]["observed_changes"]
    assert thin["value"]["status"] == memory["value"]["status"]
    assert thin["value"]["unknown_future_field"] == "Preserve me"
    assert task["prompt"] + thin["key"]["suffix"] == memory["key"]
    assert view["task"] == context["task"]
    assert byte_size(view) < byte_size(context) / 5
    assert (context, evidence) == saved


@pytest.mark.asyncio
async def test_projection_covers_reads_and_contract_retry_without_extra_model_calls():
    context, evidence = build_context({"task_id": "x", "prompt": "Public Task"})
    body = "A fully preserved supplied fact. " * 200
    evidence["source:x"] = body
    context["available_evidence_refs"] = list(evidence)
    context["memories"] = [
        {"memory_id": "m", "value": {"method": body * 3, "status": "unverified"}}
    ]
    calls, metrics = [], []

    async def model(request):
        request = expand_texts(request)
        calls.append(request)
        if len(calls) == 1:
            return {"need_evidence": ["source:x"]}
        assert request["requested_evidence"]["source:x"] == body
        return {"ok": len(calls) == 3}

    def validate(row):
        if not row.get("ok"):
            raise ValueError("Need a valid result")
        return row

    result = await ask(
        model,
        "outcome_jury_reduce",
        context,
        evidence,
        "Reduce",
        {"ok": "bool"},
        reducer_projection=True,
        record_projection=metrics.append,
        validate=validate,
        max_contract_retries=1,
        max_evidence_rounds=1,
    )
    assert result == {"ok": True}
    assert len(calls) == len(metrics) == 3
    assert [r["evidence_stage"] for r in metrics] == [0, 1, 1]
    assert metrics[-1]["retry"] == 1


@pytest.mark.asyncio
async def test_projection_does_not_call_llm_compressor_or_truncate_unique_obligations():
    context, evidence = build_context(
        {"task_id": "x", "prompt": "Unique Task obligation " * 5000}
    )

    async def model(_):
        pytest.fail("Oversize unique Task must not be sent")

    async def compress(*_):
        pytest.fail("Reducer projection must not add a compression model call")

    model.bounded_stage = True
    model.compress_request = compress
    with pytest.raises(RequestBudgetExceeded):
        await ask(
            model,
            "outcome_jury_reduce",
            context,
            evidence,
            "Reduce",
            {},
            reducer_projection=True,
            max_bytes=10000,
        )


@pytest.mark.asyncio
async def test_legacy_ask_keeps_the_original_request_contract():
    context, evidence = build_context({"task_id": "x", "prompt": "Original Task"})

    async def model(request):
        assert "reducer_projection" not in request and "shared_texts" not in request
        assert request["context"] == context
        return {"done": True}

    assert await ask(model, "outcome_jury_reduce", context, evidence, "Reduce", {}) == {
        "done": True
    }


@pytest.mark.asyncio
async def test_reducer_can_read_the_exact_indexed_memory_with_existing_evidence_protocol():
    memory = {"value": {"method": "Historical method " * 3000, "status": "not causal"}}
    context, evidence = build_context(
        {"task_id": "x", "prompt": "Original Task"}, memories=[memory]
    )
    seen = []

    async def model(request):
        request = expand_texts(request)
        seen.append(request)
        ref = request["context"]["memories"][0]["details_ref"]
        if len(seen) == 1:
            return {"need_evidence": [ref]}
        assert request["requested_evidence"][ref] == memory
        return {"done": True}

    assert await ask(
        model,
        "outcome_jury_reduce",
        context,
        evidence,
        "Reduce",
        {},
        reducer_projection=True,
    ) == {"done": True}
    assert len(seen) == 2
