
from tests.support.claw_outcome_context_roles import (
    SHARED,
    ask,
    build_context,
    definition,
    sample,
)
import asyncio
from copy import deepcopy

import pytest

from reward_engine.core.outcome import hypotheses, jury






def test_context_preserves_updates_scope_and_not_private_trace():
    context, evidence = sample()
    product = context["products"][0]
    assert [r["text"] for r in product["user_updates"]] == ["Do not send", "No. Save only."]
    assert product["artifact"]["earlier_public_deliveries"][0]["text"] == "Could we send?"
    assert "SECRET" not in str(context) + str(evidence)
    assert product["environment"]["draft"] is None


def test_evidence_expansion_is_bounded_and_explicit():
    context, evidence = sample()
    calls = []

    async def call(request):
        calls.append(request)
        return {"need_evidence": ["product_1:source:0"]} if len(calls) == 1 else {"ok": True}

    assert asyncio.run(ask(call, "test", context, evidence, "read", {})) == {"ok": True}
    assert calls[1]["requested_evidence"] == {"product_1:source:0": {"text": "source"}}

    async def bad(request):
        return {"need_evidence": ["hidden_grader"]}

    with pytest.raises(ValueError):
        asyncio.run(ask(bad, "test", context, evidence, "read", {}))


def test_adapter_budget_preserves_large_context_and_expansion_with_explicit_override():
    context, evidence = sample()
    context["memories"] = ["m" * 170000]
    evidence["product_1:source:0"] = {"text": "s" * 50000}

    class Call:
        request_budget_bytes = 240000

        def __init__(self):
            self.calls = []

        async def __call__(self, request):
            self.calls.append(request)
            return {"need_evidence": ["product_1:source:0"]} if len(self.calls) == 1 else {"ok": True}

    call = Call()
    assert asyncio.run(ask(call, "test", context, evidence, "read", {})) == {"ok": True}
    assert call.calls[1]["context"] == context
    assert call.calls[1]["requested_evidence"]["product_1:source:0"] == evidence["product_1:source:0"]
    with pytest.raises(ValueError, match="configured budget"):
        asyncio.run(ask(call, "test", context, evidence, "read", {}, max_bytes=160000))
    assert len(call.calls) == 2
    context["memories"] = ["m" * 240000]
    with pytest.raises(ValueError, match="240000 bytes"):
        asyncio.run(ask(call, "test", context, evidence, "read", {}))
    assert len(call.calls) == 2


def test_office_ranking_can_read_multiple_products_with_one_bounded_expansion():
    from reward_engine.adapters.workbuddy.models import Role

    context, evidence = build_context({"task_id": "office", "prompt": "Compare actual files"},
        [{"source_documents": [{"text": f"product {i} file {j}"} for j in range(2)]} for i in range(10)])
    refs = [ref for ref in evidence if ":source:" in ref]

    class OfficeCall(Role):
        def __init__(self, repeat=False):
            self.calls, self.repeat = [], repeat

        async def __call__(self, request):
            self.calls.append(request)
            if len(self.calls) == 1 or self.repeat:
                return {"need_evidence": refs}
            assert request["requested_evidence"] == {ref: evidence[ref] for ref in refs}
            return {"observed_files": len(request["requested_evidence"])}

    call = OfficeCall()
    assert asyncio.run(ask(call, "outcome_ranking_map", context, evidence, "Compare", {})) == {"observed_files": 20}
    assert len(call.calls) == 2 and "at most 64 supplied refs" in call.calls[0]["evidence_access"]
    repeat = OfficeCall(repeat=True)
    with pytest.raises(ValueError, match="allowance exhausted"):
        asyncio.run(ask(repeat, "outcome_ranking_map", context, evidence, "Compare", {}))
    assert len(repeat.calls) == 2
    generic = OfficeCall()
    generic.evidence_expansion_limit = 6
    with pytest.raises(ValueError, match="requested 20 refs; limit is 6"):
        asyncio.run(ask(generic, "outcome_ranking_map", context, evidence, "Compare", {}))
    assert len(generic.calls) == 1


def test_jury_and_hypothesis_share_context_no_ranking_required():
    context, evidence = sample()
    calls = []

    async def proposal(request):
        calls.append(request)
        return definition()

    async def broken(request):
        raise TimeoutError("provider")

    result = asyncio.run(
        jury.propose(context, evidence, {"a": proposal, "b": proposal, "c": broken}, proposal)
    )
    assert result["criteria"][0]["id"] == "saved"  # All products failing does not drop it.

    async def method(request):
        calls.append(request)
        return {
            "status": "propose",
            "method": "Save and inspect",
            "target": "missing draft",
            "expected_improvement": "saved draft",
            "verification": "read actual draft",
            "evidence_refs": ["task"],
            "target_gap_ids": ["missing"],
        }

    asyncio.run(hypotheses.propose(context, evidence, result, {"a": method, "b": broken}, method))
    assert all(r["shared_instruction"] == SHARED for r in calls)
    from reward_engine.core.outcome.context import compact_learning_context
    assert all(r["context"] == (compact_learning_context(context) if "hypothesis" in r["mode"] else context)
               for r in calls if r["mode"] != "outcome_expectations")
    assert all("artifact" not in p for r in calls if r["mode"] == "outcome_expectations"
               for p in r["context"]["products"])


def test_speculative_hard_and_fake_refs_rejected():
    _, evidence = sample()
    for mutation in ({"origin": "domain"}, {"evidence_refs": ["fake"]}):
        row = deepcopy(definition())
        row["criteria"][0].update(mutation)
        with pytest.raises(ValueError):
            jury.validate(row, evidence)
