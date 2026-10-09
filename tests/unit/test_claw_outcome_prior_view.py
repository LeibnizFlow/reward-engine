"""Single-test knowledge routing keeps originals readable, without changing Task scope."""

from copy import deepcopy

import pytest

from reward_engine.core.outcome.context import build_context, measurement_prior_view
from reward_engine.core.outcome.harness import defaults


def inputs():
    task = {"task_id": "note", "prompt": "Prepare a factual status note."}
    context, evidence = build_context(task, [{"artifact": {"final_response": "Label created."},
        "environment": {"current_status": "label created"}}])
    for ref in ("skill:primary", "skill:secondary"):
        content = {"kind": "task_review_prior", "authority": "domain_advice_not_user_requirement",
            "direction": ref + " direction", "relevance": "Helps distinguish plans and facts",
            "applicability": "Only when shipment status matters", "verification": "Read tracking",
            "mechanisms": [{"finding": "A label alone does not establish arrival", "quote": "exact excerpt"}]}
        evidence[ref] = deepcopy(content)
        context["skills"].append({"ref": ref, "content": content})
    context["available_evidence_refs"] = list(evidence)
    test = {"test_id": "audit", "criterion_id": "accuracy", "checker": "evidence.review",
            "rubric": {"standard": "State only supported status", "evidence_refs": ["skill:primary"]}}
    return context, evidence, test


def test_scope_retains_all_task_product_rules_and_referenced_knowledge():
    context, evidence, test = inputs()
    before = deepcopy((context, evidence, test))
    view = measurement_prior_view(context, evidence, test)
    assert view["skills"][0] == context["skills"][0]
    secondary = view["skills"][1]
    assert secondary["preview_only"] and "mechanisms" not in secondary["content"]
    assert secondary["content"]["applicability"] == context["skills"][1]["content"]["applicability"]
    assert {k: v for k, v in view.items() if k != "skills"} == {
        k: v for k, v in context.items() if k != "skills"}
    assert (context, evidence, test) == before
    # No inaccessible catalogs or guessing unknown skill formats.
    evidence.pop("skill:secondary")
    assert measurement_prior_view(context, evidence, test)["skills"][1] == context["skills"][1]
    context["skills"][1]["content"]["kind"] = "user_skill"
    assert measurement_prior_view(context, evidence, test)["skills"][1] == context["skills"][1]


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit_read", [False, True])
async def test_deferred_knowledge_is_read_before_its_citation_can_support_a_verdict(explicit_read):
    context, evidence, test = inputs()
    calls = []
    ref = "product_1:artifact:final_response"
    verdict = {"claims": [{"claim": "Only label creation is reported", "claim_refs": [ref],
        "evidence_refs": ["skill:secondary", ref], "observation": "No arrival claim", "judgment": "supported"}],
        "status": "pass", "reason": "Observed status represented correctly", "gap": "",
        "evidence_refs": [ref, "skill:secondary"]}

    async def call(request):
        calls.append(deepcopy(request))
        if len(calls) == 1:
            assert request["context"]["skills"][1]["preview_only"]
            if explicit_read:
                return {"need_evidence": ["skill:secondary"]}
        else:
            assert request["requested_evidence"]["skill:secondary"] == evidence["skill:secondary"]
        return deepcopy(verdict)

    result = await defaults(call).run(test, context, evidence)
    assert result["status"] == "pass" and len(calls) == 2


@pytest.mark.asyncio
async def test_unused_catalog_does_not_trigger_extra_read_or_model_call():
    context, evidence, test = inputs()
    calls = []

    async def call(request):
        calls.append(request)
        return {"status": "unknown", "reason": "Cannot establish arrival", "claims": [],
                "gap": "", "evidence_refs": []}

    result = await defaults(call).run(test, context, evidence)
    assert result["status"] == "unknown" and len(calls) == 1
