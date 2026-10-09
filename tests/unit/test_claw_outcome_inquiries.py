"""A targeted check receives WHY it was selected, without changing the frozen test."""

import asyncio
from copy import deepcopy

import pytest

from reward_engine.core.batch import BatchPolicy
from reward_engine.core.outcome.context import build_context, check_bundle
from reward_engine.core.outcome.delivery_triage import review
from reward_engine.core.outcome.harness import defaults, measure
from reward_engine.core.outcome.memory import Memory
from reward_engine.core.outcome.runner import DeliveryReview


@pytest.mark.asyncio
async def test_inquiry_and_claim_reach_model_and_survive_unknown_retest():
    task = {"task_id": "x", "prompt": "Draft a proposal using actual authorizations"}
    ref = "product_1:artifact:final_response"
    pack = {"task_id": "x", "task_prompt": task["prompt"], "pack_id": "p",
            "criteria": [{"id": "accuracy", "kind": "soft"}],
            "tests": [{"test_id": "review", "criterion_id": "accuracy", "checker": "evidence.review",
                       "rubric": {"id": "accuracy", "standard": "Claims are evidence-supported"}}]}
    original = deepcopy(pack)
    calls = []

    async def model(request):
        calls.append(request)
        if request["mode"] == "outcome_delivery_review":
            first = len(calls) == 1
            return {'outcome_goal': 'Fully satisfy the declared task with a correct, complete result.', 'full_score_gaps': [], 'improvements': [], "status": "target_satisfied", "reason": "No proven defect", "gaps": [],
                    "verify": [{"test_id": "review", "reason": "Was the commitment authorized?"}] if first else [],
                    "claim_checks": [{"claim": "Owner approved", "claim_ref": ref, "support_refs": [],
                        "assessment": "needs_verification", "reason": "Check approval, not just document existence",
                        "test_id": "review"}] if first else []}
        inquiry = request["context"]["measurement_inquiry"]
        assert inquiry["questions"] == ["Was the commitment authorized?"]
        assert inquiry["claim_checks"][0]["claim"] == "Owner approved"
        assert "measurement_inquiries" not in request["context"]
        assert request["test"] == original["tests"][0]
        assert "unverified question" in request["instruction"]
        return {"status": "unknown", "reason": "Approval evidence unavailable", "gap": "",
                "evidence_refs": [], "claims": []}

    registry = defaults(model)
    context, evidence = build_context(task, [{"artifact": {"final_response": "Owner approved"}}])
    first = await review(pack, registry, context, evidence)
    second = await review(pack, registry, context, evidence, previous=(context, evidence, first))
    assert len(calls) == 4
    assert second["results"][0]["measurement_inquiry"] == first["results"][0]["measurement_inquiry"]
    assert not second["delivery_clear"] and pack == original
    assert "measurement_inquiries" not in context


@pytest.mark.asyncio
async def test_each_checker_receives_only_its_own_question():
    task = {"task_id": "x", "prompt": "Review a draft"}
    context, evidence = build_context(task, [{"artifact": {"final_response": "Draft"}}])
    context["measurement_inquiries"] = {
        "one": {"questions": ["Question one"], "claim_checks": []},
        "two": {"questions": ["Question two"], "claim_checks": []},
    }
    requests = []

    async def model(request):
        requests.append(request)
        return {"status": "unknown", "reason": "Needs evidence", "gap": "", "claims": [], "evidence_refs": []}

    registry = defaults(model)
    for key in ("one", "two"):
        await registry.run({"test_id": key, "criterion_id": "c", "checker": "evidence.review"}, context, evidence)
    assert requests[0]["context"]["measurement_inquiry"]["questions"] == ["Question one"]
    assert "Question two" not in str(requests[0])
    assert "Question one" not in str(requests[1])


@pytest.mark.asyncio
async def test_new_inquiry_is_not_satisfied_by_old_cache_and_timeout_keeps_question():
    task = {"task_id": "x", "prompt": "Review the calculation"}
    context, evidence = build_context(task, [{"artifact": {"final_response": "Calculation"}}])
    inquiry = {"questions": ["Does the unchanged formula apply to these units?"], "claim_checks": []}
    pack = {"task_id": "x", "task_prompt": task["prompt"], "pack_id": "p", "criteria": [],
            "tests": [{"test_id": "test", "criterion_id": "c", "checker": "calculation"}]}
    registry = defaults(None)
    calls = []

    async def calculation(test, ctx, ev):
        calls.append(ctx)
        await asyncio.Event().wait()

    registry.register("calculation", calculation, "Test calculation", dependencies=lambda *_: "unchanged")
    previous = (deepcopy(context), deepcopy(evidence), {"results": [
        {"test_id": "test", "criterion_id": "c", "status": "pass", "reason": "Old calculation"}]})
    context["measurement_inquiries"] = {"test": inquiry}
    result = await measure(pack, registry, context, evidence, previous=previous,
                           policy=BatchPolicy(deadline_s=.02))
    assert len(calls) == 1 and calls[0]["measurement_inquiry"] == inquiry
    assert result["results"][0]["status"] == "unknown"
    assert result["results"][0]["measurement_inquiry"] == inquiry
    assert not result["results"][0].get("reused_unchanged_input")


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", ["pass", "fail", "unknown", None])
async def test_rubric_pass_cannot_erase_separate_inquiry_result(answer):
    task = {"task_id": "x", "prompt": "Prepare a draft, without making unsupported commitments"}
    ref = "product_1:artifact:final_response"
    pack = {"task_id": "x", "task_prompt": task["prompt"], "pack_id": "p",
            "criteria": [{"id": "structure", "kind": "soft", "standard": "Draft has a conclusion"}],
            "tests": [{"test_id": "review", "criterion_id": "structure", "checker": "evidence.review"}]}
    calls = []

    async def model(request):
        calls.append(request)
        if request["mode"] == "outcome_delivery_review":
            if len(calls) > 1:
                prior = request["previous_findings"]["checks"][0]
                assert prior["inquiry_result"]["status"] == (answer or "unknown")
            return {'outcome_goal': 'Fully satisfy the declared task with a correct, complete result.', 'full_score_gaps': [], 'improvements': [], "status": "target_satisfied", "reason": "Ask about the commitment", "gaps": [], "claim_checks": [],
                    "verify": [{"test_id": "review", "reason": "Is this commitment supported?"}] if len(calls) == 1 else []}
        assert "inquiry_result" in request["response"]
        result = {"status": "pass", "reason": "A conclusion exists", "gap": "", "evidence_refs": [ref],
                  "claims": [{"claim": "Conclusion exists", "claim_refs": [ref], "evidence_refs": [ref],
                              "observation": "A conclusion is present", "judgment": "supported"}]}
        if answer:
            result["inquiry_result"] = {"status": answer, "reason": "Separate commitment evaluation",
                "gap": "Remove unsupported commitment" if answer == "fail" else "",
                "evidence_refs": [ref, "task"] if answer != "unknown" else []}
        return result

    review_loop = DeliveryReview(task, pack, defaults(model))
    candidate = {"artifact": {"final_response": "Conclusion: owner already committed"}}
    feedback = await review_loop(candidate)
    row = review_loop.history[0]["results"][0]
    assert row["status"] == "pass"  # Do not falsify the narrow rubric score.
    assert row["inquiry_result"]["status"] == (answer or "unknown")
    if answer == "pass":
        assert feedback is None and review_loop.progress.verified
    else:
        assert feedback and not review_loop.progress.verified
        if answer == "fail":
            assert "Remove unsupported commitment" in feedback
        await review_loop(candidate)
        assert len(calls) == 4  # Previous unanswered question survives a later generic no_gap.


def test_inquiry_evidence_reaches_next_jury_with_correct_product_identity():
    ref = "product_1:artifact:final_response"
    measured = {"pack_id": "p", "results": [{"test_id": "t", "status": "pass", "evidence_refs": [ref],
        "measurement_inquiry": {"questions": ["product_1: is literal question text"], "claim_checks": [
            {"claim_ref": ref, "claim_refs": [ref, "product_10:artifact:final_response"],
             "support_refs": ["task"], "claim": "product_1: literal claim text"}]},
        "inquiry_result": {"status": "fail", "gap": "Unsupported commitment", "evidence_refs": [ref, "task"]},
        "private_grader": "NOT_FOR_JURY"}]}
    original = deepcopy(measured)
    bundle = check_bundle("product_3", measured, "saved-check.json")
    task = {"task_id": "x", "prompt": "Prepare a draft"}
    context, evidence = build_context(task, [{"artifact": {"final_response": "Draft"}}] * 3, checks=[bundle])
    row = context["checks"][0]
    claim = row["measurement_inquiry"]["claim_checks"][0]
    assert row["inquiry_result"]["evidence_refs"] == ["product_3:artifact:final_response", "task"]
    assert claim["claim_ref"] == "product_3:artifact:final_response"
    assert claim["claim_refs"] == ["product_3:artifact:final_response", "product_10:artifact:final_response"]
    assert claim["claim"] == "product_1: literal claim text"
    assert row["measurement_inquiry"]["questions"] == ["product_1: is literal question text"]
    assert evidence["check:0:0"]["inquiry_result"]["status"] == "fail"
    assert "NOT_FOR_JURY" not in str(context) + str(evidence)
    assert measured == original


@pytest.mark.parametrize("after_status,same_question,expected", [
    ("pass", True, True), ("unknown", True, False), ("pass", False, False)])
def test_memory_compares_only_the_same_answered_question(after_status, same_question, expected):
    task = {"task_id": "x", "prompt": "Produce an accurate draft"}
    ref = "product_1:artifact:final_response"
    before = {"pack_id": "p", "results": [{"test_id": "t", "status": "unknown", "evidence_refs": [],
        "measurement_inquiry": {"questions": ["Was the commitment authorized?"], "claim_checks": []},
        "inquiry_result": {"status": "fail", "gap": "Unsupported commitment", "evidence_refs": [ref]}}]}
    after = deepcopy(before)
    after["results"][0]["inquiry_result"].update(status=after_status, gap="")
    if not same_question:
        after["results"][0]["measurement_inquiry"]["questions"] = ["Is the delivery date suitable?"]
    memory = Memory()
    entry = memory.record(task, "Verify then revise", before, after,
                          source={"before": "before.json", "after": "after.json"}, context_note="Draft review")
    if expected:
        assert entry["value"]["observed_changes"][0]["scope"] == "inquiry"
        assert entry["value"]["observed_changes"][0]["before"] == "fail"
        assert entry["value"]["observed_changes"][0]["after"] == "pass"
        assert "Unsupported commitment" in entry["key"]
        assert entry["value"]["status"] == "observed_not_causal_or_official"
    else:
        assert entry is None and not memory.rows
