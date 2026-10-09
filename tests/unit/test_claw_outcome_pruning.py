"""Cost cuts must not manufacture passes, lose user requirements, or skip real repairs."""

from tests.support.claw_outcome_pruning import (
    TASK,
    candidate,
    pack,
)

from copy import deepcopy

import pytest
from tests.support.claw_outcome_builder_harness import setup

from reward_engine.core.outcome.builder import build
from reward_engine.core.outcome.context import (
    build_context,
    compact_learning_context,
    requirement_scope,
)
from reward_engine.core.outcome.harness import defaults, smoke
from reward_engine.core.outcome.hypotheses import compact_jury
from reward_engine.core.outcome.runner import DeliveryReview, run_delivery








@pytest.mark.asyncio
async def test_default_zero_doubts_means_zero_tests_not_all_passed():
    calls = []

    async def model(request):
        calls.append(request)
        assert request["mode"] == "outcome_delivery_review"
        assert "A verification plan is not an executed" in request["instruction"]
        assert "closing disclaimer" in request["instruction"]
        assert "never impose a search/tool-call quota" in request["instruction"]
        return {'outcome_goal': 'Fully satisfy the declared task with a correct, complete result.', 'full_score_gaps': [], 'improvements': [], "status": "target_satisfied", "reason": "No material gap observed", "claim_checks": [], "gaps": [], "verify": []}

    review = DeliveryReview(TASK, pack(), defaults(model))
    assert await review(candidate()) is None
    assert len(calls) == 1 and review.history[-1]["results"] == []
    assert review.history[-1]["coverage"] == 0
    assert review.history[-1]["ready_under_estimator"] is False
    assert review.status == "full_score_target_satisfied"
    assert all(i["status"] == "unknown" for i in review.progress.state["items"] if "test_id" in i)


@pytest.mark.asyncio
async def test_default_one_doubt_real_failure_same_session_revision_and_affected_retest():
    calls = []

    async def model(request):
        calls.append(request)
        return {'outcome_goal': 'Fully satisfy the declared task with a correct, complete result.', 'full_score_gaps': [], 'improvements': [], "status": "target_satisfied", "reason": "Syntax needs verification", "claim_checks": [], "gaps": [],
                "verify": [{"test_id": "parse", "reason": "Potential parse error"}] if len(calls) == 1 else []}

    class Actor:
        starts = 0
        resumes = 0

        async def start(self, prompt):
            self.starts += 1
            return candidate("broken")

        async def resume(self, feedback):
            self.resumes += 1
            assert "Invalid JSON" in feedback
            assert '"test_id": "other"' not in feedback
            return candidate()

    actor = Actor()
    review = DeliveryReview(TASK, pack(), defaults(model))
    assert (await run_delivery(actor, review))["artifact"]["final_response"] == "{}"
    assert actor.starts == actor.resumes == 1
    assert [[r["test_id"] for r in m["results"]] for m in review.history] == [["parse"], ["parse"]]
    assert [m["results"][0]["status"] for m in review.history] == ["fail", "pass"]
    assert len(calls) == 2 and review.history[-1]["coverage"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["unsupported_id", "error", "unknown", "too_many", "unsupported_gap", "pending"])
async def test_uncertainty_and_budget_are_not_completion(mode):
    async def model(request):
        if mode == "error":
            raise ConnectionError("offline")
        row = {'outcome_goal': 'Fully satisfy the declared task with a correct, complete result.', 'full_score_gaps': [], 'improvements': [], "status": "target_satisfied", "reason": "Inspect outcome", "claim_checks": [], "gaps": [], "verify": []}
        if mode == "unsupported_id":
            row["verify"] = [{"test_id": "imaginary", "reason": "doubt"}]
        if mode == "unknown":
            row["status"] = "unknown"
        if mode == "too_many":
            row["verify"] = [{"test_id": t["test_id"], "reason": "doubt"} for t in pack()["tests"]]
        if mode == "unsupported_gap":
            row.update(status="needs_improvement", gaps=[{"gap": "Invented", "evidence_refs": ["task"]}])
        return row

    review = DeliveryReview(TASK, pack(), defaults(model), max_revisions=0, max_tests=2)
    assert await review(candidate(pending=mode == "pending")) is None
    assert review.status == ("check_unavailable" if mode in {"error", "unsupported_id", "unsupported_gap"}
                             else "budget_exhausted_unresolved")
    assert len(review.history[-1]["results"]) <= 2
    if mode == "too_many":
        assert review.history[-1]["deferred_tests"] == ["third", "fourth"]
    else:
        assert review.history[-1]["results"] == []


@pytest.mark.asyncio
async def test_supported_gap_reaches_actor_even_with_no_check_and_does_not_require_all_tests():
    async def model(request):
        return {'outcome_goal': 'Fully satisfy the declared task with a correct, complete result.', 'full_score_gaps': [], 'improvements': [], "status": "needs_improvement", "reason": "Missing requested result", "claim_checks": [], "gaps": [
            {"gap": "Provide the requested conclusion", "evidence_refs": ["product_1:artifact:final_response"]}],
            "verify": []}

    review = DeliveryReview(TASK, pack(), defaults(model))
    feedback = await review(candidate())
    assert "Provide the requested conclusion" in feedback
    assert review.history[-1]["results"] == []


@pytest.mark.asyncio
async def test_builder_reuses_then_rebuilds_only_changed_criterion():
    context, evidence, jury, reply = setup()
    calls = []

    async def model(request):
        calls.append(request)
        assert all("artifact" not in product for product in request["context"]["products"])
        rows = request["jury"]["criteria"]
        return {"rubrics": [{"criterion_id": c["id"], "anchors": reply["rubrics"][0]["anchors"]} for c in rows],
                "tests": [{**reply["tests"][0], "criterion_id": c["id"], "test_id": "check" + c["id"]} for c in rows]}

    registry = defaults(model)
    original = await build(context, evidence, jury, registry, model)
    new_context, new_evidence = build_context(context["task"], [candidate(), candidate("[]")])
    reused = await build(new_context, new_evidence, jury, registry, model, previous=original)
    assert len(calls) == 1 and reused["tests"] == original["tests"]
    extended = {"criteria": jury["criteria"] + [{**jury["criteria"][0], "id": "new", "standard": "Explain result"}]}
    partial = await build(new_context, new_evidence, extended, registry, model, previous=reused)
    assert len(calls) == 2 and [c["id"] for c in calls[-1]["jury"]["criteria"]] == ["new"]
    assert {c["id"] for c in partial["criteria"]} == {"syntax", "new"}
    assert partial["tests"][0] == original["tests"][0]
    altered = deepcopy(extended)
    altered["criteria"][0]["standard"] = "Different requirement"
    await build(new_context, new_evidence, altered, registry, model, previous=partial)
    assert [c["id"] for c in calls[-1]["jury"]["criteria"]] == ["syntax"]
    # Changes in real user amendments invalidate reuse, even if a model repeats a criterion.
    amended = candidate()
    amended["artifact"]["public_consultation"]["earlier_exchanges"] = [{"speaker": "user", "text": "Add explanation"}]
    c, e = build_context(context["task"], [amended])
    await build(c, e, extended, registry, model, previous=partial)
    assert len(calls[-1]["jury"]["criteria"]) == 2


@pytest.mark.asyncio
async def test_builder_rejects_per_historical_product_template():
    context, evidence, jury, reply = setup()
    reply["tests"][0]["arguments"]["target_product"] = "product_3"

    async def model(_):
        return reply

    with pytest.raises(ValueError, match="historical product"):
        await build(context, evidence, jury, defaults(model), model)


@pytest.mark.asyncio
async def test_smoke_is_no_model_fixture_not_seed_measurement():
    async def forbidden(_):
        pytest.fail("Smoke must not buy semantic checks")

    result = await smoke(pack(), defaults(forbidden))
    assert result["status"] == "receipt_path_checked"
    assert result["receipt"]["test_id"] == "infra-parser"
    assert "NOT validated" in result["boundary"]


def test_compaction_keeps_user_knowledge_memory_and_pending_subrequirements():
    p = candidate("LONG_RAW_PRODUCT")
    p["artifact"]["public_consultation"]["earlier_exchanges"] = [{"speaker": "user", "text": "Actual update"}]
    c, e = build_context(TASK, [p], skills=["Important knowledge"], memories=[{"value": "Relevant gap"}])
    small = compact_learning_context(c)
    assert "LONG_RAW_PRODUCT" not in str(small)
    assert "Actual update" in str(small) and "Important knowledge" in str(small) and "Relevant gap" in str(small)
    assert e["product_1:artifact:final_response"] == "LONG_RAW_PRODUCT"
    assert small["available_evidence_refs"] == c["available_evidence_refs"]
    j = {"coverage": [{"expectation_id": "e", "status": "mapped", "subrequirements": [{"status": "deferred"}]}],
         "independent_expectations": [{"expectation_id": "e", "standard": "Essential missing knowledge"}],
         "proposal_expectations": [{"expectation_id": "done", "standard": "Redundant"}]}
    assert "Essential missing knowledge" in str(compact_jury(j)) and "Redundant" not in str(compact_jury(j))
    assert requirement_scope(c) != requirement_scope(build_context(TASK, [candidate()])[0])


def test_front_questions_preserve_meaning_without_internal_ledger():
    import json

    j = {"criteria": [], "gaps": [], "independent_expectations": [], "coverage": []}
    for source in ("seat0", "seat1"):
        j["independent_expectations"].append({
            "expectation_id": source, "standard": "Parent standard", "origin": "domain",
            "why": "May affect accuracy", "measurement_question": "Compare with the source",
            "evidence_refs": ["task", "skill:0"],
            "subrequirements": [{"part_id": source + "/0", "text": "Already covered"},
                                {"part_id": source + "/1", "text": "Check numerical precision"}]})
        j["coverage"].append({"expectation_id": source, "status": "mapped", "subrequirements": [
            {"part_id": source + "/0", "status": "mapped", "retained_text": "LONG_METADATA" * 20000},
            {"part_id": source + "/1", "status": "deferred", "reason": "Exact text mapping failed"}]})
    original = deepcopy(j)
    result = compact_jury(j)
    assert j == original and compact_jury(result) == result
    assert len(result["open_questions"]) == 1
    question = result["open_questions"][0]
    assert {k: v for k, v in question.items() if k != 'pending_sources'} == {
        "questions": ["Check numerical precision"], "origin": "domain", "why": "May affect accuracy",
        "verification": "Compare with the source", "status": "unverified_proposal",
        "evidence_refs": ["task", "skill:0"], "source_expectation_ids": ["seat0", "seat1"]}
    assert len(json.dumps(result)) < 1000
    assert "Already covered" not in str(result)
    assert all(source['parts'][0]['reason'] == 'Exact text mapping failed'
               for source in question['pending_sources'])
    assert question['status'] == 'unverified_proposal'  # Bookkeeping reason, not an observed defect.
    # A stale parent flag must not resurrect already-covered or explicitly rejected questions.
    for row in j["coverage"]:
        row["status"] = "deferred"
        row["subrequirements"][1]["status"] = "excluded"
    assert compact_jury(j)["open_questions"] == []


@pytest.mark.asyncio
async def test_hypothesis_map_reduce_and_delivery_share_unverified_questions():
    from reward_engine.core.outcome import hypotheses

    c, e = build_context(TASK, [candidate()])
    j = {"criteria": [], "gaps": [], "coverage": [{"expectation_id": "e", "status": "deferred"}],
         "independent_expectations": [{"expectation_id": "e", "standard": "Check schema relevance",
                                      "origin": "domain", "evidence_refs": ["task"]}]}
    requests = []

    async def model(request):
        requests.append(request)
        if request["mode"] == "outcome_delivery_review":
            return {'outcome_goal': 'Fully satisfy the declared task with a correct, complete result.', 'full_score_gaps': [], 'improvements': [], "status": "target_satisfied", "reason": "No supported defect", "claim_checks": [], "gaps": [], "verify": []}
        return {"status": "keep", "method": "", "clarifications": []}

    await hypotheses.propose(c, e, j, {str(i): model for i in range(3)}, model)
    p = pack()
    p.update(expectation_coverage=j["coverage"], independent_expectations=j["independent_expectations"])
    review = DeliveryReview(TASK, p, defaults(model))
    await review(candidate())
    assert len(requests) == 5
    expected = compact_jury(j)["open_questions"]
    for request in requests:
        front = request.get("jury", request)
        assert front["open_questions"] == expected
        assert "independent_expectations" not in front and "coverage" not in front
    assert review.status == "full_score_target_satisfied"  # bookkeeping alone never creates a defect


@pytest.mark.asyncio
async def test_expectation_cache_reuses_seats_not_product_judgments_and_invalidates_sources():
    from reward_engine.core.outcome import jury

    modes, cache = [], {}

    async def model(request):
        modes.append(request["mode"])
        return {"criteria": [{"id": "syntax", "standard": "Valid JSON", "kind": "hard",
                              "origin": "task", "evidence_refs": ["task"],
                              "measurement_question": "Does it parse?"}], "gaps": []}

    c, e = build_context(TASK, [candidate()])
    await jury.propose(c, e, {"a": model, "b": model, "c": model}, model, expectation_cache=cache)
    c2, e2 = build_context(TASK, [candidate(), candidate("[]")])
    await jury.propose(c2, e2, {"a": model, "b": model, "c": model}, model, expectation_cache=cache)
    assert modes.count("outcome_expectations") == 3
    assert modes.count("outcome_jury_map") == 6 and modes.count("outcome_jury_reduce") == 2
    e2["reference:task_review:new"] = {"text": "New knowledge"}
    await jury.propose(c2, e2, {"a": model, "b": model, "c": model}, model, expectation_cache=cache)
    assert modes.count("outcome_expectations") == 6
    # The real user's new requirement, not artifact content, invalidates cached expectations.
    amended = candidate()
    amended["artifact"]["public_consultation"]["earlier_exchanges"] = [{"speaker": "user", "text": "Explain JSON"}]
    c3, e3 = build_context(TASK, [amended])
    await jury.propose(c3, e3, {"a": model, "b": model, "c": model}, model, expectation_cache=cache)
    assert modes.count("outcome_expectations") == 9


@pytest.mark.asyncio
async def test_failed_expectations_do_not_cache_missing_seat():
    from reward_engine.core.outcome import jury

    cache = {}
    c, e = build_context(TASK, [candidate()])

    async def broken(request):
        raise TimeoutError("unavailable")

    with pytest.raises(RuntimeError):
        await jury.propose(c, e, {"a": broken, "b": broken}, broken, expectation_cache=cache)
    assert cache["seats"] == {}
