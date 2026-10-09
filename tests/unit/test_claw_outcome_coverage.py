"""Blind expectations, preserved omissions, real measurements and bounded review."""

import asyncio
from copy import deepcopy

import pytest

from reward_engine.core.batch import BatchPolicy
from reward_engine.core.outcome import builder, hypotheses, jury
from reward_engine.core.outcome.context import build_context, requirement_context
from reward_engine.core.outcome.coverage import report
from reward_engine.core.outcome.harness import defaults, measure
from reward_engine.core.outcome.runner import DeliveryReview
from tests.support.claw_outcome_builder_harness import setup


def expected():
    return {
        "id": "content",
        "expectation_id": "seat:0",
        "standard": "Explain the requested conclusion",
        "kind": "hard",
        "origin": "task",
        "evidence_refs": ["task"],
        "measurement_question": "Does the answer justify the conclusion?",
        "why": "A parseable response can still fail the user's question",
        "priority": "critical",
    }


@pytest.mark.asyncio
async def test_each_jury_phase_has_its_own_deadline_without_extra_calls():
    context, evidence, _, _ = setup()
    calls, logs = [], {}

    async def member(request):
        calls.append(request["mode"])
        # Two valid calls exceed one phase's budget, but each is within it.
        await asyncio.sleep(0.12)
        return {"criteria": [expected()]}

    async def reduce(request):
        assert len(request["proposals"]) == 3
        return {"criteria": [expected()]}

    await jury.propose(
        context,
        evidence,
        {str(i): member for i in range(3)},
        reduce,
        policy=BatchPolicy(deadline_s=0.2),
        record=lambda k, v: logs.update({k: v}),
    )
    assert calls.count("outcome_expectations") == 3
    assert calls.count("outcome_jury_map") == 3
    assert logs["jury_expectations_map"]["succeeded"] == 3
    assert logs["jury_map"]["succeeded"] == 3


@pytest.mark.parametrize("failed_phase", ["outcome_expectations", "outcome_jury_map"])
@pytest.mark.asyncio
async def test_one_stalled_jury_seat_does_not_erase_other_two(failed_phase):
    context, evidence, _, _ = setup()
    calls, logs = [], {}

    async def healthy(request):
        return {"criteria": [expected()]}

    async def stalled(request):
        calls.append(request["mode"])
        if request["mode"] == failed_phase:
            await asyncio.sleep(5)
        return await healthy(request)

    async def reduce(request):
        assert len(request["proposals"]) == 2
        return {"criteria": [expected()]}

    await jury.propose(
        context,
        evidence,
        {"a": healthy, "b": healthy, "c": stalled},
        reduce,
        policy=BatchPolicy(deadline_s=0.05),
        record=lambda k, v: logs.update({k: v}),
    )
    phase = (
        "jury_expectations_map"
        if failed_phase == "outcome_expectations"
        else "jury_map"
    )
    assert logs[phase]["succeeded"] == 2
    assert logs[phase]["items"][-1]["status"] == "timeout"
    assert calls.count(failed_phase) == 1  # no automatic model redraw
    if failed_phase == "outcome_expectations":
        assert calls == [failed_phase]


def test_blind_phase_cannot_read_products_even_by_evidence_expansion():
    task = {"task_id": "x", "prompt": "Explain the result"}
    context, evidence = build_context(
        task,
        [
            {
                "artifact": {
                    "final_response": "OLD_SECRET",
                    "public_consultation": {
                        "earlier_exchanges": [
                            {"speaker": "user", "text": "Use Chinese"},
                            {"speaker": "assistant", "text": "ASSISTANT_SECRET"},
                        ]
                    },
                },
                "environment": {"secret": "ENV_SECRET"},
                "source_documents": ["SOURCE_SECRET"],
            }
        ],
        skills=["Consider causal mechanisms"],
        memories=["MEMORY_SECRET"],
    )
    blind, refs = requirement_context(context, evidence)
    assert "SECRET" not in str(blind) + str(refs)
    assert "Use Chinese" in str(blind) and "causal mechanisms" in str(blind)
    assert set(refs) == {"task", "skill:0", "product_1:public:0"}


@pytest.mark.asyncio
async def test_common_omission_survives_reduce_and_builder_without_fake_check():
    context, evidence, definition, reply = setup()
    seen = []

    async def member(request):
        seen.append(request)
        if request["mode"] == "outcome_expectations":
            return {"criteria": [expected()]}
        assert "artifact" in request["context"]["products"][0]
        return {
            "criteria": [
                {
                    **definition["criteria"][0],
                    "evidence_refs": ["task"],
                    "measurement_question": "Parse",
                }
            ]
        }

    async def reducer(request):
        return await member(
            request
        )  # Simulate a reducer forgetting the independent outcome.

    output = await jury.propose(context, evidence, {"a": member, "b": member}, reducer)
    assert len(seen) == 5
    assert all(r["status"] == "deferred" for r in output["coverage"])

    async def make(_):
        return reply

    async def no_model(_):
        raise AssertionError("A parser must run, not a semantic substitute")

    registry = defaults(no_model)
    pack = await builder.build(context, evidence, output, registry, make)
    result = await measure(pack, registry, context, evidence)
    assert result["coverage"] == 1 and result["results"][0]["status"] == "pass"
    assert not result["ready_under_estimator"]
    assert all(r["status"] == "unknown" for r in result["outcome_coverage"])
    review = DeliveryReview(context["task"], pack, registry, audit_mode="full")
    feedback = await review({"artifact": {"final_response": "{}"}})
    assert "Explain the requested conclusion" in feedback
    assert "Outcome not covered" in feedback and review.status == "revision_requested"


def test_hard_expectation_cannot_be_excluded_and_phantom_mapping_rejected():
    output = {
        "criteria": [{"id": "x"}],
        "coverage": [
            {
                "expectation_id": "seat:0",
                "criterion_ids": [],
                "status": "excluded",
                "reason": "All samples omit it",
            }
        ],
    }
    assert jury.reconcile_coverage([expected()], output)[0]["status"] == "deferred"
    output["coverage"][0].update(status="mapped", criterion_ids=["invented"])
    with pytest.raises(ValueError):
        jury.reconcile_coverage([expected()], output)
    with pytest.raises(ValueError):
        jury.reconcile_coverage(
            [expected()], {**output, "coverage": output["coverage"] * 2}
        )


@pytest.mark.asyncio
async def test_hypothesis_sees_uncovered_major_outcome_and_actor_gets_verification():
    context, evidence, definition, _ = setup()
    definition["independent_expectations"] = [expected()]
    definition["coverage"] = [{"expectation_id": "seat:0", "status": "deferred"}]

    async def call(request):
        assert "coverage" not in request["jury"]
        assert request["jury"]["open_questions"][0]["status"] == "unverified_proposal"
        assert request["jury"]["open_questions"][0]["source_expectation_ids"] == [
            "seat:0"
        ]
        assert "Explain the requested conclusion" in str(request)
        return {
            "status": "propose",
            "method": "Inspect the relevant source then justify the conclusion",
            "target": "Missing explanation",
            "expected_improvement": "Supported conclusion",
            "verification": "Compare claims with the actual source",
            "evidence_refs": ["task"],
        }

    result = await hypotheses.propose(context, evidence, definition, {"a": call}, call)
    prompt = hypotheses.actor_prompt(result)
    assert "Missing explanation" in prompt and "Compare claims" in prompt


def test_missing_check_and_unknown_are_not_pass_and_exclusion_is_explicit():
    pack = {
        "tests": [],
        "independent_expectations": [expected()],
        "expectation_coverage": [
            {
                "expectation_id": "seat:0",
                "status": "mapped",
                "criterion_ids": ["content"],
                "reason": "Requested",
            }
        ],
    }
    assert report(pack, [])[0]["status"] == "unknown"


@pytest.mark.asyncio
async def test_local_revision_reuses_only_proven_unchanged_inputs():
    context, evidence, definition, reply = setup()

    async def call(_):
        return reply

    registry = defaults(call)
    pack = await builder.build(context, evidence, definition, registry, call)
    first = await measure(pack, registry, context, evidence)
    changed = deepcopy(evidence)
    changed["product_1:artifact:unrelated"] = "new note"
    second = await measure(
        pack, registry, context, changed, previous=(context, evidence, first)
    )
    assert second["results"][0]["reused_unchanged_input"]
    changed["product_1:artifact:final_response"] = "bad JSON"
    third = await measure(
        pack, registry, context, changed, previous=(context, evidence, first)
    )
    assert third["results"][0]["status"] == "fail"
    assert not third["results"][0].get("reused_unchanged_input")


@pytest.mark.asyncio
async def test_whole_task_review_finds_gap_despite_passed_checks():
    context, evidence, definition, reply = setup()
    definition.update(
        independent_expectations=[expected()],
        coverage=[
            {
                "expectation_id": "seat:0",
                "criterion_ids": ["syntax"],
                "status": "mapped",
                "reason": "Deliberately incomplete mapping to test independent final review",
            }
        ],
    )

    async def build_call(_):
        return reply

    calls = []

    async def judge(request):
        calls.append(request)
        assert request["test"]["test_id"] == "__outcome_completeness__"
        assert "complete actual Task" in str(request["test"])
        return {
            "status": "fail",
            "reason": "JSON exists but contains no requested explanation",
            "gap": "Explain the conclusion",
            "evidence_refs": ["product_1:artifact:final_response"],
            "claims": [
                {
                    "claim": "An explanation is present",
                    "claim_refs": ["task"],
                    "evidence_refs": ["product_1:artifact:final_response"],
                    "observation": "The delivered object is empty",
                    "judgment": "contradicted",
                }
            ],
        }

    registry = defaults(judge)
    pack = await builder.build(context, evidence, definition, registry, build_call)
    review = DeliveryReview(context["task"], pack, registry, audit_mode="full")
    feedback = await review({"artifact": {"final_response": "{}"}})
    assert len(calls) == 1 and "Explain the conclusion" in feedback
    assert review.history[0]["results"][0]["status"] == "pass"
    assert not review.history[0]["ready_under_estimator"]
    assert not review.progress.verified
