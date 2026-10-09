"""Bounded method refinement preserves corrections and never calls unavailable a verdict."""

import asyncio
from collections import Counter
from copy import deepcopy

import pytest

from reward_engine.core.outcome import hypotheses
from reward_engine.core.outcome.context import build_context


def inputs():
    return build_context(
        {
            "task_id": "explanation",
            "prompt": "Explain the mechanism and a usable verification.",
        },
        skills=[{"note": "Optional formatting advice; not a required outcome"}],
    )


def method(text="Explain the mechanism, then perform a discriminating source check"):
    return {
        "status": "propose",
        "method": text,
        "target": "Missing causal explanation",
        "expected_improvement": "A more useful, supportable explanation; not yet measured",
        "verification": "Compare claims with the source; contrary observations refute the method",
        "evidence_refs": ["task"],
        "target_gap_ids": [],
        "clarifications": [],
    }


def review():
    return {
        "assessment": "keep",
        "reason": "The method addresses the complete Task",
        "issues": [],
    }


@pytest.mark.asyncio
async def test_semantic_keep_still_receives_all_three_rounds_and_parallel_whole_method_audits():
    context, evidence = inputs()
    counts, started, gates, records = Counter(), Counter(), {}, {}

    async def member(request):
        counts[request["mode"]] += 1
        if request["mode"] == "outcome_hypothesis_map":
            return {"status": "keep", "method": ""}
        number = request["construction_round"]
        started[number] += 1
        gate = gates.setdefault(number, asyncio.Event())
        if started[number] == 3:
            gate.set()
        await asyncio.wait_for(gate.wait(), 1)
        assert request["current_artifact"]["status"] == "keep"
        assert "reviews" not in request
        assert request["context"]["task"] == context["task"]
        return review()

    async def reducer(request):
        counts[request["mode"]] += 1
        return {
            "status": "keep",
            "method": "",
            "revision_summary": "No supported change or unresolved issue",
        }

    result = await hypotheses.propose(
        context,
        evidence,
        {},
        {str(i): member for i in range(3)},
        reducer,
        construction_rounds=3,
        record=lambda name, value: records.update({name: deepcopy(value)}),
    )
    assert counts == {
        "outcome_hypothesis_map": 3,
        "outcome_hypothesis_reduce": 1,
        "outcome_hypothesis_audit": 6,
        "outcome_hypothesis_refine": 2,
    }
    assert result["status"] == "keep" and "availability" not in result
    assert result["construction"]["completed_rounds"] == 3
    assert result["construction"]["complete"]
    assert hypotheses.actor_prompt(result) == ""
    assert all(
        f"construction/hypothesis/round_{i}/input" in records for i in range(1, 4)
    )


@pytest.mark.asyncio
async def test_only_current_method_survives_revisions_and_inputs_are_isolated():
    context, evidence = inputs()
    jury = {
        "criteria": [
            {"id": "task_goal", "standard": "Explain and verify the mechanism"}
        ],
        "open_questions": [],
    }
    before = deepcopy((context, evidence, jury))
    seen = []

    async def member(request):
        seen.append(deepcopy(request))
        assert request["context"]["task"] == context["task"]
        assert "largest consequential" in request["instruction"]
        assert "Knowledge" in request["instruction"]
        if request["mode"] == "outcome_hypothesis_map":
            return method("ERRONEOUS_INITIAL_DRAFT")
        assert request["jury"] == jury | {"gaps": []}
        assert "reviews" not in request
        if request["construction_round"] == 3:
            assert "ERRONEOUS_INITIAL_DRAFT" not in str(request)
            assert (
                request["current_artifact"]["method"]
                == "Read the actual source then explain the mechanism"
            )
        request["context"]["task"]["prompt"] = "member mutation"
        request["current_artifact"]["method"] = "member mutation"
        return review()

    async def reducer(request):
        if request["mode"] == "outcome_hypothesis_reduce":
            return method("ERRONEOUS_INITIAL_DRAFT")
        assert "member mutation" not in str(request)
        return method("Read the actual source then explain the mechanism") | {
            "revision_summary": "Corrected the unsupported premise; verify the outcome before claiming benefit"
        }

    result = await hypotheses.propose(
        context,
        evidence,
        jury,
        {str(i): member for i in range(3)},
        reducer,
        construction_rounds=3,
    )
    assert result["construction"]["complete"]
    assert result["method"] == "Read the actual source then explain the mechanism"
    assert (context, evidence, jury) == before
    assert len(seen) == 9


@pytest.mark.asyncio
async def test_invalid_reference_retries_same_operation_with_already_read_evidence():
    context, evidence = inputs()
    attempts = []

    async def member(request):
        if request["mode"] == "outcome_hypothesis_audit":
            return review()
        attempts.append(deepcopy(request))
        if "requested_evidence" not in request:
            return {"need_evidence": ["skill:0"]}
        assert request["requested_evidence"] == {"skill:0": evidence["skill:0"]}
        if "response_recovery" not in request:
            return method() | {"evidence_refs": ["invented:source"]}
        assert request["response_recovery"]["error_type"] == "ResponseContractError"
        return method()

    async def reducer(request):
        return method() | {
            "revision_summary": "Preserved supported method and its falsification check"
        }

    result = await hypotheses.propose(
        context,
        evidence,
        {},
        {str(i): member for i in range(3)},
        reducer,
        construction_rounds=3,
    )
    assert result["construction"]["complete"]
    assert len(attempts) == 9
    assert sum("response_recovery" in row for row in attempts) == 3


@pytest.mark.asyncio
async def test_persistent_invalid_audit_stops_with_partial_history_not_semantic_keep():
    context, evidence = inputs()
    records, modes = {}, Counter()

    async def member(request):
        modes[request["mode"]] += 1
        if request["mode"] == "outcome_hypothesis_map":
            return method()
        return {
            "assessment": "revise",
            "reason": "Check this",
            "issues": [
                {
                    "scope": "current",
                    "target": "method",
                    "problem": "Unverified fact",
                    "suggested_change": "Read the source",
                    "evidence_refs": ["phantom"],
                }
            ],
        }

    async def reducer(request):
        modes[request["mode"]] += 1
        return method()

    result = await hypotheses.propose(
        context,
        evidence,
        {},
        {str(i): member for i in range(3)},
        reducer,
        construction_rounds=3,
        record=lambda name, value: records.update({name: deepcopy(value)}),
    )
    assert result["availability"] == "unavailable"
    assert result["error_type"] == "ResponseContractError"
    assert not result["construction"]["complete"]
    assert result["construction"]["completed_rounds"] == 1
    assert modes == {
        "outcome_hypothesis_map": 3,
        "outcome_hypothesis_reduce": 1,
        "outcome_hypothesis_audit": 6,
    }
    assert records["construction/hypothesis/round_1/output"]["method"]
    assert not records["construction/hypothesis/round_2/map"]["complete"]
    assert hypotheses.actor_prompt(result) == ""


@pytest.mark.asyncio
async def test_first_round_does_not_reduce_a_partial_three_seat_batch():
    context, evidence = inputs()

    async def good(request):
        return method()

    async def bad(request):
        raise TimeoutError("simulated provider failure")

    async def reducer(request):
        raise AssertionError("Cannot reduce incomplete construction")

    result = await hypotheses.propose(
        context,
        evidence,
        {},
        {"a": good, "b": good, "c": bad},
        reducer,
        construction_rounds=3,
    )
    assert result["availability"] == "unavailable"
    assert result["construction"]["completed_rounds"] == 0


def test_pending_jury_projection_uses_corrected_current_clauses_and_legacy_fallback():
    jury = {
        "independent_expectations": [
            {
                "expectation_id": "e",
                "origin": "task",
                "standard": "OLD_FALSE_STANDARD",
                "evidence_refs": ["task"],
                "why": "OLD_FALSE_RATIONALE",
                "measurement_question": "OLD_FALSE_CHECK",
                "subrequirements": [{"part_id": "p", "text": "OLD_FALSE_PART"}],
            }
        ],
        "coverage": [
            {
                "expectation_id": "e",
                "status": "deferred",
                "subrequirements": [
                    {
                        "part_id": "p",
                        "status": "deferred",
                        "current_text": "Check the actual launch dimensions",
                        "proposed_text": "OLD_FALSE_PART",
                    }
                ],
            }
        ],
    }
    before = deepcopy(jury)
    questions = hypotheses.compact_jury(jury)["open_questions"]
    assert questions[0]["questions"] == ["Check the actual launch dimensions"]
    assert "OLD_FALSE" not in str(questions)
    assert jury == before
    del jury["coverage"][0]["subrequirements"][0]["current_text"]
    jury["coverage"][0]["subrequirements"][0]["retained_text"] = "Retained correction"
    assert hypotheses.compact_jury(jury)["open_questions"][0]["questions"] == [
        "Retained correction"
    ]
    jury["coverage"][0].update(
        subrequirements=[], current_standard="Current pending requirement"
    )
    assert hypotheses.compact_jury(jury)["open_questions"][0]["questions"] == [
        "Current pending requirement"
    ]


@pytest.mark.parametrize("bad", [True, 0, 4, "3"])
@pytest.mark.asyncio
async def test_invalid_round_configuration_is_rejected_before_model_calls(bad):
    context, evidence = inputs()
    with pytest.raises(ValueError, match="construction_rounds"):
        await hypotheses.propose(
            context, evidence, {}, {}, None, construction_rounds=bad
        )


@pytest.mark.asyncio
async def test_input_fingerprint_changes_with_task_and_number_of_rounds():
    context, evidence = inputs()

    async def model(request):
        if request["mode"] == "outcome_hypothesis_audit":
            return review()
        return method() | {"revision_summary": "No change required"}

    async def run(ctx, refs, rounds):
        output = await hypotheses.propose(
            ctx,
            refs,
            {},
            {str(i): model for i in range(3)},
            model,
            construction_rounds=rounds,
        )
        return output["construction"]["input_fingerprint"]

    first = await run(context, evidence, 3)
    assert first != await run(context, evidence, 2)
    changed_context, changed_evidence = build_context(
        {"task_id": "explanation", "prompt": "Explain another task"}
    )
    assert first != await run(changed_context, changed_evidence, 3)
