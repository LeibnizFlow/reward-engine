import asyncio
from copy import deepcopy

import pytest

from reward_engine.core.outcome.jury import propose, validate
from reward_engine.core.outcome.loop import select_delivery
from tests.support.claw_outcome_gap_signals import (
    fixture,
)


def test_three_agreeing_suspicions_do_not_become_verified():
    context, evidence, result = fixture()
    calls = []

    async def call(request):
        calls.append(request)
        return deepcopy(result)

    output = asyncio.run(
        propose(context, evidence, {str(i): call for i in range(3)}, call)
    )
    assert len(calls) == 7 and output["gaps"][0]["status"] == "needs_check"
    assert (
        len(output["gaps"][0]["product_ids"]) == 3
    )  # Common failure survives zero rank contrast.
    result["gaps"][0]["status"] = "check_supported"
    with pytest.raises(ValueError, match="recorded failed check"):
        validate(result, evidence, context)


def test_real_check_support_and_unknown_not_failure():
    context, evidence, result = fixture()
    result["gaps"][0]["product_ids"] = ["product_1"]
    result["gaps"][0]["status"] = "check_supported"
    result["gaps"][0]["evidence_refs"] = ["check:0:0"]
    evidence["check:0:0"] = {
        "product_id": "product_1",
        "status": "fail",
        "error": False,
    }
    assert validate(result, evidence, context)
    evidence["check:0:0"]["status"] = "unknown"
    with pytest.raises(ValueError):
        validate(result, evidence, context)


def test_domain_hunch_allowed_but_not_a_hard_constraint():
    context, evidence, result = fixture()
    result["gaps"][0].update(
        expected_basis="domain",
        expected_refs=[],
        expected="Additional validation could improve reliability",
        status="needs_check",
    )
    assert validate(result, evidence, context)
    result["criteria"][0]["origin"] = "domain"
    with pytest.raises(ValueError):
        validate(result, evidence, context)


def test_actor_suggestion_is_not_user_requirement():
    context, evidence, result = fixture()
    evidence["product_1:assistant"] = "Let's publish it"
    result["gaps"][0].update(
        expected_basis="user", expected_refs=["product_1:assistant"]
    )
    with pytest.raises(ValueError, match="actual user"):
        validate(result, evidence, context)


@pytest.mark.parametrize("malformed", ["origin", "expected_basis", "evidence_refs"])
def test_map_draft_reaches_reduce_but_final_still_validated(malformed):
    context, evidence, correct = fixture()
    draft = deepcopy(correct)
    if malformed == "origin":
        draft["criteria"][0]["origin"] = "observed_gap"
    elif malformed == "expected_basis":
        draft["gaps"][0]["expected_basis"] = "task code semantics"
    else:
        draft["criteria"][0]["evidence_refs"] = ["invented"]

    async def member(request):
        return deepcopy(correct if request["mode"] == "outcome_expectations" else draft)

    async def reducer(request):
        assert all(p["draft_validation_issues"] for p in request["proposals"])
        return deepcopy(correct)

    result = asyncio.run(
        propose(context, evidence, {"a": member, "b": member}, reducer)
    )
    assert (
        result["criteria"] == correct["criteria"] and result["gaps"] == correct["gaps"]
    )
    assert len(result["independent_expectations"]) == 2
    if malformed == "expected_basis":
        rejected_gap = asyncio.run(
            propose(context, evidence, {"a": member, "b": member}, member)
        )
        assert rejected_gap["criteria"] == correct["criteria"]
        assert rejected_gap["gaps"] == []
        assert (
            rejected_gap["gap_validation_issues"][0]["gap_id"]
            == draft["gaps"][0]["gap_id"]
        )
        assert "expected_basis" in rejected_gap["gap_validation_issues"][0]["reason"]
    else:
        with pytest.raises(ValueError):
            asyncio.run(propose(context, evidence, {"a": member, "b": member}, member))


@pytest.mark.parametrize(
    "groups,unknown,b_status,expected",
    [
        ([["b"], ["a"], ["c"]], [], "pass", "B"),
        ([["b"], ["a"], ["c"]], [], "fail", "A"),
        ([["b"], ["a"], ["c"]], [], "unknown", "B"),
        ([["b", "a"], ["c"]], [], "pass", "A"),
        ([["b"], ["c"]], ["a"], "pass", "A"),
    ],
)
def test_original_fallback(groups, unknown, b_status, expected):
    measurements = {
        arm: {
            "results": [
                {"test_id": "valid", "status": b_status if arm == "B" else "pass"}
            ]
        }
        for arm in "ABC"
    }
    result = select_delivery(
        {"status": "compared", "groups": groups, "unknown": unknown},
        dict(zip("ABC", "abc", strict=True)),
        measurements,
    )
    assert result["arm"] == expected
