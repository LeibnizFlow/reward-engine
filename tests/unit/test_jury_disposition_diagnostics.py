"""Prospective diagnostics for native v2's shape; no native evidence or APIs."""

import socket
from copy import deepcopy

import pytest

from reward_engine.core.outcome import jury
from reward_engine.core.outcome.context import build_context
from reward_engine.core.outcome.jury_drafts import expectations
from reward_engine.core.outcome.jury_repair import apply_edits, diagnostics
from reward_engine.core.outcome.reducer_projection import expand_texts
from tests.unit.test_outcome_role_contract_recovery import roles as roles

POSITIONS = ((17, 3), (29, 2))


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Disposition tests must not connect to a provider")

    monkeypatch.setattr(socket.socket, "connect", forbidden)


def captured_shape(status="excluded"):
    """Keep the two actual indices and inconsistent fields with synthetic text."""
    context, evidence = build_context(
        {"task_id": "fixture", "prompt": "Explain the result clearly."},
        [{"artifact": {"final_response": "A short explanation."}}],
    )
    criterion = {
        "id": "organized_structure",
        "standard": "Organize the explanation clearly.",
        "kind": "soft",
        "origin": "domain",
        "evidence_refs": ["task"],
        "measurement_question": "Is the explanation organized?",
        "subrequirements": ["Headings", "Examples", "Conciseness", "Summary"],
    }
    expected = expectations([criterion] * 33, "seat", "before_products")
    coverage = [
        {
            "expectation_id": row["expectation_id"],
            "status": "mapped",
            "criterion_ids": [criterion["id"]],
            "reason": "Retain the authored organization suggestions.",
            "subrequirements": [
                {
                    "part_id": part["part_id"],
                    "status": "mapped",
                    "criterion_ids": [criterion["id"]],
                    "reason": "Explicitly retained in the final criterion.",
                    "retained_text": part["text"],
                }
                for part in row["subrequirements"]
            ],
        }
        for row in expected
    ]
    for i, j in POSITIONS:
        coverage[i]["subrequirements"][j].update(
            status=status,
            reason="This suggestion has no independently supported mapping.",
            retained_text="",
        )
    return (
        context,
        evidence,
        {
            "independent_expectations": expected,
            "proposal_expectations": [],
            "proposals": [{"member": str(i)} for i in range(3)],
            "revision_protocol": True,
        },
        {
            "criteria": [criterion],
            "gaps": [],
            "coverage": coverage,
            "uncertainties": [],
        },
    )


def edit(path, value):
    return {
        "op": "replace",
        "path": path,
        "value": value,
        "reason": "Explicit model fixture correction preserving the disposition.",
    }


def id_list_edits():
    return {
        "edits": [
            edit(["coverage", i, "subrequirements", j, "criterion_ids"], [])
            for i, j in POSITIONS
        ]
    }


def validate_shape(context, evidence, inputs, draft):
    validated = jury.validate(
        jury.normalize_reduced(draft, context, evidence), evidence, context
    )
    return jury.reconcile_coverage(inputs["independent_expectations"], validated)


@pytest.mark.parametrize("status", ["excluded", "deferred"])
@pytest.mark.parametrize(
    "targets", [["organized_structure"], "organized_structure", None, False]
)
def test_exact_empty_list_constraint_preserves_authored_status_and_input(
    status, targets
):
    context, evidence, inputs, draft = captured_shape(status)
    for i, j in POSITIONS:
        draft["coverage"][i]["subrequirements"][j]["criterion_ids"] = deepcopy(targets)
    before = deepcopy(draft)
    issues = diagnostics(draft, inputs["independent_expectations"])
    assert len(issues) == 2
    for issue, (i, j) in zip(issues, POSITIONS, strict=True):
        assert issue["path"] == ["coverage", i, "subrequirements", j, "criterion_ids"]
        assert issue["current_status"] == status
        assert issue["value"] == targets and issue["expected_value"] == []
        assert "allowed_statuses" not in issue
        assert "Preserve the authored valid status" in issue["reason"]
        assert "model must explicitly author" in issue["reason"]
        assert "remains deferred" not in issue["reason"]
    assert draft == before
    with pytest.raises(ValueError, match="Invalid subrequirement disposition"):
        validate_shape(context, evidence, inputs, draft)


@pytest.mark.parametrize("status", ["excluded", "deferred"])
def test_explicit_model_id_edits_resolve_existing_validator_without_other_changes(
    status,
):
    context, evidence, inputs, draft = captured_shape(status)
    before = deepcopy(draft)
    fixed = apply_edits(draft, id_list_edits())
    assert not diagnostics(fixed, inputs["independent_expectations"])
    resolved = validate_shape(context, evidence, inputs, fixed)
    for i, j in POSITIONS:
        original = draft["coverage"][i]["subrequirements"][j]
        assert fixed["coverage"][i]["subrequirements"][j] == {
            **original,
            "criterion_ids": [],
        }
        assert resolved[i]["subrequirements"][j]["status"] == status
        fixed["coverage"][i]["subrequirements"][j]["criterion_ids"] = deepcopy(
            original["criterion_ids"]
        )
    assert fixed == draft == before


@pytest.mark.parametrize(
    "defect",
    [
        "empty_reason",
        "unknown_status",
        "unknown_part",
        "unknown_target",
        "unknown_evidence",
        "mapped_empty",
    ],
)
def test_other_invalid_fields_remain_diagnosed_and_rejected_after_id_edits(defect):
    context, evidence, inputs, draft = captured_shape()
    part = draft["coverage"][17]["subrequirements"][3]
    if defect == "empty_reason":
        part["reason"] = " "
    elif defect == "unknown_status":
        part["status"] = "unsupported-enum"
    elif defect == "unknown_part":
        part["part_id"] = "nonexistent"
    elif defect == "unknown_target":
        draft["coverage"][0]["subrequirements"][0]["criterion_ids"] = ["nonexistent"]
    elif defect == "unknown_evidence":
        draft["criteria"][0]["evidence_refs"] = ["nonexistent"]
    else:
        draft["coverage"][0]["subrequirements"][0]["criterion_ids"] = []
    issues = diagnostics(draft, inputs["independent_expectations"])
    if defect == "empty_reason":
        row_issue = next(i for i in issues if "invalid_fields" in i)
        assert row_issue["invalid_fields"] == [
            "reason must be nonempty explanatory text"
        ]
        assert row_issue["current_status"] == "excluded"
        assert "allowed_statuses" not in row_issue
        assert "remains deferred" not in row_issue["reason"]
    fixed = apply_edits(draft, id_list_edits())
    if defect != "unknown_evidence":
        assert diagnostics(fixed, inputs["independent_expectations"])
    with pytest.raises(ValueError):
        validate_shape(context, evidence, inputs, fixed)


@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize("correct_ids", [False, True])
@pytest.mark.asyncio
async def test_actual_role_three_reply_chain_preserves_two_repair_budget(
    roles, bounded, correct_ids
):
    make, calls = roles
    context, evidence, inputs, draft = captured_shape()
    initial = deepcopy(draft)
    for row in initial["coverage"]:
        row["expectation_id"] = "criterion-name-is-not-an-expectation-id"
    original = deepcopy(initial)
    records = {}
    attempts = 0

    def reply(wire):
        nonlocal attempts
        attempts += 1
        request = expand_texts(wire)
        if attempts == 1:
            assert request["mode"] == "outcome_jury_reduce"
            return deepcopy(initial)
        assert request["mode"] == "outcome_jury_repair"
        repair = request["validation_repair"]
        if attempts == 2:
            assert repair["invalid_draft"] == initial
            return {
                "edits": [
                    edit(["coverage", i, "expectation_id"], row["expectation_id"])
                    for i, row in enumerate(draft["coverage"])
                ]
            }
        assert attempts == 3
        assert repair["invalid_draft"] == draft
        assert repair["reason"] == "Invalid subrequirement disposition"
        assert [i["path"] for i in repair["field_issues"]] == [
            ["coverage", i, "subrequirements", j, "criterion_ids"] for i, j in POSITIONS
        ]
        assert all(i["current_status"] == "excluded" for i in repair["field_issues"])
        if correct_ids:
            return id_list_edits()
        # The actual native v2 reply changed only status and must still fail.
        return {
            "edits": [
                edit(["coverage", i, "subrequirements", j, "status"], "deferred")
                for i, j in POSITIONS
            ]
        }

    reducer = make("reducer", reply, bounded=bounded)
    result = jury.reduce_proposals(
        context,
        evidence,
        inputs,
        reducer,
        record=lambda name, value: records.update({name: deepcopy(value)}),
    )
    if correct_ids:
        output = await result
        assert output["criteria"] == draft["criteria"]
        for i, j in POSITIONS:
            assert output["coverage"][i]["subrequirements"][j]["status"] == "excluded"
    else:
        with pytest.raises(ValueError, match="Invalid subrequirement disposition"):
            await result
        assert "jury" not in records
        for i, j in POSITIONS:
            part = records["jury_reduce_repair_2_raw"]["coverage"][i][
                "subrequirements"
            ][j]
            assert part["status"] == "deferred"
            assert part["criterion_ids"] == ["organized_structure"]
    assert len(calls) == attempts == 3
    assert records["jury_reduce_raw"] == initial == original
    assert records["jury_reduce_repair_1_raw"] == draft
    assert all("response_recovery" not in request for _, request in calls)
