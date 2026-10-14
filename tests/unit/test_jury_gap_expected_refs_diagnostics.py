"""Offline native-v3-shaped provenance repairs; no native result is repaired."""

import socket
from copy import deepcopy

import pytest

from reward_engine.core.outcome import jury
from reward_engine.core.outcome.context import build_context
from reward_engine.core.outcome.jury_drafts import expectations, normalize_reduced
from reward_engine.core.outcome.jury_repair import apply_edits, diagnostics
from reward_engine.core.outcome.reducer_projection import expand_texts
from tests.unit.test_claw_outcome_jury_details import criterion, decision
from tests.unit.test_outcome_role_contract_recovery import roles as roles


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Only offline MockTransport calls are permitted")

    monkeypatch.setattr(socket.socket, "connect", forbidden)


def fixture():
    context, evidence = build_context(
        {"task_id": "fixture", "prompt": "Deliver valid JSON and explain it."},
        [{"artifact": {"final_response": "{"}} for _ in range(3)],
    )
    row = {**criterion(), "origin": "task", "kind": "hard"}
    expected = expectations([row], "seat", "before_products")
    gap = {
        "gap_id": "syntax",
        "expected": "Valid JSON",
        "expected_basis": "task",
        "expected_refs": ["task"],
        "product_ids": ["product_3"],
        "observed": "An opening brace",
        "gap": "Possibly incomplete JSON",
        "evidence_refs": ["product_3:artifact:final_response"],
        "verification": "Parse the response",
        "status": "needs_check",
    }
    draft = {
        "criteria": [row],
        "gaps": [
            gap,
            {
                **deepcopy(gap),
                "gap_id": "second",
                "expected_refs": ["product_3:environment"],
            },
        ],
        "coverage": [decision(expected[0])],
        "uncertainties": [],
    }
    inputs = {
        "independent_expectations": expected,
        "proposal_expectations": [],
        "proposals": [{"member": str(i)} for i in range(3)],
        "revision_protocol": True,
    }
    return context, evidence, inputs, draft


def inspect(draft, evidence, inputs):
    return diagnostics(draft, inputs["independent_expectations"], evidence=evidence)


@pytest.mark.parametrize("missing", [False, True])
def test_exact_existing_path_and_explicit_model_edit(missing):
    context, evidence, inputs, draft = fixture()
    if missing:
        del draft["gaps"][1]["expected_refs"]
    before = deepcopy(draft)
    with pytest.raises(ValueError, match="Task expectation must refer to the Task"):
        jury.validate(draft, evidence, context)
    (issue,) = inspect(draft, evidence, inputs)
    assert issue["path"] == (["gaps", 1] if missing else ["gaps", 1, "expected_refs"])
    assert issue["field_present"] is not missing
    assert issue["current_basis"] == "task"
    value = {**draft["gaps"][1], "expected_refs": ["task"]} if missing else ["task"]
    fixed = apply_edits(
        draft,
        {
            "edits": [
                {
                    "op": "replace",
                    "path": issue["path"],
                    "value": value,
                    "reason": "The supplied Task explicitly requires valid JSON.",
                }
            ]
        },
    )
    output = jury.validate(fixed, evidence, context)
    jury.reconcile_coverage(inputs["independent_expectations"], output)
    assert output["gaps"][1]["evidence_refs"] == before["gaps"][1]["evidence_refs"]
    assert output["criteria"] == before["criteria"]
    assert draft == before
    fixed["gaps"][1]["product_ids"] = ["nonexistent"]
    with pytest.raises(ValueError, match="Gap must identify observed products"):
        jury.validate(fixed, evidence, context)


@pytest.mark.parametrize("refs", ["task", [None], {"task": True}, None])
def test_malformed_refs_are_diagnosed_without_mutation(refs):
    _, evidence, inputs, draft = fixture()
    draft["gaps"][1]["expected_refs"] = refs
    before = deepcopy(draft)
    (issue,) = inspect(draft, evidence, inputs)
    assert issue["path"] == ["gaps", 1, "expected_refs"]
    assert issue["value"] == refs
    assert draft == before


def test_alias_and_domain_empty_refs_keep_existing_acceptance():
    context, evidence, inputs, draft = fixture()
    draft["gaps"][1]["expected_refs"] = ["task:fixture"]
    assert inspect(draft, evidence, inputs) == []
    jury.validate(normalize_reduced(draft, context, evidence), evidence, context)
    draft["gaps"][1].update(expected_basis="domain", expected_refs=[])
    assert inspect(draft, evidence, inputs) == []
    jury.validate(draft, evidence, context)
    draft["gaps"][1]["expected_refs"] = ["unknown:ref"]
    with pytest.raises(ValueError, match="exact supplied evidence handle"):
        jury.validate(draft, evidence, context)


@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize("correct", [False, True])
@pytest.mark.asyncio
async def test_actual_role_correction_or_unchanged_failure_has_existing_budget(
    roles, bounded, correct
):
    context, evidence, inputs, draft = fixture()
    original = deepcopy(draft)
    make, calls = roles
    count = 0

    def reply(wire):
        nonlocal count
        count += 1
        request = expand_texts(wire)
        if count == 1:
            return deepcopy(draft)
        assert count <= 3
        repair = request["validation_repair"]
        assert repair["invalid_draft"] == original
        (issue,) = repair["field_issues"]
        assert issue["path"] == ["gaps", 1, "expected_refs"]
        return {
            "edits": [
                {
                    "op": "replace",
                    "path": issue["path"],
                    "value": ["task"] if correct else ["product_3:environment"],
                    "reason": "Explicit scripted provenance correction after inspecting Task.",
                }
            ]
        }

    reducer = make("reducer", reply, bounded=bounded)
    records = {}
    result = jury.reduce_proposals(
        context,
        evidence,
        inputs,
        reducer,
        record=lambda name, value: records.update({name: deepcopy(value)}),
    )
    if correct:
        output = await result
        assert count == 2
        assert output["criteria"] == draft["criteria"]
        assert output["gaps"][1] == {**draft["gaps"][1], "expected_refs": ["task"]}
    else:
        with pytest.raises(ValueError, match="Task expectation must refer to the Task"):
            await result
        # The unchanged second repair request reuses the Role's identical-input
        # receipt; both allowed repair slots still fail, with no fourth slot.
        assert count == 2
        assert "jury_reduce_validation_2" in records
        assert "jury_reduce_validation_3" not in records
        assert "jury" not in records
    assert len(calls) == count
    assert draft == original
