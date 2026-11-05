"""Synthetic regressions for ambiguous member-revision recovery diagnostics."""

from copy import deepcopy

import pytest

from reward_engine.core.outcome.context import ResponseContractError
from reward_engine.core.outcome.jury_drafts import expectations
from reward_engine.core.outcome.jury_repair import diagnostics
from reward_engine.core.outcome.jury_revision import materialize
from reward_engine.core.outcome.protocol_repair import ProtocolRepairCall


def test_reference_errors_do_not_hide_domain_strength_conflict():
    raw = {"criteria": [{"id": "c", "standard": "A preference", "measurement_question": "Check it",
                         "kind": "hard", "origin": "domain", "evidence_refs": []}]}
    original = deepcopy(raw)
    issues = diagnostics(raw, [])
    assert {tuple(x["path"]) for x in issues} >= {("criteria", 0, "kind"), ("criteria", 0, "evidence_refs")}
    assert raw == original


def test_clause_projection_diagnoses_all_mismatches_without_changing_draft():
    raw = {"criteria": [
        {"id": f"c{i}", "clauses": [{"text": f"Required {i}"}],
         "subrequirements": [f"Different {i}"]} for i in range(2)]}
    original = deepcopy(raw)
    with pytest.raises(ResponseContractError) as caught:
        materialize(raw, [], {})
    issues = caught.value.details["subrequirement_mismatches"]
    assert [x["path"] for x in issues] == [["criteria", i, "subrequirements"] for i in range(2)]
    assert [x["clause_texts"] for x in issues] == [[f"Required {i}"] for i in range(2)]
    assert raw == original


def test_projection_mismatch_does_not_hide_changed_retain_evidence():
    criterion = {"id": "c", "standard": "Keep source", "evidence_refs": ["task"],
                 "clauses": [{"text": "Keep source"}], "subrequirements": ["Keep source"]}
    initial = expectations([criterion], "s", "before_products")
    raw = {"criteria": [{**deepcopy(criterion), "evidence_refs": ["task", "source"],
                         "subrequirements": ["Different wording"]}],
           "revision_decisions": [{"source_parts": ["s:0/0"], "action": "retain",
                                   "targets": [{"criterion_id": "c", "part_index": 0}]}]}
    original = deepcopy(raw)
    with pytest.raises(ResponseContractError) as caught:
        materialize(raw, initial, {"task": "Keep source", "source": "Evidence"})
    details = caught.value.details
    assert details["subrequirement_mismatches"]
    assert details["downstream_diagnostics"]["retain_mismatches"]
    assert raw == original


@pytest.mark.asyncio
async def test_member_repair_exposes_missing_fields_and_uses_add():
    base = {"criteria": [{"id": "c"}], "revision_decisions": [
        {"action": "retain", "source_parts": ["s/0"], "targets": []}]}
    original = deepcopy(base)
    async def model(request):
        fields = request["protocol_repair"]["decision_fields"][0]
        assert fields["path"] == ["revision_decisions", 0]
        assert "evidence_refs" in fields["absent_fields"]
        assert fields["existing_fields"] == ["action", "source_parts", "targets"]
        assert "part_decisions" not in request["protocol_repair"]["decision_instruction"]
        return {"base_sha256": request["protocol_repair"]["base_sha256"], "edits": [
            {"op": "add", "path": [*fields["path"], "evidence_refs"],
             "value": ["task"], "reason": "Explicit model-selected support"}]}
    reply = await ProtocolRepairCall(model)({"mode": "outcome_jury_map", "response": {},
        "response_recovery": {"rejected_response": base, "details": {}}})
    assert reply["revision_decisions"][0]["evidence_refs"] == ["task"]
    assert base == original


@pytest.mark.asyncio
async def test_reference_preflight_reports_criterion_and_clause_independently():
    base = {"criteria": [{"id": "c", "evidence_refs": [], "clauses": [
        {"text": "Keep source", "evidence_refs": ["missing"]}]}]}
    async def model(request):
        issues = request["protocol_repair"]["reference_preflight"]
        assert request["protocol_repair"]["target_catalog"] == [
            {"criterion_id": "c", "part_index": 0, "text": "Keep source"}]
        assert [x["path"] for x in issues] == [
            ["criteria", 0, "evidence_refs"], ["criteria", 0, "clauses", 0, "evidence_refs"]]
        return {"base_sha256": request["protocol_repair"]["base_sha256"], "edits": [
            {"op": "replace", "path": issue["path"], "value": ["task"],
             "reason": "The supplied task explicitly requires this clause"} for issue in issues]}
    result = await ProtocolRepairCall(model)({"mode": "outcome_jury_map", "response": {},
        "context": {"available_evidence_refs": ["task"]},
        "response_recovery": {"rejected_response": base, "details": {}}})
    assert result["criteria"][0]["clauses"][0]["evidence_refs"] == ["task"]
    assert base["criteria"][0]["evidence_refs"] == []


@pytest.mark.parametrize('reason', [None, '', 7])
def test_edit_failure_identifies_its_own_reason_field_without_using_nested_reason(reason):
    from reward_engine.core.outcome.jury_repair import apply_edits

    base = {'revision_decisions': [{'action': 'retain', 'reason': 'Decision rationale'}]}
    original = deepcopy(base)
    edit = {'op': 'replace', 'path': ['revision_decisions'],
            'value': [{'action': 'rewrite', 'reason': 'Nested semantic rationale'}]}
    if reason is not None:
        edit['reason'] = reason
    with pytest.raises(ValueError) as caught:
        apply_edits(base, {'edits': [edit]}, allowed_roots={'revision_decisions'})
    details = caught.value.details
    assert details['edit_index'] == 0 and details['batch_applied'] is False
    assert details['edit_fields'] == sorted(edit)
    if reason is None:
        assert details['missing_edit_fields'] == ['reason']
    else:
        assert details['invalid_edit_fields'] == ['reason']
    assert base == original


@pytest.mark.parametrize("change", ["missing", "changed", "extra_null", "missing_null"])
@pytest.mark.asyncio
async def test_retain_repair_locates_criterion_field_and_still_requires_model_edit(change):
    from tests.unit.test_jury_retain_diagnostics import captured_shape

    initial, base = captured_shape()
    field, op = "why", "add"
    source_value = base["criteria"][0][field]
    if change == "missing":
        base["criteria"][0].pop(field)
    elif change == "changed":
        base["criteria"][0][field] = "Only partial delivery is needed."
        op = "replace"
    elif change == "extra_null":
        field, op = "condition", "remove"
        base["criteria"][0][field] = None
    else:
        field, source_value = "condition", None
        initial[0][field] = None
    original = deepcopy((initial, base))
    with pytest.raises(ResponseContractError) as caught:
        materialize(base, initial, {"task": "Task"})

    async def model(request):
        locations = request["protocol_repair"]["retain_field_locations"]
        # Two retained parts share one parent field; do not request duplicate adds.
        assert len(locations) == 1
        location = locations[0]
        assert location["path"] == ["criteria", 0, field]
        assert location["op_for_exact_restoration"] == op
        assert location["target_present"] == (op != "add")
        assert location["source_present"] == (op != "remove")
        assert "prefer ONE replace of the complete existing decision list" not in request["instruction"]
        if op != "remove":
            assert location["source_value"] == source_value
        # Re-emitting the retain decision cannot restore the missing criterion.
        with pytest.raises(ResponseContractError, match="Retain must preserve"):
            materialize(request["response_recovery"]["rejected_response"], initial, {"task": "Task"})
        edit = {"op": op, "path": location["path"], "reason": "Restore the exact original field presence and value"}
        if op != "remove":
            edit["value"] = source_value
        return {"base_sha256": request["protocol_repair"]["base_sha256"], "edits": [edit]}

    result = await ProtocolRepairCall(model)({"mode": "outcome_jury_map", "response": {},
        "response_recovery": {"rejected_response": base, "details": caught.value.details}})
    materialize(result, initial, {"task": "Task"})
    assert (initial, base) == original
