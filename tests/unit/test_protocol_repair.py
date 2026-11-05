"""Synthetic model-authored local repairs retain the exact original draft."""

from copy import deepcopy

import pytest

from reward_engine.core.outcome.context import ResponseContractError, ask
from reward_engine.core.outcome.jury_refinement import check_retired_clauses


@pytest.mark.parametrize("match_count", [0, 1, 2])
def test_binding_diagnostics_list_all_exact_matches_without_choosing(match_count):
    from reward_engine.core.outcome.jury_drafts import apply_part_decisions

    # Long near-matches must not become exact matches through diagnostic truncation.
    text = "Synthetic requirement " + "x" * 1100
    inputs = {"independent_expectations": [{"expectation_id": "s", "subrequirements": [
        {"part_id": "s/0", "text": text}]}], "proposal_expectations": []}
    raw = {"criteria": [{"id": "wrong", "clauses": [{"id": "w", "text": text + " different"}]}]
           + [{"id": f"c{i}", "clauses": [{"id": f"p{i}", "text": text}]} for i in range(match_count)],
           "part_decisions": [{"parts": ["p0"], "action": "retain", "criterion_id": "wrong", "clause_id": "w"}]}
    original = deepcopy(raw)
    with pytest.raises(ResponseContractError) as caught:
        apply_part_decisions(raw, inputs, {"task": "Synthetic"})
    issue = caught.value.details["binding_errors"][0]
    assert issue["exact_matching_clauses"] == [
        {"criterion_id": f"c{i}", "clause_id": f"p{i}", "path": ["criteria", i + 1, "clauses", 0]}
        for i in range(match_count)]
    assert raw == original


def test_part_decision_diagnostics_explain_unsupported_action_and_empty_parts():
    from reward_engine.core.outcome.jury_drafts import apply_part_decisions

    raw = {"criteria": [{"id": "c", "clauses": [{"id": "a", "text": "Synthetic"}]}],
           "part_decisions": [{"parts": [], "action": "add", "criterion_id": "c", "clause_id": "a"}]}
    original = deepcopy(raw)
    with pytest.raises(ResponseContractError) as caught:
        apply_part_decisions(raw, {"independent_expectations": [], "proposal_expectations": []}, {})
    issue = caught.value.details["decision_errors"][0]
    assert issue["invalid_action"] == "add"
    assert issue["allowed_actions"] == ["retain", "rewrite", "defer", "exclude"]
    assert issue["parts_error"] == "Provide a nonempty list of supplied selection IDs"
    assert raw == original


def test_selector_diagnostics_include_collisions_between_invalid_rows():
    from reward_engine.core.outcome.jury_drafts import apply_part_decisions

    inputs = {"independent_expectations": [{"expectation_id": "synthetic", "subrequirements": [
        {"part_id": "synthetic/0", "text": "Same requirement"},
        {"part_id": "synthetic/1", "text": "Same requirement"}]}], "proposal_expectations": []}
    raw = {"criteria": [{"id": "final"}], "selection_protocol": "exact-alias-v1",
           "part_decisions": [
               {"parts": ["p0", "p1"], "action": "defer", "reason": "Synthetic"},
               {"parts": ["p0"], "action": "defer", "reason": "Synthetic"}]}
    saved = deepcopy(raw)
    with pytest.raises(ResponseContractError) as caught:
        apply_part_decisions(raw, inputs, {"task": "Task"})
    issues = caught.value.details["selector_conflicts"]
    duplicate = next(e for e in issues if e["selection_id"] == "p0")
    assert [row["path"] for row in duplicate["occurrences"]] == [
        ["part_decisions", 0, "parts", 0], ["part_decisions", 1, "parts", 0]]
    alias = next(e for e in issues if e["selection_id"] == "p1")
    assert [row["via"] for row in alias["occurrences"]] == ["p0", "p1", "p0"]
    assert raw == saved


def test_retirement_selector_errors_identify_all_unknown_and_duplicate_paths():
    inputs = {"independent_expectations": [{"subrequirements": [{"selection_id": "unused",
        "part_id": "synthetic/0", "text": "Requirement"}]}], "proposal_expectations": []}
    raw = {"criteria": [], "part_decisions": [
        {"parts": ["p0", "missing"], "action": "defer"},
        {"parts": ["p0"], "action": "defer"}]}
    with pytest.raises(ResponseContractError) as caught:
        check_retired_clauses(raw, inputs)
    issues = caught.value.details["selector_conflicts"]
    assert {e["selection_id"] for e in issues} == {"p0", "missing"}
    assert next(e for e in issues if e["selection_id"] == "missing")["occurrences"][0]["path"] == [
        "part_decisions", 0, "parts", 1]


@pytest.mark.asyncio
async def test_repair_can_add_missing_root_and_keeps_original_contract():
    original = {"criteria": [{"id": "keep", "standard": "Preserved"}]}
    schema = {"criteria": [], "part_decisions": [{"text": "full replacement for rewrite",
                                                "evidence_refs": ["support for rewrite"]}]}
    calls = []
    class Role:
        protocol_edits = True
        async def __call__(self, request):
            calls.append(deepcopy(request))
            if len(calls) == 1:
                return deepcopy(original)
            assert request["protocol_repair"]["draft_response_schema"] == schema
            assert request["protocol_repair"]["edit_budget"]["max_edits"] == 64
            return {"base_sha256": request["protocol_repair"]["base_sha256"], "edits": [
                {"op": "add", "path": ["part_decisions"], "value": [],
                 "reason": "Supply required decision ledger; no invented decisions"}]}
    def validate(row):
        if "part_decisions" not in row:
            raise ResponseContractError("part_decisions", "Required field missing")
        assert row["criteria"] == original["criteria"]
        return row
    result = await ask(Role(), "outcome_jury_reduce", {}, {}, "Keep complete rewrite fields",
                       schema, validate=validate, max_contract_retries=1)
    assert result == {**original, "part_decisions": []}
    assert original == {"criteria": [{"id": "keep", "standard": "Preserved"}]}
    assert len(calls) == 2


@pytest.mark.parametrize("path", [["criteria"], ["criteria", 1], ["private_score"]])
def test_add_edit_never_overwrites_appends_or_escapes_allowed_roots(path):
    from reward_engine.core.outcome.jury_repair import apply_edits
    draft = {"criteria": [{"id": "keep"}]}
    with pytest.raises(ValueError):
        apply_edits(draft, {"edits": [{"op": "add", "path": path, "value": {},
                                      "reason": "Synthetic"}]}, allow_add=True)
    assert draft == {"criteria": [{"id": "keep"}]}


def test_add_is_opt_in_and_batch_is_atomic():
    from reward_engine.core.outcome.jury_repair import apply_edits
    draft = {"criteria": [{"id": "keep"}]}
    edit = {"op": "add", "path": ["criteria", 0, "kind"], "value": "hard", "reason": "Task"}
    with pytest.raises(ValueError):
        apply_edits(draft, {"edits": [edit]})
    assert apply_edits(draft, {"edits": [edit]}, allow_add=True)["criteria"][0]["kind"] == "hard"
    with pytest.raises(ValueError):
        apply_edits(draft, {"edits": [edit, {**edit, "value": "soft"}]}, allow_add=True)
    assert draft == {"criteria": [{"id": "keep"}]}


@pytest.mark.asyncio
async def test_jury_repair_applies_only_model_edits_and_revalidates():
    original = {"criteria": [{"id": "keep", "standard": "Preserved"},
                             {"id": "fix", "standard": "Grounded", "kind": "critical"}],
                "revision_decisions": []}
    calls = []
    class Role:
        protocol_edits = True
        async def __call__(self, request):
            calls.append(deepcopy(request))
            if len(calls) == 1:
                return deepcopy(original)
            assert "edit_budget" not in request["protocol_repair"]
            assert request["response"] == {"base_sha256": "exact repair base hash", "edits": [
                {"op": "replace/remove/add", "path": ["draft key", 0, "field"],
                 "value": "model-authored JSON value for replace/add", "reason": "grounded correction"}]}
            return {"base_sha256": request["protocol_repair"]["base_sha256"],
                    "edits": [{"op": "replace", "path": ["criteria", 1, "kind"],
                               "value": "hard", "reason": "Task requirement strength"}]}
    def validate(row):
        if row["criteria"][1]["kind"] != "hard":
            raise ResponseContractError("criteria[1].kind", "Must express strength")
        return row
    result = await ask(Role(), "outcome_jury_map", {}, {}, "synthetic", {"criteria": []},
                       validate=validate, max_contract_retries=1)
    assert result["criteria"][0] == original["criteria"][0]
    assert result["criteria"][1] == {**original["criteria"][1], "kind": "hard"}
    assert original["criteria"][1]["kind"] == "critical"
    assert len(calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("defect", ["extra", "missing", "hash", "not_object"])
async def test_repair_envelope_diagnoses_exact_shape_and_hash_without_applying(defect):
    from reward_engine.core.outcome.construction import fingerprint
    from reward_engine.core.outcome.protocol_repair import ProtocolRepairCall

    base = {"criteria": [{"id": "synthetic", "kind": "hard"}]}
    saved = deepcopy(base)
    answer = {"base_sha256": fingerprint(base), "edits": [
        {"op": "replace", "path": ["criteria", 0, "kind"], "value": "soft", "reason": "Synthetic"}]}
    if defect == "extra":
        answer["type"] = "repair"
    elif defect == "missing":
        del answer["edits"]
    elif defect == "hash":
        answer["base_sha256"] = "another draft"
    else:
        answer = []
    async def call(request):
        return deepcopy(answer)
    request = {"mode": "outcome_jury_reduce", "response_recovery": {
        "rejected_response": base, "details": {"field": "criteria", "reason": "Synthetic"}}}
    with pytest.raises(ResponseContractError) as caught:
        await ProtocolRepairCall(call)(request)
    details = caught.value.details
    assert details["expected_fields"] == ["base_sha256", "edits"]
    assert details["expected_base_sha256"] == fingerprint(base)
    assert details["actual_type"] == ("list" if defect == "not_object" else "dict")
    assert details["extra_fields"] == (["type"] if defect == "extra" else [])
    assert details["missing_fields"] == (
        ["base_sha256", "edits"] if defect == "not_object" else ["edits"] if defect == "missing" else [])
    assert details["base_hash_matches"] is (defect in {"extra", "missing"})
    if defect == "extra":
        assert "type" in str(caught.value) and "matches" in str(caught.value)
    assert base == saved


@pytest.mark.asyncio
async def test_repair_refuses_a_different_base():
    class Role:
        protocol_edits = True
        async def __call__(self, request):
            if "response_recovery" not in request:
                return {"criteria": [{"id": "synthetic"}]}
            return {"base_sha256": "another draft", "edits": []}
    def reject(row):
        raise ResponseContractError("criteria", "Synthetic failure")
    with pytest.raises(ResponseContractError, match="base"):
        await ask(Role(), "outcome_jury_reduce", {}, {}, "synthetic", {"criteria": []},
                  validate=reject, max_contract_retries=1)


def test_withdrawn_clause_error_identifies_text_and_conflicting_paths():
    inputs = {"independent_expectations": [{"expectation_id": "source", "subrequirements": [
        {"part_id": "source/0", "text": "Keep the exception"}]}], "proposal_expectations": []}
    raw = {"criteria": [{"id": "c", "subrequirements": ["Keep the exception"]}],
           "part_decisions": [{"parts": ["p0"], "action": "defer", "reason": "Needs evidence"}]}
    with pytest.raises(ValueError, match="withdrawn clause") as caught:
        check_retired_clauses(raw, inputs)
    assert caught.value.details["conflicts"] == [{
        "text": "Keep the exception", "criterion_paths": [["criteria", 0, "subrequirements", 0]],
        "decision_indices": [0]}]


@pytest.mark.asyncio
async def test_reduce_repair_explains_normalized_withdrawal_and_binding_errors():
    from reward_engine.core.outcome import jury

    inputs = {"proposals": [], "independent_expectations": [{"expectation_id": "source",
        "subrequirements": [{"part_id": f"source/{i}", "text": text}
                            for i, text in enumerate(["Keep A", "Keep B", "Keep C"])]}],
        "proposal_expectations": []}
    raw = {"criteria": [{"id": "final", "clauses": [
        {"id": "a", "text": "Keep A", "evidence_refs": ["task"]},
        {"id": "b", "text": "Keep B", "evidence_refs": ["task"]}]}],
        "part_decisions": [
            {"parts": ["p0", "p1"], "action": "retain", "criterion_id": "final",
             "clause_id": "a", "reason": "Group the requirements"},
            {"parts": ["p2"], "action": "retain", "criterion_id": "final", "clause_id": "missing"}]}
    saved = deepcopy(raw)
    calls = []
    class Captured(Exception):
        pass

    async def role(request):
        calls.append(deepcopy(request))
        if len(calls) == 1:
            return deepcopy(raw)
        details = request["response_recovery"]["details"]
        assert details["conflicts"][0]["decision_indices"] == [0]
        decision = details["conflicting_decisions"][0]
        assert decision["decision_index"] == 0
        assert decision["decision"]["action"] == "rewrite"
        assert decision["decision"]["binding_resolution"]["provided_action"] == "retain"
        assert decision["source_texts"] == {"p0": "Keep A", "p1": "Keep B"}
        assert details["criterion_locations"][0]["clause_id"] == "b"
        assert details["validation_errors"][1]["binding_errors"][0]["decision_index"] == 1
        raise Captured()

    role.protocol_edits = True
    with pytest.raises(Captured):
        await jury._reduce_once({"task": {"task_id": "synthetic"}}, {"task": "Task"}, inputs,
            role, record=lambda *args: None, repair={}, strict=True, clause_contract=True)
    assert len(calls) == 2 and raw == saved


@pytest.mark.asyncio
async def test_edit_syntax_retry_keeps_exact_base_and_strict_validation():
    import json

    original = {"criteria": [{"id": "same", "standard": "Unchanged"}],
                "revision_decisions": [{"action": "invalid"}]}
    calls = []
    class Role:
        protocol_edits = True
        bounded_stage = True
        stage_timeout_s = 10
        async def __call__(self, request):
            calls.append(deepcopy(request))
            if len(calls) == 1:
                return deepcopy(original)
            if len(calls) == 2:
                raise json.JSONDecodeError("truncated", '{"edits":', 9)
            assert request["response_recovery"]["rejected_response"] == original
            assert request["protocol_repair"] == calls[1]["protocol_repair"]
            assert request["response_recovery"]["details"] == calls[1]["response_recovery"]["details"]
            assert request["response_recovery"]["details"]["reason"] == "Unsupported disposition"
            assert request["response_recovery"]["syntax_error"]["error_type"] == "JSONDecodeError"
            return {"base_sha256": request["protocol_repair"]["base_sha256"],
                    "edits": [{"op": "replace", "path": ["revision_decisions", 0, "action"],
                               "value": "retain", "reason": "Preserves the supplied requirement"}]}
    def validate(row):
        if row["revision_decisions"][0]["action"] != "retain":
            raise ResponseContractError("revision_decisions", "Unsupported disposition")
        assert row["criteria"] == original["criteria"]
        return row
    result = await ask(Role(), "outcome_jury_map", {}, {}, "synthetic", {}, validate=validate)
    assert result["revision_decisions"][0]["action"] == "retain"
    assert len(calls) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("path,value", [(["private_score"], 1), (["criteria", 0, "kind"], "still invalid")])
async def test_edits_cannot_bypass_scope_or_original_validator(path, value):
    class Role:
        protocol_edits = True
        async def __call__(self, request):
            if "response_recovery" not in request:
                return {"criteria": [{"kind": "invalid"}], "private_score": 0}
            return {"base_sha256": request["protocol_repair"]["base_sha256"],
                    "edits": [{"op": "replace", "path": path, "value": value, "reason": "Synthetic"}]}
    def validate(row):
        raise ResponseContractError("criteria", "Invalid requirement")
    with pytest.raises(ResponseContractError):
        await ask(Role(), "outcome_jury_map", {}, {}, "synthetic", {},
                  validate=validate, max_contract_retries=1)


@pytest.mark.asyncio
async def test_strict_jury_wrappers_repair_inheritance_then_materialize():
    from reward_engine.core.outcome import jury
    from reward_engine.core.outcome.jury_revision import materialize
    from tests.unit.test_outcome_jury_revision import criterion, initial

    calls = []
    async def role(request):
        calls.append(request)
        if len(calls) == 1:
            return {"criteria": [{"id": "json", "inherit_from": "seat:0"}]}
        return {"base_sha256": request["protocol_repair"]["base_sha256"],
                "edits": [{"op": "replace", "path": ["criteria", 0],
                           "value": {"inherit_from": "seat:0"},
                           "reason": "Reuse the full supplied expectation without an override"}]}
    role.protocol_edits = True
    result = await jury.ask_draft(jury.field_repair_call(role), "outcome_jury_map", {},
                                  {"task": "Task"}, "synthetic", {},
                                  validate=lambda row: materialize(row, initial(), {"task": "Task"}),
                                  max_contract_retries=1)
    assert len(calls) == 2
    assert result["criteria"] == [criterion()]
    assert all(row["action"] == "retain" for row in result["revision_decisions"])


@pytest.mark.asyncio
async def test_one_bounded_repair_reports_all_inheritance_errors_to_model():
    from reward_engine.core.outcome import jury
    from reward_engine.core.outcome.jury_drafts import expectations
    from reward_engine.core.outcome.jury_revision import materialize
    from tests.unit.test_outcome_jury_revision import criterion

    criteria = [{**criterion(), "id": f"c{i}"} for i in range(3)]
    before = expectations(criteria, "seat", "before_products")
    original = {"criteria": [{"id": f"c{i}", "inherit_from": f"seat:{i}"} for i in range(3)]}
    calls = []

    async def role(request):
        calls.append(deepcopy(request))
        if len(calls) == 1:
            return deepcopy(original)
        details = request["response_recovery"]["details"]
        # The model corrects exactly the locations diagnosed by the validator.
        issues = details.get("invalid_inheritance", [details])
        return {"base_sha256": request["protocol_repair"]["base_sha256"], "edits": [
            {"op": "replace", "path": ["criteria", issue["criterion_index"]],
             "value": {"inherit_from": f"seat:{issue['criterion_index']}"},
             "reason": "Preserve the complete original criterion without overrides"}
            for issue in issues]}

    role.protocol_edits = True
    result = await jury.ask_draft(jury.field_repair_call(role), "outcome_jury_map", {},
        {"task": "Task"}, "synthetic", {},
        validate=lambda row: materialize(row, before, {"task": "Task"}),
        max_contract_retries=1)
    assert result["criteria"] == criteria
    assert len(calls) == 2
    assert all(row["action"] == "retain" for row in result["revision_decisions"])


@pytest.mark.asyncio
async def test_one_bounded_repair_includes_target_errors_hidden_by_inheritance():
    from reward_engine.core.outcome import jury
    from reward_engine.core.outcome.jury_revision import materialize
    from tests.unit.test_outcome_jury_revision import criterion, initial, target

    original = {"criteria": [{"inherit_from": "json"}], "revision_decisions": [
        {"source_parts": ["seat:0/1"], "action": "retain",
         "targets": [{"criterion_id": "json_typo", "part_index": 1}]}]}
    calls = []

    async def role(request):
        calls.append(deepcopy(request))
        if len(calls) == 1:
            return deepcopy(original)
        details = request["response_recovery"]["details"]
        assert details["invalid_inheritance"]
        assert details["reference_preflight"]["invalid_targets"]
        return {"base_sha256": request["protocol_repair"]["base_sha256"], "edits": [
            {"op": "replace", "path": ["criteria", 0],
             "value": {"inherit_from": "seat:0"}, "reason": "Use the exact expectation handle"},
            {"op": "replace", "path": ["revision_decisions", 0, "targets", 0],
             "value": target(1), "reason": "Use the actual inherited criterion ID"}]}

    role.protocol_edits = True
    result = await jury.ask_draft(jury.field_repair_call(role), "outcome_jury_map", {},
        {"task": "Task"}, "synthetic", {},
        validate=lambda row: materialize(row, initial(), {"task": "Task"}), max_contract_retries=1)
    assert len(calls) == 2
    assert result["criteria"] == [criterion()]
    assert all(row["action"] == "retain" for row in result["revision_decisions"])


@pytest.mark.asyncio
async def test_bounded_repair_gets_retain_mismatch_before_inheritance_is_resolved():
    from reward_engine.core.outcome import jury
    from reward_engine.core.outcome.jury_revision import materialize
    from tests.unit.test_outcome_jury_revision import criterion, initial

    original = {"criteria": [{"inherit_from": "json"},
        {**criterion(), "id": "changed", "priority_reason": "A proposed clarification"}],
        "revision_decisions": [{"source_parts": ["seat:0/0"], "action": "retain",
            "targets": [{"criterion_id": "changed", "part_index": 0}]}]}
    calls = []
    async def role(request):
        calls.append(deepcopy(request))
        if len(calls) == 1:
            return deepcopy(original)
        details = request["response_recovery"]["details"]
        mismatch = details["reference_preflight"]["retain_errors"][0]
        assert mismatch["retain_mismatches"]
        assert mismatch["path"] == ["revision_decisions", 0]
        return {"base_sha256": request["protocol_repair"]["base_sha256"], "edits": [
            {"op": "replace", "path": ["criteria", 0], "value": {"inherit_from": "seat:0"},
             "reason": "Use the existing expectation handle"},
            {"op": "replace", "path": mismatch["path"],
             "value": {**original["revision_decisions"][0], "action": "rewrite",
                       "reason": "Record the proposed parent rationale change", "evidence_refs": ["task"]},
             "reason": "The source and target parent fields differ"}]}
    role.protocol_edits = True
    result = await jury.ask_draft(jury.field_repair_call(role), "outcome_jury_map", {},
        {"task": "Task"}, "synthetic", {},
        validate=lambda row: materialize(row, initial(), {"task": "Task"}), max_contract_retries=1)
    assert len(calls) == 2
    assert result["revision_decisions"][0]["action"] == "rewrite"
    assert result["criteria"][1] == original["criteria"][1]


@pytest.mark.asyncio
async def test_one_bounded_repair_reports_all_invalid_retain_decisions():
    from reward_engine.core.outcome import jury
    from reward_engine.core.outcome.jury_revision import materialize
    from tests.unit.test_outcome_jury_revision import criterion, initial, target

    before = initial()
    original = {"criteria": [{**criterion(), "priority_reason": "A model-authored clarification"}],
                "revision_decisions": [
                    {"source_parts": [part["part_id"]], "action": "retain", "targets": [target(i)]}
                    for i, part in enumerate(before[0]["subrequirements"])]}
    calls = []

    async def role(request):
        calls.append(deepcopy(request))
        if len(calls) == 1:
            return deepcopy(original)
        details = request["response_recovery"]["details"]
        errors = details.get("decision_errors", [details])
        return {"base_sha256": request["protocol_repair"]["base_sha256"], "edits": [
            {"op": "replace", "path": ["revision_decisions", e["decision_index"]],
             "value": {**original["revision_decisions"][e["decision_index"]],
                       "action": "rewrite", "reason": "Explicitly propose the clarified parent explanation",
                       "evidence_refs": ["task"]}, "reason": "Declare the changed parent field"}
            for e in errors]}

    role.protocol_edits = True
    result = await jury.ask_draft(jury.field_repair_call(role), "outcome_jury_map", {},
        {"task": "Task"}, "synthetic", {},
        validate=lambda row: materialize(row, before, {"task": "Task"}), max_contract_retries=1)
    assert len(calls) == 2 and result["criteria"] == original["criteria"]
    assert [r["action"] for r in result["revision_decisions"]] == ["rewrite", "rewrite"]


@pytest.mark.asyncio
async def test_reducer_repair_sees_invalid_decisions_and_clause_bindings_together():
    from reward_engine.core.outcome.jury_drafts import apply_part_decisions

    inputs = {"independent_expectations": [{"expectation_id": "s", "subrequirements": [
        {"part_id": "s/0", "text": "Keep A"}, {"part_id": "s/1", "text": "Keep B"}]}],
        "proposal_expectations": []}
    original = {"criteria": [{"id": "final", "clauses": [
        {"id": "a", "text": "Keep A"}, {"id": "b", "text": "Keep B"}]}],
        "part_decisions": [
            {"parts": ["unknown"], "action": "retain", "criterion_id": "final", "clause_id": "a"},
            {"parts": ["p0"], "action": "retain", "criterion_id": "final", "clause_id": "b"},
            {"parts": ["p1"], "action": "invalid", "criterion_id": "final", "clause_id": "b"}]}
    saved = deepcopy(original)
    calls = []

    class Role:
        protocol_edits = True

        async def __call__(self, request):
            calls.append(deepcopy(request))
            if len(calls) == 1:
                return deepcopy(original)
            detail = request["response_recovery"]["details"]
            assert [e["decision_index"] for e in detail["decision_errors"]] == [0, 2]
            assert detail["binding_errors"][0]["selection_id"] == "p0"
            return {"base_sha256": request["protocol_repair"]["base_sha256"], "edits": [
                {"op": "replace", "path": ["part_decisions"], "value": [
                    {"parts": ["p0"], "action": "retain", "criterion_id": "final", "clause_id": "a"},
                    {"parts": ["p1"], "action": "retain", "criterion_id": "final", "clause_id": "b"}],
                 "reason": "Bind each original part to its exact unchanged clause"}]}

    result = await ask(Role(), "outcome_jury_reduce", {}, {}, "synthetic", {},
                       validate=lambda row: apply_part_decisions(row, inputs, {}), max_contract_retries=1)
    assert len(calls) == 2 and original == saved
    assert [p["status"] for p in result["coverage"][0]["subrequirements"]] == ["mapped", "mapped"]


@pytest.mark.asyncio
async def test_repair_preserves_projected_texts():
    from reward_engine.core.outcome.construction import fingerprint
    from reward_engine.core.outcome.protocol_repair import ProtocolRepairCall
    from reward_engine.core.outcome.reducer_projection import share_texts

    text = "Preserve this exact synthetic requirement. " * 12
    draft = {"criteria": [{"standard": text, "kind": "invalid"}]}
    request = share_texts({"mode": "outcome_jury_reduce", "context": {"requirement": text},
                          "response_recovery": {"rejected_response": draft}})
    assert "shared_texts" in request
    async def role(current):
        assert current["protocol_repair"]["base_sha256"] == fingerprint(draft)
        return {"base_sha256": current["protocol_repair"]["base_sha256"],
                "edits": [{"op": "replace", "path": ["criteria", 0, "kind"],
                           "value": "hard", "reason": "Task obligation"}]}
    result = await ProtocolRepairCall(role)(request)
    assert result == {"criteria": [{"standard": text, "kind": "hard"}]}


@pytest.mark.asyncio
async def test_syntax_retry_keeps_diagnostic_text_when_projection_aliases_change():
    import json

    from reward_engine.core.outcome.protocol_repair import ProtocolRepairCall

    original_text = "Preserve the exact synthetic condition"
    original = {"criteria": [{"standard": {"$reducer_text": "t0"}, "kind": "invalid"}]}
    calls = []
    async def role(request):
        calls.append(deepcopy(request))
        if len(calls) == 1:
            raise json.JSONDecodeError("bad edit syntax", '{"edits":', 9)
        assert request["response_recovery"]["details"]["source_value"] == original_text
        assert request["response_recovery"]["rejected_response"]["criteria"][0]["standard"] == original_text
        assert request["response_recovery"]["syntax_error"]["details"]["reason"] == "bad edit syntax"
        return {"base_sha256": request["protocol_repair"]["base_sha256"], "edits": [
            {"op": "replace", "path": ["criteria", 0, "kind"], "value": "hard", "reason": "Fix the diagnosed kind"}]}
    repair = ProtocolRepairCall(role)
    with pytest.raises(json.JSONDecodeError):
        await repair({"mode": "outcome_jury_reduce", "shared_texts": {"t0": original_text},
                      "response_recovery": {"error_type": "ResponseContractError",
                          "rejected_response": original, "details": {
                              "field": "kind", "source_value": {"$reducer_text": "t0"}}}})
    result = await repair({"mode": "outcome_jury_reduce", "shared_texts": {"t0": "Unrelated text"},
                          "response_recovery": {"error_type": "JSONDecodeError",
                              "details": {"reason": "bad edit syntax"}}})
    assert result == {"criteria": [{"standard": original_text, "kind": "hard"}]}
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_invalid_evidence_request_uses_existing_reference_repair():
    from reward_engine.core.outcome.protocol_repair import ProtocolRepairCall

    request = {"mode": "outcome_jury_map", "response_recovery": {
        "rejected_response": {"need_evidence": ["unknown"]}}}
    async def role(current):
        assert current == request and "protocol_repair" not in current
        return {"need_evidence": ["task"]}
    assert await ProtocolRepairCall(role)(request) == {"need_evidence": ["task"]}


@pytest.mark.asyncio
async def test_transformed_repair_obeys_the_original_stage_byte_budget():
    from reward_engine.core.outcome.context import RequestBudgetExceeded

    calls = []
    class Role:
        protocol_edits = True
        async def __call__(self, request):
            calls.append(deepcopy(request))
            return {"criteria": [{"id": "synthetic", "kind": "invalid"}]}
    def reject(_):
        raise ResponseContractError("kind", "Unsupported kind")
    # The regular request and recovery fit, but expanded edit instructions do not.
    with pytest.raises(RequestBudgetExceeded):
        await ask(Role(), "outcome_jury_map", {}, {}, "synthetic", {},
                  max_bytes=4000, validate=reject, max_contract_retries=1)
    assert len(calls) == 1
