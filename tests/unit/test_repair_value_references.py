"""Model-selected exact restoration avoids retyping provenance strings."""

from copy import deepcopy

import pytest

from reward_engine.core.outcome.context import ResponseContractError, ask


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", [None, "unknown", "wrong_target", "both_values"])
async def test_model_can_explicitly_copy_only_the_diagnosed_source_value(invalid):
    original = {"criteria": [{"id": "synthetic", "clauses": [{"id": "clause-id", "quote": "Keep file A."}]}]}
    correct = {"quote": "Keep file B."}
    calls = []

    class Role:
        protocol_edits = True

        async def __call__(self, request):
            calls.append(deepcopy(request))
            if len(calls) == 1:
                return deepcopy(original)
            offer = request["protocol_repair"]["restore_values"][0]
            assert offer["value"] == correct["quote"]
            edit = {"op": "replace", "path": offer["path"], "value_from": offer["id"],
                    "reason": "Explicitly preserve the original source wording"}
            if invalid == "unknown":
                edit["value_from"] = "unknown"
            elif invalid == "wrong_target":
                edit["path"] = ["criteria", 0, "id"]
            elif invalid == "both_values":
                edit["value"] = correct
            return {"base_sha256": request["protocol_repair"]["base_sha256"], "edits": [edit]}

    def validate(row):
        if row["criteria"][0]["clauses"][0]["quote"] != correct["quote"]:
            error = ResponseContractError("revision_decisions", "Retain must preserve source wording")
            error.details["retain_mismatches"] = [{
                "target": {"criterion_id": "synthetic", "part_index": 0},
                "field_differences": [{"field": "clause", "source_present": True,
                                       "source_value": correct}]}]
            raise error
        return row

    args = (Role(), "outcome_jury_map", {}, {}, "Preserve provenance", {"criteria": []})
    if invalid:
        with pytest.raises(ResponseContractError):
            await ask(*args, validate=validate, max_contract_retries=1)
    else:
        result = await ask(*args, validate=validate, max_contract_retries=1)
        assert result == {"criteria": [{"id": "synthetic", "clauses": [{"id": "clause-id", **correct}]}]}
    assert original["criteria"][0]["clauses"][0] == {"id": "clause-id", "quote": "Keep file A."}
    assert len(calls) == 2


def test_invalid_original_quote_is_diagnosed_not_offered_for_restoration():
    from reward_engine.core.outcome.jury_drafts import expectations
    from reward_engine.core.outcome.jury_revision import materialize
    from reward_engine.core.outcome.repair_values import restoration_values

    clause = {"id": "c", "text": "Preserve file", "use": "task_constraint",
              "evidence_refs": ["task"], "source_quotes": [{"ref": "task", "quote": "Keep file A."}],
              "conditions": [], "alternative_group": ""}
    criterion = {"id": "synthetic", "standard": "Preserve file", "clauses": [clause],
                 "subrequirements": ["Preserve file"]}
    initial = expectations([criterion], "seat", "before_products")
    draft = {"criteria": [deepcopy(criterion)], "revision_decisions": [{
        "action": "retain", "source_parts": [initial[0]["subrequirements"][0]["part_id"]],
        "targets": [{"criterion_id": "synthetic", "part_index": 0}]}]}
    draft["criteria"][0]["clauses"][0]["source_quotes"][0]["quote"] = "Keep file B."
    with pytest.raises(ResponseContractError) as caught:
        materialize(draft, initial, {"task": "Keep file B."})
    mismatch = caught.value.details["retain_mismatches"][0]
    assert mismatch["invalid_source_quotes"][0]["source_ref"] == "task"
    assert "rewrite" in mismatch["invalid_source_quotes"][0]["instruction"]
    assert restoration_values(draft, caught.value.details) == []
    assert initial[0]["clauses"][0]["source_quotes"][0]["quote"] == "Keep file A."


def test_retain_error_also_reports_invalid_quotes_in_new_clauses():
    from reward_engine.core.outcome.jury_drafts import expectations
    from reward_engine.core.outcome.jury_revision import materialize

    clause = {"id": "c", "text": "Preserve file", "use": "task_constraint",
              "evidence_refs": ["task"], "source_quotes": [{"ref": "task", "quote": "Keep file B."}],
              "conditions": [], "alternative_group": ""}
    criterion = {"id": "synthetic", "standard": "Preserve file", "clauses": [clause],
                 "measurement_question": "Inspect the file", "subrequirements": ["Preserve file"]}
    initial = expectations([criterion], "seat", "before_products")
    draft = {"criteria": [deepcopy(criterion)], "revision_decisions": [{
        "action": "retain", "source_parts": [initial[0]["subrequirements"][0]["part_id"]],
        "targets": [{"criterion_id": "synthetic", "part_index": 0}]}]}
    draft["criteria"][0]["measurement_question"] = "Changed scope"
    added = deepcopy(clause)
    added.update(id="new", text="A new requirement")
    added["source_quotes"][0]["quote"] = "Keep file A."
    draft["criteria"][0]["clauses"].append(added)
    draft["criteria"][0]["subrequirements"].append(added["text"])
    with pytest.raises(ResponseContractError) as caught:
        materialize(draft, initial, {"task": "Keep file B."})
    assert caught.value.details["retain_mismatches"]
    issue = caught.value.details["draft_quote_errors"][0]
    assert issue["path"] == ["criteria", 0, "clauses", 1, "source_quotes", 0, "quote"]
    assert issue["source_ref"] == "task"
