"""Small synthetic regressions for literal projection and complete draft preflight."""

import json
from copy import deepcopy

import pytest

from reward_engine.core.outcome import authored_json as protocol
from reward_engine.core.outcome.context import ResponseContractError, ask, build_context
from reward_engine.core.outcome.jury import SCHEMA, validate
from reward_engine.core.outcome.jury_revision import materialize
from reward_engine.core.outcome.source_projection import VERSION, project
from tests.unit.test_authored_json import criterion, enable, initial


def canonical(call):
    enable(call)
    call.role_input_projection = VERSION
    return call


def test_alias_binding_is_local_and_alias_conflict_is_precise():
    sources = initial()
    sources[0]["subrequirements"][0]["selection_id"] = "p1"
    inputs = {"independent_expectations": sources, "selection_aliases": {"p1": ["p0"]}}
    request = {**inputs, "instruction": "", "mode": "outcome_jury_reduce",
               "source_catalog": protocol.source_catalog("outcome_jury_reduce", inputs)}
    projected, audit = project(request, policy="canonical_recovery_v1")
    assert "selection_aliases" not in projected
    assert audit["local_bindings"]["selection_aliases"] == inputs["selection_aliases"]
    bad = {"criteria": [{"inherit_from": "seat:0"}], "dispositions": [
        {"source_parts": ["p0"], "action": "exclude", "reason": "duplicate", "evidence_refs": ["task"]}]}
    with pytest.raises(ResponseContractError) as caught:
        protocol.expand(bad, "outcome_jury_reduce", inputs, {"task": "Return JSON"}, canonical=True)
    issue = next(r for r in caught.value.details["issues"] if r.get("noncanonical_aliases"))
    assert not issue["unknown_sources"]
    assert issue["noncanonical_aliases"]["p0"]["canonical_source"] == "p1"
    assert issue["noncanonical_aliases"]["p0"]["already_covered"] == "criteria[0]"
    corrected = {"criteria": [{"inherit_from": "seat:0"}]}
    assert protocol.expand(corrected, "outcome_jury_reduce", inputs, {"task": "Return JSON"}, canonical=True)


def retained_reduction_inputs():
    from reward_engine.core.outcome.jury_drafts import expectations, reduction_inputs
    from reward_engine.core.outcome.jury_revision import criterion_body

    before = initial()
    draft = materialize({"criteria": [{"inherit_from": "seat:0"}]}, before,
                        {"task": "Return JSON"})
    return reduction_inputs([{
        **draft, "initial_draft": {"criteria": [criterion_body(before[0])]},
        "independent_expectations": before,
        "proposal_expectations": expectations(draft["criteria"], "seat:observed", "after_products"),
    }])


@pytest.mark.parametrize("reference", ["seat:0", "seat:observed:0"])
def test_inherit_consumes_canonical_sources_removed_by_revision_compaction(reference):
    from reward_engine.core.outcome.jury_drafts import (
        apply_part_decisions,
        compact_selection_inputs,
    )
    from reward_engine.core.outcome.jury_refinement import check_retired_clauses

    inputs = retained_reduction_inputs()
    before = deepcopy(inputs)
    compact = compact_selection_inputs(inputs)
    assert compact["independent_expectations"][0]["subrequirements"] == []
    expanded = protocol.expand({"criteria": [{"inherit_from": reference}]},
                               "outcome_jury_reduce", compact, {"task": "Return JSON"}, canonical=True)
    assert expanded["part_decisions"] == [{
        "parts": ["p1"], "action": "retain",
        "criterion_id": expanded["criteria"][0]["id"],
        "clause_id": expanded["criteria"][0]["clauses"][0]["id"],
    }]
    check_retired_clauses(expanded, inputs)
    result = apply_part_decisions(expanded, inputs, {"task": "Return JSON"}, mark_omissions=True)
    assert all(p["status"] == "mapped" for row in result["coverage"] for p in row["subrequirements"])
    assert inputs == before


@pytest.mark.parametrize("extra", [
    {"criteria": [{"inherit_from": "seat:observed:0"}]},
    {"dispositions": [{"source_parts": ["p1"], "action": "defer", "reason": "withhold"}]},
])
def test_inherited_alias_cannot_also_be_inherited_or_withheld(extra):
    from reward_engine.core.outcome.jury_drafts import compact_selection_inputs

    raw = {"criteria": [{"inherit_from": "seat:0"}, *extra.get("criteria", [])],
           "dispositions": extra.get("dispositions", [])}
    with pytest.raises(ResponseContractError) as caught:
        protocol.expand(raw, "outcome_jury_reduce", compact_selection_inputs(retained_reduction_inputs()),
                        {"task": "Return JSON"}, canonical=True)
    assert any(r.get("already_covered") for r in caught.value.details["issues"])


def test_whole_inheritance_keeps_bindings_after_literal_duplicate_compaction():
    from reward_engine.core.outcome.jury_drafts import compact_selection_inputs

    first = initial()[0]
    second = deepcopy(first)
    second["expectation_id"] = "other:0"
    second["subrequirements"][0]["part_id"] = "other:0/0"
    compact = compact_selection_inputs({"independent_expectations": [first, second],
                                        "proposal_expectations": []})
    assert compact["independent_expectations"][1]["subrequirements"] == []
    raw = {"criteria": [{"inherit_from": "other:0"}]}
    expanded = protocol.expand(raw, "outcome_jury_reduce", compact,
                               {"task": "Return JSON"}, canonical=True)
    assert expanded["part_decisions"][0]["parts"] == ["p0"]
    projected, _ = project({**compact, "instruction": ""}, policy="canonical_recovery_v1")
    assert projected["inheritance_bindings"]["other:0"] == [{"selection_id": "p0", "clause_index": 0}]


@pytest.mark.parametrize("mode", ["outcome_jury_map", "outcome_jury_reduce"])
def test_invalid_inherited_source_requires_complete_rewrite(mode):
    from reward_engine.core.outcome.jury_drafts import compact_selection_inputs

    sources = initial()
    del sources[0]["why"]  # A historical draft may predate required-field validation.
    inputs = {"independent_expectations": sources, "proposal_expectations": []}
    if mode == "outcome_jury_reduce":
        inputs = compact_selection_inputs(inputs)
    raw = {"criteria": [{"inherit_from": "seat:0"}]}
    before = deepcopy((inputs, raw))
    with pytest.raises(ResponseContractError) as caught:
        protocol.expand(raw, mode, inputs, {"task": "Return JSON"}, canonical=True)
    issue = next(r for r in caught.value.details["issues"] if r["field"] == "criteria[0].why")
    assert issue["requirement_id"] == "seat:0"
    assert issue["inherited_source_valid"] is False
    assert "complete authored criterion" in issue["inheritance_rule"]
    assert "rewrite" in issue["inheritance_rule"]
    assert protocol.diagnostics(caught.value.details)["issues"] == caught.value.details["issues"]
    assert (inputs, raw) == before

    # An override on inherit_from remains invalid, and adding a rationale is
    # a semantic change that cannot be silently labelled retain.
    with pytest.raises(ResponseContractError):
        protocol.expand({"criteria": [{"inherit_from": "seat:0", "why": "Requested output"}]},
                        mode, inputs, {"task": "Return JSON"}, canonical=True)
    item = criterion()
    item["clauses"][0].update(source_parts=["p0" if mode == "outcome_jury_reduce" else "seat:0/0"],
                              action="retain", reason="Supply missing rationale")
    with pytest.raises(ResponseContractError):
        protocol.expand({"criteria": [item]}, mode, inputs, {"task": "Return JSON"}, canonical=True)
    item["clauses"][0]["action"] = "rewrite"
    result = protocol.expand({"criteria": [item]}, mode, inputs, {"task": "Return JSON"}, canonical=True)
    assert result["criteria"][0]["why"] == "Requested output"
    assert (inputs, raw) == before


@pytest.mark.parametrize("extra_parts", [[], ["unknown"], ["p0"]])
def test_mixed_duplicate_diagnostic_preserves_unconflicted_source_obligations(extra_parts):
    first = initial()[0]
    first["subrequirements"][0]["selection_id"] = "p0"
    second = deepcopy(first)
    second["id"] = "second"
    second["expectation_id"] = "other:0"
    second["subrequirements"][0].update(part_id="other:0/0", selection_id="p1")
    inputs = {"independent_expectations": [first, second]}
    raw = {"criteria": [{"inherit_from": "seat:0"}], "dispositions": [{
        "source_parts": ["p0", "p1", *extra_parts], "action": "defer", "reason": "Needs review",
    }]}
    original = deepcopy(raw)
    with pytest.raises(ResponseContractError) as caught:
        protocol.expand(raw, "outcome_jury_reduce", inputs, {"task": "Return JSON"}, canonical=True)
    issue = next(r for r in caught.value.details["issues"] if r.get("already_covered"))
    assert issue["already_covered"] == {"p0": "criteria[0]"}
    assert issue["unconflicted_sources"] == ["p1"]
    assert "still require" in issue["coverage_rule"]
    assert raw == original
    # Dropping the entire mixed group is still rejected; only the model can
    # decide the remaining source's disposition in a complete revised draft.
    with pytest.raises(ResponseContractError, match="Complete draft failed validation") as missing:
        protocol.expand({"criteria": raw["criteria"]}, "outcome_jury_reduce", inputs,
                        {"task": "Return JSON"}, canonical=True)
    assert any(r.get("missing_sources") == ["p1"] for r in missing.value.details["issues"])
    raw["dispositions"][0]["source_parts"] = ["p1"]
    assert protocol.expand(raw, "outcome_jury_reduce", inputs, {"task": "Return JSON"}, canonical=True)


def test_projection_retains_unique_legacy_fields_and_is_reconstructable():
    sources = initial()
    sources[0]["historical_condition"] = {"only_when": "synthetic condition"}
    request = {
        "mode": "outcome_jury_map",
        "instruction": "",
        "independent_expectations": sources,
    }
    original = deepcopy(request)
    result, audit = project(request)
    row = result["source_requirements"]["seat:0"]
    assert "standard" not in row and "text" not in row["subrequirements"][0]
    assert row["clauses"][0]["text"] == "Return JSON"
    assert row["historical_condition"] == sources[0]["historical_condition"]
    assert audit["original_sources"]["independent_expectations"] == sources
    assert request == original
    sources[0]["standard"] = "A unique historical scope"
    sources[0]["subrequirements"][0]["text"] = "A unique historical detail"
    row = project(request)[0]["source_requirements"]["seat:0"]
    assert row["standard"] == "A unique historical scope"
    assert row["subrequirements"][0]["text"] == "A unique historical detail"
    assert "$reducer_text" not in json.dumps(result)


@pytest.mark.parametrize(
    "mode", ["outcome_jury_map", "outcome_jury_reduce", "outcome_expectations"]
)
def test_contract_shows_all_and_only_supported_forms(mode):
    schema = protocol.contract(mode, SCHEMA, canonical=True)["criteria"]
    assert (schema["minItems"], schema["maxItems"]) == (1, 20)
    options = schema["items"]["oneOf"]
    assert ("inherit_from" in options[0]) == (mode != "outcome_expectations")
    full = options[-1]
    assert "clauses" in full and "standard" not in full
    if mode != "outcome_expectations":
        assert full["clauses"][0]["action"] == "retain/rewrite/merge/add"


@pytest.mark.parametrize("mode", ["outcome_expectations", "outcome_jury_map", "outcome_jury_reduce"])
def test_canonical_contract_explains_quote_reference_relationship(mode):
    clause = protocol.contract(mode, SCHEMA, canonical=True)["criteria"]["items"]["oneOf"][-1]["clauses"][0]
    guidance = " ".join(clause["evidence_refs"])
    assert "source_quotes[].ref" in guidance
    assert "same clause" in guidance
    assert "parent" in guidance
    legacy = protocol.contract(mode, SCHEMA)["criteria"][0]["clauses"][0]
    assert legacy["evidence_refs"] == ["actual supplied refs, not proof of truth"]


def test_quote_reference_diagnostics_identify_both_sides_without_changing_draft():
    from reward_engine.core.outcome.draft_preflight import inspect

    row = criterion()
    row["clauses"][0].update(
        use="quality_judgment", evidence_refs=["other"],
        source_quotes=[{"ref": "task", "quote": "Return JSON"},
                       {"ref": "document", "quote": "Keep it readable"}],
    )
    draft = {"criteria": [row]}
    before = deepcopy(draft)
    issues = inspect(draft, {"task": "Return JSON", "document": "Keep it readable", "other": "Evidence"}, {})
    quotes = [i for i in issues if i["code"] == "QUOTE_REFERENCE"]
    assert len(quotes) == 2
    assert {(i.get("quoted_ref"), i.get("quote_field")) for i in quotes} == {
        ("task", "criteria[0].clauses[0].source_quotes[0].ref"),
        ("document", "criteria[0].clauses[0].source_quotes[1].ref"),
    }
    assert all(i["field"] == "criteria[0].clauses[0].evidence_refs" and i["actual"] == ["other"] for i in quotes)
    assert draft == before


@pytest.mark.asyncio
async def test_model_explicitly_corrects_quote_reference_from_precise_feedback():
    context, evidence = build_context({"task_id": "synthetic", "prompt": "Return JSON"})
    evidence["document"] = "Readable structured data"
    draft = {"criteria": [criterion()]}
    draft["criteria"][0]["clauses"][0]["evidence_refs"] = ["document"]
    before = deepcopy(draft)
    calls = []

    @canonical
    async def model(request):
        calls.append(request)
        result = deepcopy(draft)
        if len(calls) == 2:
            issue = next(i for i in request["response_recovery"]["details"]["issues"] if i["code"] == "QUOTE_REFERENCE")
            assert issue["quote_field"] == "criteria[0].clauses[0].source_quotes[0].ref"
            result["criteria"][0]["clauses"][0]["evidence_refs"].append(issue["quoted_ref"])
        return result

    result = await ask(model, "outcome_expectations", context, evidence, "", SCHEMA,
                       validate=lambda r: validate(r, evidence, context, clause_contract=True))
    assert len(calls) == 2
    assert result["criteria"][0]["clauses"][0]["evidence_refs"] == ["document", "task"]
    assert draft == before


@pytest.mark.parametrize("mode", ["outcome_jury_map", "outcome_jury_reduce"])
@pytest.mark.parametrize("change", ["none", "why", "condition", "evidence", "extra"])
def test_retain_semantics_agree_in_both_phases(mode, change):
    sources = initial()
    item = criterion()
    if change == "why":
        item["why"] = "Changed rationale"
    if change == "condition":
        item["clauses"][0]["conditions"] = ["Changed scope"]
    if change == "evidence":
        item["evidence_refs"] = ["task", "additional"]
    if change == "extra":
        item["extra_scope"] = "Changed scope"
    inputs = {"independent_expectations": sources}
    ref = "seat:0/0"
    if mode.endswith("reduce"):
        sources[0]["subrequirements"][0]["selection_id"] = "p0"
        ref = "p0"
    item["clauses"][0].update(
        action="retain", reason="Keep original", source_parts=[ref]
    )
    raw = {"criteria": [item]}
    evidence = {"task": "Return JSON", "additional": "Extra evidence"}
    if change != "none":
        with pytest.raises(ResponseContractError) as caught:
            protocol.expand(raw, mode, inputs, evidence, canonical=True)
        assert any("changed_fields" in e for e in caught.value.details["issues"])
    else:
        result = protocol.expand(raw, mode, inputs, evidence, canonical=True)
        if mode.endswith("map"):
            assert materialize(result, sources, evidence, canonical=True)["criteria"]


def test_multiple_errors_include_precise_gap_path_without_mutation():
    context, evidence = build_context({"task_id": "synthetic", "prompt": "Return JSON"})
    bad = criterion()
    bad["clauses"][0].update(
        source_parts=["seat:0/0", "seat:0/0"], action="retain", reason="Keep"
    )
    bad["why"] = "Changed"
    gaps = [
        {
            "gap_id": f"g{i}",
            "expected": "JSON",
            "observed": "None",
            "gap": "Missing",
            "verification": "Inspect",
            "product_ids": [],
            "expected_basis": "task",
            "expected_refs": [],
            "evidence_refs": ["task"],
            "status": "needs_check",
        }
        for i in range(3)
    ]
    raw = {"criteria": [bad] * 21, "gaps": gaps}
    before = deepcopy(raw)
    with pytest.raises(ResponseContractError) as caught:
        protocol.expand(
            raw,
            "outcome_jury_map",
            {"independent_expectations": initial()},
            evidence,
            canonical=True,
            context=context,
        )
    issues = caught.value.details["issues"]
    gap = next(e for e in issues if e["field"] == "gaps[2].expected_refs")
    assert (
        gap["gap_id"] == "g2"
        and gap["actual"] == []
        and 'expected_basis="task"' in gap["rule"]
    )
    assert {"CRITERIA_COUNT", "GAP_TASK_REFERENCE", "SOURCE_CONTRACT"} <= {
        e["code"] for e in issues
    }
    assert any("changed_fields" in e for e in issues)
    assert any("already_covered" in e for e in issues)
    assert raw == before


@pytest.mark.asyncio
async def test_model_authors_complete_correction_from_gap_diagnostics():
    context, evidence = build_context({"task_id": "synthetic", "prompt": "Return JSON"})
    calls = []

    @canonical
    async def model(request):
        calls.append(request)
        return {
            "criteria": [criterion()],
            "gaps": []
            if len(calls) > 1
            else [
                {
                    "gap_id": "g",
                    "expected": "JSON",
                    "observed": "None",
                    "gap": "Missing",
                    "verification": "Inspect",
                    "product_ids": [],
                    "expected_basis": "task",
                    "expected_refs": [],
                    "evidence_refs": ["task"],
                    "status": "needs_check",
                }
            ],
        }

    result = await ask(
        model,
        "outcome_expectations",
        context,
        evidence,
        "",
        SCHEMA,
        validate=lambda r: validate(r, evidence, context, clause_contract=True),
    )
    assert len(calls) == 2 and result["gaps"] == []
    assert (
        calls[1]["response_recovery"]["details"]["issues"][0]["field"]
        == "gaps[0].expected_refs"
    )
    assert "COMPLETE revised JSON" in calls[1]["response_recovery"]["instruction"]


def test_structural_failures_leave_independent_checks_available():
    raw = {"criteria": [None], "gaps": [{"gap_id": "g", "expected_basis": "task"}]}
    with pytest.raises(ResponseContractError) as caught:
        protocol.expand(raw, "outcome_expectations", {}, {}, canonical=True)
    issues = caught.value.details["issues"]
    assert any(e.get("unchecked") for e in issues)
    assert any(e["field"] == "gaps[0].expected_refs" for e in issues)


def test_check_supported_without_failed_execution_is_not_downgraded():
    raw = {
        "criteria": [criterion()],
        "gaps": [
            {
                "gap_id": "g",
                "expected": "JSON",
                "observed": "None",
                "gap": "Missing",
                "verification": "Inspect",
                "product_ids": ["p"],
                "expected_basis": "task",
                "expected_refs": ["task"],
                "evidence_refs": ["task"],
                "status": "check_supported",
            }
        ],
    }
    with pytest.raises(ResponseContractError) as caught:
        protocol.expand(
            raw,
            "outcome_expectations",
            {},
            {"task": "Return JSON"},
            canonical=True,
            context={"products": [{"product_id": "p", "user_updates": []}]},
        )
    assert any(e["code"] == "GAP_FAILED_CHECK" for e in caught.value.details["issues"])
    assert raw["gaps"][0]["status"] == "check_supported"


@pytest.mark.asyncio
async def test_projection_reuses_exact_parent_operation_after_current_validation(
    tmp_path,
):
    from reward_engine.core.outcome.completed_operations import CompletedOperations

    context, evidence = build_context({"task_id": "synthetic", "prompt": "Return JSON"})
    calls, validations = [], []

    @enable
    async def model(request):
        calls.append(request)
        return {"criteria": [criterion()]}

    def validator(row):
        validations.append(row)
        return validate(row, evidence, context, clause_contract=True)

    model.completed_operations = CompletedOperations(
        tmp_path, {"config_identity": "old"}
    )
    args = (model, "outcome_expectations", context, evidence, "Task", SCHEMA)
    await ask(*args, validate=validator)
    model.role_input_projection = VERSION
    model.completed_operations = CompletedOperations(
        tmp_path, {"config_identity": "new"}, ancestors=["old"]
    )
    result = await ask(*args, validate=validator)
    assert len(calls) == 1 and len(validations) == 2
    assert result["authored_json"]["input_projection"] == VERSION
    assert len(list(tmp_path.glob("*.reuse.json"))) == 1
    await ask(*args, validate=validator)
    assert len(calls) == 1
    changed = deepcopy(context)
    changed["task"]["prompt"] += " Changed task."
    await ask(
        model,
        "outcome_expectations",
        changed,
        evidence,
        "Task",
        SCHEMA,
        validate=validator,
    )
    assert len(calls) == 2


def test_inherited_missing_evidence_is_diagnosed_without_insertion():
    sources = initial()
    sources[0]["clauses"][0]["evidence_refs"] = []
    with pytest.raises(ResponseContractError) as caught:
        protocol.expand(
            {"criteria": [{"inherit_from": "seat:0"}]},
            "outcome_jury_map",
            {"independent_expectations": sources},
            {"task": "Return JSON"},
            canonical=True,
        )
    issues = caught.value.details["issues"]
    assert any(e["field"] == "criteria[0].clauses[0].evidence_refs" for e in issues)
    assert sources[0]["clauses"][0]["evidence_refs"] == []
