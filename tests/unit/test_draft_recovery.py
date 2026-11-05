"""Synthetic recovery requests preserve compact source lineage and budgets."""

from copy import deepcopy

import pytest

from reward_engine.core.outcome.context import ResponseContractError, ask, build_context
from reward_engine.core.outcome.draft_recovery import worklist
from reward_engine.core.outcome.jury import SCHEMA
from reward_engine.core.outcome.jury_drafts import compact_selection_inputs
from tests.unit.test_authored_json import criterion, enable, initial


def test_projected_resume_worklist_matches_current_draft_and_exposes_field_repairs():
    from reward_engine.core.outcome import authored_json
    from reward_engine.core.outcome.source_projection import project

    sources = initial()
    other = deepcopy(sources[0])
    other.update(expectation_id="seat:1", id="second")
    other["subrequirements"][0]["part_id"] = "seat:1/0"
    inputs = {"independent_expectations": [*sources, other]}
    raw = {"criteria": [criterion("Return a JSON object")]}
    raw["criteria"][0]["clauses"][0].update(
        source_parts=["seat:0/0"], action="retain", reason="Keep",
        evidence_refs=["other"], source_quotes=[{"ref": "task", "quote": "Paraphrased JSON"}],
    )
    evidence = {"task": "Return JSON", "other": "Supplemental context"}
    with pytest.raises(ResponseContractError) as caught:
        authored_json.expand(raw, "outcome_jury_map", inputs, evidence, canonical=True)
    projected, _ = project({**inputs, "mode": "outcome_jury_map", "instruction": "",
                            "source_catalog": authored_json.source_catalog("outcome_jury_map", inputs)},
                           policy="canonical_recovery_v1")
    before = deepcopy((raw, projected, caught.value.details))
    result = worklist(raw, "outcome_jury_map", projected, caught.value.details)
    assert result == worklist(raw, "outcome_jury_map", inputs, caught.value.details)
    assert result["missing_sources"] == ["seat:1/0"]
    assert {row["code"] for row in result["citation_checks"]} == {"QUOTE_REFERENCE", "QUOTE_EXACT"}
    assert all(row["quoted_ref"] == "task" for row in result["citation_checks"])
    assert result["retain_checks"][0]["source_part"] == "seat:0/0"
    assert "clause.text" in result["retain_checks"][0]["changed_fields"]
    assert (raw, projected, caught.value.details) == before


@pytest.mark.asyncio
async def test_complete_repair_resolves_quotes_retain_and_missing_source_together():
    from reward_engine.core.outcome.jury_revision import materialize

    context, evidence = build_context({"task_id": "synthetic", "prompt": "Return JSON"})
    evidence["other"] = "Supplemental context"
    sources = initial()
    other = deepcopy(sources[0])
    other.update(expectation_id="seat:1", id="second")
    other["subrequirements"][0]["part_id"] = "seat:1/0"
    sources.append(other)
    inputs = {"independent_expectations": sources}
    raw = {"criteria": [criterion("Return a JSON object")]}
    raw["criteria"][0]["clauses"][0].update(
        source_parts=["seat:0/0"], action="retain", reason="Keep",
        evidence_refs=["other"], source_quotes=[{"ref": "task", "quote": "Paraphrased JSON"}],
    )
    before = deepcopy((raw, inputs, evidence))
    calls = []

    @enable
    async def model(request):
        calls.append(deepcopy(request))
        if len(calls) == 1:
            return deepcopy(raw)
        work = request["response_recovery"]["repair_worklist"]
        assert {row["code"] for row in work["citation_checks"]} == {"QUOTE_REFERENCE", "QUOTE_EXACT"}
        assert work["retain_checks"] and work["missing_sources"] == ["seat:1/0"]
        return {"criteria": [{"inherit_from": "seat:0"}, {"inherit_from": "seat:1"}]}

    model.role_input_projection = "canonical_sources_v1"
    result = await ask(model, "outcome_jury_map", context, evidence, "", SCHEMA, extra=inputs,
                       validate=lambda row: materialize(row, sources, evidence, canonical=True))
    assert len(calls) == 2 and len(result["criteria"]) == 2
    assert all(c["clauses"][0]["text"] == "Return JSON" for c in result["criteria"])
    assert (raw, inputs, evidence) == before


@pytest.mark.asyncio
async def test_complete_repair_groups_twenty_three_sources_without_losing_clauses():
    from reward_engine.core.outcome.jury import _apply_reduced_decisions, validate

    context, evidence = build_context({"task_id": "synthetic", "prompt": "Return JSON"})
    sources = []
    for index in range(23):
        row = deepcopy(initial()[0])
        text = f"Return JSON field {index}"
        row.update(expectation_id=f"seat:{index}", id=f"requirement_{index}", standard=text)
        row["clauses"][0]["text"] = text
        row["subrequirements"][0].update(part_id=f"seat:{index}/0", selection_id=f"p{index}", text=text)
        sources.append(row)
    inputs = {"independent_expectations": sources, "proposal_expectations": []}

    def authored(index, *, invalid_quote=False):
        row = criterion(f"Return JSON field {index}")
        row["clauses"][0].update(source_parts=[f"p{index}"], action="rewrite", reason="Explicit grouping")
        if invalid_quote:
            row["clauses"][0]["source_quotes"][0]["quote"] = "Paraphrased JSON"
        return row

    raw = {"criteria": [authored(0, invalid_quote=True), authored(1, invalid_quote=True),
                         *({"inherit_from": f"seat:{i}"} for i in range(2, 23))]}
    grouped = criterion()
    grouped["clauses"] = [authored(i)["clauses"][0] for i in range(19, 23)]
    corrected = {"criteria": [authored(0), authored(1),
                               *({"inherit_from": f"seat:{i}"} for i in range(2, 19)), grouped]}
    before = deepcopy((raw, corrected, inputs))
    calls = []

    @enable
    async def model(request):
        calls.append(deepcopy(request))
        if len(calls) == 1:
            return deepcopy(raw)
        assert len(calls) == 2, request["response_recovery"]["details"]
        work = request["response_recovery"]["repair_worklist"]
        assert work["criteria_budget"]["minimum_net_reduction"] == 3
        assert len(work["citation_checks"]) == 2
        assert all(row["quoted_ref"] == "task" for row in work["citation_checks"])
        return deepcopy(corrected)

    def finalize(row):
        reduced = _apply_reduced_decisions(row, inputs, evidence, strict=True)
        validate(reduced, evidence, context, clause_contract=True)
        return reduced

    model.role_input_projection = "canonical_sources_v1"
    result = await ask(model, "outcome_jury_reduce", context, evidence, "", SCHEMA,
                       extra=inputs, validate=finalize)
    assert len(calls) == 2 and len(result["criteria"]) == 20
    assert sum(len(c["clauses"]) for c in result["criteria"]) == 23
    parts = [part for row in result["coverage"] for part in row["subrequirements"]]
    assert len(parts) == 23 and all(part["status"] == "mapped" for part in parts)
    assert (raw, corrected, inputs) == before


def test_worklist_reports_inherited_duplicates_and_all_missing_sources():
    inputs = compact_selection_inputs({"independent_expectations": initial(), "proposal_expectations": []})
    raw = {"criteria": [{"inherit_from": "seat:0"}], "dispositions": [
        {"source_parts": ["p0", "unknown"], "action": "exclude"}]}
    before = deepcopy((raw, inputs))
    work = worklist(raw, "outcome_jury_reduce", inputs, {})
    assert work["duplicate_sources"]["p0"] == [
        {"path": ["criteria", 0, "inherit_from"], "action": "retain"},
        {"path": ["dispositions", 0], "action": "exclude"},
    ]
    assert work["unknown_sources"] == ["unknown"]
    assert work["missing_sources"] == []
    assert worklist({"criteria": []}, "outcome_jury_reduce", inputs, {})["missing_sources"] == ["p0"]
    assert (raw, inputs) == before


def test_withdrawn_diagnostic_points_to_authored_fields_not_internal_decisions():
    raw = {"criteria": [criterion()], "dispositions": [{"source_parts": ["p0"], "action": "exclude"}]}
    details = {"validation_errors": [{"conflicting_decisions": [
        {"decision_index": 27, "source_texts": {"p0": "Return JSON"}}]}]}
    inputs = compact_selection_inputs({"independent_expectations": initial(), "proposal_expectations": []})
    work = worklist(raw, "outcome_jury_reduce", inputs, details)
    assert work["withdrawn_publication_conflicts"] == [{
        "source": "p0", "source_locations": [{"path": ["dispositions", 0], "action": "exclude"}],
        "published_locations": [["criteria", 0, "clauses", 0, "text"]],
    }]


@pytest.mark.parametrize("count,slots,reduction", [(19, 1, 0), (20, 0, 0), (23, 0, 3)])
def test_budget_counts_inherited_entries_without_rewriting_the_draft(count, slots, reduction):
    inputs = compact_selection_inputs({"independent_expectations": initial(), "proposal_expectations": []})
    raw = {"criteria": [{"inherit_from": "seat:0"}, {"inherit_from": "seat:0"},
                        *(criterion() for _ in range(count - 2))]}
    before = deepcopy((raw, inputs))
    budget = worklist(raw, "outcome_jury_reduce", inputs, {})["criteria_budget"]
    assert budget["counted_top_level_entries"] == count
    assert budget["inherited_entries_included"] == 2
    assert budget["available_slots"] == slots
    assert budget["minimum_net_reduction"] == reduction
    assert (raw, inputs) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("repeat", [False, True])
async def test_actual_request_preserves_rejected_draft_and_bounded_repair_limit(repeat):
    context, evidence = build_context({"task_id": "synthetic", "prompt": "Return JSON"})
    raw = {"criteria": [criterion() for _ in range(21)]}
    requests = []
    responses = []

    @enable
    async def model(request):
        requests.append(deepcopy(request))
        response = deepcopy(raw)
        if not repeat:
            response["uncertainties"] = [f"Unresolved attempt {len(requests)}"]
        responses.append(response)
        return deepcopy(response)

    model.role_input_projection = "canonical_sources_v1"
    with pytest.raises(ResponseContractError):
        await ask(model, "outcome_expectations", context, evidence, "", SCHEMA, validate=lambda x: x)
    assert len(requests) == (2 if repeat else 4)
    for request, response in zip(requests[1:], responses, strict=False):
        recovery = request["response_recovery"]
        assert recovery["rejected_response"] == response
        assert recovery["repair_worklist"]["criteria_count"] == 21
        assert recovery["repair_worklist"]["criteria_budget"]["minimum_net_reduction"] == 1
        assert "Preserve unaffected" in recovery["instruction"]
        assert "Do not truncate" in recovery["instruction"]


@pytest.mark.asyncio
async def test_withdrawn_clause_repairs_through_full_validator_without_changing_content():
    from reward_engine.core.outcome.jury import _apply_reduced_decisions

    context, evidence = build_context({"task_id": "synthetic", "prompt": "Return JSON"})
    inputs = {"independent_expectations": initial(), "proposal_expectations": []}
    raw = {"criteria": [criterion()], "dispositions": [
        {"source_parts": ["p0"], "action": "exclude", "reason": "duplicate", "evidence_refs": ["task"]}]}
    raw["criteria"][0]["clauses"][0].update(source_parts=[], action="add", reason="publish")
    requests = []

    @enable
    async def model(request):
        requests.append(deepcopy(request))
        if len(requests) == 1:
            return deepcopy(raw)
        conflict, = request["response_recovery"]["repair_worklist"]["withdrawn_publication_conflicts"]
        assert conflict["source"] == "p0"
        assert conflict["source_locations"][0]["path"] == ["dispositions", 0]
        return {"criteria": [{"inherit_from": "seat:0"}]}

    model.role_input_projection = "canonical_sources_v1"
    result = await ask(model, "outcome_jury_reduce", context, evidence, "", SCHEMA,
                       extra=compact_selection_inputs(inputs),
                       validate=lambda row: _apply_reduced_decisions(row, inputs, evidence, strict=True))
    assert len(requests) == 2
    assert result["criteria"][0]["clauses"][0]["text"] == "Return JSON"
