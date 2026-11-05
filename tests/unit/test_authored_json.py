"""Synthetic contracts: source fidelity, full correction and execution provenance."""

import json
from copy import deepcopy

import pytest

from reward_engine.core.outcome import authored_json as protocol
from reward_engine.core.outcome.context import ResponseContractError, ask, build_context
from reward_engine.core.outcome.jury import validate
from reward_engine.core.outcome.jury_drafts import (
    compact_selection_inputs,
    expectations,
)
from reward_engine.core.outcome.jury_revision import materialize


def criterion(text="Return JSON"):
    return {
        "kind": "hard",
        "origin": "task",
        "priority": "critical",
        "priority_reason": "Requested output",
        "why": "Requested output",
        "measurement_question": "Inspect the output",
        "evidence_refs": ["task"],
        "clauses": [
            {
                "text": text,
                "use": "task_constraint",
                "evidence_refs": ["task"],
                "source_quotes": [{"ref": "task", "quote": "Return JSON"}],
                "conditions": [],
                "alternative_group": "",
            }
        ],
    }


def initial():
    row = protocol.expand(
        {"criteria": [criterion()]}, "outcome_expectations", {}, {"task": "Return JSON"}
    )
    return expectations(row["criteria"], "seat", "before_products")


def enable(call):
    call.role_output_protocol = protocol.VERSION
    call.role_recovery_policy = protocol.RECOVERY
    call.bounded_stage = True
    call.protocol_edits = True  # v2 must bypass legacy atomic edits even on revisions
    return call


@pytest.mark.parametrize(
    "raw", ['{"x":1,"x":2}', '{"x":NaN}', 'Here: {"x":1}', "{} {}", "```json\n{}", "[]"]
)
def test_strict_json_rejects_ambiguous_or_non_json(raw):
    with pytest.raises(json.JSONDecodeError):
        protocol.decode(raw)


def test_complete_fence_and_document_only():
    assert protocol.decode('```json\n{"x":1}\n```') == {"x": 1}
    assert protocol.decode(' {"x":1} ') == {"x": 1}


def test_root_diagnostic_reports_actual_fields_without_unwrapping():
    draft = {"response": {"criteria": [criterion()]}}
    before = deepcopy(draft)
    with pytest.raises(ResponseContractError) as caught:
        protocol.expand(draft, "outcome_expectations", {}, {"task": "Return JSON"})
    assert caught.value.details["issues"][0]["actual_root_fields"] == ["response"]
    assert draft == before


def test_duplicate_source_diagnostic_identifies_both_decisions():
    item = criterion()
    item["clauses"][0].update(source_parts=["seat:0/0"], action="rewrite", reason="Explicit revision")
    item["clauses"].append(deepcopy(item["clauses"][0]))
    with pytest.raises(ResponseContractError) as caught:
        protocol.expand({"criteria": [item]}, "outcome_jury_map",
                        {"independent_expectations": initial()}, {"task": "Return JSON"})
    issue = next(r for r in caught.value.details["issues"] if "already_covered" in r)
    assert issue["already_covered"] == {"seat:0/0": "criteria[0].clauses[0]"}
    assert issue["field"] == "criteria[0].clauses[1]"
    assert issue["unknown_sources"] == []


def test_retain_diagnostic_identifies_changed_parent_semantics():
    item = criterion()
    item["why"] = "A different rationale"
    item["clauses"][0].update(source_parts=["seat:0/0"], action="retain", reason="Keep")
    with pytest.raises(ResponseContractError) as caught:
        protocol.expand({"criteria": [item]}, "outcome_jury_map",
                        {"independent_expectations": initial()}, {"task": "Return JSON"})
    issue = next(r for r in caught.value.details["issues"] if "changed_fields" in r)
    assert "why" in issue["changed_fields"]
    assert issue["source_part"] == "seat:0/0"
    assert issue["resolution_choices"][0]["criterion"] == {"inherit_from": "seat:0"}
    assert item["clauses"][0]["action"] == "retain"


@pytest.mark.asyncio
async def test_full_recovery_excludes_nested_legacy_edit_commands():
    context, evidence = build_context({"task_id": "synthetic", "prompt": "Return JSON"})
    requests = []

    @enable
    async def model(request):
        requests.append(deepcopy(request))
        assert "Do not put the answer inside a response object" in request["instruction"]
        return {"status": "partial" if len(requests) == 1 else "fail"}

    def validate_status(row):
        if row["status"] == "partial":
            error = ResponseContractError("status", "Expected pass/fail/unknown")
            error.details["issues"] = [{"actual": "partial", "allowed_values": ["pass", "fail", "unknown"],
                                       "repair_instruction": "Preserve every other field; do not request tools"}]
            raise error
        return row

    result = await ask(model, "outcome_measure", context, evidence, "Evaluate", {}, validate=validate_status)
    recovery = requests[1]["response_recovery"]
    assert recovery["details"]["issues"] == [{"actual": "partial", "allowed_values": ["pass", "fail", "unknown"]}]
    assert recovery["rejected_response"] == {"status": "partial"}
    assert result["status"] == "fail"


def test_projection_and_inheritance_preserve_meaning():
    source = initial()
    old = deepcopy(source)
    result = protocol.expand(
        {"criteria": [{"inherit_from": "seat:0"}], "dispositions": []},
        "outcome_jury_map",
        {"independent_expectations": source},
        {"task": "Return JSON"},
    )
    expanded = materialize(result, source, {"task": "Return JSON"})
    assert expanded["criteria"][0]["clauses"] == source[0]["clauses"]
    assert expanded["revision_decisions"][0]["action"] == "retain"
    assert source == old


def test_reduce_inherits_distinct_requirements_with_colliding_internal_ids():
    from reward_engine.core.outcome.jury_drafts import apply_part_decisions
    from reward_engine.core.outcome.jury_revision import meaning

    evidence = {"task": "Return JSON. Include a summary."}
    sources = []
    for seat, text in (("one", "Return JSON"), ("two", "Include a summary")):
        item = criterion(text)
        item["clauses"][0]["source_quotes"][0]["quote"] = text
        draft = protocol.expand({"criteria": [item]}, "outcome_expectations", {}, evidence)
        sources.extend(expectations(draft["criteria"], seat, "before_products"))
    before = deepcopy(sources)
    assert sources[0]["id"] == sources[1]["id"]
    inputs = {"independent_expectations": sources, "proposal_expectations": [], "proposals": []}
    raw = {"criteria": [{"inherit_from": row["expectation_id"]} for row in sources]}
    expanded = protocol.expand(raw, "outcome_jury_reduce", compact_selection_inputs(inputs), evidence)
    result = apply_part_decisions(expanded, inputs, evidence, mark_omissions=True)
    assert len({row["id"] for row in result["criteria"]}) == 2
    for row, bound, source in zip(expanded["criteria"], result["criteria"], sources, strict=True):
        assert meaning(row, 0) == meaning(source, 0)
        assert bound["clauses"][0]["source_part_ids"] == [source["subrequirements"][0]["part_id"]]
    assert all(row["action"] == "retain" for row in expanded["part_decisions"])
    assert sources == before
    duplicate = {"criteria": [raw["criteria"][0], raw["criteria"][0]]}
    with pytest.raises(ResponseContractError):
        protocol.expand(duplicate, "outcome_jury_reduce", compact_selection_inputs(inputs), evidence)


@pytest.mark.parametrize(
    "fault", ["missing", "duplicate", "unknown", "false_retain", "repeated_text"]
)
def test_no_guessed_lineage_or_relabelled_retain(fault):
    item = criterion()
    item["clauses"][0].update(
        source_parts=["seat:0/0"], action="rewrite", reason="Explicitly reconsidered"
    )
    raw = {"criteria": [item], "dispositions": []}
    if fault == "missing":
        item["clauses"][0]["source_parts"] = []
    if fault == "duplicate":
        raw["dispositions"] = [
            {"source_parts": ["seat:0/0"], "action": "defer", "reason": "uncertain"}
        ]
    if fault == "unknown":
        item["clauses"][0]["source_parts"] = ["missing"]
    if fault == "false_retain":
        item["clauses"][0].update(action="retain", text="Return XML")
    if fault == "repeated_text":
        item["standard"] = "Return JSON"
    before = deepcopy(raw)
    with pytest.raises(ResponseContractError):
        protocol.expand(
            raw,
            "outcome_jury_map",
            {"independent_expectations": initial()},
            {"task": "Return JSON"},
        )
    assert raw == before


def test_reduce_compiles_explicit_merge_without_copying_a_second_text():
    source = initial()
    inputs = {
        "independent_expectations": source,
        "proposal_expectations": [],
        "proposals": [],
    }
    view = compact_selection_inputs(inputs)
    item = criterion("Return a JSON document")
    item["clauses"][0].update(
        source_parts=["p0"], action="rewrite", reason="Clarifies output"
    )
    result = protocol.expand(
        {"criteria": [item], "dispositions": []},
        "outcome_jury_reduce",
        view,
        {"task": "Return JSON"},
    )
    assert result["part_decisions"][0]["text"] == item["clauses"][0]["text"]
    assert result["authored_json"]["draft"]["criteria"] == [item]


@pytest.mark.parametrize("change", ["added", "removed", "changed", "unchanged"])
def test_reduce_retain_checks_extra_parent_semantics(change):
    evidence = {"task": "Return JSON"}
    source = criterion()
    if change != "added":
        source["quality_condition"] = "Preserve all requested fields"
    initial_draft = protocol.expand({"criteria": [source]}, "outcome_expectations", {}, evidence)
    inputs = compact_selection_inputs({
        "independent_expectations": expectations(initial_draft["criteria"], "seat", "before_products"),
        "proposal_expectations": [], "proposals": [],
    })
    item = deepcopy(source)
    if change == "removed":
        item.pop("quality_condition")
    elif change != "unchanged":
        item["quality_condition"] = "Include only selected fields"
    item["clauses"][0].update(source_parts=["p0"], action="retain", reason="Keep")
    raw = {"criteria": [item]}
    before = deepcopy(raw)
    if change == "unchanged":
        result = protocol.expand(raw, "outcome_jury_reduce", inputs, evidence)
        assert result["criteria"][0]["quality_condition"] == source["quality_condition"]
        assert result["part_decisions"][0]["action"] == "retain"
    else:
        with pytest.raises(ResponseContractError) as caught:
            protocol.expand(raw, "outcome_jury_reduce", inputs, evidence)
        issue = next(row for row in caught.value.details["issues"] if "changed_fields" in row)
        assert "parent.quality_condition" in issue["changed_fields"]
    assert raw == before


@pytest.mark.asyncio
async def test_full_correction_preserves_original_context_and_stops_identical_failure():
    context, evidence = build_context({"task_id": "synthetic", "prompt": "Return JSON"})
    requests = []

    @enable
    async def model(request):
        requests.append(deepcopy(request))
        if len(requests) > 1:
            assert request["response_recovery"]["rejected_response"] == {"criteria": []}
            assert request["response_recovery"]["draft_identity"]
            assert "edits" not in request["response"]
        return {"criteria": []}

    with pytest.raises(ResponseContractError):
        await ask(
            model,
            "outcome_expectations",
            context,
            evidence,
            "Task",
            {"criteria": [criterion()]},
            validate=lambda row: validate(row, evidence, context, clause_contract=True),
        )
    assert len(requests) == 2
    assert requests[0]["context"] == requests[1]["context"]


@pytest.mark.asyncio
async def test_three_content_corrections_then_complete_valid_draft():
    context, evidence = build_context({"task_id": "synthetic", "prompt": "Return JSON"})
    calls = []

    @enable
    async def model(request):
        calls.append(request)
        value = criterion()
        if len(calls) <= 3:
            value["kind"] = f"invalid{len(calls)}"
        return {"criteria": [value]}

    result = await ask(
        model,
        "outcome_expectations",
        context,
        evidence,
        "Task",
        {"criteria": [criterion()]},
        validate=lambda row: validate(row, evidence, context, clause_contract=True),
    )
    assert len(calls) == 4
    assert result["criteria"][0]["standard"] == "Return JSON"


@pytest.mark.asyncio
async def test_early_final_reply_keeps_real_recovery_tool_slot():
    from reward_engine.core.outcome.harness import defaults
    from tests.unit.test_claw_outcome_measurement_tools import case, verdict

    context, evidence, _, test = case()
    source = "product_1:artifact:final_response"
    requests = []

    @enable
    async def model(request):
        requests.append(deepcopy(request))
        n = len(requests)
        if n in (1, 3):
            if n == 3:
                assert request["measurement_budget"]["remaining_tool_waves"] == 1
                assert request["measurement_recovery_required"][
                    "replacement_execution_allowed"
                ]
            return {
                "need_tools": [
                    {
                        "tool": "code.run",
                        "arguments": {
                            "program": "print(2+3)" if n == 1 else "result = 2+3",
                            "input_refs": [source],
                        },
                    }
                ]
            }
        result = verdict("tool:check:1" if n == 4 else None)
        if n == 2:
            result.pop("status")
        if n == 4:
            result["measurement_recovery"] = [
                {
                    "failed_ref": "tool:check:0",
                    "replacement_refs": ["tool:check:1"],
                    "reason": "Corrected real execution",
                }
            ]
        return result

    result = await defaults(model).run(test, context, evidence)
    assert result["status"] == "pass", result
    assert result["execution_usage"] == {"model_requests": 4, "tool_calls": 2}
    assert result["tool_observations"]["tool:check:0"]["status"] != "observed"
    assert result["tool_observations"]["tool:check:1"]["status"] == "observed"


@pytest.mark.asyncio
async def test_completed_seats_survive_another_seat_failure(tmp_path):
    from reward_engine.core.outcome.completed_operations import CompletedOperations
    from reward_engine.core.outcome.jury import _propose_once

    context, evidence = build_context({"task_id": "synthetic", "prompt": "Return JSON"})
    counts = {str(n): [] for n in range(1, 4)}
    broken = True
    members = {}
    for seat in counts:

        @enable
        async def call(request, seat=seat):
            counts[seat].append(request["mode"])
            if request["mode"] == "outcome_expectations":
                return {"criteria": [criterion()]}
            if broken and seat == "3":
                return {"criteria": []}
            return {
                "criteria": [
                    {
                        "inherit_from": request["independent_expectations"][0][
                            "expectation_id"
                        ]
                    }
                ],
                "dispositions": [],
                "gaps": [],
            }

        call.completed_operations = CompletedOperations(
            tmp_path / seat, {"revision": "same"}
        )
        members[seat] = call

    @enable
    async def reducer(request):
        item = criterion()
        source = [
            p["selection_id"]
            for group in ("independent_expectations", "proposal_expectations")
            for row in request[group]
            for p in row["subrequirements"]
        ]
        item["clauses"][0].update(
            source_parts=source, action="merge", reason="Same explicit task goal"
        )
        return {"criteria": [item], "dispositions": [], "gaps": []}

    with pytest.raises(RuntimeError, match="three usable outcome proposals"):
        await _propose_once(
            context,
            evidence,
            members,
            reducer,
            strict=True,
            clause_contract=True,
            revision_protocol=True,
        )
    first = deepcopy(counts)
    broken = False
    result = await _propose_once(
        context,
        evidence,
        members,
        reducer,
        strict=True,
        clause_contract=True,
        revision_protocol=True,
    )
    assert counts["1"] == first["1"] and counts["2"] == first["2"]
    assert counts["3"].count("outcome_expectations") == 1
    assert result["criteria"][0]["standard"] == "Return JSON"


@pytest.mark.asyncio
async def test_completed_measurement_has_zero_repeated_model_or_tool_calls(tmp_path):
    from reward_engine.core.outcome.completed_operations import CompletedOperations
    from reward_engine.core.outcome.harness import defaults
    from tests.unit.test_claw_outcome_measurement_tools import case, verdict

    context, evidence, _, test = case()
    calls = []

    @enable
    async def model(request):
        calls.append(request)
        return verdict()

    model.completed_operations = CompletedOperations(tmp_path, {"revision": "one"})
    first = await defaults(model).run(test, context, evidence)
    second = await defaults(model).run(test, context, evidence)
    assert not first.get("error"), first
    assert first == second and len(calls) == 1
    changed = deepcopy(test)
    changed["question"] = "A different measurement question"
    await defaults(model).run(changed, context, evidence)
    assert len(calls) == 2
    binding = {
        "test": test,
        "context": context,
        "evidence": evidence,
        "tools": defaults(model).tools.catalog(),
    }
    checkpoint = tmp_path / (
        model.completed_operations.key("measurement", binding) + ".json"
    )
    data = json.loads(checkpoint.read_text())
    data["result"]["reason"] = "tampered"
    checkpoint.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="checkpoint identity or content"):
        model.completed_operations.load(
            "measurement",
            {
                "test": test,
                "context": context,
                "evidence": evidence,
                "tools": defaults(model).tools.catalog(),
            },
        )


def test_all_missing_citations_are_located_without_filling_them():
    first, second = criterion(), criterion()
    first['clauses'][0]['evidence_refs'] = []
    second.pop('evidence_refs')
    raw = {'criteria': [first, second]}
    old = deepcopy(raw)
    with pytest.raises(ResponseContractError) as caught:
        protocol.expand(raw, 'outcome_expectations', {}, {'task': 'Return JSON'})
    assert {i['field'] for i in caught.value.details['issues']} == {
        'criteria[0].clauses[0].evidence_refs', 'criteria[1].evidence_refs'}
    assert raw == old
    catalog = protocol.source_catalog('outcome_jury_map', {'independent_expectations': initial()})
    assert set(catalog) == {'seat:0/0'} and 'task' not in catalog


@pytest.mark.asyncio
async def test_parent_revision_completion_requires_current_full_validation(tmp_path):
    from reward_engine.core.outcome.completed_operations import CompletedOperations

    context, evidence = build_context({'task_id': 'synthetic', 'prompt': 'Return JSON'})
    calls, validations = [], []
    @enable
    async def model(request):
        calls.append(request)
        return {'criteria': [criterion()]}
    def validator(row):
        validations.append(deepcopy(row))
        return validate(row, evidence, context, clause_contract=True)
    schema = {'criteria': [criterion()]}
    model.completed_operations = CompletedOperations(tmp_path, {'config_identity': 'old'})
    first = await ask(model, 'outcome_expectations', context, evidence, 'Task', schema, validate=validator)
    model.completed_operations = CompletedOperations(tmp_path, {'config_identity': 'new'}, ancestors=['old'])
    second = await ask(model, 'outcome_expectations', context, evidence, 'Task', schema, validate=validator)
    assert first == second and len(calls) == 1 and len(validations) == 2
    assert len(list(tmp_path.glob('*.reuse.json'))) == 1
    model.completed_operations = CompletedOperations(tmp_path, {'config_identity': 'unrelated'})
    await ask(model, 'outcome_expectations', context, evidence, 'Task', schema, validate=validator)
    assert len(calls) == 2
