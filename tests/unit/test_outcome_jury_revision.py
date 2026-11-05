"""A member revision cannot silently retire, weaken or relabel original requirements."""

from copy import deepcopy

import pytest

from reward_engine.core.outcome import jury
from reward_engine.core.outcome.context import ResponseContractError, build_context
from reward_engine.core.outcome.jury_drafts import (
    apply_part_decisions,
    compact_selection_inputs,
    expectations,
    reduction_inputs,
)
from reward_engine.core.outcome.jury_revision import VERSION, materialize
from reward_engine.core.outcome.reducer_projection import expand_texts


def criterion():
    return {
        "id": "json",
        "standard": "Produce complete JSON",
        "kind": "hard",
        "origin": "task",
        "priority": "critical",
        "evidence_refs": ["task"],
        "measurement_question": "Check JSON and exceptions",
        "subrequirements": ["Valid JSON", "Keep every exception"],
    }


def initial():
    return expectations([criterion()], "seat", "before_products")


def target(index):
    return {"criterion_id": "json", "part_index": index}


def proposal(prefix="seat"):
    before = expectations([criterion()], prefix, "before_products")
    draft = materialize(
        {"criteria": [{"inherit_from": before[0]["expectation_id"]}]},
        before,
        {"task": "Task"},
    )
    return {
        **draft,
        "initial_draft": {"criteria": [criterion()]},
        "independent_expectations": before,
        "proposal_expectations": expectations(
            draft["criteria"], prefix + ":observed", "after_products"
        ),
    }


def test_inheritance_is_complete_and_never_mutates_original():
    before = initial()
    saved = deepcopy(before)
    result = materialize(
        {"criteria": [{"inherit_from": "seat:0"}]}, before, {"task": "Task"}
    )
    assert result["criteria"] == [criterion()]
    assert all(d["action"] == "retain" for d in result["revision_decisions"])
    assert before == saved
    assert result["revision_protocol"] == VERSION


@pytest.mark.parametrize("row", [{"id": "json", "inherit_from": "seat:0"}, {"inherit_from": "json"}])
def test_inheritance_repair_diagnostic_distinguishes_ids_without_editing(row):
    before = initial()
    raw = {"criteria": [deepcopy(row)]}
    with pytest.raises(ResponseContractError) as caught:
        materialize(raw, before, {"task": "Task"})
    detail = caught.value.details
    assert detail["criterion_index"] == 0
    assert detail["actual"] == row
    assert detail["allowed_inheritance"] == [{"inherit_from": "seat:0"}]
    assert "criterion id" in detail["instruction"]
    assert raw == {"criteria": [row]}


def test_inheritance_diagnostic_lists_all_invalid_references_and_overrides():
    before = initial()
    raw = {"criteria": [{"id": "override", "inherit_from": "seat:0"},
                        {"inherit_from": "seat:0"}, {"inherit_from": "missing"},
                        {"inherit_from": ["seat:0"]}]}
    saved = deepcopy(raw)
    with pytest.raises(ResponseContractError) as caught:
        materialize(raw, before, {"task": "Task"})
    assert caught.value.details["invalid_inheritance"] == [
        {"criterion_index": i, "actual": raw["criteria"][i]} for i in (0, 2, 3)
    ]
    assert raw == saved


def test_invalid_revision_target_reports_actual_and_allowed_targets():
    raw = {"criteria": [criterion()], "revision_decisions": [
        {"source_parts": [initial()[0]["subrequirements"][0]["part_id"]],
         "action": "retain", "targets": [{"criterion_id": "json", "part_index": 99}]}]}
    with pytest.raises(ResponseContractError) as caught:
        materialize(raw, initial(), {"task": "Task"})
    assert caught.value.details["actual"] == {"criterion_id": "json", "part_index": 99}
    assert caught.value.details["allowed_targets"] == [target(0), target(1)]


def test_invalid_inheritance_does_not_hide_independent_target_errors():
    raw = {"criteria": [{"inherit_from": "json"},
                        {**criterion(), "id": "extra"}],
           "revision_decisions": [
               {"source_parts": ["seat:0/0"], "action": "retain", "targets": [target(0)]},
               {"source_parts": ["seat:0/1"], "action": "rewrite",
                "targets": [{"criterion_id": "extra_typo", "part_index": 1}],
                "reason": "Clarify", "evidence_refs": ["task"]}]}
    saved = deepcopy(raw)
    with pytest.raises(ResponseContractError) as caught:
        materialize(raw, initial(), {"task": "Task"})
    details = caught.value.details
    assert details["invalid_inheritance"]
    references = details["reference_preflight"]
    assert references["invalid_targets"] == [
        {"decision_index": 1, "target_index": 0,
         "path": ["revision_decisions", 1, "targets", 0],
         "actual": {"criterion_id": "extra_typo", "part_index": 1}}]
    assert target(0) in references["possible_targets"]
    assert {"criterion_id": "extra", "part_index": 1} in references["possible_targets"]
    assert raw == saved


def test_inheritance_preflight_includes_independent_retain_meaning_errors():
    before = initial()
    raw = {"criteria": [{"inherit_from": "missing"},
                        {**criterion(), "id": "changed", "priority_reason": "A changed scope"}],
           "revision_decisions": [{"source_parts": ["seat:0/0"], "action": "retain",
               "targets": [{"criterion_id": "changed", "part_index": 0}]}]}
    saved = deepcopy(raw)
    with pytest.raises(ResponseContractError) as caught:
        materialize(raw, before, {"task": "Task"})
    errors = caught.value.details["reference_preflight"]["retain_errors"]
    assert errors[0]["path"] == ["revision_decisions", 0]
    assert errors[0]["retain_mismatches"][0]["field_differences"][0]["field"] == "priority_reason"
    assert raw == saved


def test_inheritance_preflight_does_not_guess_unresolved_target_part_counts():
    raw = {"criteria": [{"inherit_from": "json", "clauses": [
        {"text": "Valid JSON"}, {"text": "Keep every exception"}, {"text": "A proposed addition"}]}],
        "revision_decisions": [{"source_parts": [], "action": "add",
            "targets": [{"criterion_id": "json", "part_index": 2}],
            "reason": "A new proposal", "evidence_refs": ["task"]}]}
    with pytest.raises(ResponseContractError) as caught:
        materialize(raw, initial(), {"task": "Task"})
    # Whether the author converts the override into a complete revised criterion
    # is unresolved. Its part count cannot yet be diagnosed as an invalid target.
    assert caught.value.details["reference_preflight"]["invalid_targets"] == []


def test_revision_diagnostics_cover_multiple_decisions_and_targets():
    before = initial()
    raw = {"criteria": [{**criterion(), "priority_reason": "New explanation"}],
           "revision_decisions": [
               {"source_parts": [before[0]["subrequirements"][0]["part_id"]], "action": "retain",
                "targets": [{"criterion_id": "wrong", "part_index": 0},
                            {"criterion_id": "wrong", "part_index": 1}]},
               {"source_parts": [before[0]["subrequirements"][1]["part_id"]], "action": "retain",
                "targets": [target(1)]},
           ]}
    saved = deepcopy(raw)
    with pytest.raises(ResponseContractError) as caught:
        materialize(raw, before, {"task": "Task"})
    errors = caught.value.details["decision_errors"]
    assert [e["decision_index"] for e in errors] == [0, 1]
    assert [e["target_index"] for e in errors[0]["invalid_targets"]] == [0, 1]
    assert errors[1]["retain_mismatches"]
    assert raw == saved


def test_retain_diagnostics_locate_nested_literal_quote_difference():
    base = criterion()
    base["clauses"] = [{"id": str(i), "text": text,
                        "source_quotes": [{"ref": "task", "quote": "Use A"}]}
                       for i, text in enumerate(base["subrequirements"])]
    before = expectations([base], "seat", "before_products")
    changed = deepcopy(base)
    changed["clauses"][0]["source_quotes"][0]["quote"] = "Use B"
    raw = {"criteria": [changed], "revision_decisions": [
        {"source_parts": ["seat:0/0"], "action": "retain", "targets": [target(0)]}]}
    with pytest.raises(ResponseContractError) as caught:
        materialize(raw, before, {"task": "Use A"})
    field = caught.value.details["retain_mismatches"][0]["field_differences"][0]
    assert field["field"] == "clause"
    assert field["nested_differences"] == [{
        "path": ["source_quotes", 0, "quote"], "source_present": True, "target_present": True,
        "source_value": "Use A", "target_value": "Use B"}]
    assert raw["criteria"][0]["clauses"][0]["source_quotes"][0]["quote"] == "Use B"


@pytest.mark.asyncio
async def test_field_repair_preserves_inherited_criteria():
    before = initial()
    reference = {"inherit_from": before[0]["expectation_id"]}
    added = {**criterion(), "id": "style", "kind": "critical", "origin": "domain"}
    original = {"criteria": [reference, added]}
    saved = deepcopy(original)
    request = {
        "mode": "outcome_jury_map",
        "response_recovery": {
            "details": {"criterion_field_repairs": [{"id": "style"}]},
            "rejected_response": original,
        },
    }

    async def repair(request):
        assert request["mode"] == "outcome_jury_map_field_repair"
        return {"updates": [{"id": "style", "kind": "soft", "origin": "domain"}]}

    result = await jury.field_repair_call(repair)(request)
    assert original == saved
    assert result["criteria"] == [reference, {**added, "kind": "soft"}]
    expanded = materialize(result, before, {"task": "Task"})
    assert expanded["criteria"][0] == criterion()
    assert expanded["criteria"][1]["kind"] == "soft"


def test_field_repair_keeps_provider_request_and_evidence_limits():
    async def call(request):
        return request

    call.request_budget_bytes = 250000
    call.evidence_expansion_limit = 12
    wrapped = jury.field_repair_call(call)
    assert wrapped.request_budget_bytes == 250000
    assert wrapped.evidence_expansion_limit == 12


@pytest.mark.parametrize(
    "change",
    [
        {"priority": "minor"},
        {"origin": "domain"},
        {"measurement_question": "Only check syntax"},
        {"standard": "Ignore exceptions"},
        {"conditions": ["Only if convenient"]},
        {"subrequirements": ["Valid JSON", "Keep some exceptions"]},
    ],
)
def test_retain_cannot_hide_changed_meaning(change):
    raw = {
        "criteria": [{**criterion(), **change}],
        "revision_decisions": [
            {"source_parts": ["seat:0/1"], "action": "retain", "targets": [target(1)]}
        ],
    }
    with pytest.raises(ResponseContractError, match="Retain must preserve"):
        materialize(raw, initial(), {"task": "Task"})


def test_rewrite_delete_and_omission_remain_reviewable():
    raw = {
        "criteria": [{**criterion(), "subrequirements": ["Use parseable JSON"]}],
        "revision_decisions": [
            {
                "source_parts": ["seat:0/0"],
                "action": "rewrite",
                "targets": [target(0)],
                "reason": "Clarify syntax",
                "evidence_refs": ["task"],
            }
        ],
    }
    result = materialize(raw, initial(), {"task": "Task"})
    missing = result["revision_decisions"][1]
    assert missing["source_parts"] == ["seat:0/1"] and missing["action"] == "defer"
    assert missing["code_deferred"]
    raw["revision_decisions"].append(
        {
            "source_parts": ["seat:0/1"],
            "action": "delete",
            "targets": [],
            "reason": "Member proposes exclusion",
            "evidence_refs": ["task"],
        }
    )
    result = materialize(raw, initial(), {"task": "Task"})
    assert result["revision_decisions"][1]["action"] == "delete"
    # Deletion is a draft decision, not final coverage or permission to drop text.
    assert "coverage" not in result


@pytest.mark.parametrize(
    "decision",
    [
        {
            "source_parts": ["invented"],
            "action": "defer",
            "targets": [],
            "reason": "Unknown",
        },
        {
            "source_parts": ["seat:0/0"],
            "action": "delete",
            "targets": [],
            "reason": "No evidence",
        },
        {
            "source_parts": ["seat:0/0"],
            "action": "rewrite",
            "targets": [target(20)],
            "reason": "Unknown target",
            "evidence_refs": ["task"],
        },
    ],
)
def test_invalid_revision_cannot_be_published(decision):
    with pytest.raises(ValueError):
        materialize(
            {"criteria": [criterion()], "revision_decisions": [decision]},
            initial(),
            {"task": "Task"},
        )


def test_aliases_preserve_all_original_and_revised_coverage_across_three_members():
    inputs = reduction_inputs([proposal(str(i)) for i in range(3)])
    view = compact_selection_inputs(inputs)
    visible = [
        p
        for group in ("independent_expectations", "proposal_expectations")
        for row in view[group]
        for p in row["subrequirements"]
    ]
    assert (
        len(visible) == 2
    )  # Two unique parts, twelve independently scoped occurrences.
    raw = {
        "criteria": [criterion()],
        "selection_protocol": "exact-alias-v1",
        "part_decisions": [
            {"parts": [p["selection_id"]], "action": "retain", "criterion_id": "json"}
            for p in visible
        ],
    }
    result = apply_part_decisions(raw, inputs, {"task": "Task"})
    assert len(result["coverage"]) == 6
    assert all(
        p["status"] == "mapped"
        for r in result["coverage"]
        for p in r["subrequirements"]
    )


def test_new_parts_and_merge_have_explicit_basis():
    raw = {
        "criteria": [
            {
                **criterion(),
                "subrequirements": [
                    "Valid JSON retaining exceptions",
                    "Explain unknowns",
                ],
            }
        ],
        "revision_decisions": [
            {
                "source_parts": ["seat:0/0", "seat:0/1"],
                "action": "merge",
                "targets": [target(0)],
                "reason": "Combine without weakening either obligation",
                "evidence_refs": ["task"],
            },
            {
                "source_parts": [],
                "action": "add",
                "targets": [target(1)],
                "reason": "Public task asks to explain uncertainty",
                "evidence_refs": ["task"],
            },
        ],
    }
    result = materialize(raw, initial(), {"task": "Task"})
    assert not result["unclaimed_additions"]


@pytest.mark.asyncio
async def test_real_revision_pipeline_keeps_blinding_cache_and_complete_coverage():
    context, evidence = build_context(
        {"task_id": "json", "prompt": "Return valid JSON and keep every exception."},
        [{"artifact": {"final_response": "PRIVATE_PRODUCT"}}],
    )
    calls, records, cache = [], {}, {}

    async def member(request):
        calls.append(request["mode"])
        if request["mode"] == "outcome_expectations":
            assert "PRIVATE_PRODUCT" not in str(request)
            return {"criteria": [criterion()]}
        assert request["revision_protocol"] == VERSION
        return {
            "criteria": [
                {
                    "inherit_from": request["independent_expectations"][0][
                        "expectation_id"
                    ]
                }
            ]
        }

    async def reduce(request):
        request = expand_texts(request)
        assert request["revision_protocol"] == VERSION
        visible = [
            p
            for group in ("independent_expectations", "proposal_expectations")
            for row in request[group]
            for p in row["subrequirements"]
        ]
        return {
            "criteria": [criterion()],
            "selection_protocol": "exact-alias-v1",
            "part_decisions": [
                {
                    "parts": [p["selection_id"]],
                    "action": "retain",
                    "criterion_id": "json",
                }
                for p in visible
            ],
        }

    for _ in range(2):
        result = await jury.propose(
            context,
            evidence,
            {"a": member, "b": member},
            reduce,
            revision_protocol=True,
            expectation_cache=cache,
            record=records.__setitem__,
        )
        assert len(result["coverage"]) == 4
        assert all(r["status"] == "mapped" for r in result["coverage"])
    assert calls.count("outcome_expectations") == 2
    assert calls.count("outcome_jury_map") == 4
    assert (
        records["jury_reduce_attempt_1/reducer_request_projections"][0][
            "additional_compression_calls"
        ]
        == 0
    )


@pytest.mark.asyncio
async def test_common_omission_is_not_hidden_when_member_omits_revision_ledger():
    context, evidence = build_context(
        {"task_id": "json", "prompt": "Return valid JSON and keep every exception."}
    )

    async def member(request):
        row = criterion()
        if request["mode"] == "outcome_jury_map":
            row["subrequirements"] = ["Valid JSON"]
        return {"criteria": [row]}

    async def reduce(request):
        request = expand_texts(request)
        assert any(
            p["text"] == "Keep every exception"
            for r in request["independent_expectations"]
            for p in r["subrequirements"]
        )
        assert all(
            d["action"] == "defer"
            for m in request["proposals"]
            for d in m["revision_decisions"]
        )
        return {"criteria": [criterion()]}

    result = await jury.propose(
        context, evidence, {"a": member, "b": member}, reduce, revision_protocol=True
    )
    assert all(r["status"] == "deferred" for r in result["coverage"])


def test_tampered_checkpoint_cannot_alias_changed_priority():
    inputs = reduction_inputs([proposal("a"), proposal("b")])
    inputs["proposal_expectations"][0]["priority"] = "minor"
    with pytest.raises(ValueError, match="canonical source"):
        compact_selection_inputs(inputs)


@pytest.mark.asyncio
async def test_structured_inheritance_keeps_conditions_quotes_and_full_coverage():
    task = "Valid JSON. Keep every exception."
    context, evidence = build_context({"task_id": "json", "prompt": task})
    row = criterion()
    row["clauses"] = [
        {
            "id": f"c{i}",
            "text": text,
            "use": "task_constraint",
            "evidence_refs": ["task"],
            "source_quotes": [{"ref": "task", "quote": text}],
            "conditions": ["When requested"],
            "alternative_group": "",
        }
        for i, text in enumerate(row["subrequirements"])
    ]

    async def member(request):
        if request["mode"] == "outcome_expectations":
            return {"criteria": [deepcopy(row)]}
        return {
            "criteria": [
                {
                    "inherit_from": request["independent_expectations"][0][
                        "expectation_id"
                    ]
                }
            ]
        }

    async def reduce(request):
        request = expand_texts(request)
        parts = [
            p
            for g in ("independent_expectations", "proposal_expectations")
            for r in request[g]
            for p in r["subrequirements"]
        ]
        return {
            "criteria": [deepcopy(row)],
            "selection_protocol": "exact-alias-v1",
            "part_decisions": [
                {
                    "parts": [p["selection_id"]],
                    "action": "retain",
                    "criterion_id": "json",
                    "clause_id": next(
                        c["id"] for c in row["clauses"] if c["text"] == p["text"]
                    ),
                }
                for p in parts
            ],
        }

    result = await jury._propose_once(
        context,
        evidence,
        {str(i): member for i in range(3)},
        reduce,
        strict=True,
        clause_contract=True,
        revision_protocol=True,
    )
    for expected, actual in zip(
        row["clauses"], result["criteria"][0]["clauses"], strict=True
    ):
        assert {k: v for k, v in actual.items() if k != "source_part_ids"} == expected
        assert len(actual["source_part_ids"]) == 6
    assert len(result["coverage"]) == 6
    assert all(r["status"] == "mapped" for r in result["coverage"])


@pytest.mark.asyncio
async def test_targeted_office_repairs_keep_existing_two_repair_bound_and_all_metrics():
    context, evidence = build_context(
        {"task_id": "json", "prompt": "Valid JSON and keep every exception."}
    )
    inputs = reduction_inputs([proposal("a"), proposal("b")])
    requests, records = [], {}

    async def reducer(request):
        request = expand_texts(request)
        requests.append(request)
        if len(requests) == 1:
            return {"criteria": [{**criterion(), "origin": "unsupported-origin"}]}
        return {
            "edits": [
                {
                    "op": "replace",
                    "path": ["criteria", 0, "origin"],
                    "value": "task" if len(requests) == 3 else "still-invalid",
                    "reason": "Task supplies this requirement",
                }
            ]
        }

    reducer.jury_repair_mode = "targeted"
    output = await jury.reduce_proposals(
        context, evidence, inputs, reducer, record=records.__setitem__
    )
    assert [r["mode"] for r in requests] == [
        "outcome_jury_reduce",
        "outcome_jury_repair",
        "outcome_jury_repair",
    ]
    assert len(records["reducer_request_projections"]) == 3
    assert all(
        r["within_budget"] and r["request_sha256"]
        for r in records["reducer_request_projections"]
    )
    assert len(output["coverage"]) == 4


@pytest.mark.asyncio
async def test_revised_draft_wrapper_preserves_office_budget_and_evidence_limits():
    context, evidence = build_context(
        {"task_id": "json", "prompt": "Public Task " * 16000}
    )
    for i in range(7):
        evidence[f"source:{i}"] = "Full source " * 3000
    calls = []

    async def member(request):
        calls.append(request)
        if len(calls) == 1:
            return {"need_evidence": [f"source:{i}" for i in range(7)]}
        assert request["requested_evidence"] == {
            f"source:{i}": evidence[f"source:{i}"] for i in range(7)
        }
        return {"criteria": [criterion()]}

    member.request_budget_bytes = 1_000_000
    member.evidence_expansion_limit = 64
    result = await jury.ask_draft(
        member, "outcome_jury_map", context, evidence, "Revise", jury.SCHEMA
    )
    assert result["criteria"] == [criterion()]
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_legacy_revision_map_gets_one_contract_correction_without_new_expectations():
    context, evidence = build_context(
        {"task_id": "json", "prompt": "Return valid JSON and keep every exception."}
    )
    calls = []

    async def member(request):
        calls.append(request)
        if request["mode"] == "outcome_expectations":
            return {"criteria": [criterion()]}
        ref = request["independent_expectations"][0]["expectation_id"]
        if "response_recovery" not in request:
            return {"criteria": [{"inherit_from": ref, "id": "forbidden_override"}]}
        assert "inherit_from" in str(request["response_recovery"])
        return {"criteria": [{"inherit_from": ref}]}

    async def reduce(request):
        return {"criteria": [criterion()]}

    result = await jury.propose(
        context, evidence, {"a": member, "b": member}, reduce, revision_protocol=True
    )
    assert result["criteria"]
    assert sum(r["mode"] == "outcome_expectations" for r in calls) == 2
    assert sum("response_recovery" in r for r in calls) == 2
    assert len(calls) == 6


@pytest.mark.asyncio
async def test_invalid_draft_ledger_cannot_retire_originals_after_bounded_correction():
    context, evidence = build_context(
        {"task_id": "json", "prompt": "Return valid JSON and keep every exception."}
    )
    calls, records = [], {}

    async def member(request):
        calls.append(request)
        row = criterion()
        if request["mode"] == "outcome_expectations":
            return {"criteria": [row]}
        row["priority"] = "minor"
        source = request["independent_expectations"][0]["subrequirements"][0]["part_id"]
        return {
            "criteria": [row],
            "revision_decisions": [
                {
                    "source_parts": [source],
                    "action": "retain",
                    "targets": [{"criterion_id": "json", "part_index": 0}],
                }
            ],
        }

    async def reduce(request):
        request = expand_texts(request)
        for proposal in request["proposals"]:
            assert proposal["revision_ledger_failure"]
            assert proposal["rejected_revision_decisions"][0]["action"] == "retain"
            assert all(d["action"] == "defer" for d in proposal["revision_decisions"])
            assert len(proposal["unclaimed_additions"]) == 2
        return {"criteria": [criterion()]}

    result = await jury.propose(
        context,
        evidence,
        {"a": member, "b": member},
        reduce,
        revision_protocol=True,
        record=records.__setitem__,
    )
    assert result["criteria"][0]["priority"] == "critical"
    assert all(r["status"] == "deferred" for r in result["coverage"])
    assert sum("response_recovery" in r for r in calls) == 2


@pytest.mark.asyncio
async def test_invalid_corrected_criteria_falls_back_to_validated_expectations():
    context, evidence = build_context(
        {"task_id": "json", "prompt": "Return valid JSON and keep every exception."}
    )
    calls = []

    async def member(request):
        calls.append(request)
        if request["mode"] == "outcome_expectations":
            return {"criteria": [criterion()]}
        ref = request["independent_expectations"][0]["expectation_id"]
        if "response_recovery" not in request:
            return {"criteria": [{"inherit_from": ref, "id": "invalid_override"}]}
        return {
            "criteria": [{**criterion(), "subrequirements": [""]}],
            "revision_decisions": [],
        }

    async def reduce(request):
        observed = {
            row["expectation_id"]: row for row in request["proposal_expectations"]
        }
        for draft in request["proposals"]:
            rows = [observed[ref] for ref in draft["criterion_refs"]]
            assert [row["standard"] for row in rows] == [criterion()["standard"]]
            assert draft["rejected_revision_criteria"][0]["subrequirements"] == [""]
            assert draft["revision_criteria_failure"]["field"] == "revision_decisions"
            assert all(row["action"] == "defer" for row in draft["revision_decisions"])
        return {"criteria": [criterion()]}

    result = await jury.propose(
        context, evidence, {"a": member, "b": member}, reduce, revision_protocol=True
    )
    assert result["criteria"] == [criterion()]
    assert sum("response_recovery" in request for request in calls) == 2
