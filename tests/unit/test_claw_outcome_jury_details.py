"""Draft tolerance and per-detail accountability, without another model layer."""

from copy import deepcopy

import pytest

from reward_engine.core.outcome import builder, jury
from reward_engine.core.outcome.coverage import complete, final_test, report
from reward_engine.core.outcome.harness import defaults, measure
from reward_engine.core.outcome.jury_drafts import (
    expectations,
    keep_draft,
    normalize_reduced,
    reduction_inputs,
)
from reward_engine.core.outcome.task_progress import TaskProgress
from tests.support.claw_outcome_builder_harness import setup


def criterion():
    return {
        "id": "syntax",
        "standard": "Valid JSON",
        "kind": "soft",
        "origin": "domain",
        "evidence_refs": ["task"],
        "measurement_question": "Can the JSON parse?",
        "subrequirements": ["Valid JSON", "Explain the result", "Optional diagram"],
    }


def decision(expected):
    return {
        "expectation_id": expected["expectation_id"],
        "status": "mapped",
        "criterion_ids": ["syntax"],
        "reason": "Keep syntax, account for other suggestions",
        "subrequirements": [
            {
                "part_id": p["part_id"],
                "status": "mapped",
                "criterion_ids": ["syntax"],
                "reason": "Explicitly retained",
                "retained_text": p["text"],
            }
            for p in expected["subrequirements"]
        ],
    }


@pytest.mark.parametrize(
    "change",
    [
        {"kind": "important"},
        {"kind": "hard", "origin": "domain"},
        {"evidence_refs": ["nonexistent"]},
        {"measurement_question": None},
    ],
)
@pytest.mark.asyncio
async def test_bad_initial_fields_reach_reduce_without_invented_corrections(change):
    context, evidence, _, _ = setup()
    raw = {"criteria": [{**criterion(), **change}]}
    calls = []

    async def member(request):
        calls.append(request["mode"])
        return deepcopy(raw)

    async def reduce(request):
        assert len(request["proposals"]) == 2
        by_id = {r["expectation_id"]: r for r in request["independent_expectations"]}
        for proposal in request["proposals"]:
            row = by_id[proposal["initial_draft"]["criterion_refs"][0]]
            assert row["draft_validation_issues"]
            assert all(row[k] == v for k, v in change.items())
        assert len(request["proposal_expectations"]) == 2
        return {"criteria": [criterion()]}

    output = await jury.propose(context, evidence, {"a": member, "b": member}, reduce)
    assert len(calls) == 4
    assert len(output["coverage"]) == 4
    assert all(r["status"] == "deferred" for r in output["coverage"])
    with pytest.raises((ValueError, TypeError)):
        jury.validate(
            raw, evidence, context
        )  # Tolerant drafts are NOT published criteria.
    assert "draft_validation_issues" not in raw["criteria"][0]


def test_reduce_deduplicates_criteria_without_losing_details_or_disagreements():
    before = expectations([criterion()], "a", "before_products")
    after = expectations(
        [{**criterion(), "standard": "A competing interpretation"}],
        "a:observed",
        "after_products",
    )
    proposal = {
        "criteria": [criterion()],
        "gaps": [{"gap": "Important disagreement"}],
        "initial_draft": {
            "criteria": [criterion()],
            "draft_validation_issues": ["bad enum"],
        },
        "independent_expectations": before,
        "proposal_expectations": after,
        "uncertainties": ["Need a real measurement"],
    }
    saved = deepcopy(proposal)
    result = reduction_inputs([proposal])
    assert result["independent_expectations"] == before
    assert result["proposal_expectations"] == after
    member = result["proposals"][0]
    assert "criteria" not in member and "criteria" not in member["initial_draft"]
    assert member["gaps"] == proposal["gaps"]
    assert member["uncertainties"] == proposal["uncertainties"]
    assert member["initial_draft"]["draft_validation_issues"] == ["bad enum"]
    assert member["criterion_refs"] == [after[0]["expectation_id"]]
    assert member["initial_draft"]["criterion_refs"] == [before[0]["expectation_id"]]
    assert proposal == saved


def test_reduce_task_alias_and_check_claim_are_conservative_not_fabricated():
    ctx = {"task": {"task_id": "current"}}
    evidence = {"task": "Actual task", "product_1:artifact": "Actual delivery"}
    raw = {
        "criteria": [{"evidence_refs": ["task:current", "task:other"]}],
        "gaps": [
            {
                "gap_id": "gap",
                "status": "check_supported",
                "product_ids": ["product_1"],
                "evidence_refs": ["product_1:artifact"],
                "gap": "Model's real reasoning is retained",
            }
        ],
    }
    out = normalize_reduced(raw, ctx, evidence)
    assert out["criteria"][0]["evidence_refs"] == ["task", "task:other"]
    assert out["gaps"][0]["status"] == "needs_check"
    assert out["gaps"][0]["gap"] == raw["gaps"][0]["gap"]
    assert raw["gaps"][0]["status"] == "check_supported"
    assert len(out["admission_notes"]) == 2
    # A genuinely registered handle wins over alias interpretation.
    out = normalize_reduced(
        raw, ctx, {**evidence, "task:current": "Different registered source"}
    )
    assert out["criteria"][0]["evidence_refs"][0] == "task:current"


@pytest.mark.parametrize(
    "failed,error,product,expected",
    [
        ("fail", False, "product_1", "check_supported"),
        ("pass", False, "product_1", "needs_check"),
        ("fail", True, "product_1", "needs_check"),
        ("fail", False, "product_2", "needs_check"),
    ],
)
def test_only_actual_failed_check_can_support_gap(failed, error, product, expected):
    raw = {
        "gaps": [
            {
                "gap_id": "gap",
                "status": "check_supported",
                "product_ids": ["product_1"],
                "evidence_refs": ["check:1"],
            }
        ]
    }
    ctx = {"task": {"task_id": "current"}}
    evidence = {"check:1": {"status": failed, "error": error, "product_id": product}}
    assert normalize_reduced(raw, ctx, evidence)["gaps"][0]["status"] == expected


def test_empty_proposals_reject_but_mixed_rows_preserved():
    context, evidence, _, _ = setup()
    for raw in (None, [], {}, {"criteria": [{}]}, {"criteria": ["bad"]}):
        with pytest.raises(ValueError):
            keep_draft(raw, evidence, context, jury.validate)
    raw = {"criteria": [criterion(), {"id": "unfinished"}]}
    result = keep_draft(raw, evidence, context, jury.validate)
    assert result["unusable_criteria"] == [{"id": "unfinished"}]
    assert len(result["criteria"]) == 1


@pytest.mark.parametrize("gaps", [None, "malformed", [None]])
def test_bad_gap_shape_does_not_erase_substantive_criteria(gaps):
    context, evidence, _, _ = setup()
    raw = {"criteria": [criterion()], "gaps": gaps}
    result = keep_draft(raw, evidence, context, jury.validate)
    assert result["criteria"] == raw["criteria"]
    assert result["gaps"] == gaps and result["draft_validation_issues"]
    with pytest.raises(ValueError):
        jury.validate(raw, evidence, context)


def test_parent_id_cannot_hide_missing_details():
    exp = expectations([criterion()], "a", "before_products")[0]
    row = decision(exp)
    row.pop("subrequirements")
    result = jury.reconcile_coverage(
        [exp], {"criteria": [criterion()], "coverage": [row]}
    )
    assert result[0]["status"] == "deferred"
    assert len(result[0]["subrequirements"]) == 3
    assert all(p["status"] == "deferred" for p in result[0]["subrequirements"])


def test_explicit_omission_allowed_but_fake_retention_is_not():
    exp = expectations([criterion()], "a", "after_products")[0]
    row = decision(exp)
    row["subrequirements"][2].update(
        status="excluded",
        criterion_ids=[],
        reason="Optional diagram not necessary for the requested outcome",
    )
    result = jury.reconcile_coverage(
        [exp], {"criteria": [criterion()], "coverage": [row]}
    )
    assert result[0]["status"] == "mapped"
    assert result[0]["subrequirements"][2]["status"] == "excluded"
    row["subrequirements"][0]["retained_text"] = "Invented retained clause"
    assert (
        jury.reconcile_coverage([exp], {"criteria": [criterion()], "coverage": [row]})[
            0
        ]["status"]
        == "deferred"
    )


@pytest.mark.parametrize("bad", ["duplicate", "unknown", "no_reason"])
def test_invalid_detail_receipts_reject(bad):
    exp = expectations([criterion()], "a", "before_products")[0]
    row = decision(exp)
    if bad == "duplicate":
        row["subrequirements"].append(deepcopy(row["subrequirements"][0]))
    elif bad == "unknown":
        row["subrequirements"][0]["part_id"] = "phantom"
    else:
        row["subrequirements"][0]["reason"] = ""
    with pytest.raises(ValueError):
        jury.reconcile_coverage([exp], {"criteria": [criterion()], "coverage": [row]})


@pytest.mark.asyncio
async def test_later_details_survive_builder_measurement_and_actor_checklist():
    context, evidence, definition, reply = setup()
    # An incomplete source field must not crash coverage or acquire invented authority.
    exp = expectations(
        [{"standard": "Explain the result"}], "a:observed", "after_products"
    )
    definition.update(proposal_expectations=exp)
    definition["coverage"] = jury.reconcile_coverage(exp, definition)

    async def call(_):
        if _["mode"] == "outcome_builder":
            assert "proposal_expectations" not in _["jury"]
            assert "coverage" not in _["jury"]
            assert _["jury"]["criteria"] == definition["criteria"]
        return reply

    registry = defaults(call)
    pack = await builder.build(context, evidence, definition, registry, call)
    assert pack["proposal_expectations"] == exp
    result = await measure(pack, registry, context, evidence)
    assert result["results"][0]["status"] == "pass"
    assert not result["ready_under_estimator"]
    assert not complete(result["outcome_coverage"])
    assert report(pack, [])[0]["origin"] == "unresolved"
    assert final_test(pack)["rubric"]["expectations"] == exp
    progress = TaskProgress(context["task"], pack)
    progress.begin(context, 1, "test")
    progress.finish(result)
    assert "Explain the result" in str(progress.for_actor()["remaining"])
    assert not progress.verified


def test_parent_exclusion_cannot_hide_retained_children():
    exp = expectations([criterion()], "a", "before_products")[0]
    row = decision(exp)
    row.update(status="excluded", criterion_ids=[])
    result = jury.reconcile_coverage(
        [exp], {"criteria": [criterion()], "coverage": [row]}
    )
    assert result[0]["status"] == "deferred"


def test_malformed_detail_list_preserves_original_prose():
    raw = {**criterion(), "subrequirements": [None]}
    exp = expectations([raw], "a", "before_products")[0]
    assert exp["subrequirements"] == [{"part_id": "a:0/0", "text": "Valid JSON"}]


@pytest.mark.asyncio
@pytest.mark.parametrize("repairable", [True, False])
async def test_reduce_repairs_missing_product_evidence_with_bounded_strict_validation(
    repairable,
):
    from reward_engine.core.outcome.context import build_context

    context, evidence = build_context(
        {"task_id": "x", "prompt": "Deliver JSON"},
        [{"artifact": {"final_response": "{"}}, {"artifact": {"final_response": "{"}}],
    )
    invalid = {
        "criteria": [criterion()],
        "gaps": [
            {
                "gap_id": "syntax",
                "product_ids": ["product_1", "product_2"],
                "expected": "Valid JSON",
                "expected_basis": "task",
                "expected_refs": ["task"],
                "observed": "Both responses contain only an opening brace",
                "gap": "Possibly invalid JSON",
                "evidence_refs": ["product_1:artifact:final_response"],
                "verification": "Parse each response",
                "status": "needs_check",
            }
        ],
    }
    calls, records = [], {}

    async def member(request):
        return {"criteria": [criterion()]}

    async def reducer(request):
        calls.append(deepcopy(request))
        if len(calls) > 1:
            repair = request["validation_repair"]
            assert (
                repair["reason"] == "Gap must cite evidence for every affected product"
            )
            assert repair["invalid_draft"] == invalid
            assert request["context"] == calls[0]["context"]
            assert request["proposals"] == calls[0]["proposals"]
            if repairable:
                return {
                    "edits": [
                        {
                            "op": "replace",
                            "path": ["gaps", 0, "evidence_refs"],
                            "value": [
                                "product_1:artifact:final_response",
                                "product_2:artifact:final_response",
                            ],
                            "reason": "Cite the actual response for each observed product",
                        }
                    ]
                }
        return deepcopy(invalid)

    reducer.jury_repair_mode = "targeted"
    run = jury.propose(
        context,
        evidence,
        {"a": member, "b": member},
        reducer,
        record=lambda name, value: records.update({name: deepcopy(value)}),
    )
    if repairable:
        output = await run
        assert len(calls) == 2
        assert jury.validate(output, evidence, context) == output
        assert output["gaps"][0]["status"] == "needs_check"
        assert "jury_reduce_repair_1_raw" in records
    else:
        with pytest.raises(ValueError, match="every affected product"):
            await run
        assert len(calls) == 3
        assert "jury" not in records and "jury_reduce_validation_2" in records
    assert records["jury_reduce_raw"] == invalid
    assert invalid["gaps"][0]["evidence_refs"] == ["product_1:artifact:final_response"]


@pytest.mark.asyncio
async def test_large_coverage_repair_preserves_full_inputs_and_identifies_unknown_ids():
    import json

    context, evidence, _, _ = setup()
    invalid = {
        "criteria": [criterion()],
        "uncertainties": ["x" * 165000],
        "coverage": [
            {
                "expectation_id": "invented",
                "status": "deferred",
                "criterion_ids": [],
                "reason": "unresolved",
            }
        ],
    }
    calls = []

    async def member(request):
        return {"criteria": [criterion()]}

    async def reducer(request):
        calls.append(deepcopy(request))
        if "validation_repair" not in request:
            return deepcopy(invalid)
        assert 160000 < len(json.dumps(request, ensure_ascii=False).encode()) < 320000
        assert request["validation_repair"]["invalid_draft"] == invalid
        assert request["validation_repair"]["unknown_expectation_ids"] == ["invented"]
        for key in (
            "context",
            "proposals",
            "independent_expectations",
            "proposal_expectations",
        ):
            assert request[key] == calls[0][key]
        return {"criteria": [criterion()]}

    reducer.jury_repair_mode = "targeted"
    result = await jury.propose(context, evidence, {"a": member, "b": member}, reducer)
    assert len(calls) == 2
    assert len(result["coverage"]) == 4
    assert all(row["status"] == "deferred" for row in result["coverage"])


@pytest.mark.asyncio
async def test_provenance_repair_identifies_invalid_field_without_inventing_authority():
    context, evidence, _, _ = setup()
    invalid = {
        "criteria": [criterion()],
        "gaps": [
            {
                "gap_id": "parse",
                "product_ids": ["product_1"],
                "expected": "A measured parsing result",
                "expected_basis": "observed_gap",
                "expected_refs": ["product_1:artifact:final_response"],
                "observed": "No recorded parser check",
                "gap": "Unknown syntax",
                "evidence_refs": ["product_1:artifact:final_response"],
                "verification": "Run a parser",
                "status": "needs_check",
            }
        ],
    }
    calls = []

    async def member(request):
        return {"criteria": [criterion()]}

    async def reducer(request):
        calls.append(request)
        if "validation_repair" not in request:
            return deepcopy(invalid)
        repair = request["validation_repair"]
        assert repair["invalid_draft"] == invalid
        field = repair["invalid_fields"][0]
        assert (
            field["path"] == "gaps[0].expected_basis"
            and field["value"] == "observed_gap"
        )
        assert field["allowed_values"] == ["task", "user", "domain"]
        return {
            "edits": [
                {
                    "op": "replace",
                    "path": ["gaps", 0, "expected_basis"],
                    "value": "domain",
                    "reason": "This is a checking prior, not an additional task instruction",
                },
                {
                    "op": "replace",
                    "path": ["gaps", 0, "expected_refs"],
                    "value": [],
                    "reason": "The artifact does not supply a task or user instruction",
                },
            ]
        }

    reducer.jury_repair_mode = "targeted"
    result = await jury.propose(context, evidence, {"a": member, "b": member}, reducer)
    assert len(calls) == 2 and result["gaps"][0]["expected_basis"] == "domain"
    assert invalid["gaps"][0]["expected_basis"] == "observed_gap"


def test_repair_diagnoses_hidden_coverage_errors_and_applies_edits_atomically():
    from reward_engine.core.outcome.jury_repair import apply_edits, diagnostics

    expected = expectations([criterion()], "a", "before_products")
    draft = {
        "criteria": [criterion()],
        "gaps": [{"expected_basis": "observed_gap"}],
        "coverage": [
            {"expectation_id": "a:0", "subrequirements": [{"part_id": "invented"}]}
        ],
    }
    issues = diagnostics(draft, expected)
    assert [i["path"] for i in issues] == [
        ["gaps", 0, "expected_basis"],
        ["coverage", 0, "subrequirements", 0],
    ]
    assert issues[1]["allowed_ids"] == [
        p["part_id"] for p in expected[0]["subrequirements"]
    ]
    edit = {
        "op": "replace",
        "path": ["gaps", 0, "expected_basis"],
        "value": "domain",
        "reason": "prior",
    }
    result = apply_edits(draft, {"edits": [edit]})
    assert result["gaps"][0]["expected_basis"] == "domain"
    assert draft["gaps"][0]["expected_basis"] == "observed_gap"
    for path in (
        ["gaps", -1],
        ["gaps", True],
        ["gaps", 4],
        ["task"],
        ["coverage", 0, "missing"],
    ):
        with pytest.raises(ValueError):
            apply_edits(
                draft,
                {"edits": [edit, {"op": "remove", "path": path, "reason": "bad"}]},
            )
        assert draft["gaps"][0]["expected_basis"] == "observed_gap"


def test_repair_locates_published_subrequirements_without_changing_coverage_schema():
    from reward_engine.core.outcome.jury_repair import diagnostics

    expected = expectations([criterion()], "a", "before_products")
    parts = [
        {
            "part_id": p["part_id"],
            "retained_text": p["text"],
            "status": "deferred",
            "reason": "Pending evidence",
        }
        for p in expected[0]["subrequirements"]
    ]
    draft = {
        "criteria": [
            {
                **criterion(),
                "subrequirements": [{"id": "local", "text": "Keep all text"}],
            }
        ],
        "coverage": [{"expectation_id": "a:0", "subrequirements": parts}],
    }
    issues = diagnostics(draft, expected)
    assert [i["path"] for i in issues] == [["criteria", 0, "subrequirements"]]
    assert "coverage decisions remain objects" in issues[0]["reason"]
    assert draft["criteria"][0]["subrequirements"][0]["text"] == "Keep all text"


def test_repair_diagnoses_part_ids_used_as_criterion_targets():
    from reward_engine.core.outcome.jury_repair import diagnostics

    expected = expectations([criterion()], "a", "before_products")
    draft = {
        "criteria": [criterion()],
        "coverage": [
            {
                "expectation_id": "a:0",
                "subrequirements": [
                    {
                        "part_id": expected[0]["subrequirements"][0]["part_id"],
                        "criterion_ids": [criterion()["id"] + "/0"],
                        "status": "mapped",
                        "reason": "Retained clause",
                    }
                ],
            }
        ],
    }
    issues = diagnostics(draft, expected)
    assert len(issues) == 1
    assert issues[0]["path"] == ["coverage", 0, "subrequirements"]
    assert issues[0]["allowed_criterion_ids"] == [criterion()["id"]]


def test_repair_locates_empty_criterion_refs_before_unrelated_gap_edits():
    from reward_engine.core.outcome.jury_repair import diagnostics

    draft = {"criteria": [{**criterion(), "evidence_refs": []}]}
    issues = diagnostics(draft, [])
    assert [row["path"] for row in issues] == [["criteria", 0, "evidence_refs"]]


@pytest.mark.parametrize(
    "change",
    [
        {"status": "unmapped"},
        {"status": "mapped", "criterion_ids": []},
        {"status": "deferred", "criterion_ids": ["criterion"]},
        {"reason": ""},
        {"criterion_ids": "criterion"},
    ],
)
def test_repair_locates_invalid_disposition_without_changing_judgment(change):
    from reward_engine.core.outcome.jury_repair import diagnostics

    expected = expectations([criterion()], "a", "before_products")
    part = {
        "part_id": expected[0]["subrequirements"][0]["part_id"],
        "status": "deferred",
        "criterion_ids": [],
        "reason": "Need evidence",
        **change,
    }
    draft = {
        "criteria": [{**criterion(), "id": "criterion"}],
        "coverage": [{"expectation_id": "a:0", "subrequirements": [part]}],
    }
    original = deepcopy(draft)
    issues = diagnostics(draft, expected)
    assert len(issues) == 1 and issues[0]["path"] == [
        "coverage",
        0,
        "subrequirements",
        0,
    ]
    assert issues[0]["allowed_statuses"] == ["mapped", "deferred", "excluded"]
    assert draft == original


def test_missing_criterion_identity_does_not_erase_supported_coverage():
    from reward_engine.core.outcome.jury_repair import apply_edits, diagnostics

    context, evidence, _, _ = setup()
    item = criterion()
    expected = expectations([item], "a", "before_products")
    draft = {"criteria": [deepcopy(item)], "coverage": [decision(expected[0])]}
    draft["criteria"][0]["criterion_id"] = draft["criteria"][0].pop("id")
    before = deepcopy(draft)
    issues = diagnostics(draft, expected)
    assert any(
        row["path"] == ["criteria", 0] and row.get("invalid_required_fields") == ["id"]
        for row in issues
    )
    assert not any("invalid_target_indices" in row for row in issues)
    assert draft == before
    repaired = apply_edits(
        draft,
        {
            "edits": [
                {
                    "op": "replace",
                    "path": ["criteria", 0],
                    "value": item,
                    "reason": "Restore the required identity field without changing coverage",
                }
            ]
        },
    )
    jury.validate(repaired, evidence, context)
    assert repaired["coverage"] == before["coverage"]
    assert diagnostics(repaired, expected) == []


@pytest.mark.parametrize(
    "field,value", [("id", ""), ("standard", None), ("measurement_question", 3)]
)
def test_missing_criterion_fields_offer_an_existing_parent_repair_path(field, value):
    from reward_engine.core.outcome.jury_repair import diagnostics

    item = {**criterion(), field: value}
    issues = diagnostics({"criteria": [item]}, [])
    assert any(
        row["path"] == ["criteria", 0]
        and field in row.get("invalid_required_fields", [])
        for row in issues
    )


def test_partial_criterion_id_set_defers_reference_diagnostics_until_identities_are_complete():
    from reward_engine.core.outcome.jury_repair import diagnostics

    item = criterion()
    expected = expectations([item], "a", "before_products")
    other = {**criterion(), "criterion_id": "other"}
    other.pop("id")
    mapped = decision(expected[0])
    for part in mapped["subrequirements"]:
        part["criterion_ids"] = ["other"]
    draft = {"criteria": [item, other], "coverage": [mapped]}
    assert not any(
        "invalid_target_indices" in row for row in diagnostics(draft, expected)
    )
    other["id"] = other.pop("criterion_id")
    assert not any(
        "invalid_target_indices" in row for row in diagnostics(draft, expected)
    )
    mapped["subrequirements"][0]["criterion_ids"] = ["unregistered"]
    assert any("invalid_target_indices" in row for row in diagnostics(draft, expected))
