"""Captured WB missing-standard failure: precise diagnostics, unchanged admission.

Preserving IDs/count/content is instructed during the existing single retry, not
an enforced cross-response lock. Scripted correction tests do not prove a model
will comply, or that its proposed requirement is true.
"""

import hashlib
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest

from reward_engine.core.outcome import jury
from reward_engine.core.outcome.context import ResponseContractError, ask, build_context
from reward_engine.core.outcome.jury_drafts import keep_draft
from tests.unit.test_outcome_role_contract_recovery import roles as roles

FIXTURE = (
    Path(__file__).parents[1]
    / "fixtures/outcome/workbuddy_v8_expectations_missing_standard.json"
)


def captured():
    return json.loads(FIXTURE.read_text())


def inputs():
    return build_context(captured()["task"], [])


def model_correction():
    """Explicit offline model output: no runtime projection from other fields."""
    value = json.loads(captured()["responses"][0]["raw_output"])
    authored = [
        "Deliver the requested Chinese Markdown manual in the publicly specified task-directory file.",
        "Explain CLI entry points, commands, options and examples using supplied workspace evidence.",
        "Explain the Python API entry points and calling patterns and compare them with CLI usage.",
        "Cross-check documented configuration and output behavior against the supplied source and examples.",
        "Distinguish supported behavior from experimental, obsolete or uncertain behavior.",
        "Respect the public read-only and external-access boundaries while preparing the permitted deliverable.",
        "Cite supplied workspace paths for substantive claims.",
        "Organize the Chinese manual so developers can locate usage, parameters, evidence and caveats.",
    ]
    for row, text in zip(value["criteria"], authored, strict=True):
        row["standard"] = text
    return value


@pytest.mark.parametrize("index", [0, 1], ids=["original", "actual-repeated-omission"])
def test_both_exact_captured_responses_remain_invalid(index):
    source = captured()["responses"][index]
    assert (
        hashlib.sha256(source["raw_output"].encode()).hexdigest()
        == source["raw_output_sha256"]
    )
    raw = json.loads(source["raw_output"])
    before = deepcopy(raw)
    assert len(raw["criteria"]) == 8
    assert all("standard" not in r and "clauses" not in r for r in raw["criteria"])
    context, evidence = inputs()
    with pytest.raises(ResponseContractError) as caught:
        keep_draft(raw, evidence, context, jury.validate)
    details = caught.value.details
    assert details["criteria_count"] == 8
    assert details["criterion_ids"] == [r["id"] for r in raw["criteria"]]
    assert [i["field"] for i in details["issues"]] == [
        f"criteria[{i}].standard" for i in range(8)
    ]
    assert [i["path"] for i in details["issues"]] == [
        ["criteria", i, "standard"] for i in range(8)
    ]
    assert [i["clause_issues"][0]["field"] for i in details["issues"]] == [
        f"criteria[{i}].clauses" for i in range(8)
    ]
    assert (
        "same criterion count, order and existing id"
        in caught.value.recovery_instruction
    )
    assert "Preserve every other existing field" in caught.value.recovery_instruction
    assert raw == before  # Diagnostic construction never inserts standards.


@pytest.mark.parametrize("standard", [None, "", " \t", 7, ["text"], {"text": "text"}])
def test_invalid_standard_and_malformed_clauses_get_exact_paths(standard):
    context, evidence = inputs()
    raw = {
        "criteria": [
            {
                "id": "unchanged",
                "standard": standard,
                "why": "Rationale is not a standard",
                "measurement_question": "Question?",
                "subrequirements": ["Detail is not a standard"],
                "clauses": [{"text": ""}, {"text": 1}, "not an object"],
            }
        ]
    }
    before = deepcopy(raw)
    with pytest.raises(ResponseContractError) as caught:
        keep_draft(raw, evidence, context, jury.validate)
    issue = caught.value.details["issues"][0]
    assert issue["field"] == "criteria[0].standard"
    assert [c["field"] for c in issue["clause_issues"]] == [
        f"criteria[0].clauses[{i}].text" for i in range(3)
    ]
    assert raw == before


def test_admission_semantics_and_valid_content_are_unchanged():
    context, evidence = inputs()
    corrected = model_correction()
    before = deepcopy(corrected)
    draft = keep_draft(corrected, evidence, context, jury.validate)
    assert corrected == before
    assert len(draft["criteria"]) == 8
    for row, source in zip(draft["criteria"], before["criteria"], strict=True):
        assert {k: row[k] for k in source} == source
        assert "draft_field_projections" not in row
    # Captured kind='important' rows still fail publication validation. A new
    # standard makes these drafts substantive, not automatically valid standards.
    assert draft["draft_validation_issues"]
    with pytest.raises(ValueError):
        jury.validate(deepcopy(corrected), evidence, context)
    unfinished = {"id": "unfinished", "why": "Keep this raw proposal"}
    mixed = keep_draft(
        {"criteria": [corrected["criteria"][0], unfinished]},
        evidence,
        context,
        jury.validate,
    )
    assert mixed["unusable_criteria"] == [unfinished]
    assert mixed["criteria"][0]["standard"] == corrected["criteria"][0]["standard"]
    explicit = {
        "criteria": [
            {"id": "clauses", "clauses": [{"text": "Already authored definition"}]}
        ]
    }
    draft = keep_draft(explicit, evidence, context, jury.validate)
    assert draft["criteria"][0]["standard"] == "Already authored definition"
    assert draft["criteria"][0]["draft_field_projections"] == [
        "standard_from_existing_clause_texts"
    ]
    assert "standard" not in explicit["criteria"][0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "corrected",
    [False, True],
    ids=["captured-repeat-still-fails", "authored-standard-draft"],
)
async def test_actual_role_uses_only_existing_single_contract_retry(
    roles, tmp_path, corrected
):
    make, calls = roles
    fixture = captured()
    original = json.loads(fixture["responses"][0]["raw_output"])
    new_text = model_correction()

    def response(request):
        recovery = request.get("response_recovery")
        if not recovery:
            return fixture["responses"][0]["raw_output"]
        assert recovery["attempt"] == 1
        assert recovery["rejected_response"] == original
        assert recovery["details"]["criteria_count"] == len(original["criteria"])
        assert recovery["details"]["criterion_ids"] == [
            r["id"] for r in original["criteria"]
        ]
        assert len(recovery["details"]["issues"]) == 8
        assert "literal standard key" in recovery["instruction"]
        assert "drop criteria, merge rows or change IDs" in recovery["instruction"]
        return new_text if corrected else fixture["responses"][1]["raw_output"]

    role = make("jury_member_1", response)
    context, evidence = inputs()

    async def run():
        return await ask(
            role,
            "outcome_expectations",
            context,
            evidence,
            "Propose Task-grounded draft requirements.",
            {
                "criteria": [
                    {"id": "open string", "standard": "Task outcome requirement"}
                ]
            },
            validate=lambda row: keep_draft(row, evidence, context, jury.validate),
            max_contract_retries=1,
        )

    if corrected:
        result = await run()
        assert [r["standard"] for r in result["criteria"]] == [
            r["standard"] for r in new_text["criteria"]
        ]
        assert [
            {
                k: v
                for k, v in row.items()
                if k not in {"standard", "draft_validation_issues"}
            }
            for row in result["criteria"]
        ] == original["criteria"]
        assert result["draft_validation_issues"]
    else:
        with pytest.raises(ResponseContractError):
            await run()
    assert len(calls) == 2
    assert [q["mode"] for _, q in calls] == ["outcome_expectations"] * 2
    assert calls[0][1]["context"] == calls[1][1]["context"]
    saved = [
        json.loads(p.read_text())
        for p in (tmp_path / "jury_member_1").glob("*/response.json")
    ]
    assert len(saved) == 2
    assert fixture["responses"][0]["raw_output"] in [r["raw_output"] for r in saved]
    if not corrected:
        assert fixture["responses"][1]["raw_output"] in [r["raw_output"] for r in saved]


@pytest.mark.asyncio
async def test_missing_seat_remains_missing_at_three_seat_boundary(roles):
    make, calls = roles
    fixture = captured()

    def missing(request):
        assert request["mode"] == "outcome_expectations"
        return fixture["responses"][int(bool(request.get("response_recovery")))][
            "raw_output"
        ]

    def good(request):
        return {
            "criteria": [
                {
                    "id": "real",
                    "standard": "Explain the supplied CLI",
                    "kind": "hard",
                    "origin": "task",
                    "evidence_refs": ["task"],
                    "measurement_question": "Is the explanation grounded?",
                }
            ]
        }

    async def forbidden_reduce(_):
        raise AssertionError("Missing seat must not reach three-seat Reduce")

    records = {}
    context, evidence = inputs()
    with pytest.raises(
        RuntimeError, match="Fewer than three usable independent expectations"
    ):
        await jury._propose_once(
            context,
            evidence,
            {
                "1": make("missing", missing),
                "2": make("second", good),
                "3": make("third", good),
            },
            forbidden_reduce,
            minimum_members=3,
            record=lambda name, value: records.update({name: deepcopy(value)}),
        )
    assert records["jury_expectations_map"]["total"] == 3
    assert records["jury_expectations_map"]["succeeded"] == 2
    assert records["jury_expectations_map"]["complete"] is False
    assert "expectations/1" not in records and "jury_map" not in records
    assert Counter(name for name, _ in calls) == {"missing": 2, "second": 1, "third": 1}
