"""Synthetic contract-validation cases using offline role transports.

No native grading or live provider calls are performed.
"""

import json
from copy import deepcopy

import httpx
import pytest

from reward_engine.adapters.workbuddy.models import Role
from reward_engine.core.outcome.context import ResponseContractError, ask, build_context
from reward_engine.core.outcome.delivery_triage import validate_review
from reward_engine.core.outcome.improvement import INSTRUCTION, schema, validate
from tests.support.contract_responses import IMPROVEMENT_GAP_ID_DIAGNOSTICS_CASE as CASE
from tests.support.role_transport import encode, event
from tests.support.role_transport import transport as transport

EXACT_IDS = [
    "G1_status_field_omission",
    "G2_category_rule_edge_case",
    "G3_recommendation_specificity",
    "G4_currency_notation",
]


def evidence_case():
    context, evidence = build_context(
        {"task_id": "offline-review", "prompt": "Inspect the delivered report"},
        [
            {
                "artifact": {"final_response": "Offline structural replay"},
                "source_documents": [
                    {"text": f"Synthetic source {i}"} for i in range(13)
                ],
            }
        ],
    )
    evidence["reference:outcome_question:5"] = (
        "Synthetic question for reference validation"
    )
    context["available_evidence_refs"] = list(evidence)
    return context, evidence


def corrected_case():
    row = deepcopy(CASE)
    # Explicit SCRIPTED MODEL replacements, not host inference or alias mapping.
    row["improvements"][0]["gap_ids"] = ["G1_status_field_omission"]
    row["improvements"][1]["gap_ids"] = ["G2_category_rule_edge_case"]
    row["improvements"][2]["gap_ids"] = ["G3_recommendation_specificity"]
    return row


def test_synthetic_case_reports_all_invalid_actions_and_allowed_own_ledger_ids():
    _, evidence = evidence_case()
    raw = deepcopy(CASE)
    with pytest.raises(ResponseContractError) as caught:
        validate(raw, evidence)
    details = caught.value.details
    assert details["allowed_ids"] == EXACT_IDS
    assert details["field"] == "improvements[0].gap_ids"
    assert [i["field"] for i in details["issues"]] == [
        f"improvements[{n}].gap_ids" for n in range(3)
    ]
    assert [i["actual"] for i in details["issues"]] == [["G1"], ["G2"], ["G3"]]
    assert all(
        i["invalid_indices"] == [0] and not i["missing"] for i in details["issues"]
    )
    assert all(g in details["reason"] for g in EXACT_IDS)
    assert "Correct every listed" in caught.value.recovery_instruction
    assert raw == CASE


@pytest.mark.parametrize(
    "bad",
    [
        None,
        [],
        "G1_status_field_omission",
        ["G1"],
        ["g1_status_field_omission"],
        ["G1_status_field_omission "],
        ["G1_status"],
        ["I1_add_cross_check_dimensions"],
        ["product_1:source:4"],
        ["G1_status_field_omission", "G2", 7, {}],
    ],
)
def test_exact_reference_rule_rejects_aliases_and_malformed_values_without_mutation(
    bad,
):
    _, evidence = evidence_case()
    raw = corrected_case()
    raw["improvements"][0]["gap_ids"] = deepcopy(bad)
    before = deepcopy(raw)
    with pytest.raises(ResponseContractError) as caught:
        validate(raw, evidence)
    issue = caught.value.details["issues"][0]
    assert issue["actual"] == bad
    assert issue["invalid_indices"] == (
        [1, 2, 3]
        if isinstance(bad, list) and len(bad) == 4
        else [0]
        if isinstance(bad, list) and bad
        else []
    )
    assert raw == before


def test_missing_references_are_distinguished_and_exact_multiple_ids_still_pass():
    _, evidence = evidence_case()
    row = corrected_case()
    del row["improvements"][1]["gap_ids"]
    with pytest.raises(ResponseContractError) as caught:
        validate(row, evidence)
    assert caught.value.details["issues"] == [
        {
            "field": "improvements[1].gap_ids",
            "actual": None,
            "missing": True,
            "invalid_indices": [],
        }
    ]
    row = corrected_case()
    row["improvements"][0]["gap_ids"] = [
        "G1_status_field_omission",
        "G4_currency_notation",
    ]
    before = deepcopy(row)
    assert validate(row, evidence) == before and row == before


@pytest.mark.asyncio
@pytest.mark.parametrize("repair", ["exact_ids", "identical", "only_first"])
async def test_actual_role_single_repair_receives_all_indices_and_preserves_raw_replies(
    transport, repair
):
    original, responses, calls = transport
    role = Role(
        {**original.config, "role_bounded_stage": True}, "judge", original.output
    )
    first = deepcopy(CASE)
    second = corrected_case() if repair == "exact_ids" else deepcopy(CASE)
    if repair == "only_first":
        second["improvements"][0]["gap_ids"] = ["G1_status_field_omission"]
    responses.extend(
        httpx.Response(
            200,
            text=encode(
                [event({"content": json.dumps(row, ensure_ascii=False)}, "stop")]
            ),
        )
        for row in (first, second)
    )
    context, evidence = evidence_case()
    tests = {item["test_id"] for item in first["verify"]}

    async def run():
        return await ask(
            role,
            "outcome_delivery_review",
            context,
            evidence,
            INSTRUCTION,
            {
                **schema(),
                "status": "review status",
                "reason": "review conclusion",
                "gaps": [],
                "verify": [],
                "claim_checks": [],
            },
            verify_used_previews=True,
            max_evidence_rounds=2,
            close_evidence=True,
            max_contract_retries=1,
            validate=lambda value: validate(
                validate_review(value, evidence, tests), evidence
            ),
        )

    if repair == "exact_ids":
        result = await run()
        assert result["full_score_gaps"] == first["full_score_gaps"]
        assert result["improvements"] == second["improvements"]
        assert result["status"] == first["status"] == "needs_improvement"
        assert result["reason"] == first["reason"]
        assert result["claim_checks"] == first["claim_checks"]
        for old, new in zip(first["improvements"], result["improvements"], strict=True):
            assert {k: v for k, v in old.items() if k != "gap_ids"} == {
                k: v for k, v in new.items() if k != "gap_ids"
            }
    else:
        with pytest.raises(ResponseContractError) as caught:
            await run()
        expected = range(1, 3) if repair == "only_first" else range(3)
        assert [i["field"] for i in caught.value.details["issues"]] == [
            f"improvements[{i}].gap_ids" for i in expected
        ]
    assert len(calls) == 2 and not responses
    requests = [json.loads(call["messages"][1]["content"]) for call in calls]
    recovery = requests[1]["response_recovery"]
    assert recovery["details"]["allowed_ids"] == EXACT_IDS
    assert len(recovery["details"]["issues"]) == 3
    assert recovery["rejected_response"] == first
    assert requests[0]["context"] == requests[1]["context"]
    assert requests[0].get("requested_evidence") == requests[1].get(
        "requested_evidence"
    )
    assert "copy each complete ID string" in requests[0]["instruction"]
    assert (
        "exact complete full_score_gaps[].id"
        in requests[0]["response"]["improvements"][0]["gap_ids"][0]
    )
    assert "Correct every listed improvements[i].gap_ids" in recovery["instruction"]
    assert "Preserve full_score_gaps IDs" in recovery["instruction"]
    assert "host does not infer or map aliases" in recovery["instruction"]
    receipts = [json.loads(p.read_text()) for p in role.output.glob("*/response.json")]
    assert len(receipts) == 2
    assert sorted(r["raw_output"] for r in receipts) == sorted(
        json.dumps(row, ensure_ascii=False) for row in (first, second)
    )
    assert first == CASE
