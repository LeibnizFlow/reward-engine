"""Offline actual-Role repair of the retained Claw v10 judgment omission.

Captured responses are exact; source evidence in these tests is synthetic. This
establishes contract behavior, not semantic correctness or repair of that run.
The read-only older harness has the enum check but no field-correction protocol;
its inspected hash and the original native receipt hashes are in the fixture.
"""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from reward_engine.core.outcome.context import build_context
from tests.unit.test_claim_contract_repair import replay
from tests.unit.test_workbuddy_stream_terminal_replay import transport as transport

FIXTURE = (
    Path(__file__).parents[1] / "fixtures/outcome/claw_v10_missing_claim_judgment.json"
)
CAPTURE = json.loads(FIXTURE.read_text())
FIRST, EXPANDED, HISTORICAL_REPAIR = [
    a["response"] for a in CAPTURE["response_validation"]["attempts"]
]
FIELD = "claims[1].judgment"


def evidence_case(*, further_preview=False):
    return build_context(
        {
            "task_id": "offline-claw-v10-judgment",
            "prompt": "Inspect the recorded methodology",
        },
        [
            {
                "artifact": {
                    "final_response": "Synthetic methodology structural replay"
                },
                "source_documents": [
                    {
                        "text": (
                            "Preview data " * 180
                            if i in ({2, 3, 8} if further_preview else {2, 3})
                            else f"Offline source {i}"
                        )
                    }
                    for i in range(9)
                ],
            }
        ],
    )


def patch(value="unverified"):
    return {"field_corrections": {FIELD: value}}


def test_capture_integrity_and_exact_original_failure():
    provenance = CAPTURE["provenance"]

    def digest(value):
        return hashlib.sha256(
            json.dumps(
                value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()

    assert [digest(r) for r in (FIRST, EXPANDED, HISTORICAL_REPAIR)] == provenance[
        "response_sha256"
    ]
    assert [len(r["claims"]) for r in (FIRST, EXPANDED, HISTORICAL_REPAIR)] == [5, 2, 5]
    assert "judgment" not in EXPANDED["claims"][1]
    assert EXPANDED["claims"][1]["observation"].endswith("that invoi")
    attempts = CAPTURE["response_validation"]["attempts"]
    assert attempts[0]["errors"] == {}
    assert set(attempts[0]["pending_evidence"]["global"]) == {
        "product_1:source:2",
        "product_1:source:3",
    }
    assert attempts[1]["issues"] == [
        {
            "scope": "global",
            "field": FIELD,
            "reason": "Invalid claim judgment",
            "claim_index": 1,
            "claim_field": "judgment",
            "actual": None,
            "missing": True,
        }
    ]
    assert attempts[2]["errors"] == {
        "global": "Structure repair must preserve claim count and order"
    }
    expanded, repair = provenance["provider_receipts"][1:]
    assert expanded["finish_reason"] == "stop" and expanded["completion_tokens"] == 571
    assert len(repair["request_diagnostics"]["provisional_check"]["claims"]) == 5
    assert (
        len(
            repair["request_diagnostics"]["response_recovery"]["rejected_response"][
                "claims"
            ]
        )
        == 2
    )
    assert "reassess" in repair["request_diagnostics"]["evidence_followup"]


@pytest.mark.asyncio
@pytest.mark.parametrize("syntax_retry", [False, True])
@pytest.mark.parametrize("judgment", ["supported", "contradicted", "unverified"])
async def test_actual_role_expansion_then_one_explicit_judgment_correction(
    transport, syntax_retry, judgment
):
    correction = patch(judgment)
    replies = (
        [FIRST, EXPANDED]
        + (['{"field_corrections":'] if syntax_retry else [])
        + [correction]
    )
    result, requests = await replay(transport, replies, *evidence_case())
    assert (
        requests[1]["evidence_followup"]
        and len(requests[1]["provisional_check"]["claims"]) == 5
    )
    assert set(requests[1]["requested_evidence"]) == {
        "product_1:source:2",
        "product_1:source:3",
    }
    for request in requests[2:]:
        offer = request["field_correction_reply"]
        assert offer["allowed_fields"] == offer["required_fields"] == [FIELD]
        assert offer["allowed_values"] == {
            FIELD: ["supported", "contradicted", "unverified"]
        }
        assert offer["base_attempt"] == 2
        assert "no defaults" in offer["instruction"]
        assert "Do not rewrite" in offer["instruction"]
        assert request["response_recovery"]["instruction"].startswith(
            "Preferred repair reply"
        )
        assert "evidence_followup" not in request and "provisional_check" not in request
        assert request["requested_evidence"] == requests[1]["requested_evidence"]
        assert request["context"] == requests[1]["context"]
        assert request["measurement_budget"]["remaining_tool_calls"] == 0
    assert requests[2]["response_recovery"]["rejected_response"] == EXPANDED
    assert result["status"] == "fail" and not result.get("error")
    assert result["response_validation"]["selected_attempts"] == {
        "global": len(replies)
    }
    attempts = result["response_validation"]["attempts"]
    assert attempts[0]["response"] == FIRST and attempts[0]["pending_evidence"]
    assert attempts[1]["response"] == EXPANDED and attempts[1]["errors"]
    composed = attempts[-1]["composed_response"]
    expected = deepcopy(EXPANDED)
    expected["claims"][1]["judgment"] = judgment
    assert composed == expected and result["claims"] == expected["claims"]
    assert attempts[-1]["response"] == correction and not attempts[-1]["errors"]
    provenance = attempts[-1]["field_provenance"]
    assert provenance["base_attempt"] == provenance["claim_order_source_attempt"] == 2
    assert provenance["fields"][FIELD] == {
        "source_attempt": len(replies),
        "source_field": f"field_corrections.{FIELD}",
        "kind": "replacement",
    }
    assert all(
        v["source_attempt"] == 2 and v["kind"] == "preserved"
        for k, v in provenance["fields"].items()
        if k != FIELD
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", [None, "", "uncertain", [], {}])
async def test_invalid_existing_judgment_can_be_corrected_but_not_other_fields(
    transport, invalid
):
    expanded = deepcopy(EXPANDED)
    expanded["claims"][1]["judgment"] = invalid
    result, requests = await replay(
        transport, [FIRST, expanded, patch()], *evidence_case()
    )
    assert requests[2]["field_correction_reply"]["required_fields"] == [FIELD]
    assert not result.get("error") and result["claims"][1]["judgment"] == "unverified"
    assert result["response_validation"]["attempts"][1]["response"] == expanded


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad",
    [
        "valid_judgment",
        "claim_count",
        "citation",
        "sibling",
        "bad_path",
        "missing_field",
        "null",
        "enum",
        "full_valid_judgment",
        "full_citation",
        "full_order",
        "historical_full_reply",
    ],
)
async def test_correction_cannot_edit_valid_fields_or_restore_older_claims(
    transport, bad
):
    correction = patch()
    values = correction["field_corrections"]
    if bad == "valid_judgment":
        values["claims[0].judgment"] = "unverified"
    elif bad == "claim_count":
        values["claims"] = deepcopy(HISTORICAL_REPAIR["claims"])
    elif bad == "citation":
        values["claims[1].evidence_refs"] = ["product_1:source:2"]
    elif bad == "sibling":
        correction["status"] = "fail"
    elif bad == "bad_path":
        values["claims[-1].judgment"] = values.pop(FIELD)
    elif bad == "missing_field":
        values.clear()
    elif bad in ("null", "enum"):
        values[FIELD] = None if bad == "null" else "uncertain"
    elif bad == "historical_full_reply":
        correction = HISTORICAL_REPAIR
    else:
        correction = deepcopy(EXPANDED)
        correction["claims"][1]["judgment"] = "unverified"
        if bad == "full_valid_judgment":
            correction["claims"][0]["judgment"] = "unverified"
        elif bad == "full_citation":
            correction["claims"][1]["evidence_refs"] = ["product_1:source:2"]
        elif bad == "full_order":
            correction["claims"].reverse()
    result, requests = await replay(
        transport, [FIRST, EXPANDED, correction], *evidence_case()
    )
    assert requests[2]["field_correction_reply"]["required_fields"] == [FIELD]
    assert result["status"] == "unknown" and result["error"]
    assert result["response_validation"]["selected_attempts"] == {}
    assert result["response_validation"]["attempts"][-1]["response"] == correction
    assert result["response_validation"]["attempts"][-1]["errors"]
    if bad == "historical_full_reply":
        assert "preserve claim count" in result["diagnostic"]


@pytest.mark.asyncio
async def test_full_reply_compatibility_preserves_every_other_original_field(transport):
    corrected = deepcopy(EXPANDED)
    corrected["claims"][1]["judgment"] = "unverified"
    result, _ = await replay(transport, [FIRST, EXPANDED, corrected], *evidence_case())
    assert result["status"] == "fail" and not result.get("error")
    assert result["claims"] == corrected["claims"]


@pytest.mark.asyncio
@pytest.mark.parametrize("dependency", ["contradictory_pass", "missing_support"])
async def test_corrected_judgment_still_requires_existing_status_and_evidence_rules(
    transport, dependency
):
    expanded = deepcopy(EXPANDED)
    if dependency == "contradictory_pass":
        expanded["status"] = "pass"
        correction = patch("unverified")
    else:
        expanded["claims"][1]["evidence_refs"] = []
        correction = patch("supported")
    result, requests = await replay(
        transport, [FIRST, expanded, correction], *evidence_case()
    )
    assert requests[2]["field_correction_reply"]["required_fields"] == [FIELD]
    assert result["error"] and result["response_validation"]["selected_attempts"] == {}
    attempted = result["response_validation"]["attempts"][-1]
    assert (
        attempted["composed_response"]["claims"][1]["judgment"]
        == correction["field_corrections"][FIELD]
    )
    assert attempted["composed_response"]["status"] == expanded["status"]
    assert (
        "silently become pass" in str(attempted["errors"])
        if dependency == "contradictory_pass"
        else "needs observed evidence" in str(attempted["errors"])
    )


@pytest.mark.asyncio
async def test_simultaneous_existing_status_error_keeps_all_enum_constraints(transport):
    expanded = deepcopy(EXPANDED)
    expanded["status"] = "pass"
    expanded["claims"][0]["judgment"] = "unverified"
    correction = patch()
    correction["field_corrections"].update(
        status="unknown",
        reason="The preserved material assessment is unverified.",
        gap="",
    )
    result, requests = await replay(
        transport, [FIRST, expanded, correction], *evidence_case()
    )
    assert requests[2]["field_correction_reply"]["allowed_values"] == {
        FIELD: ["supported", "contradicted", "unverified"],
        "status": ["fail", "unknown"],
    }
    assert result["status"] == "unknown" and not result.get("error")
    assert result["claims"][0] == expanded["claims"][0]


@pytest.mark.asyncio
async def test_no_offer_does_not_invent_original_response_or_judgment(transport):
    result, requests = await replay(transport, [patch(), patch()], *evidence_case())
    assert all("field_correction_reply" not in request for request in requests)
    assert result["error"] and result["response_validation"]["selected_attempts"] == {}
    assert all(
        "composed_response" not in a for a in result["response_validation"]["attempts"]
    )


@pytest.mark.asyncio
async def test_further_genuine_preview_expansion_resets_repair_after_field_correction(
    transport,
):
    expanded = deepcopy(EXPANDED)
    expanded["claims"][1]["evidence_refs"].append("product_1:source:8")
    expanded["evidence_refs"].append("product_1:source:8")
    fresh = deepcopy(HISTORICAL_REPAIR)
    fresh["claims"] = [deepcopy(FIRST["claims"][0])]
    fresh.update(
        status="pass",
        reason="Newly visible synthetic record changes this offline assessment.",
        gap="",
    )
    result, requests = await replay(
        transport,
        [FIRST, expanded, patch(), fresh],
        *evidence_case(further_preview=True),
    )
    assert (
        "evidence_followup" not in requests[2]
        and "provisional_check" not in requests[2]
    )
    assert requests[3]["evidence_followup"] and "response_recovery" not in requests[3]
    assert "field_correction_reply" not in requests[3]
    assert "product_1:source:8" in requests[3]["requested_evidence"]
    assert len(requests[3]["provisional_check"]["claims"]) == 2
    assert result["status"] == "pass" and not result.get("error")
    assert result["claims"] == fresh["claims"]
    assert result["response_validation"]["selected_attempts"] == {"global": 4}
