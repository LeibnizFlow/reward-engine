"""Exact WB v8 misplaced envelope, replayed offline through actual bounded Role.

This optional protocol relocates a model-selected existing object, never chooses
its verdict. Invalid nested assessments and ambiguous root siblings deliberately
have no relocation offer and fail closed for the same-evidence repair. Supporting
nested field correction is outside this bounded change. Synthetic preview data
exercises transport/validation, not benchmark truth or repair of the frozen run.
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
    Path(__file__).parents[1] / "fixtures/outcome/workbuddy_v8_wrapped_measurement.json"
)
CAPTURE = json.loads(FIXTURE.read_text())
WRAPPED = CAPTURE["response_validation"]["attempts"][0]["response"]
UNWRAP = {"response_unwrap": "response"}


def relocated():
    return {
        **deepcopy(WRAPPED["response"]),
        "inquiry_result": deepcopy(WRAPPED["inquiry_result"]),
    }


def evidence_case(*, requested=False):
    context, evidence = build_context(
        {
            "task_id": "offline-wb-v8-wrapper",
            "prompt": "Audit examples and output coverage",
        },
        [
            {
                "artifact": {
                    "final_response": "Offline actor summary",
                    "changed_files": "Offline file list",
                },
                "source_documents": [
                    {"text": "Offline source"},
                    {"text": "Offline source"},
                    {"text": "Full synthetic document " * 100},
                ],
            }
        ],
    )
    if requested:
        context["measurement_inquiries"] = {
            "contract": {"questions": ["Are examples covered?"]}
        }
    return context, evidence


def fresh_assessment():
    fresh = relocated()
    fresh["claims"] = [deepcopy(fresh["claims"][0]), deepcopy(fresh["claims"][4])]
    fresh.update(
        status="pass",
        reason="New full synthetic evidence supports this offline assessment.",
        gap="",
    )
    return fresh


def test_fixture_retains_identical_failed_envelopes_and_source_hashes():
    a, b = CAPTURE["response_validation"]["attempts"]
    assert a["response"] == b["response"] and len(WRAPPED["response"]["claims"]) == 5
    assert "status" not in WRAPPED and WRAPPED["response"]["status"] == "unknown"
    digest = hashlib.sha256(
        json.dumps(
            WRAPPED, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    assert CAPTURE["provenance"]["response_sha256"] == [digest, digest]
    assert all("typed status" in attempt["errors"]["global"] for attempt in (a, b))


@pytest.mark.asyncio
@pytest.mark.parametrize("syntax_retry", [False, True])
@pytest.mark.parametrize("full_reply", [False, True])
@pytest.mark.parametrize("requested", [False, True])
async def test_explicit_unwrap_or_faithful_full_reply_then_mandatory_expansion(
    transport, syntax_retry, full_reply, requested
):
    correction = relocated() if full_reply else UNWRAP
    replies = (
        [WRAPPED]
        + (['{"response_unwrap":'] if syntax_retry else [])
        + [correction, fresh_assessment()]
    )
    result, requests = await replay(
        transport, replies, *evidence_case(requested=requested)
    )
    for request in requests[1:-1]:
        offer = request["response_unwrap_reply"]
        assert offer["schema"] == "measurement-response-unwrap/v1"
        assert offer["base_attempt"] == 1 and offer["reply"] == UNWRAP
        assert "No replacements/defaults/tools/reads" in offer["instruction"]
        assert request["response_recovery"]["instruction"].startswith(
            "Preferred envelope repair"
        )
        assert request["measurement_budget"]["remaining_tool_calls"] == 0
        assert "requested_evidence" not in request
        assert "field_correction_reply" not in request
    details = requests[1]["response_recovery"]["details"]
    issue = next(i for i in details["issues"] if i["field"] == "response")
    assert issue["missing_root_fields"] == [
        "claims",
        "status",
        "reason",
        "gap",
        "evidence_refs",
    ]
    assert issue["actual_root_fields"] == ["inquiry_result", "response"]
    assert issue["nested_assessment_valid"] is True
    assert requests[1]["response_recovery"]["rejected_response"] == WRAPPED
    # Unknown remains provisional. Reading the actual cited preview, not merely
    # repairing an envelope or surviving syntax recovery, unlocks reassessment.
    final = requests[-1]
    assert (
        final["evidence_followup"]
        and "product_1:source:2" in final["requested_evidence"]
    )
    assert "response_recovery" not in final and "response_unwrap_reply" not in final
    assert len(final["provisional_check"]["claims"]) == 5
    assert final["measurement_budget"]["remaining_tool_calls"] == 4
    assert result["status"] == "pass" and not result.get("error")
    assert len(result["claims"]) == 2  # Genuine new evidence, not a same-evidence edit.
    attempts = result["response_validation"]["attempts"]
    assert attempts[0]["response"] == WRAPPED and attempts[0]["errors"]
    corrected = attempts[-2]
    assert corrected["response"] == correction and corrected["pending_evidence"][
        "global"
    ] == ["product_1:source:2"]
    if not full_reply:
        assert corrected["response_kind"] == "response_unwrap"
        assert corrected["composed_response"] == relocated()
        provenance = corrected["field_provenance"]
        assert (
            provenance["base_attempt"] == provenance["claim_order_source_attempt"] == 1
        )
        assert provenance["operation"] == {
            "source_attempt": len(replies) - 1,
            "source_field": "response_unwrap",
            "value": "response",
        }
        assert provenance["fields"]["claims[1].judgment"] == {
            "source_attempt": 1,
            "source_field": "response.claims[1].judgment",
            "kind": "relocated",
        }
        assert provenance["fields"]["inquiry_result"] == {
            "source_attempt": 1,
            "source_field": "inquiry_result",
            "kind": "preserved",
        }
        assert all(f["source_attempt"] == 1 for f in provenance["fields"].values())
    assert result["response_validation"]["selected_attempts"]["global"] == len(replies)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [
        "judgment",
        "relevance",
        "count",
        "order",
        "status",
        "reason",
        "citation",
        "inquiry",
        "extra",
        "operation_sibling",
        "operation_path",
        "repeat_wrapper",
    ],
)
async def test_unwrap_repair_cannot_reinterpret_or_edit_nested_assessment(
    transport, change
):
    second = relocated()
    if change == "judgment":
        second["claims"][1]["judgment"] = "supported"
    elif change == "relevance":
        second["claims"][1]["verdict_relevance"] = "advisory"
    elif change == "count":
        second["claims"].pop()
    elif change == "order":
        second["claims"].reverse()
    elif change == "status":
        second["status"] = "fail"
    elif change == "reason":
        second["reason"] = "Rewritten summary"
    elif change == "citation":
        second["claims"][1]["evidence_refs"] = ["task"]
    elif change == "inquiry":
        second["inquiry_result"]["reason"] = "Changed inquiry"
    elif change == "extra":
        second["unrequested"] = True
    elif change == "operation_sibling":
        second = {**UNWRAP, "status": "pass"}
    elif change == "operation_path":
        second = {"response_unwrap": "response.claims"}
    elif change == "repeat_wrapper":
        second = WRAPPED
    result, requests = await replay(transport, [WRAPPED, second], *evidence_case())
    assert "response_unwrap_reply" in requests[1]
    assert "requested_evidence" not in requests[1]
    assert result["error"] and result["status"] == "unknown"
    assert result["response_validation"]["selected_attempts"] == {}
    assert result["response_validation"]["attempts"][-1]["response"] == second
    assert result["response_validation"]["attempts"][-1]["errors"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind", ["mixed_status", "mixed_claims", "extra_wrapper", "invalid_nested"]
)
async def test_ambiguous_or_invalid_nested_envelope_is_intentionally_not_relocated(
    transport, kind
):
    first = deepcopy(WRAPPED)
    if kind == "mixed_status":
        first["status"] = "pass"
    elif kind == "mixed_claims":
        first["claims"] = deepcopy(first["response"]["claims"])
    elif kind == "extra_wrapper":
        first["alternative"] = deepcopy(first["response"])
    else:
        del first["response"]["claims"][1]["judgment"]
    # Even the otherwise valid model-authored flat reply must not silently
    # choose which ambiguous assessment is authoritative or repair nested facts.
    result, requests = await replay(transport, [first, relocated()], *evidence_case())
    assert "response_unwrap_reply" not in requests[1]
    assert result["error"] and result["response_validation"]["selected_attempts"] == {}
    assert "ambiguous source selection" in result["diagnostic"]


@pytest.mark.asyncio
async def test_unrequested_unwrap_operation_is_not_an_implicit_host_default(transport):
    result, requests = await replay(transport, [UNWRAP, UNWRAP], *evidence_case())
    assert all("response_unwrap_reply" not in r for r in requests)
    assert result["error"] and result["response_validation"]["selected_attempts"] == {}
    assert all(
        "composed_response" not in a for a in result["response_validation"]["attempts"]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("request_kind", ["need_evidence", "need_tools"])
async def test_syntax_retry_does_not_reopen_reads_or_tools_after_envelope_rejection(
    transport, request_kind
):
    hostile = {
        request_kind: ["product_1:source:2"]
        if request_kind == "need_evidence"
        else [{"tool": "calculate", "arguments": {"expression": "1+1"}}]
    }
    result, requests = await replay(
        transport, [WRAPPED, '{"response_unwrap":', hostile], *evidence_case()
    )
    assert requests[2]["measurement_budget"]["remaining_tool_calls"] == 0
    assert requests[2]["response_unwrap_reply"] == requests[1]["response_unwrap_reply"]
    assert "requested_evidence" not in requests[2]
    assert result["error"] and result["response_validation"]["selected_attempts"] == {}
    assert "Structure repair must use existing evidence" in result["diagnostic"]
