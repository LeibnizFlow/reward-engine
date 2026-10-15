"""Synthetic contract-validation cases using offline role transports.

No native grading or live provider calls are performed.
"""

from copy import deepcopy

import pytest

from reward_engine.core.outcome.context import build_context
from tests.support.contract_responses import (
    MEASUREMENT_POST_PREVIEW_CONSISTENCY_REPLIES as REPLIES,
)
from tests.support.role_transport import replay
from tests.support.role_transport import transport as transport

# Synthetic replies deliberately exercise summary and claim consistency.

REMINDER = "After assessing the newly supplied evidence"


def evidence_case(*, inquiry=False):
    context, evidence = build_context(
        {"task_id": "offline-wbv9-status", "prompt": "Contract replay only"},
        [
            {
                "artifact": {"final_response": "Synthetic contract replay"},
                "source_documents": [
                    {
                        "text": "Synthetic preview data " * 180
                        if i == 2
                        else f"Offline source {i}"
                    }
                    for i in range(3)
                ],
            }
        ],
    )
    evidence["tool:language_format_usability_check:0"] = {
        "text": "Synthetic recorded observation; no tool executes"
    }
    if inquiry:
        context["measurement_inquiries"] = {
            "contract": {"questions": ["Separate local question"]}
        }
    return context, evidence


def corrected():
    return {**deepcopy(REPLIES[0]), **deepcopy(REPLIES[1]["field_corrections"])}


def assert_reminder(requests):
    assert all(REMINDER not in r.get("evidence_followup", "") for r in requests[:-1])
    final = requests[-1]
    text = final["evidence_followup"]
    for fragment in (
        REMINDER,
        "Omitted verdict_relevance means material",
        "calling a claim nonmaterial only in prose does not change",
        "model-authored relevance_reason tied to THIS criterion",
        "Do not change relevance just to obtain pass",
        "New evidence may legitimately change judgment or relevance when supported",
        "prior provisional verdict is not fixed",
        "global verdict and any inquiry_result independent",
    ):
        assert fragment in text
    assert set(final["requested_evidence"]) == {"product_1:source:2"}
    assert "response_recovery" not in final and "field_correction_reply" not in final
    assert final["measurement_budget"]["remaining_tool_calls"] == 4
    return final


@pytest.mark.asyncio
@pytest.mark.parametrize("conservative", [False, True])
@pytest.mark.parametrize("syntax_retry", [False, True])
async def test_synthetic_failure_and_conservative_control(
    transport, conservative, syntax_retry
):
    replies = deepcopy(REPLIES)
    if conservative:
        replies[-1] = (
            corrected()
        )  # Scripted reuse of an authored unknown, not host choice.
    if syntax_retry:
        replies.insert(1, '{"field_corrections":')
    result, requests = await replay(transport, replies, *evidence_case())
    final = assert_reminder(requests)
    assert final["provisional_check"] == corrected()
    for request in requests[1:-1]:
        assert request["measurement_budget"]["remaining_tool_calls"] == 0
    attempts = result["response_validation"]["attempts"]
    assert attempts[-2]["errors"] == {}
    assert attempts[-2]["composed_response"] == corrected()
    assert attempts[-2]["response"] == REPLIES[1]
    assert attempts[-2]["pending_evidence"] == {"global": ["product_1:source:2"]}
    assert attempts[-1]["response"] == replies[-1]
    assert result["status"] == "unknown"
    if conservative:
        assert not result.get("error") and attempts[-1]["errors"] == {}
    else:
        assert result["error"]
        assert "claims[3].judgment" in attempts[-1]["errors"]["global"]
    # replay verifies exact physical call count and raw receipts: no fourth
    # contract attempt (syntax control adds only its existing syntax allowance).


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault", ["advisory_without_reason", "prose_only", "material_contradicted"]
)
async def test_reminder_does_not_accept_inconsistent_fresh_assessments(
    transport, fault
):
    replies = deepcopy(REPLIES)
    claim = replies[-1]["claims"][3]
    if fault == "advisory_without_reason":
        claim["verdict_relevance"] = "advisory"
    elif fault == "prose_only":
        claim["relevance_reason"] = (
            "Scripted prose says nonmaterial but field is omitted."
        )
    else:
        claim["judgment"] = "contradicted"
    result, requests = await replay(transport, replies, *evidence_case())
    assert_reminder(requests)
    assert result["error"] and result["response_validation"]["attempts"][-1]["errors"]


@pytest.mark.asyncio
@pytest.mark.parametrize("fresh_change", ["advisory", "supported"])
@pytest.mark.parametrize("local_status", [None, "unknown", "fail"])
async def test_scripted_new_evidence_allows_fresh_assessment_and_independent_inquiry(
    transport, fresh_change, local_status
):
    first = corrected()
    final = deepcopy(REPLIES[2])
    claim = final["claims"][3]
    if fresh_change == "advisory":
        claim["verdict_relevance"] = "advisory"
        claim["relevance_reason"] = (
            "Scripted simulation: the newly read full document establishes the required "
            "format and trace; independent runtime reproduction is outside this criterion."
        )
    else:
        claim["judgment"] = "supported"
        claim["observation"] = (
            "Scripted simulation: newly expanded evidence supports this claim."
        )
        final["reason"] = "Scripted simulation: all material claims are now supported."
    if local_status:
        local = {
            "status": local_status,
            "reason": "Separate scripted inquiry assessment",
            "gap": "Separate local issue",
            "evidence_refs": ["product_1:artifact:final_response"],
        }
        first["inquiry_result"] = deepcopy(local)
        final["inquiry_result"] = deepcopy(local)
    result, requests = await replay(
        transport, [first, final], *evidence_case(inquiry=bool(local_status))
    )
    assert_reminder(requests)
    assert requests[-1]["provisional_check"]["status"] == "unknown"
    assert result["status"] == "pass" and not result.get("error")
    if local_status:
        assert result["inquiry_result"] == local
    assert result["claims"][3] == claim


@pytest.mark.asyncio
@pytest.mark.parametrize("syntax_retry", [False, True])
async def test_same_evidence_relabel_still_rejected_without_fresh_reminder(
    transport, syntax_retry
):
    repair = deepcopy(REPLIES[2])
    repair["claims"][3].update(
        verdict_relevance="advisory", relevance_reason="Force scripted pass"
    )
    replies = [REPLIES[0]] + (['{"claims":'] if syntax_retry else []) + [repair]
    result, requests = await replay(transport, replies, *evidence_case())
    assert all(REMINDER not in r.get("evidence_followup", "") for r in requests)
    assert result["error"]
    assert (
        "preserve"
        in result["response_validation"]["attempts"][-1]["errors"]["global"].lower()
    )
    assert all(
        r["measurement_budget"]["remaining_tool_calls"] == 0 for r in requests[1:]
    )
