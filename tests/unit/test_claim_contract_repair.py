"""Synthetic contract-validation cases using offline role transports.

No native grading or live provider calls are performed.
"""

import json
from copy import deepcopy

import pytest

from reward_engine.core.outcome.context import ResponseContractError, build_context
from reward_engine.core.outcome.harness import validate_claims
from tests.support.contract_responses import (
    CLAIM_CONTRACT_REPAIR_CLAW_CASE as CLAW_CASE,
)
from tests.support.role_transport import replay
from tests.support.role_transport import transport as transport
from tests.unit.test_claim_status_diagnostics import CASE as WB_CASE
from tests.unit.test_claim_status_diagnostics import evidence_case as wb_evidence


def claw_evidence():
    return build_context(
        {"task_id": "offline-claw", "prompt": "Inspect the recorded report"},
        [
            {
                "artifact": {"final_response": "Synthetic structural replay"},
                "source_documents": [
                    {"text": f"Offline source {i}"} for i in range(11)
                ],
            }
        ],
    )


def test_exact_twelve_missing_refs_and_pass_dependency_are_all_reported():
    _, evidence = claw_evidence()
    raw = deepcopy(CLAW_CASE)
    with pytest.raises(ResponseContractError) as caught:
        validate_claims(raw, evidence)
    issues = caught.value.details["issues"]
    missing = [i for i in issues if i.get("claim_field") == "evidence_refs"]
    assert len(issues) == 13 and len(missing) == 12
    assert {i["field"] for i in missing} == {
        f"claims[{n}].evidence_refs" for n in range(12)
    }
    assert all(i["missing"] and i["actual"] is None for i in missing)
    assert next(i for i in issues if i["field"] == "status")["depends_on"] == {
        "claims[4].judgment": "unverified",
        "claims[4].verdict_relevance": "material",
    }
    assert raw == CLAW_CASE


@pytest.mark.parametrize("bad_refs", [None, "product_1:source:2", {}, ["absent"]])
def test_nonlist_or_unknown_refs_do_not_hide_other_claims_or_status(bad_refs):
    _, evidence = claw_evidence()
    raw = deepcopy(CLAW_CASE)
    raw["claims"][0]["evidence_refs"] = bad_refs
    before = deepcopy(raw)
    with pytest.raises(ResponseContractError) as caught:
        validate_claims(raw, evidence)
    issues = caught.value.details["issues"]
    assert len(issues) == 13
    assert issues[0]["field"] == "claims[0].evidence_refs"
    assert issues[0]["actual"] == bad_refs and not issues[0]["missing"]
    assert any(i["field"] == "status" for i in issues)
    assert raw == before


@pytest.mark.asyncio
@pytest.mark.parametrize("repair", ["all_fields", "refs_only", "identical", "upgrade"])
async def test_all_missing_fields_and_status_share_one_actual_role_repair(
    transport, repair
):
    first, second = deepcopy(CLAW_CASE), deepcopy(CLAW_CASE)
    if repair != "identical":
        # Authored by the scripted model, not inferred/filled by the harness.
        for claim in second["claims"]:
            claim["evidence_refs"] = (
                []
                if claim["judgment"] == "unverified"
                else ["product_1:artifact:final_response"]
            )
    if repair == "all_fields":
        second.update(
            status="unknown",
            reason="The material TXN-704 assessment remains unverified.",
            gap="The transaction record has not been observed.",
        )
    elif repair == "upgrade":
        second["claims"][4].update(
            judgment="supported", evidence_refs=["product_1:artifact:final_response"]
        )
    result, requests = await replay(transport, [first, second], *claw_evidence())
    recovery = requests[1]["response_recovery"]
    assert recovery["rejected_response"] == first
    assert len(recovery["details"]["issues"]) == 13
    assert "claims[4].judgment" in json.dumps(recovery["details"]["issues"])
    assert "minimally correct status/reason/gap" in recovery["instruction"]
    assert requests[0]["context"] == requests[1]["context"]
    assert requests[1]["measurement_budget"]["remaining_tool_calls"] == 0
    attempts = result["response_validation"]["attempts"]
    assert [a["response"] for a in attempts] == [first, second]
    assert result["status"] == "unknown"
    if repair == "all_fields":
        assert not result.get("error")
        assert result["claims"] == second["claims"]
        assert result["reason"] == second["reason"]
        assert [
            {k: v for k, v in c.items() if k != "evidence_refs"}
            for c in result["claims"]
        ] == first["claims"]
        assert result["response_validation"]["selected_attempts"] == {"global": 2}
    else:
        assert result["error"] and attempts[1]["errors"]["global"]
        assert result["response_validation"]["selected_attempts"] == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [
        "upgrade",
        "downgrade",
        "delete",
        "refs",
        "reorder",
        "observation",
        "conservative",
    ],
)
@pytest.mark.parametrize("syntax_retry", [False, True])
async def test_cs1_actual_role_preserves_assessment_across_same_evidence_and_syntax(
    transport, change, syntax_retry
):
    first, second = deepcopy(WB_CASE), deepcopy(WB_CASE)
    if change == "upgrade":
        second["claims"][2]["judgment"] = "supported"
    elif change == "downgrade":
        second["claims"][2].update(
            verdict_relevance="advisory",
            relevance_reason="Reclassified outside the standard",
        )
    elif change == "delete":
        del second["claims"][2]
    elif change == "reorder":
        second["claims"][0], second["claims"][1] = (
            second["claims"][1],
            second["claims"][0],
        )
    else:
        second.update(
            status="unknown", reason="The material comparison remains unverified."
        )
        if change == "refs":
            second["claims"][2]["evidence_refs"] = ["product_1:artifact:final_response"]
        elif change == "observation":
            second["claims"][2]["observation"] = "A different assessment"
    replies = [first, '{"status":', second] if syntax_retry else [first, second]
    result, requests = await replay(transport, replies, *wb_evidence())
    assert all(r["context"] == requests[0]["context"] for r in requests)
    assert all(
        r["measurement_budget"]["remaining_tool_calls"] == 0 for r in requests[1:]
    )
    attempts = result["response_validation"]["attempts"]
    assert attempts[0]["response"] == first and attempts[-1]["response"] == second
    if syntax_retry:
        assert attempts[1]["raw_response"] == replies[1]
        assert requests[-1]["response_recovery"]["error_type"] == "JSONDecodeError"
    assert result["status"] == "unknown"
    if change == "conservative":
        assert not result.get("error") and result["claims"] == first["claims"]
    else:
        assert result["error"]
        assert "preserve" in attempts[-1]["errors"]["global"]
        assert result["response_validation"]["selected_attempts"] == {}


def preview_case():
    context, evidence = build_context(
        {"task_id": "offline-preview", "prompt": "Check the delivered fact"},
        [
            {
                "artifact": {"final_response": "Claim needing verification"},
                "source_documents": [
                    {"text": "x" * 1000 + " Newly visible confirming observation."}
                ],
            }
        ],
    )
    ref = "product_1:source:0"
    raw = {
        "claims": [
            {
                "claim": "The delivered fact holds",
                "claim_refs": ["product_1:artifact:final_response"],
                "evidence_refs": [ref],
                "observation": "The preview does not establish this fact",
                "judgment": "unverified",
                "verdict_relevance": "material",
            }
        ],
        "status": "pass",
        "reason": "Internally inconsistent summary",
        "gap": "",
        "evidence_refs": [ref],
    }
    return context, evidence, ref, raw


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_refs", [False, True])
async def test_new_preview_evidence_after_consistent_repair_permits_fresh_assessment(
    transport,
    missing_refs,
):
    context, evidence, ref, first = preview_case()
    second = {
        **deepcopy(first),
        "status": "unknown",
        "reason": "The material fact is unverified.",
    }
    third = deepcopy(first)
    third.update(reason="The newly supplied record establishes the fact.")
    third["claims"][0].update(
        judgment="supported", observation="The complete supplied record confirms it."
    )
    if missing_refs:
        del first["claims"][0]["evidence_refs"]
    result, requests = await replay(
        transport, [first, second, third], context, evidence
    )
    assert requests[1]["response_recovery"]["rejected_response"] == first
    assert ref not in requests[1].get("requested_evidence", {})
    assert requests[2]["requested_evidence"][ref] == evidence[ref]
    assert requests[2]["evidence_followup"] and not requests[2].get("response_recovery")
    assert result["status"] == "pass" and not result.get("error")
    assert result["claims"] == third["claims"]
    assert result["response_validation"]["selected_attempts"] == {"global": 3}


@pytest.mark.asyncio
async def test_all_advisory_pass_can_repair_summary_without_changing_claims(transport):
    first = deepcopy(WB_CASE)
    for claim in first["claims"]:
        claim.update(
            verdict_relevance="advisory", relevance_reason="An ancillary observation"
        )
    second = {
        **deepcopy(first),
        "status": "unknown",
        "reason": "No material outcome was established.",
    }
    result, requests = await replay(transport, [first, second], *wb_evidence())
    assert any(
        i["field"] == "status"
        for i in requests[1]["response_recovery"]["details"]["issues"]
    )
    assert result["status"] == "unknown" and not result.get("error")
    assert result["claims"] == first["claims"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "read_kind", ["already_inline", "duplicate", "new_during_repair"]
)
async def test_read_request_cannot_unlock_same_evidence_repair(transport, read_kind):
    context, evidence, ref, first = preview_case()
    if read_kind == "already_inline":
        context, evidence = wb_evidence()
        first = deepcopy(WB_CASE)
        ref = "product_1:source:2"
    replies = ([{"need_evidence": [ref]}] if read_kind == "duplicate" else []) + [
        first,
        {"need_evidence": [ref]},
    ]
    result, requests = await replay(transport, replies, context, evidence)
    assert result["status"] == "unknown" and result["error"]
    assert requests[-1]["measurement_budget"]["remaining_tool_calls"] == 0
    if read_kind == "duplicate":
        assert requests[-1]["requested_evidence"][ref] == evidence[ref]
    attempts = result["response_validation"]["attempts"]
    assert attempts[-1]["response"] == replies[-1]
    assert "existing evidence" in attempts[-1]["errors"]["repair"]
    assert result["response_validation"]["selected_attempts"] == {}
