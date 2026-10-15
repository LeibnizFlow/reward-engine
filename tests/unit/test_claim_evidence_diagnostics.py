"""Synthetic contract-validation cases using offline role transports.

No native grading or live provider calls are performed.
"""

import json
from copy import deepcopy

import httpx
import pytest

from reward_engine.adapters.workbuddy.models import Role
from reward_engine.core.outcome.context import ResponseContractError, build_context
from reward_engine.core.outcome.harness import defaults, validate_claims
from tests.support.contract_responses import CLAIM_EVIDENCE_DIAGNOSTICS_CASE as CASE
from tests.support.role_transport import encode, event
from tests.support.role_transport import transport as transport


def evidence_case():
    context, evidence = build_context(
        {
            "task_id": "offline-capture",
            "prompt": "Inspect traceability of the delivered report",
        },
        [
            {
                "artifact": {
                    "final_response": "Offline structural replay, not native task evidence"
                },
                "source_documents": [
                    {"text": f"Offline source handle {i}"} for i in range(8)
                ],
            }
        ],
    )
    return context, evidence


def test_synthetic_eighth_claim_error_identifies_field_without_mutating_reply():
    _, evidence = evidence_case()
    raw = deepcopy(CASE)
    assert len(raw["claims"]) == 8
    with pytest.raises(ResponseContractError) as caught:
        validate_claims(raw, evidence)
    issue = caught.value.details
    assert issue["field"] == "claims[7].evidence_refs"
    assert issue["claim_index"] == 7
    assert issue["judgment"] == "contradicted" and issue["actual"] == []
    assert "supplied evidence handles" in issue["reason"]
    assert "flip the judgment" in issue["reason"]
    assert raw == CASE


@pytest.mark.parametrize("judgment", ["supported", "contradicted", "unverified"])
def test_evidence_requirement_and_acceptance_remain_unchanged(judgment):
    _, evidence = evidence_case()
    raw = deepcopy(CASE)
    raw["claims"][7]["judgment"] = judgment
    before = deepcopy(raw)
    if judgment == "unverified":
        assert validate_claims(raw, evidence) == before
    else:
        with pytest.raises(ResponseContractError, match=r"claims\[7\].evidence_refs"):
            validate_claims(raw, evidence)
    assert raw == before


@pytest.mark.asyncio
@pytest.mark.parametrize("repair", ["model_citation", "identical", "invented_citation"])
async def test_actual_role_receives_indexed_same_evidence_repair_and_keeps_raw_attempts(
    transport, repair
):
    original, responses, calls = transport
    role = Role(
        {**original.config, "role_bounded_stage": True}, "judge", original.output
    )
    first, second = deepcopy(CASE), deepcopy(CASE)
    if repair == "model_citation":
        # Only the scripted model response supplies the previously absent refs.
        second["claims"][7]["evidence_refs"] = ["product_1:artifact:final_response"]
    elif repair == "invented_citation":
        second["claims"][7]["evidence_refs"] = ["invented:approval"]
    responses.extend(
        httpx.Response(
            200,
            text=encode(
                [event({"content": json.dumps(reply, ensure_ascii=False)}, "stop")]
            ),
        )
        for reply in (first, second)
    )
    context, evidence = evidence_case()
    result = await defaults(role).run(
        {
            "test_id": "traceability",
            "criterion_id": "traceability",
            "checker": "evidence.review",
        },
        context,
        evidence,
    )
    assert len(calls) == 2
    requests = [json.loads(call["messages"][1]["content"]) for call in calls]
    recovery = requests[1]["response_recovery"]
    assert recovery["rejected_response"] == first
    issue = recovery["details"]["issues"][0]
    assert issue["scope"] == "global"
    assert issue["field"] == "claims[7].evidence_refs"
    assert issue["claim_index"] == 7 and issue["judgment"] == "contradicted"
    assert issue["actual"] == []
    assert "claims[7].evidence_refs" in recovery["details"]["invalid_fields"]["global"]
    assert "same evidence" in recovery["instruction"]
    assert requests[1]["context"] == requests[0]["context"]
    assert requests[1]["measurement_budget"]["remaining_tool_calls"] == 0
    assert "nonempty evidence_refs" in requests[0]["instruction"]
    assert (
        "empty allowed only for unverified"
        in requests[0]["response"]["claims"][0]["evidence_refs"][0]
    )
    assert result["execution_usage"] == {"model_requests": 2, "tool_calls": 0}
    attempts = result["response_validation"]["attempts"]
    assert attempts[0]["response"] == first and attempts[1]["response"] == second
    assert attempts[0]["issues"][0] == issue
    # The actual Role persists both provider replies even when validation fails.
    receipts = [json.loads(p.read_text()) for p in role.output.glob("*/response.json")]
    assert len(receipts) == 2
    assert sorted(
        json.dumps(json.loads(r["raw_output"]), sort_keys=True) for r in receipts
    ) == sorted(json.dumps(r, sort_keys=True) for r in (first, second))
    if repair == "model_citation":
        assert result["status"] == "fail" and not result.get("error")
        assert result["claims"] == second["claims"]
        assert result["reason"] == first["reason"] and result["gap"] == first["gap"]
        assert result["response_validation"]["selected_attempts"] == {"global": 2}
    else:
        assert result["status"] == "unknown" and result["error"]
        assert attempts[1]["errors"]["global"]
        assert result["response_validation"]["selected_attempts"] == {}
    assert CASE["claims"][7]["evidence_refs"] == []
    assert first == CASE
