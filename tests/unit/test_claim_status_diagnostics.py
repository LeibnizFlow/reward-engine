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
from tests.support.contract_responses import CLAIM_STATUS_DIAGNOSTICS_CASE as CASE
from tests.support.role_transport import encode, event
from tests.support.role_transport import transport as transport


def evidence_case():
    return build_context(
        {"task_id": "offline-capture", "prompt": "Audit the supplied standard"},
        [
            {
                "artifact": {
                    "final_response": "Synthetic structural replay, not native task evidence",
                    "changed_files": "Offline placeholder",
                },
                "source_documents": [{"text": f"Offline source {i}"} for i in range(3)],
            }
        ],
    )


def test_synthetic_material_claim_status_conflict_is_indexed_and_not_mutated():
    _, evidence = evidence_case()
    raw = deepcopy(CASE)
    with pytest.raises(ResponseContractError) as caught:
        validate_claims(raw, evidence)
    issue = caught.value.details
    assert issue["field"] == "status" and issue["actual"] == "pass"
    assert issue["claim_index"] == 2
    assert issue["depends_on"] == {
        "claims[2].judgment": "unverified",
        "claims[2].verdict_relevance": "material",
    }
    assert raw == CASE


@pytest.mark.parametrize(
    "judgment,status,accepted",
    [
        ("unverified", "pass", False),
        ("contradicted", "pass", False),
        ("supported", "pass", True),
        ("unverified", "unknown", True),
        ("contradicted", "fail", True),
    ],
)
def test_existing_status_claim_acceptance_is_unchanged(judgment, status, accepted):
    _, evidence = evidence_case()
    raw = deepcopy(CASE)
    raw["claims"][2]["judgment"] = judgment
    raw["status"] = status
    before = deepcopy(raw)
    if accepted:
        assert validate_claims(raw, evidence) == before
    else:
        with pytest.raises(ResponseContractError):
            validate_claims(raw, evidence)
    assert raw == before


@pytest.mark.asyncio
@pytest.mark.parametrize("repair", ["model_conservative", "identical"])
async def test_actual_role_repairs_only_inconsistent_summary_or_rejects_repeat(
    transport, repair
):
    original, responses, calls = transport
    role = Role(
        {**original.config, "role_bounded_stage": True}, "judge", original.output
    )
    first, second = deepcopy(CASE), deepcopy(CASE)
    if repair == "model_conservative":
        # Scripted MODEL output authors a conservative summary, preserving every
        # claim and citation. The host must not select/replace the verdict.
        second.update(
            status="unknown",
            reason="The material source/schema comparison remains unverified.",
            gap="Independent source contents are unavailable in this evidence.",
        )
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
        {"test_id": "caveats", "criterion_id": "caveats", "checker": "evidence.review"},
        context,
        evidence,
    )
    assert len(calls) == 2 and not responses
    requests = [json.loads(call["messages"][1]["content"]) for call in calls]
    recovery = requests[1]["response_recovery"]
    assert recovery["rejected_response"] == first
    issue = recovery["details"]["issues"][0]
    assert issue["scope"] == "global" and issue["claim_index"] == 2
    assert issue["depends_on"]["claims[2].judgment"] == "unverified"
    assert "claims[2].judgment" in recovery["details"]["invalid_fields"]["global"]
    instruction = recovery["instruction"]
    assert issue["repair_instruction"] in instruction
    assert "same evidence" in instruction
    assert "minimally correct status/reason/gap" in instruction
    assert "Preserve claims, judgments, relevance and evidence refs" in instruction
    for forbidden in (
        "Do not upgrade an unsupported claim to supported",
        "downgrade material to advisory",
        "invent proof",
    ):
        assert forbidden in instruction
        assert forbidden in requests[0]["instruction"]
    assert (
        "pass requires ALL material claims to be supported"
        in requests[0]["instruction"]
    )
    assert "nonempty evidence_refs" in requests[0]["instruction"]
    assert requests[1]["context"] == requests[0]["context"]
    assert requests[1]["measurement_budget"]["remaining_tool_calls"] == 0
    assert result["execution_usage"] == {"model_requests": 2, "tool_calls": 0}
    attempts = result["response_validation"]["attempts"]
    assert [a["response"] for a in attempts] == [first, second]
    assert attempts[0]["issues"][0] == issue
    receipts = [json.loads(p.read_text()) for p in role.output.glob("*/response.json")]
    assert len(receipts) == 2
    assert sorted(
        json.dumps(json.loads(r["raw_output"]), sort_keys=True) for r in receipts
    ) == sorted(json.dumps(r, sort_keys=True) for r in (first, second))
    assert result["status"] == "unknown"
    if repair == "model_conservative":
        assert not result.get("error")
        assert result["claims"] == first["claims"]
        assert result["reason"] == second["reason"] and result["gap"] == second["gap"]
        assert result["response_validation"]["selected_attempts"] == {"global": 2}
    else:
        assert result["error"]
        assert attempts[1]["issues"][0] == issue
        assert result["response_validation"]["selected_attempts"] == {}
    assert first == CASE
