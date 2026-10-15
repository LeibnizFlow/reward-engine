"""Synthetic contract-validation cases using offline role transports.

No native grading or live provider calls are performed.
"""

from copy import deepcopy

import pytest

from reward_engine.core.outcome.context import ResponseContractError
from reward_engine.core.outcome.harness import validate_claims, validate_result
from tests.support.contract_responses import (
    CLAIM_SUMMARY_ROOT_DIAGNOSTICS_CASES as CASES,
)
from tests.support.contract_responses import (
    CLAIM_SUMMARY_ROOT_DIAGNOSTICS_WB_CASE as WB_CASE,
)
from tests.support.role_transport import transport as transport
from tests.unit.test_claim_contract_repair import claw_evidence, replay

CLARITY = "report_clarity_and_language_check"
STRUCTURE = "reconciliation_report_structure_check"


def test_synthetic_cases_identify_all_advisory_status_and_missing_root_refs():
    _, evidence = claw_evidence()
    first = deepcopy(CASES[CLARITY][0])
    assert len(first["claims"]) == 5
    assert all(c["verdict_relevance"] == "advisory" for c in first["claims"])
    with pytest.raises(ResponseContractError) as caught:
        validate_claims(first, evidence)
    issue = caught.value.details
    assert issue["field"] == "status" and issue["actual"] == "pass"
    assert issue["material_claim_count"] == 0
    assert issue["depends_on"] == {
        f"claims[{i}].verdict_relevance": "advisory" for i in range(5)
    }
    assert "minimally correct status/reason/gap" in issue["repair_instruction"]
    assert first == CASES[CLARITY][0]
    first = deepcopy(CASES[STRUCTURE][0])
    assert "evidence_refs" not in first and all(
        c["evidence_refs"] for c in first["claims"]
    )
    with pytest.raises(ResponseContractError) as caught:
        validate_result(first, evidence)
    issue = caught.value.details
    assert (
        issue["field"] == "evidence_refs"
        and issue["actual"] is None
        and issue["missing"]
    )
    assert "Top-level evidence_refs" in issue["reason"]
    assert (
        "Claim-level evidence_refs do not replace this root field"
        in issue["repair_instruction"]
    )
    assert first == CASES[STRUCTURE][0]


@pytest.mark.asyncio
@pytest.mark.parametrize("repair", ["summary_only", "scripted_relabeling", "identical"])
async def test_all_advisory_repair_corrects_summary_without_reclassifying_claims(
    transport, repair
):
    first = deepcopy(CASES[CLARITY][0])
    second = deepcopy(CASES[CLARITY][1] if repair == "scripted_relabeling" else first)
    if repair == "summary_only":
        # Authored by the scripted model; the host never substitutes a verdict.
        second.update(
            status="unknown",
            reason="Only advisory findings were assessed; no material basis establishes this criterion.",
            gap="",
        )
    result, requests = await replay(transport, [first, second], *claw_evidence())
    recovery = requests[1]["response_recovery"]
    assert recovery["details"]["issues"][0]["material_claim_count"] == 0
    instruction = recovery["instruction"]
    assert "minimally correct status/reason/gap" in instruction
    assert "do not promote advisory claims to material" in instruction
    assert "same evidence" in instruction
    assert (
        "Material means needed to judge THIS criterion, including a soft criterion"
        in requests[0]["instruction"]
    )
    assert (
        "A soft criterion does not make its claims advisory"
        in requests[0]["instruction"]
    )
    assert requests[0]["context"] == requests[1]["context"]
    assert requests[1]["measurement_budget"]["remaining_tool_calls"] == 0
    attempts = result["response_validation"]["attempts"]
    assert [a["response"] for a in attempts] == [first, second]
    assert result["status"] == "unknown"
    if repair == "summary_only":
        assert not result.get("error") and result["claims"] == first["claims"]
        assert result["reason"] == second["reason"]
        assert result["response_validation"]["selected_attempts"] == {"global": 2}
    else:
        assert (
            result["error"] and result["response_validation"]["selected_attempts"] == {}
        )
        assert attempts[1]["errors"]["global"]
        if repair == "scripted_relabeling":
            assert "preserve already valid fields" in attempts[1]["errors"]["global"]


@pytest.mark.asyncio
@pytest.mark.parametrize("repair", ["root_only", "scripted_rewrite", "identical"])
async def test_missing_root_refs_repair_keeps_exact_assessment(transport, repair):
    first = deepcopy(CASES[STRUCTURE][0])
    second = deepcopy(CASES[STRUCTURE][1] if repair == "scripted_rewrite" else first)
    if repair == "root_only":
        second["evidence_refs"] = ["task", "product_1:artifact:final_response"]
    result, requests = await replay(transport, [first, second], *claw_evidence())
    recovery = requests[1]["response_recovery"]
    issue = recovery["details"]["issues"][0]
    assert issue["scope"] == "global" and issue["field"] == "evidence_refs"
    assert issue["missing"] and issue["actual"] is None
    assert issue["repair_instruction"] in recovery["instruction"]
    assert (
        "Repair only the invalid top-level evidence_refs field"
        in recovery["instruction"]
    )
    assert "do not regenerate claims" in recovery["instruction"]
    assert "host does not fill them" in recovery["instruction"]
    assert requests[0]["context"] == requests[1]["context"]
    assert requests[1]["measurement_budget"]["remaining_tool_calls"] == 0
    attempts = result["response_validation"]["attempts"]
    assert [a["response"] for a in attempts] == [first, second]
    if repair == "root_only":
        assert result["status"] == "pass" and not result.get("error")
        for field in ("claims", "status", "reason", "gap"):
            assert result[field] == first[field]
        assert result["evidence_refs"] == second["evidence_refs"]
        assert result["response_validation"]["selected_attempts"] == {"global": 2}
    else:
        assert result["status"] == "unknown" and result["error"]
        assert result["response_validation"]["selected_attempts"] == {}
        assert attempts[1]["errors"]["global"]
        if repair == "scripted_rewrite":
            assert "preserve already valid fields" in attempts[1]["errors"]["global"]


@pytest.mark.parametrize(
    "refs", [None, [], "product_1:artifact:final_response", ["invented:proof"]]
)
def test_bad_root_refs_keep_exact_actual_without_mutation(refs):
    _, evidence = claw_evidence()
    first = deepcopy(CASES[STRUCTURE][0])
    first["evidence_refs"] = refs
    before = deepcopy(first)
    with pytest.raises(ResponseContractError) as caught:
        validate_result(first, evidence)
    issue = caught.value.details
    assert issue["field"] == "evidence_refs" and issue["actual"] == refs
    assert issue["missing"] is False
    assert first == before


def test_existing_root_refs_acceptance_unchanged():
    _, evidence = claw_evidence()
    first = deepcopy(CASES[STRUCTURE][0])
    first["evidence_refs"] = ["product_1:artifact:final_response"]
    assert validate_result(first, evidence) == first
    first["status"] = "unknown"
    del first["evidence_refs"]
    assert validate_result(first, evidence) == first


# WBv6 construction row 2: cross_validation_evidence_citation_check.


@pytest.mark.asyncio
@pytest.mark.parametrize("repair", ["summary_only", "scripted_rewrite"])
async def test_valid_claim_text_and_observation_are_immutable(transport, repair):
    from tests.unit.test_claim_status_diagnostics import evidence_case

    first, second = deepcopy(WB_CASE)
    changed = {
        k for k in first["claims"][3] if first["claims"][3][k] != second["claims"][3][k]
    }
    assert changed == {"claim", "observation"}
    assert (
        first["claims"][3]["judgment"]
        == second["claims"][3]["judgment"]
        == "unverified"
    )
    if repair == "summary_only":
        # The scripted model authors the summary without revising its assessment.
        second = {
            **deepcopy(first),
            **{k: second[k] for k in ("status", "reason", "gap")},
        }
    result, requests = await replay(transport, [first, second], *evidence_case())
    recovery = requests[1]["response_recovery"]
    assert recovery["details"]["issues"][0]["claim_index"] == 3
    assert (
        "Copy already valid claim fields exactly from rejected_response"
        in recovery["instruction"]
    )
    assert (
        "depends_on fields are constraints to preserve, not repair targets"
        in recovery["instruction"]
    )
    assert (
        "Do not paraphrase or clarify valid claim text or observations"
        in recovery["instruction"]
    )
    assert requests[0]["context"] == requests[1]["context"]
    attempts = result["response_validation"]["attempts"]
    assert [a["response"] for a in attempts] == [first, second]
    assert result["status"] == "unknown"
    if repair == "summary_only":
        assert not result.get("error") and result["claims"] == first["claims"]
        assert result["reason"] == second["reason"] and result["gap"] == second["gap"]
    else:
        assert result["error"]
        assert "claims[3]" in attempts[1]["errors"]["global"]
        assert result["response_validation"]["selected_attempts"] == {}
