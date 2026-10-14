"""Offline v5-shaped root citation repairs; native receipts are never read or changed."""

import json
import socket
from copy import deepcopy

import pytest

from reward_engine.adapters.workbuddy.measurement import WorkBuddyTools
from reward_engine.core.outcome.context import build_context
from reward_engine.core.outcome.harness import defaults, has_product_evidence
from tests.unit.test_outcome_role_contract_recovery import roles as roles

TEST_ID = "t_no_workspace_modification"
TOOL_REF = f"tool:{TEST_ID}:0"
PRODUCT_REF = "product_1:environment"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Only offline MockTransport is permitted")

    monkeypatch.setattr(socket.socket, "connect", forbidden)


def case():
    files = [
        {
            "path": f"input/source_{i}.txt",
            "changed": False,
            "content": {"text": "source"},
        }
        for i in range(27)
    ] + [{"path": "final_answer.md", "changed": True, "content": {"text": "Manual"}}]
    text = json.dumps(files)
    context, evidence = build_context(
        {
            "task_id": "offline-v5",
            "prompt": "Write final_answer.md without modifying input files.",
        },
        [
            {
                "artifact": {"final_response": "Manual saved."},
                "environment": {"file_count": 28, "deleted_files": []},
                "source_documents": [
                    {
                        "source": "workspace",
                        "offset": 0,
                        "total_chars": len(text),
                        "text": text,
                    }
                ],
            }
        ],
    )
    first = {
        "claims": [
            {
                "claim": claim,
                "claim_refs": ["task", TOOL_REF],
                "claim_location": "Task instruction",
                "evidence_refs": [TOOL_REF],
                "observation": observation,
                "judgment": "supported",
            }
            for claim, observation in [
                (
                    "Input files remain unchanged.",
                    "The inventory lists all 27 input files with changed=false.",
                ),
                (
                    "Only the required manual was created.",
                    "Only final_answer.md has changed=true.",
                ),
            ]
        ],
        "status": "pass",
        "reason": "The observed inventory satisfies the stated boundary.",
        "gap": None,
        "evidence_refs": [TOOL_REF, "task"],
    }
    test = {
        "test_id": TEST_ID,
        "criterion_id": "workspace_boundary",
        "checker": "evidence.review",
        "execution_recipe": {
            "probes": [{"tool": "workspace.inspect", "arguments": {"paths": ["*"]}}]
        },
    }
    return context, evidence, test, first


async def measured(
    roles, first, second=None, *, inquiry=False, tool="workspace.inspect"
):
    make, calls = roles
    context, evidence, test, _ = case()
    test["execution_recipe"]["probes"][0] = {
        "tool": tool,
        "arguments": {
            "paths": ["final_answer.md"] if tool == "workspace.read" else ["*"]
        },
    }
    if inquiry:
        context["measurement_inquiries"] = {
            TEST_ID: {"questions": ["Were input files preserved?"]}
        }
    replies = [deepcopy(first)] + ([deepcopy(second)] if second is not None else [])
    original = deepcopy((replies, context, evidence))
    requests = []

    def reply(request):
        requests.append(deepcopy(request))
        assert len(requests) <= len(replies), "No extra semantic retries"
        return deepcopy(replies[len(requests) - 1])

    role = make("judge", reply)
    result = await defaults(role, tools=WorkBuddyTools()).run(test, context, evidence)
    assert len(requests) == len(calls) == len(replies)
    assert result["execution_usage"] == {
        "model_requests": len(replies),
        "tool_calls": 1,
    }
    assert (replies, context, evidence) == original
    for request in requests[1:]:
        assert request["measurement_budget"]["remaining_tool_calls"] == 0
        assert request["measurement_budget"]["remaining_tool_waves"] == 0
    return result, requests


@pytest.mark.asyncio
@pytest.mark.parametrize("full_reply", [False, True])
async def test_v5_root_only_correction_preserves_exact_assessment(roles, full_reply):
    _, _, _, first = case()
    refs = [TOOL_REF, PRODUCT_REF]
    second = (
        {**deepcopy(first), "evidence_refs": refs}
        if full_reply
        else {"field_corrections": {"evidence_refs": refs}}
    )
    result, requests = await measured(roles, first, second)
    offer = requests[1]["field_correction_reply"]
    assert offer["allowed_fields"] == offer["required_fields"] == ["evidence_refs"]
    (issue,) = requests[1]["response_recovery"]["details"]["issues"]
    assert issue["scope"] == "global" and issue["field"] == "evidence_refs"
    assert issue["actual"] == first["evidence_refs"]
    assert not issue["missing"]
    assert "workspace.inspect" in issue["repair_instruction"]
    assert result["claims"] == first["claims"]
    assert result["status"] == first["status"] and result["reason"] == first["reason"]
    assert result["gap"] == ""  # Existing validate_result normalization of null.
    assert result["evidence_refs"] == refs and not result.get("error")
    receipt = result["tool_observations"][TOOL_REF]
    assert receipt["tool"] == "workspace.inspect" and receipt["result"]["evidence_refs"]
    if not full_reply:
        attempt = result["response_validation"]["attempts"][1]
        assert attempt["response"] == second
        assert attempt["composed_response"] == {**first, "evidence_refs": refs}
        provenance = attempt["field_provenance"]["fields"]
        assert provenance["evidence_refs"]["source_attempt"] == 2
        assert all(
            v["source_attempt"] == 1
            for k, v in provenance.items()
            if k != "evidence_refs"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad",
    [
        "unknown",
        "task",
        "inspect",
        "empty",
        "alias_path",
        "extra_claim",
        "rewrite_claims",
    ],
)
async def test_unsupported_refs_and_claim_rewrites_still_reject(roles, bad):
    _, _, _, first = case()
    values = {"evidence_refs": [TOOL_REF, PRODUCT_REF]}
    second = {"field_corrections": values}
    bad_refs = {
        "unknown": ["product_1:invented"],
        "task": ["task"],
        "inspect": [TOOL_REF, "task"],
        "empty": [],
    }
    if bad in bad_refs:
        values["evidence_refs"] = bad_refs[bad]
    elif bad == "alias_path":
        values["response.evidence_refs"] = values.pop("evidence_refs")
    elif bad == "extra_claim":
        values["claims[0].evidence_refs"] = [PRODUCT_REF]
    else:
        second = {**deepcopy(first), "evidence_refs": values["evidence_refs"]}
        for claim in second["claims"]:
            claim["evidence_refs"].append(PRODUCT_REF)
            claim["observation"] += " The environment confirms this."
    result, _ = await measured(roles, first, second)
    assert result["status"] == "unknown" and result["error"]
    if bad == "rewrite_claims":
        assert "preserve already valid fields of claims[0]" in result["diagnostic"]


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", ["workspace.inspect", "workspace.read"])
async def test_existing_direct_product_and_qualified_read_acceptance(roles, tool):
    _, _, _, first = case()
    if tool == "workspace.inspect":
        first["evidence_refs"] = [TOOL_REF, PRODUCT_REF]
    result, _ = await measured(roles, first, tool=tool)
    assert result["status"] == "pass" and not result.get("error")
    assert result["claims"] == first["claims"]


def test_inspect_receipt_cannot_masquerade_as_qualified_read():
    _, evidence, _, first = case()
    receipt = {
        "tool": "workspace.inspect",
        "status": "observed",
        "result": {"text": "Inventory", "observed_product_refs": [PRODUCT_REF]},
    }
    assert not has_product_evidence(
        first["evidence_refs"], evidence, {TOOL_REF: receipt}
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("rewrite_global", [False, True])
@pytest.mark.parametrize(
    "refs", [[TOOL_REF, PRODUCT_REF], [TOOL_REF, "task"], ["product_1:invented"]]
)
async def test_inquiry_root_diagnostic_and_global_isolation(
    roles, refs, rewrite_global
):
    _, _, _, first = case()
    first["evidence_refs"] = [TOOL_REF, PRODUCT_REF]
    first["inquiry_result"] = {
        "status": "pass",
        "reason": "Input inventory is unchanged.",
        "gap": "",
        "evidence_refs": [TOOL_REF, "task"],
    }
    second = deepcopy(first)
    second["inquiry_result"]["evidence_refs"] = refs
    if rewrite_global:
        second.update(
            status="fail", reason="Attempted global reassessment", gap="New gap"
        )
        second["claims"][0]["observation"] = (
            "Attempted rewrite of a validated global claim"
        )
    result, requests = await measured(roles, first, second, inquiry=True)
    details = requests[1]["response_recovery"]["details"]
    (issue,) = details["issues"]
    assert (
        issue["scope"] == "inquiry" and issue["field"] == "inquiry_result.evidence_refs"
    )
    assert issue["actual"] == first["inquiry_result"]["evidence_refs"]
    assert details["preserved_fields"] == ["global"]
    assert "field_correction_reply" not in requests[1]
    assert result["status"] == first["status"] and result["reason"] == first["reason"]
    assert (
        result["claims"] == first["claims"]
        and result["evidence_refs"] == first["evidence_refs"]
    )
    assert result["response_validation"]["selected_attempts"]["global"] == 1
    if PRODUCT_REF in refs:
        assert result["inquiry_result"] == second["inquiry_result"]
    else:
        assert (
            result["inquiry_result"]["status"] == "unknown"
            and result["inquiry_result"]["error"]
        )


@pytest.mark.asyncio
async def test_global_root_correction_preserves_valid_inquiry(roles):
    _, _, _, first = case()
    first["inquiry_result"] = {
        "status": "fail",
        "reason": "A separate issue remains.",
        "gap": "Investigate the separate issue.",
        "evidence_refs": [PRODUCT_REF],
    }
    result, requests = await measured(
        roles,
        first,
        {"field_corrections": {"evidence_refs": [TOOL_REF, PRODUCT_REF]}},
        inquiry=True,
    )
    assert requests[1]["response_recovery"]["details"]["preserved_fields"] == [
        "inquiry"
    ]
    assert (
        result["status"] == "pass"
        and result["inquiry_result"] == first["inquiry_result"]
    )
    assert result["response_validation"]["selected_attempts"] == {
        "inquiry": 1,
        "global": 2,
    }
