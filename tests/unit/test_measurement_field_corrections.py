"""Optional bounded field corrections use actual Role with offline HTTP only.

The original captures remain in their focused regression modules. This protocol
composes model-authored replacements; it does not certify those model judgments.
"""

from copy import deepcopy

import pytest

from tests.unit.test_claim_contract_repair import (
    CLAW_CAPTURE,
    claw_evidence,
    preview_case,
    replay,
)
from tests.unit.test_claim_status_diagnostics import evidence_case as wb_evidence
from tests.unit.test_claim_summary_root_diagnostics import (
    CAPTURES,
    CLARITY,
    STRUCTURE,
    WB_CAPTURE,
)
from tests.unit.test_workbuddy_stream_terminal_replay import transport as transport


def case(kind):
    if kind == "root_refs":
        first = deepcopy(CAPTURES[STRUCTURE][0])
        values = {"evidence_refs": ["product_1:artifact:final_response"]}
        context, evidence = claw_evidence()
    else:
        values = {
            "status": "unknown",
            "reason": "The preserved assessment does not establish a material pass.",
            "gap": "",
        }
        if kind == "all_advisory":
            first = deepcopy(CAPTURES[CLARITY][0])
            context, evidence = claw_evidence()
        elif kind == "unverified":
            first = deepcopy(WB_CAPTURE[0])
            context, evidence = wb_evidence()
        elif kind == "twelve_refs":
            first = deepcopy(CLAW_CAPTURE)
            context, evidence = claw_evidence()
            for i, claim in enumerate(first["claims"]):
                values[f"claims[{i}].evidence_refs"] = (
                    []
                    if claim["judgment"] == "unverified"
                    else ["product_1:artifact:final_response"]
                )
        else:
            raise AssertionError(kind)
    return first, {"field_corrections": values}, context, evidence


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind", ["root_refs", "all_advisory", "unverified", "twelve_refs"]
)
@pytest.mark.parametrize("syntax_retry", [False, True])
async def test_captured_failures_can_use_explicit_model_field_values_with_provenance(
    transport, kind, syntax_retry
):
    first, correction, context, evidence = case(kind)
    replies = (
        [first, '{"field_corrections":', correction]
        if syntax_retry
        else [first, correction]
    )
    result, requests = await replay(transport, replies, context, evidence)
    offer = requests[-1]["field_correction_reply"]
    assert offer["schema"] == "measurement-field-correction/v1"
    assert offer["instruction"].startswith("Preferred repair reply")
    assert all(
        r["response_recovery"]["instruction"].startswith("Preferred repair reply")
        for r in requests[1:]
    )
    assert (
        "Do not regenerate unchanged claims"
        in requests[-1]["response_recovery"]["instruction"]
    )
    assert offer["base_attempt"] == 1
    if kind == "root_refs":
        assert "allowed_values" not in offer
    else:
        assert offer["allowed_values"] == {"status": ["fail", "unknown"]}
        assert "For this locked assessment, pass is invalid" in offer["instruction"]
        assert all(
            r["field_correction_reply"]["allowed_values"] == offer["allowed_values"]
            for r in requests[1:]
        )
    assert (
        set(offer["allowed_fields"])
        == set(offer["required_fields"])
        == set(correction["field_corrections"])
    )
    assert requests[1]["response_recovery"]["rejected_response"] == first
    assert (
        requests[1]["response_recovery"]["details"]["field_correction_reply"] == offer
    )
    assert all(r["context"] == requests[0]["context"] for r in requests)
    assert all(
        r["measurement_budget"]["remaining_tool_calls"] == 0 for r in requests[1:]
    )
    attempts = result["response_validation"]["attempts"]
    assert attempts[0]["response"] == first
    corrected = attempts[-1]
    assert corrected["response"] == correction
    assert corrected["response_kind"] == "field_correction"
    assert not corrected["errors"]
    composed = corrected["composed_response"]
    provenance = corrected["field_provenance"]
    attempt = len(replies)
    assert provenance["base_attempt"] == provenance["claim_order_source_attempt"] == 1
    assert provenance["correction_attempt"] == attempt
    assert len(composed["claims"]) == len(first["claims"])
    for field, value in first.items():
        if field != "claims" and field not in correction["field_corrections"]:
            assert composed[field] == value
            assert provenance["fields"][field] == {
                "source_attempt": 1,
                "source_field": field,
                "kind": "preserved",
            }
    for i, claim in enumerate(first["claims"]):
        for name, value in claim.items():
            path = f"claims[{i}].{name}"
            assert composed["claims"][i][name] == value
            assert provenance["fields"][path] == {
                "source_attempt": 1,
                "source_field": path,
                "kind": "preserved",
            }
    for path, value in correction["field_corrections"].items():
        assert provenance["fields"][path] == {
            "source_attempt": attempt,
            "source_field": f"field_corrections.{path}",
            "kind": "replacement",
        }
        if path in ("status", "reason", "gap", "evidence_refs"):
            assert composed[path] == value and result[path] == value
    assert result["claims"] == composed["claims"]
    if kind == "twelve_refs":
        for i, claim in enumerate(result["claims"]):
            assert (
                claim["evidence_refs"]
                == correction["field_corrections"][f"claims[{i}].evidence_refs"]
            )
    assert result["status"] == ("pass" if kind == "root_refs" else "unknown")
    assert not result.get("error")
    assert result["response_validation"]["selected_attempts"] == {"global": attempt}
    if syntax_retry:
        assert attempts[1]["raw_response"] == replies[1]
        assert requests[-1]["response_recovery"]["error_type"] == "JSONDecodeError"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad",
    [
        "upgrade",
        "relevance",
        "claim_text",
        "claim_list",
        "delete",
        "bad_index",
        "bad_path",
        "extra_root",
        "missing_status",
        "missing_reason",
        "missing_gap",
        "null_gap",
        "invalid_status",
        "missing_citation",
        "unknown_refs",
        "nonproduct_refs",
        "unknown_claim_refs",
        "empty_supported_refs",
        "extra_valid_refs",
        "nonobject",
        "tools",
        "reads",
        "still_pass",
    ],
)
async def test_corrections_reject_unadvertised_edits_or_incomplete_invalid_values(
    transport, bad
):
    kind = (
        "twelve_refs"
        if bad in ("missing_citation", "unknown_claim_refs", "empty_supported_refs")
        else "root_refs"
        if bad in ("unknown_refs", "nonproduct_refs")
        else "unverified"
    )
    first, correction, context, evidence = case(kind)
    values = correction["field_corrections"]
    edits = {
        "upgrade": ("claims[3].judgment", "supported"),
        "relevance": ("claims[3].verdict_relevance", "advisory"),
        "claim_text": ("claims[3].claim", "Rewritten assessment"),
        "claim_list": ("claims", []),
        "delete": ("claims[3]", None),
        "bad_index": ("claims[-1].evidence_refs", []),
        "bad_path": ("claims[3].__class__.__dict__", {}),
        "extra_valid_refs": (
            "claims[3].evidence_refs",
            ["product_1:artifact:final_response"],
        ),
    }
    if bad in edits:
        key, value = edits[bad]
        values[key] = value
    elif bad == "extra_root":
        correction["claims"] = deepcopy(first["claims"])
    elif bad.startswith("missing_"):
        key = (
            "claims[0].evidence_refs"
            if bad == "missing_citation"
            else bad.removeprefix("missing_")
        )
        del values[key]
    elif bad == "null_gap":
        values["gap"] = None
    elif bad == "invalid_status":
        values["status"] = "partial"
    elif bad == "unknown_refs":
        values["evidence_refs"] = ["invented:proof"]
    elif bad == "nonproduct_refs":
        values["evidence_refs"] = ["task"]
    elif bad == "unknown_claim_refs":
        values["claims[0].evidence_refs"] = ["invented:proof"]
    elif bad == "empty_supported_refs":
        values["claims[0].evidence_refs"] = []
    elif bad == "nonobject":
        correction["field_corrections"] = []
    elif bad == "tools":
        correction["need_tools"] = [
            {"tool": "calculate", "arguments": {"expression": "2+3"}}
        ]
    elif bad == "reads":
        correction["need_evidence"] = ["product_1:source:2"]
    elif bad == "still_pass":
        values["status"] = "pass"
    result, requests = await replay(transport, [first, correction], context, evidence)
    assert result["status"] == "unknown" and result["error"]
    assert result["response_validation"]["selected_attempts"] == {}
    attempts = result["response_validation"]["attempts"]
    assert attempts[0]["response"] == first and attempts[-1]["response"] == correction
    assert attempts[-1]["errors"]
    assert requests[1]["measurement_budget"]["remaining_tool_calls"] == 0
    if bad in ("invalid_status", "still_pass"):
        assert attempts[-1]["composed_response"]["status"] == values["status"]
        assert (
            attempts[-1]["field_provenance"]["fields"]["status"]["source_attempt"] == 2
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("next_reply", ["fresh_full", "stale_correction"])
async def test_genuine_preview_expansion_requires_fresh_full_assessment(
    transport, next_reply
):
    context, evidence, ref, first = preview_case()
    del first["claims"][0]["evidence_refs"]
    correction = {
        "field_corrections": {
            "claims[0].evidence_refs": [ref],
            "status": "unknown",
            "reason": "The material claim remains unverified in the preview.",
            "gap": "",
        }
    }
    third = deepcopy(correction)
    if next_reply == "fresh_full":
        third = deepcopy(first)
        third["claims"][0].update(
            evidence_refs=[ref],
            judgment="supported",
            observation="The newly supplied record establishes the fact.",
        )
        third["reason"] = "Verified against the newly supplied record."
    result, requests = await replay(
        transport, [first, correction, third], context, evidence
    )
    assert requests[2]["requested_evidence"][ref] == evidence[ref]
    assert requests[2]["evidence_followup"] and not requests[2].get("response_recovery")
    assert "field_correction_reply" not in requests[2]
    attempts = result["response_validation"]["attempts"]
    assert (
        attempts[1]["response"] == correction
        and attempts[1]["composed_response"]["status"] == "unknown"
    )
    if next_reply == "fresh_full":
        assert result["status"] == "pass" and not result.get("error")
        assert result["claims"] == third["claims"]
    else:
        assert result["status"] == "unknown" and result["error"]
        assert attempts[-1]["response"] == third
        assert "composed_response" not in attempts[-1]


@pytest.mark.asyncio
async def test_valid_inquiry_survives_rejected_field_correction(transport):
    first, correction, context, evidence = case("unverified")
    context["measurement_inquiries"] = {
        "contract": {"questions": ["Is there a defect?"]}
    }
    first["inquiry_result"] = {
        "status": "fail",
        "reason": "Observed independent defect",
        "gap": "Missing explanation",
        "evidence_refs": ["product_1:artifact:final_response"],
    }
    correction["field_corrections"]["claims[3].judgment"] = "supported"
    result, _ = await replay(transport, [first, correction], context, evidence)
    assert result["status"] == "unknown" and result["error"]
    assert result["inquiry_result"] == first["inquiry_result"]
    assert result["response_validation"]["selected_attempts"] == {"inquiry": 1}


@pytest.mark.asyncio
async def test_unbounded_legacy_role_does_not_advertise_or_compose_corrections():
    from reward_engine.core.outcome.harness import defaults

    first, correction, context, evidence = case("unverified")
    calls = []

    async def model(request):
        calls.append(deepcopy(request))
        return correction

    result = await defaults(model).run(
        {"test_id": "legacy", "criterion_id": "legacy", "checker": "evidence.review"},
        context,
        evidence,
    )
    assert len(calls) == 1 and "field_correction_reply" not in calls[0]
    assert result["status"] == "unknown" and result["error"]
    assert "claims" not in correction


# Exact raw correction from live_field_correction_probe_4d4210fc/all_advisory/probe_result.json.
# Hybrid injected first reply + live Role repair; NOT a native pipeline result.
HYBRID_PASS_SHA256 = "0549213dce566e714d67fe17f303d28f3d5e23e61423b1564e3c42c9c2d25256"
HYBRID_PASS_CORRECTION = {
    "field_corrections": {
        "status": "pass",
        "reason": "The report is written in Chinese, uses the exact labels 匹配正常 / 金额不符 / 状态异常, and presents entries in scannable tables/bullets with the key facts (amount, invoice, status, verdict). The preserved claim assessment records all five claims as supported by the supplied artifact; the all-advisory relevance reflects that this is a soft, minor usability standard rather than a hard correctness requirement, so the supported findings are sufficient to satisfy it.",
        "gap": "",
    }
}


@pytest.mark.asyncio
async def test_hybrid_probe_unsupported_pass_remains_rejected_without_host_correction(
    transport,
):
    import hashlib
    import json

    first, _, context, evidence = case("all_advisory")
    correction = deepcopy(HYBRID_PASS_CORRECTION)
    assert (
        hashlib.sha256(
            json.dumps(
                correction, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        == HYBRID_PASS_SHA256
    )
    result, requests = await replay(transport, [first, correction], context, evidence)
    assert requests[1]["field_correction_reply"]["allowed_values"] == {
        "status": ["fail", "unknown"]
    }
    assert result["status"] == "unknown" and result["error"]
    attempts = result["response_validation"]["attempts"]
    assert attempts[0]["response"] == first and attempts[1]["response"] == correction
    assert attempts[1]["composed_response"]["status"] == "pass"
    assert attempts[1]["composed_response"]["claims"] == first["claims"]
    assert "pass needs supported material claims" in attempts[1]["errors"]["global"]
    assert attempts[1]["field_provenance"]["fields"]["status"]["source_attempt"] == 2
    assert result["response_validation"]["selected_attempts"] == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["fail", "unknown"])
async def test_allowed_summary_status_remains_model_authored(transport, status):
    first, correction, context, evidence = case("unverified")
    correction["field_corrections"].update(
        status=status,
        reason="The model judges that the required evidence has not been established.",
        gap="The caveat lacks independently observed source support."
        if status == "fail"
        else "",
    )
    result, requests = await replay(transport, [first, correction], context, evidence)
    assert requests[1]["field_correction_reply"]["allowed_values"] == {
        "status": ["fail", "unknown"]
    }
    assert result["status"] == status and not result.get("error")
    assert result["claims"] == first["claims"]
    for field, value in correction["field_corrections"].items():
        assert result[field] == value
