"""Independent global/local verdicts share one structure-repair budget."""

import json
from copy import deepcopy

import pytest

from reward_engine.core.outcome.context import build_context
from reward_engine.core.outcome.harness import defaults
from tests.unit.test_claw_outcome_measurement_tools import case, verdict


def inquiry(status="fail"):
    return {
        "status": status,
        "reason": "The local requested explanation is missing",
        "gap": "Explain the calculation" if status == "fail" else "",
        "evidence_refs": ["product_1:artifact:final_response"],
    }


async def measured(replies, *, bounded=True):
    context, evidence, _, test = case()
    context["measurement_inquiries"] = {
        test["test_id"]: {"questions": ["Does the answer explain the calculation?"]}
    }
    test["execution_recipe"] = {
        "probes": [{"tool": "calculate", "arguments": {"expression": "2+3"}}]
    }
    requests = []

    async def model(request):
        requests.append(deepcopy(request))
        assert len(requests) <= len(replies), "Unexpected semantic retry"
        reply = replies[len(requests) - 1]
        if isinstance(reply, Exception):
            raise reply
        return deepcopy(reply)

    model.bounded_stage = bounded
    result = await defaults(model).run(test, context, evidence)
    return result, requests


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad", [None, "pass", {}, {"status": "partial", "reason": "Maybe"}]
)
async def test_malformed_requested_inquiry_gets_one_structure_repair(bad):
    first = {**verdict("tool:check:0"), "inquiry_result": bad}
    second = {**verdict("tool:check:0"), "inquiry_result": inquiry()}
    # The valid global counterpart is not a fresh semantic choice during repair.
    second.update(
        status="fail", reason="Reconsidered the global verdict", gap="Changed judgment"
    )
    original = deepcopy((first, second))
    result, requests = await measured([first, second])
    assert len(requests) == 2
    recovery = requests[1]["response_recovery"]
    assert set(recovery["details"]["invalid_fields"]) == {"inquiry"}
    assert recovery["details"]["preserved_fields"] == ["global"]
    assert recovery["rejected_response"] == first
    assert result["status"] == "pass" and result["reason"] == first["reason"]
    assert result["inquiry_result"] == inquiry()
    audit = result["response_validation"]
    assert audit["attempts"][0]["response"] == first
    assert audit["attempts"][0]["errors"]["inquiry"]
    assert audit["attempts"][1]["response"] == second
    assert audit["selected_attempts"] == {"global": 1, "inquiry": 2}
    assert result["execution_usage"] == {"model_requests": 2, "tool_calls": 1}
    assert result["tool_observations"]["tool:check:0"]["result"] == 5
    assert (first, second) == original


@pytest.mark.asyncio
@pytest.mark.parametrize("repairable", [False, True])
@pytest.mark.parametrize("malformed", ["status", "reason", "claims"])
async def test_valid_local_survives_malformed_global_and_exhausted_repair(
    repairable, malformed
):
    first = {**verdict("tool:check:0"), "inquiry_result": inquiry()}
    if malformed == "status":
        first["status"] = "partial"
    elif malformed == "reason":
        first.pop("reason")
    else:
        first["claims"] = ["invalid claim row"]
    second = verdict("tool:check:0") if repairable else deepcopy(first)
    second["inquiry_result"] = "malformed during repair"
    result, requests = await measured([first, second])
    assert len(requests) == 2
    assert result["inquiry_result"] == inquiry()
    assert result["status"] == ("pass" if repairable else "unknown")
    assert bool(result.get("error")) is (not repairable)
    audit = result["response_validation"]
    assert audit["attempts"][0]["errors"]["global"]
    assert audit["attempts"][1]["errors"]["inquiry"]
    assert audit["selected_attempts"]["inquiry"] == 1
    assert ("global" in audit["selected_attempts"]) is repairable
    assert result["execution_usage"] == {"model_requests": 2, "tool_calls": 1}


@pytest.mark.asyncio
async def test_valid_global_survives_exhausted_inquiry_repair():
    first = {**verdict("tool:check:0"), "inquiry_result": "unknown"}
    result, requests = await measured([first, first])
    assert len(requests) == 2
    assert result["status"] == "pass" and not result.get("error")
    assert result["inquiry_result"]["status"] == "unknown"
    assert result["inquiry_result"]["error"] is True
    assert "verdict object" in result["inquiry_result"]["diagnostic"]
    assert all(
        a["errors"]["inquiry"] for a in result["response_validation"]["attempts"]
    )


@pytest.mark.asyncio
async def test_both_invalid_stay_explicit_unknown_after_single_repair():
    first = {"status": "partial", "inquiry_result": "fail"}
    result, requests = await measured([first, first])
    assert len(requests) == 2
    assert result["status"] == "unknown" and result["error"]
    assert (
        result["inquiry_result"]["status"] == "unknown"
        and result["inquiry_result"]["error"]
    )
    assert result["response_validation"]["selected_attempts"] == {}
    assert all(
        set(a["errors"]) == {"global", "inquiry"}
        for a in result["response_validation"]["attempts"]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["pass", "fail", "unknown"])
async def test_valid_semantic_outcomes_do_not_trigger_repair(status):
    first = {**verdict("tool:check:0"), "inquiry_result": inquiry(status)}
    first.update(status=status, gap="Missing explanation" if status == "fail" else "")
    result, requests = await measured([first])
    assert len(requests) == 1
    assert result["status"] == status
    assert result["inquiry_result"] == inquiry(status)
    assert "response_validation" not in result


@pytest.mark.asyncio
async def test_unbounded_role_does_not_gain_repair_but_preserves_valid_local():
    first = {
        **verdict("tool:check:0"),
        "status": "partial",
        "inquiry_result": inquiry(),
    }
    result, requests = await measured([first], bounded=False)
    assert len(requests) == 1
    assert result["status"] == "unknown" and result["error"]
    assert result["inquiry_result"] == inquiry()


@pytest.mark.asyncio
@pytest.mark.parametrize("bounded", [False, True])
async def test_omitted_inquiry_is_legacy_uncertainty_but_bounded_contract_error(
    bounded,
):
    first = verdict("tool:check:0")
    result, requests = await measured(
        [first, first] if bounded else [first], bounded=bounded
    )
    assert len(requests) == (2 if bounded else 1)
    assert result["status"] == "pass" and not result.get("error")
    assert result["inquiry_result"]["status"] == "unknown"
    assert result["inquiry_result"]["error"] is bounded
    if bounded:
        assert result["response_validation"]["attempts"][0]["errors"]["inquiry"]
    else:
        assert "not explicitly answered" in result["inquiry_result"]["reason"]


@pytest.mark.asyncio
async def test_transport_failure_during_repair_preserves_local_without_retry():
    first = {
        **verdict("tool:check:0"),
        "status": "partial",
        "inquiry_result": inquiry(),
    }
    result, requests = await measured([first, TimeoutError("scripted timeout")])
    assert len(requests) == 2
    assert result["status"] == "unknown" and result["error"]
    assert result["failure_layer"] == "infrastructure"
    assert result["inquiry_result"] == inquiry()
    assert result["response_validation"]["attempts"][-1]["error_type"] == "TimeoutError"


@pytest.mark.asyncio
async def test_syntax_failure_during_repair_retains_raw_attempt_and_valid_local():
    first = {
        **verdict("tool:check:0"),
        "status": "partial",
        "inquiry_result": inquiry(),
    }
    broken = '{"status":'
    syntax_error = json.JSONDecodeError("Incomplete response", broken, len(broken))
    result, requests = await measured([first, syntax_error, first])
    # Existing bounded ask allows one syntax repair and one contract repair,
    # shared across the whole stage. Neither permits another semantic attempt.
    assert len(requests) == 3
    assert result["status"] == "unknown" and result["error"]
    assert result["inquiry_result"] == inquiry()
    audit = result["response_validation"]
    assert audit["selected_attempts"] == {"inquiry": 1}
    assert audit["attempts"][0]["response"] == first
    assert audit["attempts"][1]["error_type"] == "JSONDecodeError"
    assert audit["attempts"][1]["raw_response"] == broken
    assert audit["attempts"][2]["errors"]["global"]
    assert result["execution_usage"] == {"model_requests": 3, "tool_calls": 1}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "repair_response",
    [
        {"need_tools": [{"tool": "calculate", "arguments": {"expression": "9+9"}}]},
        {"need_evidence": ["product_1:artifact:final_response"]},
    ],
)
async def test_structure_repair_cannot_restart_measurements_or_evidence_reads(
    repair_response,
):
    first = {**verdict("tool:check:0"), "inquiry_result": "fail"}
    result, requests = await measured([first, repair_response])
    assert len(requests) == 2
    assert requests[1]["measurement_budget"]["final_verdict_required"]
    assert requests[1]["measurement_budget"]["remaining_tool_calls"] == 0
    assert result["execution_usage"] == {"model_requests": 2, "tool_calls": 1}
    assert result["status"] == "pass"
    assert result["inquiry_result"]["status"] == "unknown"
    assert result["inquiry_result"]["error"]
    assert result["response_validation"]["attempts"][-1]["response"] == repair_response
    assert result["response_validation"]["attempts"][-1]["errors"]["repair"]


@pytest.mark.asyncio
@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize(
    "new_request",
    [
        {"need_tools": [{"tool": "calculate", "arguments": {"expression": "9+9"}}]},
        {"need_evidence": ["product_1:artifact:final_response"]},
    ],
)
async def test_syntax_retry_cannot_unlock_structure_repair(new_request, wrapped):
    if wrapped:
        new_request = {"response": new_request}
    first = {**verdict("tool:check:0"), "inquiry_result": "fail"}
    broken = '{"status":'
    result, requests = await measured(
        [
            first,
            json.JSONDecodeError("broken", broken, len(broken)),
            new_request,
            {**verdict("tool:check:0"), "inquiry_result": inquiry()},
        ]
    )
    assert len(requests) == 3
    assert requests[-1]["measurement_budget"]["remaining_tool_calls"] == 0
    assert requests[-1]["measurement_budget"]["remaining_tool_waves"] == 0
    assert result["execution_usage"] == {"model_requests": 3, "tool_calls": 1}
    assert len(result["tool_observations"]) == 1
    assert result["inquiry_result"]["status"] == "unknown"
    assert result["inquiry_result"]["error"]
    assert result["response_validation"]["attempts"][-1]["response"] == new_request


async def preview_measurement(scope="both", *, bounded=True, exhaust=False):
    context, evidence = build_context(
        {"task_id": "audit", "prompt": "Check the whole report and its calculation."},
        [
            {
                "artifact": {"final_response": "The answer is 5"},
                "source_documents": [
                    {"text": "x" * 1000 + " The answer is actually 6."}
                ],
            }
        ],
    )
    _, _, _, test = case()
    context["measurement_inquiries"] = {
        "check": {"questions": ["Is the calculation explained?"]}
    }
    ref = "product_1:source:0"
    assert context["products"][0]["sources"][0]["preview_only"]
    raw = verdict()
    raw["evidence_refs"] = [ref]
    raw["claims"][0]["evidence_refs"] = [ref]
    raw["inquiry_result"] = {**inquiry("pass"), "evidence_refs": [ref]}
    if scope == "global":
        raw["inquiry_result"] = "fail"
    elif scope == "inquiry":
        raw["status"] = "partial"
    requests = []

    async def model(request):
        requests.append(deepcopy(request))
        if exhaust and len(requests) == 1:
            return {"need_evidence": ["task"]}
        if exhaust and len(requests) == 2:
            return {"need_evidence": ["product_1:artifact:final_response"]}
        assert len(requests) <= (4 if exhaust else 2 if bounded else 1)
        return deepcopy(raw)

    model.bounded_stage = bounded
    result = await defaults(model).run(test, context, evidence)
    return result, requests, ref


@pytest.mark.asyncio
async def test_preview_read_exhaustion_cannot_restore_passes_or_erase_error():
    from experiments.pipeline_audit import check_result

    result, requests, ref = await preview_measurement(exhaust=True)
    assert len(requests) == 4 and requests[-1]["evidence_access_closed"]
    assert all(ref not in r.get("requested_evidence", {}) for r in requests)
    assert result["status"] == "unknown" and result["error"]
    assert result["unresolved_evidence_refs"] == [ref]
    assert result["inquiry_result"]["status"] == "unknown"
    assert result["inquiry_result"]["error"]
    with pytest.raises(ValueError):
        check_result(result, "global")
    with pytest.raises(ValueError):
        check_result(result["inquiry_result"], "inquiry")
    assert result["response_validation"]["selected_attempts"] == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["global", "inquiry"])
@pytest.mark.parametrize("bounded", [False, True])
async def test_malformed_companion_cannot_preserve_unread_preview_verdict(
    scope, bounded
):
    result, requests, ref = await preview_measurement(scope, bounded=bounded)
    assert all(ref not in r.get("requested_evidence", {}) for r in requests)
    assert result["status"] == "unknown" and result["error"]
    assert (
        result["inquiry_result"]["status"] == "unknown"
        and result["inquiry_result"]["error"]
    )
    assert ref in result["unresolved_evidence_refs"]
    assert result["response_validation"]["selected_attempts"] == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("malformed_scope", [None, "global", "inquiry"])
async def test_preview_counterpart_is_preservable_only_after_source_supplied(
    malformed_scope,
):
    context, evidence = build_context(
        {"task_id": "audit", "prompt": "Check the recorded calculation"},
        [
            {
                "artifact": {"final_response": "The answer is 5"},
                "source_documents": [{"text": "x" * 1000 + " End of actual source."}],
            }
        ],
    )
    _, _, _, test = case()
    context["measurement_inquiries"] = {"check": {"questions": ["Is it explained?"]}}
    ref = "product_1:source:0"
    requests = []

    async def model(request):
        requests.append(deepcopy(request))
        # An explicit read before a malformed companion, or an implicit
        # preview followup for a well-formed pair, supplies the actual source.
        if malformed_scope and len(requests) == 1:
            return {"need_evidence": [ref]}
        raw = verdict()
        raw["evidence_refs"] = raw["claims"][0]["evidence_refs"] = [ref]
        raw["inquiry_result"] = {**inquiry(), "evidence_refs": [ref]}
        if len(requests) > 1:
            assert request["requested_evidence"][ref] == evidence[ref]
        if malformed_scope == "global":
            raw["status"] = "partial"
        elif malformed_scope == "inquiry":
            raw["inquiry_result"] = "fail"
        return raw

    model.bounded_stage = True
    result = await defaults(model).run(test, context, evidence)
    assert len(requests) == (3 if malformed_scope else 2)
    assert not result.get("unresolved_evidence_refs")
    assert result["status"] == ("unknown" if malformed_scope == "global" else "pass")
    assert result["inquiry_result"]["status"] == (
        "unknown" if malformed_scope == "inquiry" else "fail"
    )
    if malformed_scope:
        part = result if malformed_scope == "global" else result["inquiry_result"]
        assert part["error"]
    else:
        assert not result.get("error") and not result["inquiry_result"].get("error")


@pytest.mark.asyncio
async def test_read_limit_error_not_replaced_even_by_evidence_complete_global():
    context, evidence, _, test = case()
    context["measurement_inquiries"] = {"check": {"questions": ["Explain it"]}}
    calls = []

    async def model(request):
        calls.append(request)
        if len(calls) == 1:
            return {**verdict(), "inquiry_result": "fail"}
        # The exception can originate from a source-reading wrapper as well as
        # ask. A valid earlier global part must not erase the terminal failure.
        from reward_engine.core.outcome.context import EvidenceReadLimit

        raise EvidenceReadLimit(["product_1:artifact:final_response"])

    model.bounded_stage = True
    result = await defaults(model).run(test, context, evidence)
    assert len(calls) == 2
    assert result["status"] == "unknown" and result["error"]
    assert result["unresolved_evidence_refs"] == ["product_1:artifact:final_response"]
    assert "global" not in result["response_validation"]["selected_attempts"]
