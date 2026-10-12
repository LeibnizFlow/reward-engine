"""Independent global/local verdicts share one structure-repair budget."""

import json
from copy import deepcopy

import pytest

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
