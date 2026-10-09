from copy import deepcopy
from types import SimpleNamespace

import pytest

from reward_engine.core.outcome.context import (
    ResponseContractError,
    ask,
    build_context,
    response_payload,
)
from reward_engine.core.outcome.counterexamples import verify_issue
from reward_engine.core.outcome.measurement_tools import MeasurementTools
from tests.support.claw_outcome_feedback_arms import PACK, TASK, candidate
from tests.support.outcome_attack_execution_integrity import issue, parser_reference


@pytest.mark.asyncio
@pytest.mark.parametrize("empty_read_list", [False, True])
async def test_reference_response_envelope_needs_no_paid_retry_and_keeps_all_verdicts(
    empty_read_list,
):
    context, evidence = build_context(TASK, [candidate("{}")])
    calls, raw = [], []

    async def reference(request):
        calls.append(deepcopy(request))
        reply = {"response": await parser_reference(request)}
        if empty_read_list:
            reply["need_evidence"] = []
        raw.append(deepcopy(reply))
        return reply

    async def run(test, c, e):
        verdict = (
            "fail" if c["products"][0]["artifact"]["final_response"] == "{" else "pass"
        )
        return {"test_id": test["test_id"], "criterion_id": "syntax", "status": verdict}

    result = await verify_issue(
        issue("whole_check"),
        PACK,
        "reward",
        context,
        evidence,
        SimpleNamespace(run=run, tools=MeasurementTools()),
        reference,
        record=lambda *_: None,
        adversarial_review_enabled=True,
    )
    assert len(calls) == 1
    assert result["status"] == "supported_on_cases"
    assert sorted(c["reference"]["status"] for c in result["cases"]) == [
        "fail",
        "pass",
        "pass",
    ]
    assert set(raw[0]) == (
        {"response", "need_evidence"} if empty_read_list else {"response"}
    )
    assert "response_recovery" not in calls[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "defect", ["missing_case", "unknown_case", "invalid_verdict", "invented_evidence"]
)
async def test_unwrapping_does_not_bypass_case_or_evidence_validation(defect):
    context, evidence = build_context(TASK, [candidate("{}")])
    calls = []

    async def reference(request):
        calls.append(deepcopy(request))
        payload = await parser_reference(request)
        if defect == "missing_case":
            payload["cases"].pop()
        elif defect == "unknown_case":
            payload["cases"][0]["id"] = "not-a-supplied-case"
        elif defect == "invalid_verdict":
            payload["cases"][0]["status"] = "approved"
        else:
            payload["cases"][0]["evidence_refs"] = ["fabricated-source"]
        return {"response": payload}

    async def run(*args):
        pytest.fail("Rejected reference must not dispatch checkers")

    with pytest.raises(ResponseContractError) as error:
        await verify_issue(
            issue("whole_check"),
            PACK,
            "reward",
            context,
            evidence,
            SimpleNamespace(run=run, tools=MeasurementTools()),
            reference,
            record=lambda *_: None,
            adversarial_review_enabled=True,
        )
    assert len(calls) == 2
    assert set(error.value.rejected_response) == {"response"}
    assert set(calls[1]["response_recovery"]["rejected_response"]) == {"response"}


@pytest.mark.parametrize(
    "reply,schema",
    [
        ({"response": {"cases": []}, "cases": ["conflict"]}, {"cases": []}),
        ({"response": {"response": {"cases": []}}}, {"cases": []}),
        ({"response": []}, {"cases": []}),
        ({"response": {"cases": []}}, {"response": {"cases": []}}),
    ],
)
def test_ambiguous_or_schema_owned_envelopes_are_not_reinterpreted(reply, schema):
    before = deepcopy(reply)
    assert response_payload(reply, schema) == before
    assert reply == before


@pytest.mark.asyncio
async def test_wrapped_evidence_request_uses_the_same_exact_evidence_protocol():
    context, evidence = build_context(TASK, [candidate("{}")])
    calls = []

    async def model(request):
        calls.append(deepcopy(request))
        return (
            {"response": {"need_evidence": ["task"]}}
            if len(calls) == 1
            else {"response": {"value": "done"}}
        )

    result = await ask(
        model, "fixture", context, evidence, "Return value", {"value": "text"}
    )
    assert result == {"value": "done"}
    assert calls[1]["requested_evidence"]["task"] == evidence["task"]
    assert "response_recovery" not in calls[1]
