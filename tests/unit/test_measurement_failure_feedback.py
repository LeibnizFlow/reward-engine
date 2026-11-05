"""Executed failures retain output and guide an existing bounded correction wave."""

import json
from copy import deepcopy

import httpx
import pytest

from reward_engine.core.outcome.harness import defaults
from reward_engine.core.outcome.measurement_tools import MeasurementTools
from tests.support.role_transport import transport as transport
from tests.unit.test_claw_outcome_measurement_tools import case, verdict


@pytest.mark.asyncio
async def test_missing_result_keeps_stdout_without_promoting_it_to_a_result():
    receipt = await MeasurementTools().run(
        {"tool": "code.run", "arguments": {"program": "print(2 + 3)"}}, {}
    )
    assert receipt["status"] == "unknown" and not receipt["request_rejected"]
    assert receipt["reason"] == "Program must assign result"
    assert receipt["stdout"] == "5\n"
    assert "result" not in receipt


@pytest.mark.asyncio
@pytest.mark.parametrize("repair", [True, False])
async def test_failed_execution_guides_existing_tool_wave_without_waiving_failure(repair):
    context, evidence, _, test = case()
    source = "product_1:artifact:final_response"
    requests = []

    async def model(request):
        requests.append(deepcopy(request))
        if len(requests) == 1:
            return {"need_tools": [{"tool": "code.run", "arguments": {
                "program": "print(inputs['product_1:artifact:final_response'])", "input_refs": [source]}}]}
        feedback = request["measurement_recovery_required"]
        assert feedback["failed_measurements"][0]["failed_ref"] == "tool:check:0"
        assert feedback["failed_measurements"][0]["reason"] == "Program must assign result"
        assert feedback["failed_measurements"][0]["input_refs"] == [source]
        if repair and len(requests) == 2:
            assert request["measurement_budget"]["remaining_tool_waves"] == 1
            assert feedback["replacement_execution_allowed"] is True
            assert feedback["remaining_tool_calls"] == 3
            return {"need_tools": [{"tool": "code.run", "arguments": {
                "program": "result = 2 + 3", "input_refs": [source]}}]}
        answer = verdict("tool:check:1" if repair else None)
        if repair:
            answer["measurement_recovery"] = [{"failed_ref": "tool:check:0",
                "replacement_refs": ["tool:check:1"], "reason": "Executed corrected calculation on the same inputs"}]
        return answer

    model.bounded_stage = True
    result = await defaults(model).run(test, context, evidence)
    assert len(requests) == (3 if repair else 2)
    assert result["status"] == ("pass" if repair else "unknown")
    assert result["unresolved_measurement_refs"] == ([] if repair else ["tool:check:0"])
    assert result["tool_observations"]["tool:check:0"]["status"] == "unknown"
    assert result["execution_usage"]["tool_calls"] == (2 if repair else 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("broken_summary", [False, True])
async def test_incomplete_final_measurement_does_not_pin_failed_reply_on_resume(transport, broken_summary):
    role, responses, calls = transport
    role.config["role_stream"] = False
    role.bounded_stage = True
    context, evidence, _, test = case()
    source = "product_1:artifact:final_response"
    def reply(answer):
        return httpx.Response(200, json={"model": "test-model-a", "choices": [{
            "index": 0, "finish_reason": "stop", "message": {"content": json.dumps(answer)}}]})
    def probe(program):
        return {"need_tools": [{"tool": "code.run", "arguments": {"program": program, "input_refs": [source]}}]}
    final = verdict("tool:check:1")
    final["measurement_recovery"] = [{"failed_ref": "tool:check:0", "replacement_refs": ["tool:check:1"],
                                      "reason": "Executed corrected program over the same inputs"}]
    incomplete = verdict()
    if broken_summary:
        incomplete["status"] = "partial"
    responses.extend([reply(probe("print(2+3)")), reply(incomplete)])
    if broken_summary:
        responses.append(reply({"field_corrections": {"status": "partial", "reason": "Still invalid", "gap": ""}}))
    responses.extend([reply(probe("result = 2+3")), reply(final)])
    first = await defaults(role).run(test, context, deepcopy(evidence))
    assert first["unresolved_measurement_refs"] == ["tool:check:0"]
    originals = {p: p.read_bytes() for p in role.output.glob('*/response.json')}
    assert len(calls) == 2 + int(broken_summary)
    second = await defaults(role).run(test, context, deepcopy(evidence))
    assert second["status"] == "pass" and second["unresolved_measurement_refs"] == []
    assert len(calls) == 4 + int(broken_summary)
    assert all(p.read_bytes() == raw for p, raw in originals.items())
    assert len(list(role.output.glob('*/contract_rejected.json'))) == 1 + int(broken_summary)


@pytest.mark.asyncio
async def test_closed_source_reads_do_not_hide_remaining_replacement_tool_slots():
    from reward_engine.core.outcome.context import build_context

    context, evidence = build_context({"task_id": "synthetic", "prompt": "Check arithmetic"}, [{
        "artifact": {"final_response": "The answer is 5"},
        "source_documents": [{"text": "x" * 1000} for _ in range(3)]}])
    _, _, _, test = case()
    requests = []
    source = "product_1:artifact:final_response"
    async def model(request):
        requests.append(deepcopy(request))
        count = len(requests)
        if count == 1 or count == 5:
            if count == 5:
                assert request["evidence_access_closed"]
                assert request["measurement_recovery_required"]["replacement_execution_allowed"]
                assert request["measurement_recovery_required"]["remaining_tool_calls"] == 3
            return {"need_tools": [{"tool": "code.run", "arguments": {
                "program": "print(2+3)" if count == 1 else "result = 2+3", "input_refs": [source]}}]}
        if count < 5:
            return {"need_evidence": [f"product_1:source:{count - 2}"]}
        result = verdict("tool:check:1")
        result["measurement_recovery"] = [{"failed_ref": "tool:check:0",
            "replacement_refs": ["tool:check:1"], "reason": "Corrected computation on the same inputs"}]
        return result
    model.bounded_stage = True
    result = await defaults(model).run(test, context, evidence)
    assert result["status"] == "pass" and result["unresolved_measurement_refs"] == []
    assert result["execution_usage"] == {"model_requests": 6, "tool_calls": 2}
    assert "product_1:source:2" not in requests[-1]["requested_evidence"]
