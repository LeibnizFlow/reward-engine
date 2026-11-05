"""Synthetic regressions at the tool, recovery and complete assessment seams."""

import json
from copy import deepcopy

import pytest

from reward_engine.adapters.workbuddy.measurement import WorkBuddyTools
from reward_engine.core.outcome.authored_json import RECOVERY, VERSION
from reward_engine.core.outcome.harness import (
    defaults,
    measure,
    unresolved_measurements,
)
from reward_engine.core.outcome.measurement_tools import MeasurementTools
from tests.unit.test_claw_outcome_measurement_tools import case, verdict

POLICY = "canonical_recovery_v1"


def configured(cls=MeasurementTools):
    tools = cls()
    tools.role_contract_policy = POLICY
    return tools


@pytest.mark.asyncio
@pytest.mark.parametrize("tool,args", [
    ("calculate", {"expression": "1+2;3+4"}),
    ("calculate", {}),
    ("code.run", {"program": "result = ("}),
    ("code.run", {"program": "result = 1", "input_refs": ["r"], "input_paths": {"r": ["missing"]}}),
    ("workspace.read", {"paths": 42}),
    ("workspace.read", {"paths": ["*"], "offset": -1}),
])
async def test_non_executable_probe_is_rejected_before_execution(tool, args):
    receipt = await configured(WorkBuddyTools).run_checked(
        {"tool": tool, "arguments": args}, {"r": {"a": [1]}})
    assert receipt["request_rejected"] is True
    assert receipt["execution_phase"] == "preflight"
    assert receipt["diagnostics"][0]["field"].startswith("arguments")


@pytest.mark.asyncio
async def test_real_execution_failure_cannot_be_reclassified():
    receipt = await configured().run_checked(
        {"tool": "calculate", "arguments": {"expression": "1/0"}}, {})
    assert receipt["status"] == "unknown" and not receipt["request_rejected"]
    assert receipt["execution_phase"] == "execution"


@pytest.mark.asyncio
async def test_invalid_offset_after_external_fetch_is_an_executed_failure():
    calls = []
    async def reader(url):
        calls.append(url)
        return "short"
    tools = MeasurementTools(document_reader=reader, role_contract_policy=POLICY)
    receipt = await tools.run_checked({"tool": "document.read", "arguments": {
        "url": "https://synthetic.example/source", "offset": 100}},
        {"source": {"url": "https://synthetic.example/source"}})
    assert len(calls) == 1 and receipt["status"] == "unknown" and not receipt["request_rejected"]


@pytest.mark.asyncio
@pytest.mark.parametrize("path,accepted", [("a", True), ("b", False)])
async def test_recovery_binds_selected_data_not_only_reference(path, accepted):
    tools, evidence = configured(), {"r": {"a": [1], "b": [2]}}
    def probe(program, selection):
        return {"tool": "code.run", "arguments": {"program": program,
                "input_refs": ["r"], "input_paths": {"r": [selection]}}}
    failed = await tools.run_checked(probe("result = 1/0", "a"), evidence)
    later = await tools.run_checked(probe("result = inputs['r'][0]", path), evidence)
    result = {"evidence_refs": ["later"], "measurement_recovery": [{
        "failed_ref": "failed", "replacement_refs": ["later"], "reason": "Corrected program"}]}
    assert (not unresolved_measurements(result, {"failed": failed, "later": later})) == accepted


@pytest.mark.asyncio
async def test_recursive_glob_includes_root_and_nested_files():
    raw = json.dumps([{"path": p, "content": {"text": p}}
                      for p in ("report.md", "nested/report.md", "other.txt")])
    evidence = {"product_1:source:0": {"source": "synthetic", "offset": 0,
                "total_chars": len(raw), "text": raw}}
    receipt = await configured(WorkBuddyTools).run_checked(
        {"tool": "workspace.read", "arguments": {"paths": ["**/*.md"]}}, evidence)
    assert {r["path"] for r in json.loads(receipt["result"]["text"])["files"]} == {
        "report.md", "nested/report.md"}


@pytest.mark.asyncio
@pytest.mark.parametrize("pattern,reusable", [("**/*.md", False), ("nested/*.md", True)])
async def test_historical_glob_reuse_requires_equal_selection_without_reexecution(pattern, reusable):
    from reward_engine.core.outcome.resume_inputs import validate_measurement

    _, evidence, _, test = case()
    raw = json.dumps([{"path": p, "content": {"text": p}}
                      for p in ("report.md", "nested/report.md")])
    evidence["product_1:source:0"] = {"source": "synthetic", "offset": 0, "total_chars": len(raw), "text": raw}
    receipt = await WorkBuddyTools().run_checked({"tool": "workspace.read", "arguments": {"paths": [pattern]}}, evidence)
    row = {**verdict(), "test_id": test["test_id"], "criterion_id": test["criterion_id"],
           "checker": test["checker"], "measurement_kind": "model_assisted", "tool_observations": {"tool:check:0": receipt}}
    class NoExecution(WorkBuddyTools):
        async def run(self, request, evidence):
            pytest.fail("Historical validation must not execute tools")
    tools = NoExecution(role_contract_policy=POLICY)
    if reusable:
        assert validate_measurement(row, test, evidence, policy=POLICY, tools=tools) == row
    else:
        with pytest.raises(ValueError, match="different files"):
            validate_measurement(row, test, evidence, policy=POLICY, tools=tools)


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_expression", ["2+3;4+5", "undefined_cost+3"])
async def test_syntax_rejection_then_full_verdict_correction_preserves_receipts(bad_expression):
    context, evidence, criterion, test = case()
    requests = []
    async def model(request):
        requests.append(deepcopy(request))
        if len(requests) <= 2:
            if len(requests) == 2:
                assert request["measurement_budget"]["remaining_tool_calls"] == 4
                assert request["context"]["tool_observations"]["tool:check:0"]["request_rejected"]
            return {"need_tools": [{"tool": "calculate", "arguments": {
                "expression": bad_expression if len(requests) == 1 else "2+3"}}]}
        answer = verdict("tool:check:1")
        if len(requests) == 3:
            answer["status"] = "partial"
        return answer
    model.role_output_protocol, model.role_recovery_policy = VERSION, RECOVERY
    model.role_contract_policy = POLICY
    model.bounded_stage = True
    registry = defaults(model)
    pack = {"task_id": context["task"]["task_id"], "task_prompt": context["task"]["prompt"],
            "pack_id": "synthetic", "criteria": [criterion], "tests": [test]}
    measured = await measure(pack, registry, context, evidence)
    result = measured["results"][0]
    assert result["status"] == "pass"
    assert result["execution_usage"] == {"model_requests": 4, "tool_calls": 1}
    assert result["tool_observations"]["tool:check:0"]["request_rejected"]
    assert not result["unresolved_measurement_refs"]
    resumed = await measure(pack, registry, context, evidence, completed=measured)
    assert len(requests) == 4 and resumed["results"][0]["reused_completed_measurement"]
    changed = {**evidence, "task": "changed task input"}
    with pytest.raises(ValueError, match="inputs changed"):
        await measure(pack, registry, context, changed, completed=measured)
    assert len(requests) == 4


@pytest.mark.asyncio
async def test_exhausted_unrecoverable_execution_stops_without_futile_corrections():
    context, evidence, _, test = case()
    calls = []
    async def model(request):
        calls.append(deepcopy(request))
        if len(calls) <= 2:
            return {"need_tools": [{"tool": "calculate", "arguments": {"expression": "1/0"}}]}
        return verdict()
    model.role_output_protocol, model.role_recovery_policy = VERSION, RECOVERY
    model.role_contract_policy, model.bounded_stage = POLICY, True
    result = await defaults(model).run(test, context, evidence)
    assert len(calls) == 3
    assert result["error"] and result["status"] == "unknown"
    assert len(result["tool_observations"]) == 2
    assert result["diagnostic_details"]["code"] == "MEASUREMENT_RECOVERY_EXHAUSTED"


@pytest.mark.asyncio
async def test_transport_failure_keeps_preceding_measurements():
    context, evidence, _, test = case()
    calls = []
    async def model(request):
        calls.append(request)
        if len(calls) == 1:
            return {"need_tools": [{"tool": "calculate", "arguments": {"expression": "2+3"}}]}
        raise TimeoutError("synthetic transport timeout")
    model.role_output_protocol, model.role_recovery_policy = VERSION, RECOVERY
    model.role_contract_policy, model.bounded_stage = POLICY, True
    result = await defaults(model).run(test, context, evidence)
    assert result["error"] and result["error_type"] == "TimeoutError"
    assert result["tool_observations"]["tool:check:0"]["result"] == 5


@pytest.mark.asyncio
async def test_completed_measurement_revalidates_without_calls(tmp_path):
    from reward_engine.core.outcome.completed_operations import CompletedOperations
    from reward_engine.core.outcome.resume_inputs import validate_measurement

    context, evidence, _, test = case()
    calls = []
    async def model(request):
        calls.append(request)
        return verdict()
    model.role_output_protocol, model.role_recovery_policy = VERSION, RECOVERY
    model.role_contract_policy, model.bounded_stage = POLICY, True
    model.completed_operations = CompletedOperations(tmp_path/'operations', {"config_identity": "synthetic"})
    registry = defaults(model)
    first = await registry.run(test, context, evidence)
    assert not first.get("error")
    assert await registry.run(test, context, evidence) == first
    assert len(calls) == 1
    bad = deepcopy(first)
    bad["evidence_refs"] = ["task"]
    with pytest.raises(ValueError, match="product evidence"):
        validate_measurement(bad, test, evidence, policy=POLICY, tools=registry.tools)


@pytest.mark.asyncio
async def test_resume_reconstructs_exact_installed_standard_view():
    from reward_engine.core.outcome.context import measurement_test_view
    from reward_engine.core.outcome.resume_inputs import validate_measurement

    context, evidence, _, test = case()
    test["rubric"]["source_expectation_ids"] = ["historical-binding"]
    visible = measurement_test_view(test)
    supplied = {**evidence, test["test_id"]: visible, visible["rubric"]["id"]: visible["rubric"]}
    tools = configured()
    receipt = await tools.run_checked({"tool": "code.run", "arguments": {
        "input_refs": [test["test_id"]], "program": "result = len(inputs['check']['rubric']['standard'])"}}, supplied)
    assert receipt["status"] == "observed"
    row = {**verdict(), "test_id": test["test_id"], "criterion_id": test["criterion_id"],
           "checker": test["checker"], "measurement_kind": "model_assisted", "tool_observations": {"tool:check:0": receipt}}
    assert validate_measurement(row, test, evidence, policy=POLICY, tools=tools) == row
