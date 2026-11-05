"""The same input check proves live and historical adapter-level rejections."""
import json
from copy import deepcopy

import pytest

from reward_engine.adapters.workbuddy.measurement import WorkBuddyTools
from reward_engine.core.outcome.authored_json import RECOVERY, VERSION
from reward_engine.core.outcome.construction import fingerprint
from reward_engine.core.outcome.harness import defaults
from reward_engine.core.outcome.measurement_requests import POLICY
from reward_engine.core.outcome.resume_inputs import validate_measurement
from tests.unit.test_claw_outcome_measurement_tools import case, verdict


def inputs():
    context, evidence, _, test = case()
    raw = json.dumps([{'path': 'report.md', 'changed': True,
                       'content': {'text': 'A short synthetic report'}}])
    evidence['product_1:source:0'] = {'source': 'synthetic', 'offset': 0,
                                     'total_chars': len(raw), 'text': raw}
    return context, evidence, test


@pytest.mark.asyncio
@pytest.mark.parametrize('tool', ['workspace.read', 'workspace.inspect'])
async def test_offset_rejection_is_proven_without_reexecuting_a_read(tool):
    _, evidence, test = inputs()
    tools = WorkBuddyTools(role_contract_policy=POLICY)
    receipt = await tools.run_checked({'tool': tool, 'arguments': {'paths': ['report.md'], 'offset': 10000}}, evidence)
    assert receipt['request_rejected'] and receipt['execution_phase'] == 'preflight'
    row = {**verdict(), **{k: test[k] for k in ('test_id', 'criterion_id', 'checker')},
           'measurement_kind': 'model_assisted', 'tool_observations': {'tool:check:0': receipt}}
    original = deepcopy(row)

    class NoExecution(WorkBuddyTools):
        async def run(self, *args):
            pytest.fail('Revalidation must not execute a tool')

    audit = NoExecution(role_contract_policy=POLICY)
    assert validate_measurement(row, test, evidence, policy=POLICY, tools=audit) == original
    assert row == original
    forged = deepcopy(row)
    forged['tool_observations']['tool:check:0']['arguments']['offset'] = 0
    with pytest.raises(ValueError, match='Historical rejection is not proven'):
        validate_measurement(forged, test, evidence, policy=POLICY, tools=audit)


@pytest.mark.asyncio
async def test_out_of_range_workspace_offset_never_enters_execution():
    _, evidence, _ = inputs()

    class NoExecution(WorkBuddyTools):
        async def run(self, *args):
            pytest.fail('Invalid workspace offset must be caught at admission')

    receipt = await NoExecution(role_contract_policy=POLICY).run_checked(
        {'tool': 'workspace.read', 'arguments': {'offset': 10000}}, evidence)
    assert receipt['request_rejected'] and receipt['execution_phase'] == 'preflight'


@pytest.mark.asyncio
async def test_adapter_rejection_does_not_block_the_last_valid_execution_slot():
    context, evidence, test = inputs()
    test['execution_recipe'] = {'probes': [
        {'tool': 'calculate', 'arguments': {'expression': '1+1'}} for _ in range(3)]}
    calls = []

    async def model(request):
        calls.append(deepcopy(request))
        if len(calls) == 1:
            assert request['measurement_budget']['remaining_tool_calls'] == 1
            return {'need_tools': [
                {'tool': 'workspace.read', 'arguments': {'offset': 10000}},
                {'tool': 'calculate', 'arguments': {'expression': '2+3'}}]}
        assert request['context']['tool_observations']['tool:check:3']['request_rejected']
        assert request['context']['tool_observations']['tool:check:4']['result'] == 5
        return verdict('tool:check:4')

    model.role_output_protocol, model.role_recovery_policy = VERSION, RECOVERY
    model.role_contract_policy, model.bounded_stage = POLICY, True
    result = await defaults(model, tools=WorkBuddyTools()).run(test, context, evidence)
    assert result['status'] == 'pass' and not result.get('error')
    assert len(calls) == 2 and result['execution_usage']['tool_calls'] == 4


@pytest.mark.asyncio
@pytest.mark.parametrize('changed_selection', [False, True])
async def test_stronger_admission_keeps_proven_recovery_of_an_old_executed_failure(changed_selection):
    _, evidence, test = inputs()
    source = 'product_1:artifact:final_response'
    historical = {'tool': 'code.run', 'arguments': {
        'program': 'result = []\nfor key, value in [[1, 2]]:\n    result.append(value)',
        'input_refs': [source]}, 'status': 'unknown', 'request_rejected': False,
        'execution_phase': 'execution', 'contract_policy': POLICY,
        'reason': 'Unsupported statement: For', 'error_type': 'ValueError',
        'input_bindings': [{'ref': source, 'path': [], 'sha256': fingerprint(evidence[source])}]}
    tools = WorkBuddyTools(role_contract_policy=POLICY)
    replacement = await tools.run_checked({'tool': 'code.run', 'arguments': {
        'program': 'result = 2', 'input_refs': ['task'] if changed_selection else [source]}}, evidence)
    row = {**verdict('tool:check:1'), **{k: test[k] for k in ('test_id', 'criterion_id', 'checker')},
           'measurement_kind': 'model_assisted',
           'tool_observations': {'tool:check:0': historical, 'tool:check:1': replacement},
           'measurement_recovery': [{'failed_ref': 'tool:check:0', 'replacement_refs': ['tool:check:1'],
                                    'reason': 'Corrected the program with the same inputs'}]}
    original = deepcopy(row)
    if changed_selection:
        with pytest.raises(ValueError, match='Historical execution recovery is not proven'):
            validate_measurement(row, test, evidence, policy=POLICY, tools=tools)
    else:
        assert validate_measurement(row, test, evidence, policy=POLICY, tools=tools) == original
    assert row == original
    # An allegedly successful unsupported program still fails current validation.
    forged = deepcopy(row)
    forged['tool_observations']['tool:check:0']['status'] = 'observed'
    with pytest.raises(ValueError, match='Historical executed result fails current preflight'):
        validate_measurement(forged, test, evidence, policy=POLICY, tools=tools)
