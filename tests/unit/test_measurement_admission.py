"""Invalid programs do not spend execution waves; real failures still need proof."""
from copy import deepcopy

import pytest

from reward_engine.core.outcome.authored_json import RECOVERY, VERSION
from reward_engine.core.outcome.harness import defaults
from reward_engine.core.outcome.measurement_requests import POLICY
from reward_engine.core.outcome.measurement_tools import MeasurementTools
from tests.unit.test_claw_outcome_measurement_tools import case, verdict


@pytest.mark.asyncio
@pytest.mark.parametrize('program', [
    "result = []\nfor key, value in [['a', 1]]:\n    result.append(value)",
    "result = ','.join(['a', 'b'])",
    "result = 1\nresult += 2",
    "result = sum([1, 2])",
    "import math\nresult = 1",
])
async def test_unsupported_program_is_rejected_without_execution(monkeypatch, program):
    def forbidden(*args, **kwargs):
        pytest.fail('Unsupported syntax must not execute')
    monkeypatch.setattr('reward_engine.core.outcome.measurement_tools.run_numeric', forbidden)
    request = {'tool': 'code.run', 'arguments': {'program': program}}
    original = deepcopy(request)
    receipt = await MeasurementTools(role_contract_policy=POLICY).run_checked(request, {})
    assert receipt['request_rejected'] and receipt['execution_phase'] == 'preflight'
    assert receipt['diagnostics'][0]['field'] == 'arguments.program'
    assert request == original


@pytest.mark.asyncio
async def test_rejected_wave_preserves_both_execution_waves_for_actual_recovery():
    context, evidence, _, test = case()
    requests = []
    source = 'product_1:artifact:final_response'

    async def model(request):
        requests.append(deepcopy(request))
        number = len(requests)
        if number <= 3:
            if number == 2:
                assert request['measurement_budget']['remaining_tool_waves'] == 2
            if number == 3:
                assert request['measurement_budget']['remaining_tool_waves'] == 1
            return {'need_tools': [{'tool': 'code.run', 'arguments': {
                'program': ['print(5)', 'result = 1/0', 'result = 5'][number - 1],
                'input_refs': [source]}}]}
        answer = verdict('tool:check:2')
        answer['measurement_recovery'] = [{'failed_ref': 'tool:check:1',
            'replacement_refs': ['tool:check:2'], 'reason': 'Corrected execution on the same inputs'}]
        return answer

    model.role_output_protocol, model.role_recovery_policy = VERSION, RECOVERY
    model.role_contract_policy, model.bounded_stage = POLICY, True
    result = await defaults(model).run(test, context, evidence)
    assert result['status'] == 'pass' and not result.get('error')
    assert len(requests) == 4
    assert result['execution_usage'] == {'model_requests': 4, 'tool_calls': 2}
    assert result['tool_observations']['tool:check:0']['request_rejected']
    assert not result['tool_observations']['tool:check:1']['request_rejected']
    assert not result['unresolved_measurement_refs']


@pytest.mark.asyncio
async def test_repeated_preflight_rejections_remain_bounded():
    context, evidence, _, test = case()
    calls = []

    async def model(request):
        calls.append(request)
        return {'need_tools': [{'tool': 'code.run', 'arguments': {'program': 'print(1)'}}]}

    model.role_output_protocol, model.role_recovery_policy = VERSION, RECOVERY
    model.role_contract_policy, model.bounded_stage = POLICY, True
    result = await defaults(model).run(test, context, evidence)
    assert result['error'] and result['status'] == 'unknown'
    assert len(calls) <= 8
    assert result['execution_usage']['tool_calls'] == 0
    assert len(result['tool_observations']) <= 8


@pytest.mark.asyncio
@pytest.mark.parametrize('program,expected', [
    ('result = 1 if True else missing()', 1),
    ('result = False and missing()', False),
    ('result = [missing() for item in []]', []),
    ('result = 1\nif False:\n    import missing', 1),
    ('result = []\nfor x in [1, 2]:\n    result.append(x)', [1, 2]),
])
async def test_preflight_does_not_speculate_about_unexecuted_branches(program, expected):
    receipt = await MeasurementTools(role_contract_policy=POLICY).run_checked(
        {'tool': 'code.run', 'arguments': {'program': program}}, {})
    assert receipt['status'] == 'observed' and receipt['result'] == expected
