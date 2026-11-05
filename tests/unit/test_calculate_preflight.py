"""Reject provably unbound calculator arguments without erasing execution failures."""
from copy import deepcopy

import pytest

from reward_engine.core.outcome.harness import unresolved_measurements
from reward_engine.core.outcome.measurement_requests import POLICY, preflight
from reward_engine.core.outcome.measurement_tools import (
    MeasurementTools,
    validate_probe_arguments,
)


@pytest.mark.asyncio
async def test_unbound_calculation_never_executes_and_keeps_symbolic_recipe(monkeypatch):
    request = {'tool': 'calculate', 'arguments': {'expression': 'abs(cost-price)/price'}}
    original = deepcopy(request)
    # Published recipes are retained. Their concrete invocation is validated at runtime.
    validate_probe_arguments(request, policy=POLICY)
    assert preflight(request) == []
    def forbidden(*args, **kwargs):
        pytest.fail('An unbound calculator request must not execute')
    monkeypatch.setattr('reward_engine.core.outcome.measurement_tools.run_numeric', forbidden)
    receipt = await MeasurementTools(role_contract_policy=POLICY).run_checked(request, {})
    assert receipt['request_rejected'] and receipt['execution_phase'] == 'preflight'
    assert receipt['diagnostics'][0]['actual'] == ['cost', 'price']
    assert receipt['diagnostics'][0]['field'] == 'arguments.expression'
    assert request == original


@pytest.mark.asyncio
async def test_literal_correction_cannot_erase_a_historical_executed_failure():
    tools = MeasurementTools(role_contract_policy=POLICY)
    request = {'tool': 'calculate', 'arguments': {'expression': 'cost-price'}}
    rejected = await tools.run_checked(request, {})
    observed = await tools.run_checked({'tool': 'calculate', 'arguments': {'expression': '12-10'}}, {})
    assert observed['status'] == 'observed' and observed['result'] == 2
    result = {'evidence_refs': ['later'], 'measurement_recovery': [
        {'failed_ref': 'before', 'replacement_refs': ['later'], 'reason': 'Corrected the request'}]}
    assert not unresolved_measurements(result, {'before': rejected, 'later': observed})
    historical = {**request, 'status': 'unknown', 'request_rejected': False,
                  'execution_phase': 'execution', 'contract_policy': POLICY,
                  'input_bindings': [], 'error_type': 'KeyError', 'reason': 'cost'}
    original = deepcopy(historical)
    assert unresolved_measurements(result, {'before': historical, 'later': observed}) == {'before'}
    assert historical == original


@pytest.mark.asyncio
@pytest.mark.parametrize('expression,expected', [
    ('abs(12-10)/10', 0.2),
    ('[x*x for x in [1,2,3]]', [1, 4, 9]),
    ('[missing for x in []]', []),
    ('1 if True else missing', 1),
    ('False and missing', False),
    ('True or missing', True),
    ('3 < 2 < missing', False),
    ('len(inputs)', 0),
])
async def test_successful_expressions_remain_valid(expression, expected):
    request = {'tool': 'calculate', 'arguments': {'expression': expression}}
    receipt = await MeasurementTools(role_contract_policy=POLICY).run_checked(request, {})
    assert receipt['status'] == 'observed' and receipt['result'] == expected
    assert not receipt['request_rejected']


@pytest.mark.asyncio
async def test_division_by_zero_remains_an_executed_failure():
    receipt = await MeasurementTools(role_contract_policy=POLICY).run_checked(
        {'tool': 'calculate', 'arguments': {'expression': '1/0'}}, {})
    assert receipt['status'] == 'unknown' and not receipt['request_rejected']
    assert receipt['execution_phase'] == 'execution'


@pytest.mark.asyncio
@pytest.mark.parametrize('expression', [
    '(300/200)^(1/2)-1', '3.0 & 2', '2 | 1.5', '2 << 1.5', '2.0 >> 1',
])
async def test_float_bitwise_request_is_rejected_without_execution(monkeypatch, expression):
    request = {'tool': 'calculate', 'arguments': {'expression': expression}}
    validate_probe_arguments(request, policy=POLICY)
    def forbidden(*args, **kwargs):
        pytest.fail('Statically invalid bitwise arithmetic must not execute')
    monkeypatch.setattr('reward_engine.core.outcome.measurement_tools.run_numeric', forbidden)
    receipt = await MeasurementTools(role_contract_policy=POLICY).run_checked(request, {})
    assert receipt['request_rejected'] and receipt['execution_phase'] == 'preflight'
    assert receipt['diagnostics'][0]['field'] == 'arguments.expression'


@pytest.mark.asyncio
@pytest.mark.parametrize('expression,expected', [
    ('3 ^ 1', 2), ('3 & 1', 1), ('2 | 1', 3), ('2 << 1', 4),
    ('2 >> 1', 1), ('int(3.5) ^ 1', 2), ('(300/200)**(1/2)-1', (1.5 ** 0.5) - 1),
    ('1 if True else 1.5 ^ 2', 1), ('False and (1.5 ^ 2)', False),
    ('[1.5 ^ x for x in []]', []),
])
async def test_bitwise_and_conditional_calculations_stay_valid(expression, expected):
    receipt = await MeasurementTools(role_contract_policy=POLICY).run_checked(
        {'tool': 'calculate', 'arguments': {'expression': expression}}, {})
    assert receipt['status'] == 'observed' and receipt['result'] == expected


@pytest.mark.asyncio
async def test_preflight_power_correction_does_not_reclassify_historical_xor_failure():
    tools = MeasurementTools(role_contract_policy=POLICY)
    bad = {'tool': 'calculate', 'arguments': {'expression': '(300/200)^(1/2)-1'}}
    rejected = await tools.run_checked(bad, {})
    corrected = await tools.run_checked(
        {'tool': 'calculate', 'arguments': {'expression': '(300/200)**(1/2)-1'}}, {})
    result = {'evidence_refs': ['later'], 'measurement_recovery': [
        {'failed_ref': 'before', 'replacement_refs': ['later'], 'reason': 'Corrected power syntax'}]}
    assert not unresolved_measurements(result, {'before': rejected, 'later': corrected})
    historical = {**bad, 'status': 'unknown', 'request_rejected': False,
                  'execution_phase': 'execution', 'contract_policy': POLICY,
                  'input_bindings': [], 'error_type': 'TypeError', 'reason': 'float operands'}
    assert unresolved_measurements(result, {'before': historical, 'later': corrected}) == {'before'}
