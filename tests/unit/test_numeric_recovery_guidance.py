"""Executed numerical repairs retain the exact evidence selection."""
from copy import deepcopy

import pytest

from reward_engine.core.outcome.harness import unresolved_measurements
from reward_engine.core.outcome.measurement_requests import POLICY, preflight
from reward_engine.core.outcome.measurement_tools import MeasurementTools


@pytest.mark.asyncio
@pytest.mark.parametrize('selected', [False, True])
async def test_large_string_repair_uses_bound_inputs_and_selects_inside_program(selected):
    tools = MeasurementTools(role_contract_policy=POLICY)
    evidence = {'source': {'document': {'content': [{'text': 'x' * 12000 + 'needle'}]}}}
    paths = {'source': ['document']} if selected else {}
    args = {'program': "result = str(inputs['source'])", 'input_refs': ['source'],
            'input_paths': paths}
    failed = await tools.run_checked({'tool': 'code.run', 'arguments': args}, evidence)
    assert failed['status'] == 'unknown' and not failed['request_rejected']
    original = deepcopy(failed)
    hint = failed['recovery_hint']
    assert hint['required_input_bindings'] == failed['input_bindings']
    assert hint['selection_policy'] == 'preserve_failed_selection_select_inside_program'
    access = "inputs['source']" + ('' if selected else "['document']")
    corrected = await tools.run_checked({'tool': 'code.run', 'arguments': {
        **args, 'program': f"text = {access}['content'][0]['text']\nresult = text.find('needle')"}}, evidence)
    assert corrected['status'] == 'observed' and corrected['result'] == 12000
    result = {'evidence_refs': ['later'], 'measurement_recovery': [{
        'failed_ref': 'before', 'replacement_refs': ['later'], 'reason': 'Inspected bounded data directly'}]}
    assert not unresolved_measurements(result, {'before': failed, 'later': corrected})
    assert failed == original


@pytest.mark.asyncio
async def test_print_only_request_is_rejected_before_execution_and_can_be_corrected(monkeypatch):
    tools = MeasurementTools(role_contract_policy=POLICY)
    request = {'tool': 'code.run', 'arguments': {
        'program': 'values = [2, 3]\nprint(values)\nprint(len(values))', 'input_refs': []}}
    assert preflight(request) == []
    def forbidden(*args, **kwargs):
        pytest.fail('A program with no result binding must not execute')
    with monkeypatch.context() as patch:
        patch.setattr('reward_engine.core.outcome.measurement_tools.run_numeric', forbidden)
        rejected = await tools.run_checked(request, {})
    assert rejected['request_rejected'] and rejected['execution_phase'] == 'preflight'
    assert rejected['diagnostics'][0]['field'] == 'arguments.program'
    corrected = await tools.run_checked({'tool': 'code.run', 'arguments': {
        **request['arguments'], 'program': request['arguments']['program'] + '\nresult = len(values)'}}, {})
    assert corrected['status'] == 'observed'
    result = {'evidence_refs': ['later'], 'measurement_recovery': [{
        'failed_ref': 'before', 'replacement_refs': ['later'], 'reason': 'Assigned computed result'}]}
    assert not unresolved_measurements(result, {'before': rejected, 'later': corrected})
    historical = {**request, 'status': 'unknown', 'request_rejected': False,
                  'execution_phase': 'execution', 'contract_policy': POLICY, 'input_bindings': [],
                  'error_type': 'ValueError', 'reason': 'Program must assign result'}
    assert unresolved_measurements(result, {'before': historical, 'later': corrected}) == {'before'}


@pytest.mark.asyncio
async def test_loop_result_binding_remains_valid_and_unreached_binding_still_fails():
    tools = MeasurementTools(role_contract_policy=POLICY)
    valid = await tools.run_checked({'tool': 'code.run', 'arguments': {
        'program': 'for result in [1, 2]:\n    value = result', 'input_refs': []}}, {})
    assert valid['status'] == 'observed' and valid['result'] == 2
    failed = await tools.run_checked({'tool': 'code.run', 'arguments': {
        'program': 'if False:\n    result = 2', 'input_refs': []}}, {})
    assert failed['status'] == 'unknown' and not failed['request_rejected']
