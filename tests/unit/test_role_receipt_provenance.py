"""A missing client acknowledgement requires complete server admission proof."""
import pytest

from reward_engine.adapters.workbuddy.io import read, write
from reward_engine.adapters.workbuddy.role_audit import request_audit


@pytest.fixture
def correlated_request(tmp_path):
    operation, physical = 'a' * 32, 'b' * 32
    gateway = tmp_path / 'transport/roles' / ('c' * 32) / 'receipt.json'
    budget = tmp_path / 'task/operation_budgets' / (operation + '.json')
    ledger = tmp_path / 'transport/role_operations' / operation / 'admissions.json'
    write(gateway, {'logical_operation_id': operation, 'physical_request_id': physical,
        'contract_policy': 'canonical_recovery_v1', 'config_identity': 'synthetic',
        'status': 'completed', 'attempts': 1, 'upstream_attempts': 1,
        'operation_started_at': 100, 'deadline': 1900, 'admitted_at': 110,
        'upstream_started_at': 120, 'finished_at': 150})
    write(budget, {'logical_operation_id': operation, 'config_identity': 'synthetic',
        'contract_policy': 'canonical_recovery_v1', 'requests': 1,
        'started_at': 100, 'deadline': 1900, 'receipts': [{
            'logical_operation_id': operation, 'physical_request_id': physical,
            'status': 'dispatched', 'deadline': 1900, 'gateway_calls': 1}]})
    write(ledger, {'deadline': 1900, 'started_at': 100, 'config_identity': 'synthetic',
        'requests': {physical: str(gateway)}, 'admitted_at': {physical: 110}})
    write(tmp_path / 'transport/role_budget_index' / (operation + '.json'),
          {'receipt': str(budget), 'config_identity': 'synthetic'})
    return tmp_path, operation, physical, gateway, budget, ledger


def test_lost_client_pointer_is_proven_from_unchanged_admission_ledger(correlated_request):
    root, operation, _, gateway, budget, ledger = correlated_request
    original = {p: p.read_bytes() for p in (gateway, budget, ledger)}
    audit = request_audit(root, config_identity='synthetic')
    assert audit['all_current_contract_budgets_verified'], audit['operations'][operation]['audit_issues']
    assert audit['operations'][operation]['recovered_gateway_pointers']
    assert all(p.read_bytes() == content for p, content in original.items())


@pytest.mark.parametrize('mutation', ['no_ledger', 'wrong_path', 'wrong_physical',
    'wrong_operation', 'wrong_config', 'wrong_deadline', 'wrong_admission',
    'outside_campaign', 'unfinished', 'bad_attempts', 'late_success', 'explicit_wrong_pointer'])
def test_incomplete_or_conflicting_provenance_still_fails(correlated_request, mutation):
    root, _, physical, gateway, budget, ledger = correlated_request
    if mutation == 'no_ledger':
        ledger.unlink()
    elif mutation in {'wrong_path', 'outside_campaign'}:
        value = read(ledger)
        value['requests'][physical] = str(root / ('missing.json' if mutation == 'wrong_path' else '../outside.json'))
        write(ledger, value)
    elif mutation == 'wrong_admission':
        value = read(ledger)
        value['admitted_at'][physical] = 99
        write(ledger, value)
    elif mutation == 'explicit_wrong_pointer':
        value = read(budget)
        value['receipts'][0]['gateway_receipt'] = str(root / 'missing.json')
        write(budget, value)
    else:
        field, value = {'wrong_physical': ('physical_request_id', 'd' * 32),
            'wrong_operation': ('logical_operation_id', 'd' * 32),
            'wrong_config': ('config_identity', 'different'),
            'wrong_deadline': ('deadline', 2000), 'unfinished': ('status', 'running'),
            'bad_attempts': ('upstream_attempts', 2), 'late_success': ('finished_at', 1901)}[mutation]
        write(gateway, {**read(gateway), field: value})
    assert not request_audit(root, config_identity='synthetic')['all_current_contract_budgets_verified']
