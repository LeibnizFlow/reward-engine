"""Task verification need not rescan unrelated campaign accounting."""
import pytest

from reward_engine.adapters.jiuwenswarm.checkpoints import artifact_bindings
from reward_engine.adapters.jiuwenswarm.report import report
from reward_engine.adapters.workbuddy.io import file_hash, write
from reward_engine.core.outcome.construction import fingerprint


def config(tmp_path):
    return {'output': str(tmp_path), 'task_ids': ['synthetic'], 'rounds': 1}


def test_task_only_report_does_not_read_unrelated_transport(tmp_path):
    receipt = tmp_path / 'transport/roles/unrelated/receipt.json'
    receipt.parent.mkdir(parents=True)
    receipt.write_text('unrelated incomplete accounting write')
    result = report(config(tmp_path), tmp_path / 'task-report', campaign_accounting=False)
    assert result['expected_tasks'] == 1 and result['completed_tasks'] == 0
    assert result['tasks'][0]['task_id'] == 'synthetic'
    assert result['usage'] is None and result['role_request_audit'] is None
    assert 'full campaign audit remains required' in result['campaign_accounting_scope']
    with pytest.raises(ValueError):
        report(config(tmp_path), tmp_path / 'full-report')


def test_default_campaign_accounting_still_collects_usage_and_audit(tmp_path):
    write(tmp_path / 'transport/roles/physical/receipt.json', {
        'status': 'completed', 'attempts': 1,
        'usage': {'prompt_tokens': 5, 'completion_tokens': 7},
    })
    result = report(config(tmp_path), tmp_path / 'default')
    assert result == report(config(tmp_path), tmp_path / 'explicit', campaign_accounting=True)
    assert result['usage']['transport']['observed_prompt_tokens'] == 5
    assert result['usage']['transport']['observed_completion_tokens'] == 7
    assert isinstance(result['role_request_audit'], dict)
    assert 'campaign_accounting_scope' not in result


@pytest.mark.parametrize('mutation', ['stage_hash', 'native_file'])
def test_task_only_report_retains_stage_and_native_byte_checks(tmp_path, mutation):
    root = tmp_path / 'tasks/synthetic'
    actor = root / 'seeds/1'
    snapshot = actor / 'snapshots/1'
    (snapshot / 'workspace').mkdir(parents=True)
    delivery = snapshot / 'workspace/output.txt'
    delivery.write_text('synthetic delivery')
    candidate = {'artifact': {'final_response': 'synthetic'}}
    write(snapshot / 'candidate.json', candidate)
    write(snapshot / 'inventory.json', [{'path': 'output.txt', 'sha256': file_hash(delivery)}])
    value = {'runtime_directory': str(actor), 'session': 'synthetic-session',
             'candidate': candidate, 'resources': {'turns': 1},
             'artifact_bindings': artifact_bindings(actor)}
    write(actor / 'RESULT.json', value)
    stage = {'status': 'completed', 'value': value, 'value_sha256': fingerprint(value)}
    write(root / 'seed_0/stage.json', stage)
    assert report(config(tmp_path), tmp_path / 'valid', campaign_accounting=False)['completed_actors'] == 1
    if mutation == 'stage_hash':
        write(root / 'seed_0/stage.json', {**stage, 'value_sha256': 'changed'})
    else:
        delivery.write_text('changed after completion')
    with pytest.raises(ValueError):
        report(config(tmp_path), tmp_path / 'invalid', campaign_accounting=False)
