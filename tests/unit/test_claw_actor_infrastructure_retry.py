"""Another infrastructure execution needs its own bound, single-use approval."""
import pytest

from reward_engine.adapters.claw_eval.actor_retry import (
    ActorRetries,
    verify_actor_retry,
)
from reward_engine.adapters.claw_eval.actor_transport_retry import (
    propose_transport_retry,
)
from reward_engine.adapters.workbuddy.io import read, write
from reward_engine.core.outcome.three_b import Stages
from tests.support.actor_infrastructure import third_failed_actor
from tests.support.actor_retry import approved, file_inventory

POLICY = 'one_terminated_infrastructure_actor_retry_v1'


def test_gateway_queue_timing_excludes_setup_before_process_start(tmp_path):
    config, _, _, _, third, _ = third_failed_actor(tmp_path, 'queue')
    write(third / 'STARTED.json', {'identity': 'actor', 'at': 80})
    proposal = propose_transport_retry(config, 'synthetic_task', 'round_1/B1', policy=POLICY)
    assert proposal['gateway_evidence']['fraction_native_budget'] == pytest.approx(.95)


@pytest.mark.asyncio
@pytest.mark.parametrize('kind,failed_cost', [('upstream', 38.1), ('queue', 57.0)])
async def test_authorized_third_failure_recovery_preserves_all_failed_costs(tmp_path, kind, failed_cost):
    config, root, original, second, third, stage = third_failed_actor(tmp_path, kind)
    before = [file_inventory(path) for path in (original, second, third)]
    inventory = file_inventory(tmp_path)
    proposal = propose_transport_retry(config, 'synthetic_task', 'round_1/B1', policy=POLICY)
    assert file_inventory(tmp_path) == inventory
    assert proposal['original_attempt'] == 3 and proposal['max_additional_attempts'] == 1
    assert proposal['failed_execution_s'] == pytest.approx(failed_cost)
    assert len(proposal['failed_executions_s']) == 3
    assert proposal['observed_session_id'] == 'third-native-session'
    assert proposal['native_session_id'] == ('third-native-session' if kind == 'upstream' else None)
    manager = ActorRetries(config, [approved(proposal)])
    stages = Stages({}, lambda k, v: write(root / (k + '.json'), v),
                    lambda k: read(root / (k + '.json')) if (root / (k + '.json')).exists() else None,
                    recover_actor=lambda name, saved: manager.recover('synthetic_task', name, saved))
    calls = []

    async def execute():
        slot = manager.slot('synthetic_task', 'round_1/B1', original)
        assert slot not in (original, second, third) and not slot.exists()
        assert manager.protocol_identity(slot, 'new-source') == 'earlier'
        manager.validate_slot(slot, 'actor')
        calls.append(slot)
        write(slot / 'RESULT.json', {'status': 'completed'})
        return {'status': 'completed', 'identity': 'actor', 'runtime_directory': str(slot),
                'actor_execution_s': 4, **manager.result_fields(slot, 4)}

    row = await stages.run('round_1/B1', {}, execute, actor=True)
    assert read(root / 'round_1/B1/stage.json')['attempt'] == 4
    assert row['actor_retry']['additional_attempts'] == 3
    assert row['actor_retry']['this_authorization_attempts'] == 1
    assert row['actor_retry']['total_execution_s'] == pytest.approx(failed_cost + 4)
    assert [file_inventory(path) for path in (original, second, third)] == before
    write(root / 'state.json', {'status': 'completed'})
    verify_actor_retry(row)
    assert await stages.run('round_1/B1', {}, execute, actor=True) == row and len(calls) == 1
    with pytest.raises(ValueError, match='consumed'):
        ActorRetries(config, [approved(proposal)]).recover('synthetic_task', 'round_1/B1', stage)
    row['actor_retry']['failed_execution_s'] = 0
    with pytest.raises(ValueError, match='cost'):
        verify_actor_retry(row)


@pytest.mark.parametrize('change', ['completed', 'snapshot', 'identity', 'history', 'prior_history',
                                   'prior_authorization', 'request_budget', 'host_budget', 'cleanup',
                                   'signals', 'nonserver', 'session', 'attempt'])
def test_third_native_failure_rejects_changed_or_delivered_evidence(tmp_path, change):
    config, root, _, _, third, stage = third_failed_actor(tmp_path)
    process = third / 'jiuwenswarm/process'
    if change in ('completed', 'snapshot'):
        write(third / ('RESULT.json' if change == 'completed' else 'snapshots/1/delivery.json'), {})
    else:
        path, edits = {
            'identity': (third / 'STARTED.json', {'identity': 'changed'}),
            'history': (root / 'round_1/B1/failures/3.json', {'reason': 'changed'}),
            'prior_history': (root / 'round_1/B1/failures/2.json', {'reason': 'changed'}),
            'prior_authorization': (third.parent / 'authorization.json', {'approval': None}),
            'request_budget': (process / 'turn-1.request.json', {'timeout_seconds': 11}),
            'host_budget': (process / 'turn-1.execution.json', {'host_timeout_s': 26}),
            'cleanup': (process / 'turn-1.execution.json', {'cleanup_error': 'failed'}),
            'signals': (process / 'turn-1.execution.json', {'signals': ['SIGTERM']}),
            'nonserver': (process / 'turn-1.result.json', {'error': {'code': 'RUNTIME_ERROR', 'message': 'Task failed'}}),
            'session': (third / 'FAILURE.json', {'session': 'changed'}),
            'attempt': (root / 'round_1/B1/stage.json', {'attempt': 4}),
        }[change]
        write(path, {**read(path), **edits})
    with pytest.raises(ValueError):
        propose_transport_retry(config, 'synthetic_task', 'round_1/B1', policy=POLICY)


@pytest.mark.parametrize('change', ['low_wait', 'overlap', 'wrong_actor', 'wrong_model', 'missing_receipt',
                                   'ordinary_timeout', 'completed_task'])
def test_host_timeout_requires_bound_gateway_queue_starvation(tmp_path, change):
    config, root, _, _, third, _ = third_failed_actor(tmp_path, 'queue')
    gateway = tmp_path / 'transport' / ('actor-' + '3' * 32)
    if change == 'ordinary_timeout':
        (third / 'trace.jsonl').unlink()
    elif change == 'completed_task':
        write(root / 'state.json', {'status': 'completed'})
    elif change == 'missing_receipt':
        (gateway / 'first/receipt.json').unlink()
    else:
        path, edits = {
            'low_wait': (gateway / 'last/receipt.json', {'finished_at': 107.5}),
            'overlap': (gateway / 'last/receipt.json', {'queued_at': 105}),
            'wrong_actor': (gateway / 'first/receipt.json', {'channel': 'actor-unrelated'}),
            'wrong_model': (gateway / 'first/receipt.json', {'model': 'unrelated-model'}),
        }[change]
        write(path, {**read(path), **edits})
    with pytest.raises(ValueError):
        propose_transport_retry(config, 'synthetic_task', 'round_1/B1', policy=POLICY)


def test_new_approval_cannot_be_inferred_or_rebound_to_another_budget(tmp_path):
    config, root, original, _, _, stage = third_failed_actor(tmp_path)
    proposal = propose_transport_retry(config, 'synthetic_task', 'round_1/B1', policy=POLICY)
    assert not ActorRetries(config, []).recover('synthetic_task', 'round_1/B1', stage)
    with pytest.raises(ValueError, match='approval'):
        ActorRetries(config, [{'proposal': proposal, 'approval': None}])
    with pytest.raises(ValueError, match='budget'):
        ActorRetries({**config, 'actor_timeout_s': 11}, [approved(proposal)])
    (original / 'new-source.txt').write_text('Changed original evidence')
    with pytest.raises(ValueError):
        ActorRetries(config, [approved(proposal)]).recover('synthetic_task', 'round_1/B1', stage)


@pytest.mark.parametrize('change', ['gateway', 'archived_failure', 'original_history'])
def test_completed_retry_rejects_changes_to_external_or_archived_evidence(tmp_path, change):
    config, root, original, _, _, stage = third_failed_actor(tmp_path, 'queue')
    proposal = propose_transport_retry(config, 'synthetic_task', 'round_1/B1', policy=POLICY)
    manager = ActorRetries(config, [approved(proposal)])
    manager.recover('synthetic_task', 'round_1/B1', stage)
    slot = manager.slot('synthetic_task', 'round_1/B1', original)
    row = {'identity': 'actor', 'runtime_directory': str(slot), 'actor_execution_s': 4,
           **manager.result_fields(slot, 4)}
    verify_actor_retry(row)
    path = {
        'gateway': tmp_path / 'transport' / ('actor-' + '3' * 32) / 'first/receipt.json',
        'archived_failure': slot.parent / 'previous_failure.json',
        'original_history': root / 'round_1/B1/failures/3.json',
    }[change]
    write(path, {**read(path), 'unexpected': 'Evidence changed after completion'})
    with pytest.raises(ValueError):
        verify_actor_retry(row)
