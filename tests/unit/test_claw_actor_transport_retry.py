"""A new transport retry preserves both failures and cannot silently repeat."""
import json
import shutil

import pytest

from reward_engine.adapters.claw_eval.actor_retry import (
    UPSTREAM_POLICY,
    ActorRetries,
    verify_actor_retry,
)
from reward_engine.adapters.claw_eval.actor_transport_retry import (
    propose_transport_retry,
)
from reward_engine.adapters.workbuddy.io import read, write
from reward_engine.core.outcome.three_b import Stages
from tests.support.actor_retry import (
    approved,
    file_inventory,
    native_failure,
    native_upstream_failure,
)


@pytest.mark.asyncio
async def test_authorized_transport_retry_preserves_lineage_and_total_cost(tmp_path):
    config, root, original, failed, stage = native_failure(tmp_path)
    original_files, failed_files = file_inventory(original), file_inventory(failed)
    proposal = propose_transport_retry(config, 'synthetic_task', 'round_1/B1')
    assert proposal['failed_execution_s'] == 31.6
    manager = ActorRetries(config, [approved(proposal)])
    stages = Stages({}, lambda k, v: write(root / (k + '.json'), v),
                    lambda k: read(root / (k + '.json')) if (root / (k + '.json')).exists() else None,
                    recover_actor=lambda name, saved: manager.recover('synthetic_task', name, saved))
    calls = []

    async def execute():
        slot = manager.slot('synthetic_task', 'round_1/B1', original)
        assert slot not in (original, failed) and not slot.exists()
        assert manager.protocol_identity(slot, 'latest-code') == 'earlier'
        manager.validate_slot(slot, 'actor')
        calls.append(slot)
        write(slot / 'RESULT.json', {'status': 'completed'})
        return {'status': 'completed', 'identity': 'actor', 'runtime_directory': str(slot),
                'actor_execution_s': 4, **manager.result_fields(slot, 4)}

    result = await stages.run('round_1/B1', {}, execute, actor=True)
    assert read(root / 'round_1/B1/stage.json')['attempt'] == 3
    assert result['actor_retry']['additional_attempts'] == 2
    assert result['actor_retry']['this_authorization_attempts'] == 1
    assert result['actor_retry']['total_execution_s'] == 35.6
    assert file_inventory(original) == original_files
    assert file_inventory(failed) == failed_files
    verify_actor_retry(result)
    assert await stages.run('round_1/B1', {}, execute, actor=True) == result
    assert len(calls) == 1
    with pytest.raises(ValueError, match='consumed'):
        ActorRetries(config, [approved(proposal)]).recover('synthetic_task', 'round_1/B1', stage)
    result['actor_retry']['failed_execution_s'] = 6.5
    with pytest.raises(ValueError, match='cost'):
        verify_actor_retry(result)


def test_transport_proposal_does_not_approve_or_launch(tmp_path):
    config, root, _, _, stage = native_failure(tmp_path)
    before = file_inventory(root)
    proposal = propose_transport_retry(config, 'synthetic_task', 'round_1/B1')
    assert file_inventory(root) == before
    assert not ActorRetries(config, []).recover('synthetic_task', 'round_1/B1', stage)
    with pytest.raises(ValueError, match='approval'):
        ActorRetries(config, [{'proposal': proposal, 'approval': None}])
    with pytest.raises(ValueError, match='budget'):
        ActorRetries({**config, 'actor_timeout_s': 11}, [approved(proposal)])


@pytest.mark.parametrize('change', ['delivered', 'session_archive', 'identity', 'duration',
                                    'request_budget', 'cleanup', 'signals', 'result', 'history', 'prior_authorization'])
def test_transport_retry_rejects_ambiguous_or_changed_failure(tmp_path, change):
    config, root, _, slot, _ = native_failure(tmp_path)
    process = slot / 'jiuwenswarm/process'
    if change in ('delivered', 'session_archive'):
        write(slot / ('RESULT.json' if change == 'delivered' else 'sessions/1/native.json'), {})
    else:
        path, edits = {
            'identity': (slot / 'STARTED.json', {'identity': 'changed'}),
            'duration': (process / 'turn-1.execution.json', {'duration_s': 10}),
            'request_budget': (process / 'turn-1.request.json', {'timeout_seconds': 11}),
            'cleanup': (process / 'turn-1.execution.json', {'cleanup_error': 'failed'}),
            'signals': (process / 'turn-1.execution.json', {'signals': ['SIGTERM']}),
            'result': (process / 'turn-1.result.json', {'status': 'completed'}),
            'history': (root / 'round_1/B1/failures/2.json', {'reason': 'changed'}),
            'prior_authorization': (slot.parent / 'authorization.json', {'approval': None}),
        }[change]
        write(path, {**read(path), **edits})
    with pytest.raises(ValueError):
        propose_transport_retry(config, 'synthetic_task', 'round_1/B1')


def test_approval_is_bound_to_current_source_files(tmp_path):
    config, _, _, failed, stage = native_failure(tmp_path)
    proposal = propose_transport_retry(config, 'synthetic_task', 'round_1/B1')
    (failed / 'additional-evidence.txt').write_text('changed after approval')
    with pytest.raises(ValueError, match='evidence changed'):
        ActorRetries(config, [approved(proposal)]).recover('synthetic_task', 'round_1/B1', stage)


@pytest.mark.parametrize('message', ['Task validation failed',
                                    '[181001] model call failed, reason: unrelated failure',
                                    '[181001] model call failed, reason: LLM stream timeout: stream frame timeout: '
                                    'stage=first_chunk, timeout=3.0s, chunk_count=0, first_chunk_elapsed=3.00s, '
                                    'total_elapsed=3.00s, model=another-model'])
def test_consistently_recorded_nonmatching_error_is_not_retryable(tmp_path, message):
    config, root, _, slot, _ = native_failure(tmp_path)
    process = slot / 'jiuwenswarm/process'
    result = read(process / 'turn-1.result.json')
    result['error']['message'] = message
    write(process / 'turn-1.result.json', result)
    (process / 'turn-1.jsonl').write_text(json.dumps(result) + '\n')
    reason = 'JiuwenSwarm failed: ' + str(result['error'])
    for path in (slot / 'FAILURE.json', root / 'round_1/B1/failure.json', root / 'round_1/B1/failures/2.json'):
        write(path, {**read(path), 'reason': reason})
    with pytest.raises(ValueError, match='transport failure'):
        propose_transport_retry(config, 'synthetic_task', 'round_1/B1')




@pytest.mark.asyncio
async def test_separately_authorized_upstream_retry_preserves_partial_session_and_all_costs(tmp_path):
    config, root, original, failed, stage = native_upstream_failure(tmp_path)
    before_original, before_failed = file_inventory(original), file_inventory(failed)
    with pytest.raises(ValueError, match='recoverable session state'):
        propose_transport_retry(config, 'synthetic_task', 'round_1/B1')
    proposal = propose_transport_retry(config, 'synthetic_task', 'round_1/B1', policy=UPSTREAM_POLICY)
    assert proposal['recovery_mode'] == 'fresh_actor_with_retained_failed_session'
    assert proposal['native_session_id'] == 'native-session'
    assert proposal['failed_execution_s'] == 31.6
    assert not ActorRetries(config, []).recover('synthetic_task', 'round_1/B1', stage)
    manager = ActorRetries(config, [approved(proposal)])
    stages = Stages({}, lambda k, v: write(root / (k + '.json'), v),
                    lambda k: read(root / (k + '.json')) if (root / (k + '.json')).exists() else None,
                    recover_actor=lambda name, saved: manager.recover('synthetic_task', name, saved))
    calls = []

    async def execute():
        slot = manager.slot('synthetic_task', 'round_1/B1', original)
        assert slot not in (original, failed) and not slot.exists()
        assert manager.protocol_identity(slot, 'latest-code') == 'earlier'
        manager.validate_slot(slot, 'actor')
        calls.append(slot)
        write(slot / 'RESULT.json', {'status': 'completed'})
        return {'status': 'completed', 'identity': 'actor', 'runtime_directory': str(slot),
                'actor_execution_s': 4, **manager.result_fields(slot, 4)}

    row = await stages.run('round_1/B1', {}, execute, actor=True)
    assert read(root / 'round_1/B1/stage.json')['attempt'] == 3
    assert row['actor_retry']['policy'] == UPSTREAM_POLICY
    assert row['actor_retry']['recovery_mode'] == proposal['recovery_mode']
    assert row['actor_retry']['retained_native_session_id'] == 'native-session'
    assert row['actor_retry']['additional_attempts'] == 2
    assert row['actor_retry']['total_execution_s'] == 35.6
    assert file_inventory(original) == before_original
    assert file_inventory(failed) == before_failed
    verify_actor_retry(row)
    assert await stages.run('round_1/B1', {}, execute, actor=True) == row
    assert len(calls) == 1
    with pytest.raises(ValueError, match='consumed'):
        ActorRetries(config, [approved(proposal)]).recover('synthetic_task', 'round_1/B1', stage)
    row['actor_retry']['total_execution_s'] = 4
    with pytest.raises(ValueError, match='cost'):
        verify_actor_retry(row)


@pytest.mark.parametrize('status', [500, 502, 503, 504])
def test_explicit_upstream_retry_accepts_only_bound_native_server_failure(tmp_path, status):
    config, _, _, slot, _ = native_upstream_failure(tmp_path, status=status)
    before = file_inventory(slot)
    proposal = propose_transport_retry(config, 'synthetic_task', 'round_1/B1', policy=UPSTREAM_POLICY)
    assert proposal['policy'] == UPSTREAM_POLICY and proposal['max_additional_attempts'] == 1
    assert file_inventory(slot) == before
    with pytest.raises(ValueError, match='approval'):
        ActorRetries(config, [{'proposal': proposal, 'approval': None}])


@pytest.mark.parametrize('change', ['completed', 'snapshot', 'actor_state', 'missing_session',
                                    'metadata_session', 'actor_session', 'host_budget', 'duration',
                                    'cleanup', 'signals', 'history', 'nonserver_error', 'extra_archive',
                                    'missing_archive'])
def test_upstream_retry_rejects_completed_ambiguous_or_changed_native_evidence(tmp_path, change):
    config, root, _, slot, _ = native_upstream_failure(tmp_path)
    process = slot / 'jiuwenswarm/process'
    if change in ('completed', 'snapshot', 'actor_state'):
        write(slot / {'completed': 'RESULT.json', 'snapshot': 'snapshots/1/delivery.json',
                      'actor_state': 'actor_state.json'}[change], {})
    elif change == 'missing_session':
        (slot / 'sessions/1/data/agent/sessions/native-session/history.jsonl').unlink()
    elif change == 'extra_archive':
        write(slot / 'sessions/2/metadata.json', {'session_id': 'another-session'})
    elif change == 'missing_archive':
        shutil.rmtree(slot / 'sessions')
    else:
        path, edit = {
            'metadata_session': (slot / 'sessions/1/data/agent/sessions/native-session/metadata.json', {'session_id': 'another-session'}),
            'actor_session': (slot / 'FAILURE.json', {'session': None}),
            'host_budget': (process / 'turn-1.execution.json', {'host_timeout_s': 26}),
            'duration': (process / 'turn-1.execution.json', {'duration_s': 10}),
            'cleanup': (process / 'turn-1.execution.json', {'cleanup_error': 'synthetic failed cleanup'}),
            'signals': (process / 'turn-1.execution.json', {'signals': ['SIGTERM']}),
            'history': (root / 'round_1/B1/failures/2.json', {'reason': 'changed'}),
            'nonserver_error': (process / 'turn-1.result.json', {'error': {'code': 'RUNTIME_ERROR', 'message': 'Task validation failed', 'retryable': False, 'details': {}}}),
        }[change]
        write(path, {**read(path), **edit})
        if change == 'nonserver_error':
            event = (process / 'turn-1.jsonl').read_text().splitlines()[0]
            result = read(path)
            (process / 'turn-1.jsonl').write_text(event + '\n' + json.dumps(result) + '\n')
            reason = 'JiuwenSwarm failed: ' + str(result['error'])
            for p in (slot / 'FAILURE.json', root / 'round_1/B1/failure.json', root / 'round_1/B1/failures/2.json'):
                write(p, {**read(p), 'reason': reason})
    with pytest.raises(ValueError):
        propose_transport_retry(config, 'synthetic_task', 'round_1/B1', policy=UPSTREAM_POLICY)


def test_upstream_retry_approval_cannot_discard_new_partial_evidence(tmp_path):
    config, _, _, failed, stage = native_upstream_failure(tmp_path)
    proposal = propose_transport_retry(config, 'synthetic_task', 'round_1/B1', policy=UPSTREAM_POLICY)
    (failed / 'workspace/late-evidence.txt').write_text('Synthetic extra retained evidence.')
    with pytest.raises(ValueError, match='evidence changed'):
        ActorRetries(config, [approved(proposal)]).recover('synthetic_task', 'round_1/B1', stage)
