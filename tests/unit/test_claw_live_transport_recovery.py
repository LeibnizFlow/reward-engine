"""Recover a native server failure in the live Actor, preserving time and state."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from experiments.claw_eval.opencode import runtime
from reward_engine.adapters.jiuwenswarm import container
from reward_engine.adapters.jiuwenswarm.process import ProcessResult
from reward_engine.adapters.jiuwenswarm.recovery import verify
from reward_engine.adapters.workbuddy.io import file_hash, read, write


def live_actor(tmp_path, monkeypatch, *, status=502, second_fails=False, export_fails=False,
               missing_session=False, bad_metadata=False, exhausted=False, legacy=False):
    actor = object.__new__(runtime.Actor)
    actor.config = {'actor_timeout_s': 100, 'generation_protocol':
                    'legacy' if legacy else 'three_b_fixed_baselines_v1'}
    actor.execution_s, actor.turn, actor.session = 0, 0, None
    actor.usage, actor.raw_events = [], []
    actor.output, actor.container = tmp_path, 'synthetic-live-container'
    actor.trace_message = Mock()
    actor._snapshot = AsyncMock(return_value={'delivery': 'synthetic completed work'})
    clock = iter([1000, 1100 if exhausted else 1010, 1010, 1015])
    monkeypatch.setattr(runtime, 'time', SimpleNamespace(monotonic=lambda: next(clock), time=lambda: 1))
    calls = []

    async def run(prompt, **kwargs):
        number = len(calls)
        calls.append((prompt, kwargs))
        failed = number == 0 or second_fails
        session = None if missing_session else 'synthetic-session'
        message = ('[181001] model call failed, reason: openAI API async stream error: '
                   f"InternalServerError: Error code: {status} - "
                   "{'detail': 'Upstream request failed; see campaign transport receipt'}")
        request = {'schema_version': '0.1', 'type': 'run', 'request_id': f'request-{number}',
                   'session_id': kwargs['session_id'], 'input': prompt,
                   'agent': {'name': 'synthetic'}, 'mode': 'agent.code.normal',
                   'workspace': {'cwd': '/workspace'}, 'timeout_seconds': kwargs['timeout_s']}
        result = {'schema_version': '0.1', 'type': 'result', 'sequence': 0,
                  'request_id': request['request_id'], 'session_id': session,
                  'status': 'failed' if failed else 'completed', 'exit_code': 1 if failed else 0,
                  'output': None if failed else 'synthetic final response',
                  'error': {'code': 'RUNTIME_ERROR', 'message': message, 'retryable': False, 'details': {}}
                  if failed else None, 'usage': {'input_tokens': 7 if number == 0 else 11}}
        label = kwargs['label']
        base = tmp_path / 'jiuwenswarm/process'
        write(base / f'{label}.request.json', request)
        write(base / f'{label}.result.json', result)
        write(base / f'{label}.execution.json', {'status': 'exited', 'returncode': result['exit_code'],
              'requested_timeout_s': kwargs['timeout_s'], 'host_timeout_s': kwargs['timeout_s'] + 15,
              'duration_s': 10 if number == 0 else 5, 'signals': []})
        raw = json.dumps(result) + '\n'
        (base / f'{label}.jsonl').write_text(raw)
        return ProcessResult(request, (), result, raw, '', result['exit_code'], 10 if number == 0 else 5)

    actor.process = SimpleNamespace(run=run, child_root='/synthetic-native')

    async def export(*args):
        assert args[:2] == ('cp', 'synthetic-live-container:/synthetic-native')
        if export_fails:
            raise RuntimeError('synthetic export failed')
        archive = Path(args[2])
        folder = archive / 'data/agent/sessions/synthetic-session'
        write(folder / 'metadata.json', {'session_id': 'wrong' if bad_metadata else 'synthetic-session',
                                         'channel_id': 'process_cli', 'mode': 'agent.code.normal'})
        (folder / 'history.jsonl').write_text('{"role":"user","content":"synthetic task"}\n')

    monkeypatch.setattr(container, 'docker', export)
    return actor, calls


@pytest.mark.asyncio
@pytest.mark.parametrize('status', [500, 502, 503, 504])
async def test_server_failure_continues_same_live_session_with_remaining_budget(tmp_path, monkeypatch, status):
    actor, calls = live_actor(tmp_path, monkeypatch, status=status)
    assert await actor._jiuwenswarm_turn('synthetic original guidance') == {'delivery': 'synthetic completed work'}
    assert len(calls) == 2
    assert calls[0][1]['session_id'] is None and calls[0][1]['timeout_s'] == 100
    assert calls[1][1]['session_id'] == 'synthetic-session' and calls[1][1]['timeout_s'] == 90
    assert calls[0][1]['workspace'] == calls[1][1]['workspace']
    assert calls[0][1]['label'] == 'turn-1' and calls[1][1]['label'] == 'turn-1-recovery-1'
    assert actor.turn == 1 and actor.execution_s == 15
    assert actor.usage == [{'input_tokens': 7}, {'input_tokens': 11}]
    assert (tmp_path / 'sessions/1/data/agent/sessions/synthetic-session/history.jsonl').is_file()
    assert (tmp_path / 'sessions/1-recovery-1/data/agent/sessions/synthetic-session/history.jsonl').is_file()
    proof = read(tmp_path / 'jiuwenswarm/process/turn-1.recovery.json')
    assert proof['remaining_s'] == 90 and proof['execution_s_before_recovery'] == 10
    actor._snapshot.assert_awaited_once_with('synthetic final response')


@pytest.mark.asyncio
@pytest.mark.parametrize('change', ['nonserver', 'no_session', 'export', 'metadata', 'exhausted', 'legacy'])
async def test_ineligible_native_failure_never_starts_recovery(tmp_path, monkeypatch, change):
    actor, calls = live_actor(tmp_path, monkeypatch, **{
        'nonserver': {'status': 429}, 'no_session': {'missing_session': True},
        'export': {'export_fails': True}, 'metadata': {'bad_metadata': True},
        'exhausted': {'exhausted': True}, 'legacy': {'legacy': True},
    }[change])
    with pytest.raises(RuntimeError, match='JiuwenSwarm failed'):
        await actor._jiuwenswarm_turn('synthetic task')
    assert len(calls) == 1 and actor.turn == 1
    assert actor.execution_s == (100 if change == 'exhausted' else 10)
    actor._snapshot.assert_not_awaited()


@pytest.mark.asyncio
async def test_second_native_failure_is_terminal_and_retains_both_costs(tmp_path, monkeypatch):
    actor, calls = live_actor(tmp_path, monkeypatch, second_fails=True)
    with pytest.raises(RuntimeError, match='JiuwenSwarm failed'):
        await actor._jiuwenswarm_turn('synthetic task')
    assert len(calls) == 2 and actor.execution_s == 15 and actor.turn == 1
    assert (tmp_path / 'jiuwenswarm/process/turn-1.result.json').is_file()
    assert (tmp_path / 'jiuwenswarm/process/turn-1-recovery-1.result.json').is_file()
    actor._snapshot.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('change', [None, 'count', 'budget', 'spent', 'session', 'prompt',
                                   'settings', 'deadline', 'archive', 'missing_proof', 'extra_execution'])
async def test_completed_recovery_verifies_original_failure_session_and_budget(tmp_path, monkeypatch, change):
    actor, _ = live_actor(tmp_path, monkeypatch)
    await actor._jiuwenswarm_turn('synthetic task')
    row = {'resources': {'turns': 1, 'transport_recoveries': 1,
                         'process_executions': 2, 'actor_timeout_s': 100}}
    process = tmp_path / 'jiuwenswarm/process'
    proof = process / 'turn-1.recovery.json'
    if change == 'count':
        row['resources']['transport_recoveries'] = 0
    elif change == 'budget':
        row['resources']['actor_timeout_s'] = 200
    elif change == 'spent':
        write(proof, {**read(proof), 'execution_s_before_recovery': 0})
    elif change in ('session', 'prompt', 'settings', 'deadline'):
        request = process / 'turn-1-recovery-1.request.json'
        edit = {'session': {'session_id': 'changed'}, 'prompt': {'input': 'changed'},
                'settings': {'agent': {'name': 'changed'}}, 'deadline': {'timeout_seconds': 100}}[change]
        write(request, {**read(request), **edit})
    elif change == 'archive':
        history = tmp_path / 'sessions/1/data/agent/sessions/synthetic-session/history.jsonl'
        history.write_text('changed after failed execution')
    elif change == 'missing_proof':
        proof.unlink()
        row['resources']['transport_recoveries'] = 0
    elif change == 'extra_execution':
        write(process / 'turn-1-recovery-2.request.json', {})
    if change is None:
        verify(tmp_path, row)
    else:
        with pytest.raises((ValueError, RuntimeError)):
            verify(tmp_path, row)


@pytest.mark.asyncio
@pytest.mark.parametrize('error', [TimeoutError('synthetic host deadline'), asyncio.CancelledError()])
async def test_recovery_host_timeout_or_cancellation_never_delivers_partial_work(tmp_path, monkeypatch, error):
    actor, calls = live_actor(tmp_path, monkeypatch)
    run = actor.process.run

    async def interrupted(prompt, **kwargs):
        if calls:
            calls.append((prompt, kwargs))
            raise error
        return await run(prompt, **kwargs)

    actor.process.run = interrupted
    with pytest.raises(type(error)):
        await actor._jiuwenswarm_turn('synthetic task')
    assert len(calls) == 2 and actor.execution_s == 15 and actor.turn == 1
    assert read(tmp_path / 'jiuwenswarm/process/turn-1.result.json')['status'] == 'failed'
    assert read(tmp_path / 'jiuwenswarm/process/turn-1.recovery.json')['remaining_s'] == 90
    actor._snapshot.assert_not_awaited()


@pytest.mark.asyncio
async def test_sample_records_recovery_without_spending_a_critic_revision(tmp_path, monkeypatch):
    from reward_engine.adapters.jiuwenswarm.checkpoints import verify_actor

    slot = tmp_path / 'actor'
    actor, _ = live_actor(slot, monkeypatch)
    actor.events = []
    actor.setup, actor.finalize, actor.close = AsyncMock(), AsyncMock(), AsyncMock()

    async def snapshot(answer):
        candidate = {'artifact': {'final_response': answer}}
        folder = slot / 'snapshots/1'
        (folder / 'workspace').mkdir(parents=True)
        delivered = folder / 'workspace/output.txt'
        delivered.write_text(answer)
        write(folder / 'candidate.json', candidate)
        write(folder / 'inventory.json', [{'path': 'output.txt', 'sha256': file_hash(delivered)}])
        return candidate

    actor._snapshot = snapshot
    runner = object.__new__(runtime.Runtime)
    runner.config = {**actor.config, 'identity': 'synthetic', 'actor_backend': 'jiuwenswarm'}
    runner.check_resources = lambda: None
    monkeypatch.setattr(runtime, 'Actor', lambda *args: actor)

    async def deliver(actor, review, *, prompt):
        return await actor._jiuwenswarm_turn(prompt)

    monkeypatch.setattr(runtime, 'run_delivery', deliver)
    row = await runner.sample('synthetic_task', slot, guidance='synthetic task')
    assert row['resources'] == {'turns': 1, 'revision_turns': 0,
                                'transport_recoveries': 1, 'process_executions': 2, 'actor_timeout_s': 100}
    assert row['actor_execution_s'] == 15
    assert row['usage'] == [{'input_tokens': 7}, {'input_tokens': 11}]
    verify_actor(row)
    assert await runner.sample('synthetic_task', slot, guidance='synthetic task') == row
    actor.setup.assert_awaited_once()
