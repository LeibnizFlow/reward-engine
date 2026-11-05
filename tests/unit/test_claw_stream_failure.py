"""Keep native failures resumable without granting more execution time."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import yaml

from experiments.claw_eval.opencode import runtime
from reward_engine.adapters.jiuwenswarm import container
from reward_engine.adapters.jiuwenswarm.process import ProcessResult, make_instance
from reward_engine.adapters.workbuddy.io import read


@pytest.mark.asyncio
async def test_container_first_output_wait_uses_existing_actor_budget(tmp_path, monkeypatch):
    template = tmp_path / container.TEMPLATE
    template.parent.mkdir(parents=True)
    template.write_text("models: {}\nmodes: {}\n")
    monkeypatch.setattr(container, 'bundle_path', lambda config: tmp_path)
    monkeypatch.setattr(container, 'capabilities', lambda *args: {})
    monkeypatch.setattr(container, 'mode', lambda config: 'agent.code.normal')
    monkeypatch.setattr(container, 'docker', AsyncMock(return_value=''))
    config = {'actor_timeout_s': 1800}
    await container.prepare('synthetic', tmp_path / 'actor', config,
                            {'base_url': 'http://synthetic/v1', 'model': 'synthetic',
                             'api_key': 'synthetic-secret'})
    document = yaml.safe_load((tmp_path / 'actor/jiuwenswarm-config/data/config/config.yaml').read_text())
    client = document['models']['defaults'][0]['model_client_config']
    assert client['stream_first_chunk_timeout'] == config['actor_timeout_s']
    assert client['timeout'] >= client['stream_first_chunk_timeout']
    assert client['stream_idle_timeout'] == 120
    assert config == {'actor_timeout_s': 1800}


@pytest.mark.parametrize('budget', [0, -1, float('nan'), float('inf'), True])
def test_instance_rejects_invalid_execution_budget_before_writing(tmp_path, budget):
    with pytest.raises(ValueError, match='execution_timeout_s'):
        make_instance(tmp_path / 'instance', model={}, template=tmp_path / 'unused',
                      execution_timeout_s=budget)
    assert not (tmp_path / 'instance').exists()


@pytest.mark.asyncio
@pytest.mark.parametrize('export_fails', [False, True])
async def test_failed_turn_preserves_session_usage_and_debits_time(tmp_path, monkeypatch, export_fails):
    receipt = ProcessResult(request={}, events=({'event_type': 'synthetic_event'},),
                            result={'status': 'failed', 'session_id': 'native-session',
                                    'usage': {'input_tokens': 42},
                                    'error': {'message': 'synthetic first chunk timeout'}},
                            stdout='', stderr='', returncode=1, duration_s=12)
    actor = object.__new__(runtime.Actor)
    actor.config = {'actor_timeout_s': 1800}
    actor.execution_s, actor.turn, actor.session = 100, 0, None
    actor.usage, actor.raw_events = [], []
    actor.output, actor.container = tmp_path, 'synthetic-container'
    actor.process = SimpleNamespace(run=AsyncMock(return_value=receipt), child_root='/native')
    actor._snapshot, actor.trace_message = AsyncMock(), Mock()
    clock = iter([1000, 1012])
    monkeypatch.setattr(runtime, 'time', SimpleNamespace(monotonic=lambda: next(clock)))

    async def export(*args):
        if export_fails:
            raise RuntimeError('synthetic export failure')
        assert args[:2] == ('cp', 'synthetic-container:/native')
        destination = tmp_path / 'sessions/1'
        destination.mkdir()
        (destination / 'native-session.json').write_text('{"native": true}')

    monkeypatch.setattr(container, 'docker', export)
    with pytest.raises(RuntimeError, match='synthetic first chunk timeout'):
        await actor._jiuwenswarm_turn('original guidance')
    assert actor.execution_s == 112
    assert actor.process.run.call_args.kwargs['timeout_s'] == 1700
    assert actor.session == 'native-session'
    assert actor.usage == [{'input_tokens': 42}]
    assert actor.raw_events == list(receipt.events)
    actor._snapshot.assert_not_awaited()
    actor.trace_message.assert_not_called()
    assert not (tmp_path / 'RESULT.json').exists()
    assert not (tmp_path / 'actor_state.json').exists()
    if export_fails:
        assert read(tmp_path / 'sessions/1.export-failure.json')['reason'] == 'synthetic export failure'
    else:
        assert read(tmp_path / 'sessions/1/native-session.json') == {'native': True}
