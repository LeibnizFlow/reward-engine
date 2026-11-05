"""Synthetic third failures after the two separately authorized Actor retries."""
import json
import shutil

from reward_engine.adapters.claw_eval.actor_retry import UPSTREAM_POLICY, ActorRetries
from reward_engine.adapters.claw_eval.actor_transport_retry import (
    propose_transport_retry,
)
from reward_engine.adapters.workbuddy.io import read, write

from .actor_retry import approved, native_upstream_failure


def third_failed_actor(tmp_path, kind='upstream'):
    config, root, original, second, stage = native_upstream_failure(tmp_path)
    prior = propose_transport_retry(config, 'synthetic_task', 'round_1/B1', policy=UPSTREAM_POLICY)
    manager = ActorRetries(config, [approved(prior)])
    manager.recover('synthetic_task', 'round_1/B1', stage)
    third = manager.slot('synthetic_task', 'round_1/B1', original)
    shutil.copytree(second, third)
    process = third / 'jiuwenswarm/process'
    request = read(process / 'turn-1.request.json')
    request['request_id'] = 'third'
    write(process / 'turn-1.request.json', request)
    result = read(process / 'turn-1.result.json')
    result.update(request_id='third', session_id='third-native-session')
    write(process / 'turn-1.result.json', result)
    events = [json.loads(line) for line in (process / 'turn-1.jsonl').read_text().splitlines()]
    for event in events:
        event.update(request_id='third', session_id='third-native-session')
    (process / 'turn-1.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in events))
    session_root = third / 'sessions/1/data/agent/sessions'
    (session_root / 'native-session').rename(session_root / 'third-native-session')
    write(session_root / 'third-native-session/metadata.json', {'session_id': 'third-native-session'})
    write(third / 'STARTED.json', {'identity': 'actor', 'at': 100})
    write(third / 'FAILURE.json', {**read(third / 'FAILURE.json'), 'session': 'third-native-session'})
    if kind == 'queue':
        shutil.rmtree(third / 'sessions')
        (process / 'turn-1.result.json').unlink()
        (process / 'turn-1.jsonl').write_text(json.dumps(events[0]) + '\n')
        write(process / 'turn-1.execution.json', {'status': 'timed_out', 'error': 'TimeoutError',
              'returncode': 124, 'requested_timeout_s': 10, 'host_timeout_s': 25,
              'duration_s': 25.4, 'timeout_elapsed_s': 25.01, 'timeout_source': 'host', 'signals': []})
        write(third / 'FAILURE.json', {'identity': 'actor', 'session': None,
              'error_type': 'TimeoutError', 'reason': 'Synthetic host timeout after gateway queue wait'})
        trace = {'type': 'trace_start', 'trace_id': '3' * 32, 'task_id': 'synthetic_task',
                 'model': 'synthetic-model', 'timestamp': '1970-01-01T00:01:40+00:00'}
        (third / 'trace.jsonl').write_text(json.dumps(trace) + '\n')
        write(third / 'public_task.json', {'task_id': 'synthetic_task'})
        gateway = tmp_path / 'transport' / ('actor-' + '3' * 32)
        write(gateway / 'first/receipt.json', {'channel': gateway.name, 'model': 'synthetic-model',
              'actual_model': 'synthetic-model', 'queued_at': 101, 'queue_s': 5,
              'elapsed_s': .5, 'finished_at': 106.5, 'stream_timeout_s': 10,
              'attempts': 1, 'status': 'completed'})
        write(gateway / 'last/receipt.json', {'channel': gateway.name, 'model': 'synthetic-model',
              'queued_at': 107, 'finished_at': 111.5, 'status': 'rejected_before_execution',
              'error_type': 'CancelledError', 'reason': 'Operation ended while waiting for gateway capacity'})
    stage = {**stage, 'attempt': 3, 'implementation': 'third-implementation'}
    actor_failure = read(third / 'FAILURE.json')
    failure = {**stage, **{k: actor_failure[k] for k in ('error_type', 'reason')}}
    for name, value in [('stage.json', stage), ('failure.json', failure), ('failures/3.json', failure)]:
        write(root / 'round_1/B1' / name, value)
    write(root / 'state.json', {'status': 'failed'})
    return config, root, original, second, third, stage
