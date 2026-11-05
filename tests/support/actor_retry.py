"""Synthetic failed Actor evidence shared by retry regressions."""
import json

from reward_engine.adapters.claw_eval.actor_retry import (
    ActorRetries,
    propose_actor_retry,
)
from reward_engine.adapters.workbuddy.io import file_hash, read, write
from reward_engine.core.outcome.construction import fingerprint


def failed_actor(tmp_path):
    config = {"output": str(tmp_path), "identity": "implementation",
              "checkpoint_config_identity": "frozen-experiment",
              "actor_backend": "jiuwenswarm",
              "generation_protocol": "three_b_fixed_baselines_v1",
              "actor_timeout_s": 10, "task_ids": ["synthetic_task"]}
    root = tmp_path / "tasks/synthetic_task"
    stage = {"status": "started", "identity": fingerprint({"run": {}, "inputs": {}}),
             "attempt": 1, "implementation": "earlier"}
    failure = {**stage, "error_type": "TimeoutError", "reason": ""}
    write(root / "round_1/B1/stage.json", stage)
    write(root / "round_1/B1/failure.json", failure)
    write(root / "round_1/B1/failures/1.json", failure)
    slot = root / "round_1/actors/B1"
    write(slot / "STARTED.json", {"identity": "actor", "at": 1})
    write(slot / "FAILURE.json", {"identity": "actor", "session": None,
                                 "error_type": "TimeoutError", "reason": ""})
    process = slot / "jiuwenswarm/process"
    write(process / "turn-1.request.json", {"request_id": "first", "session_id": None,
                                          "timeout_seconds": 10})
    write(process / "turn-1.execution.json", {"status": "timed_out", "error": "TimeoutError",
                                            "returncode": 137, "duration_s": 25.1})
    (process / "turn-1.jsonl").write_text(json.dumps({
        "type": "event", "sequence": 0, "request_id": "first",
        "session_id": "old-session", "event_type": "chat.tool_call"}) + "\n")
    (process / "turn-1.stderr").write_text("Synthetic active work, no final delivery.\n")
    return config, root, slot, stage


def approved(proposal):
    return {"proposal": proposal, "approval": {
        "decision": "approve", "proposal_sha256": fingerprint(proposal)}}


def native_timed_out_actor(tmp_path):
    config, root, slot, stage = failed_actor(tmp_path)
    process = slot / 'jiuwenswarm/process'
    write(process / 'turn-1.execution.json', {
        'status': 'timed_out', 'error': 'TimeoutError', 'returncode': 124,
        'duration_s': 36.4, 'requested_timeout_s': 10, 'host_timeout_s': 25,
        'timeout_source': 'host', 'timeout_elapsed_s': 25.01, 'signals': [],
    })
    event = {'schema_version': '0.1', 'type': 'event', 'sequence': 0,
             'request_id': 'first', 'session_id': 'old-session',
             'event_type': 'chat.tool_call', 'payload': {}}
    terminal = {'schema_version': '0.1', 'type': 'result', 'sequence': 1,
                'request_id': 'first', 'session_id': 'old-session',
                'status': 'timed_out', 'exit_code': 124,
                'output': 'Synthetic partial work before the execution deadline.',
                'error': {'code': 'TIMEOUT', 'message': 'Command timed out.',
                          'retryable': False, 'details': {}}, 'usage': {}}
    (process / 'turn-1.jsonl').write_text(json.dumps(event) + '\n' + json.dumps(terminal) + '\n')
    return config, root, slot, stage


def file_inventory(root):
    return {str(p.relative_to(root)): file_hash(p) for p in root.rglob("*") if p.is_file()}




def native_failure(tmp_path):
    config, root, original, first_stage = failed_actor(tmp_path)
    config['roles'] = {'actor': 'synthetic-model'}
    prior = approved(propose_actor_retry(config, 'synthetic_task', 'round_1/B1'))
    manager = ActorRetries(config, [prior])
    manager.recover('synthetic_task', 'round_1/B1', first_stage)
    slot = manager.slot('synthetic_task', 'round_1/B1', original)
    message = ('[181001] model call failed, reason: LLM stream timeout: stream frame timeout: '
               'stage=first_chunk, timeout=3.0s, chunk_count=0, first_chunk_elapsed=3.00s, '
               'total_elapsed=3.00s, model=synthetic-model')
    error = {'code': 'RUNTIME_ERROR', 'message': message, 'retryable': False, 'details': {}}
    stage = {**first_stage, 'attempt': 2, 'implementation': 'changed-implementation'}
    failure = {**stage, 'error_type': 'RuntimeError', 'reason': 'JiuwenSwarm failed: ' + str(error)}
    write(root / 'round_1/B1/stage.json', stage)
    write(root / 'round_1/B1/failure.json', failure)
    write(root / 'round_1/B1/failures/2.json', failure)
    write(slot / 'STARTED.json', {'identity': 'actor'})
    write(slot / 'FAILURE.json', {'identity': 'actor', 'session': None,
                                 'error_type': 'RuntimeError', 'reason': failure['reason']})
    process = slot / 'jiuwenswarm/process'
    write(process / 'turn-1.request.json', {'request_id': 'second', 'session_id': None,
                                           'timeout_seconds': 10})
    write(process / 'turn-1.execution.json', {'status': 'exited', 'returncode': 1,
                                             'requested_timeout_s': 10, 'duration_s': 6.5, 'signals': []})
    result = {'schema_version': '0.1', 'type': 'result', 'sequence': 0, 'request_id': 'second',
              'session_id': 'native-session', 'status': 'failed', 'exit_code': 1,
              'error': error, 'usage': {'input_tokens': 42}}
    write(process / 'turn-1.result.json', result)
    (process / 'turn-1.jsonl').write_text(json.dumps(result) + '\n')
    return config, root, original, slot, stage

def native_upstream_failure(tmp_path, *, status=502):
    """Synthetic native failure after a tool call, with exported partial session."""
    config, root, original, slot, stage = native_failure(tmp_path)
    process = slot / 'jiuwenswarm/process'
    result = read(process / 'turn-1.result.json')
    result['sequence'] = 1
    result['error']['message'] = (
        '[181001] model call failed, reason: openAI API async stream error: '
        f"InternalServerError: Error code: {status} - {{'detail': 'Upstream request failed; see campaign transport receipt'}}")
    write(process / 'turn-1.result.json', result)
    event = {'schema_version': '0.1', 'type': 'event', 'sequence': 0,
             'request_id': 'second', 'session_id': 'native-session',
             'event_type': 'chat.tool_call', 'payload': {'name': 'Bash'}}
    (process / 'turn-1.jsonl').write_text(json.dumps(event) + '\n' + json.dumps(result) + '\n')
    write(process / 'turn-1.execution.json', {
        **read(process / 'turn-1.execution.json'), 'host_timeout_s': 25})
    reason = 'JiuwenSwarm failed: ' + str(result['error'])
    for path in (slot / 'FAILURE.json', root / 'round_1/B1/failure.json', root / 'round_1/B1/failures/2.json'):
        write(path, {**read(path), 'reason': reason})
    write(slot / 'FAILURE.json', {**read(slot / 'FAILURE.json'), 'session': 'native-session'})
    session = slot / 'sessions/1/data/agent/sessions/native-session'
    write(session / 'metadata.json', {'session_id': 'native-session', 'status': 'idle'})
    (session / 'history.jsonl').write_text('{"role":"user","content":"Synthetic task"}\n')
    (slot / 'workspace').mkdir()
    (slot / 'workspace/partial.txt').write_text('Synthetic incomplete work remains evidence.\n')
    return config, root, original, slot, stage
