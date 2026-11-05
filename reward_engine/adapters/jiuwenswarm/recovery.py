"""One live session continuation after a proven native server failure."""
import math
import re
from pathlib import Path

from reward_engine.adapters.workbuddy.io import file_hash, read

from .process import _validate_record_stream

POLICY = 'same_session_native_upstream_recovery_v1'
CONTINUE = ('Continue the same task and workspace after the temporary model service failure; '
            'finish the pending work, preserving existing work and verifying the result.')
SERVER_ERROR = re.compile(
    r'\[181001\] model call failed, reason: openAI API async stream error: '
    r'InternalServerError: Error code: (?:500|502|503|504) - '
    r"\{'detail': 'Upstream request failed; see campaign transport receipt'\}")


def _plain(path, root):
    return (path.is_file() and not path.is_symlink()
            and path.resolve().is_relative_to(root.resolve())
            and not any(p.is_symlink() for p in path.parents if p != root.parent))


def proposal(root, turn, budget, execution_s):
    """Read closed native files only; an incomplete archive never grants recovery."""
    if (type(turn) is not int or turn < 1
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in (budget, execution_s))
            or not 0 < execution_s < budget):
        return None
    root = Path(root)
    label = f'turn-{turn}'
    process = root / 'jiuwenswarm/process'
    paths = [process / (label + suffix) for suffix in
             ('.request.json', '.execution.json', '.result.json', '.jsonl')]
    if not all(_plain(path, root) for path in paths):
        return None
    request, execution, terminal = [read(path) for path in paths[:3]]
    _, result = _validate_record_stream(paths[3].read_text(), request, execution.get('returncode'))
    session = result.get('session_id')
    error = result.get('error') or {}
    requested = request.get('timeout_seconds')
    duration = execution.get('duration_s')
    if (result != terminal or result.get('status') != 'failed' or result.get('exit_code') != 1
            or error.get('code') != 'RUNTIME_ERROR' or not SERVER_ERROR.fullmatch(error.get('message', ''))
            or not isinstance(session, str) or not session or Path(session).name != session
            or session in {'.', '..'} or execution.get('status') != 'exited'
            or execution.get('returncode') != 1 or execution.get('signals')
            or 'cleanup_error' in execution
            or type(requested) not in (int, float) or not 0 < requested <= budget
            or execution.get('requested_timeout_s') != requested
            or execution.get('host_timeout_s') != requested + 15
            or type(duration) not in (int, float) or not math.isfinite(duration)
            or not 0 <= duration <= execution_s):
        return None
    session_root = root / 'sessions' / str(turn) / 'data/agent/sessions' / session
    metadata, history = session_root / 'metadata.json', session_root / 'history.jsonl'
    if not all(_plain(path, root) for path in (metadata, history)):
        return None
    saved = read(metadata)
    if (saved.get('session_id') != session or saved.get('channel_id') != 'process_cli'
            or saved.get('mode') != request.get('mode') or not history.read_text().strip()):
        return None
    return {'policy': POLICY, 'turn': turn, 'failed_label': label,
            'recovery_label': label + '-recovery-1', 'session_id': session,
            'actor_timeout_s': budget, 'execution_s_before_recovery': execution_s,
            'remaining_s': budget - execution_s,
            'source_hashes': {str(path.relative_to(root)): file_hash(path)
                              for path in (*paths, metadata, history)}}


def verify(root, row):
    """Replay completed recovery provenance and the remaining-budget request."""
    root = Path(root)
    process = root / 'jiuwenswarm/process'
    receipts = list(process.glob('turn-*.recovery.json'))
    resources = row.get('resources', {})
    if resources.get('transport_recoveries', 0) != len(receipts):
        raise ValueError('Native recovery count changed')
    if receipts and resources.get('process_executions') != resources.get('turns', 0) + len(receipts):
        raise ValueError('Native Process execution count changed')
    if any(not _plain(path, root) for path in receipts):
        raise ValueError('Native recovery receipt is linked or missing')
    expected = {f'turn-{read(path).get("turn")}-recovery-1.request.json' for path in receipts}
    if {path.name for path in process.glob('turn-*-recovery-*.request.json')} != expected:
        raise ValueError('Native recovery authorization receipt missing or extra execution present')
    for path in receipts:
        proof = read(path)
        turn, budget = proof.get('turn'), resources.get('actor_timeout_s')
        if (proof != proposal(root, turn, budget, proof.get('execution_s_before_recovery'))
                or path.name != f'turn-{turn}.recovery.json'
                or turn > resources.get('turns', 0)):
            raise ValueError('Native recovery provenance or execution budget changed')
        label = proof['recovery_label']
        paths = [process / (label + suffix) for suffix in
                 ('.request.json', '.execution.json', '.result.json', '.jsonl')]
        if not all(_plain(p, root) for p in paths):
            raise ValueError('Native recovery execution evidence missing')
        request, execution, result = [read(p) for p in paths[:3]]
        _, terminal = _validate_record_stream(paths[3].read_text(), request, execution.get('returncode'))
        original = read(process / (proof['failed_label'] + '.request.json'))
        if (result != terminal or result.get('session_id') != proof['session_id']
                or result.get('status') != 'completed' or execution.get('returncode') != 0
                or request.get('session_id') != proof['session_id']
                or request.get('input') != CONTINUE
                or any(request.get(k) != original.get(k) for k in ('mode', 'workspace', 'agent'))
                or request.get('timeout_seconds') != proof['remaining_s']
                or execution.get('requested_timeout_s') != proof['remaining_s']
                or execution.get('host_timeout_s') != proof['remaining_s'] + 15
                or execution.get('status') != 'exited' or execution.get('signals')
                or 'cleanup_error' in execution):
            raise ValueError('Native recovery changed session, settings, or remaining budget')
