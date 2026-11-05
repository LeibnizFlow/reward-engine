"""One explicitly authorized fresh Actor after a third proven infrastructure failure."""

import json
import re
from datetime import datetime
from pathlib import Path

from reward_engine.adapters.jiuwenswarm.process import _validate_record_stream
from reward_engine.adapters.workbuddy.io import file_hash, read

from .actor_retry import (
    INFRASTRUCTURE_POLICY,
    UPSTREAM_POLICY,
    _approved,
    _finite_nonnegative,
    _inventory,
    _locations,
    _native_timeout_terminal,
    _plain_file,
    archive_authorization,
)
from .actor_transport_retry import claim_path, propose_transport_retry


def _gateway_evidence(config, slot, execution):
    campaign = Path(config["output"]).resolve()
    for path in (slot / "trace.jsonl", slot / "public_task.json"):
        _plain_file(path, slot)
    traces = [
        json.loads(line) for line in (slot / "trace.jsonl").read_text().splitlines()
    ]
    starts = [
        row
        for row in traces
        if isinstance(row, dict) and row.get("type") == "trace_start"
    ]
    if len(starts) != 1 or any(
        not isinstance(row, dict) or row.get("type") == "trace_end" for row in traces
    ):
        raise ValueError("Gateway queue recovery requires one unfinished Actor trace")
    trace = starts[0]
    trace_id = trace.get("trace_id")
    started = read(slot / "STARTED.json").get("at")
    if (
        not isinstance(trace_id, str)
        or not re.fullmatch("[a-f0-9]{32}", trace_id)
        or trace.get("task_id") != read(slot / "public_task.json").get("task_id")
        or trace.get("task_id") != slot.parents[2].name
        or trace.get("model") != config["roles"]["actor"]
        or not _finite_nonnegative(started)
    ):
        raise ValueError("Gateway queue evidence is not bound to this Actor")
    try:
        trace_at = datetime.fromisoformat(trace["timestamp"])
        if trace_at.tzinfo is None or trace_at.timestamp() < started:
            raise ValueError("Actor trace predates its execution")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Actor queue trace timestamp is invalid") from error
    channel = campaign / "transport" / ("actor-" + trace_id)
    paths = list(channel.glob("*/receipt.json"))
    if len(paths) < 2 or channel.is_symlink():
        raise ValueError("Gateway queue receipts are missing or linked")
    receipts = []
    for path in paths:
        _plain_file(path, campaign)
        row = read(path)
        if (
            not isinstance(row, dict)
            or row.get("channel") != channel.name
            or row.get("model") != config["roles"]["actor"]
            or not all(
                _finite_nonnegative(row.get(k)) for k in ("queued_at", "finished_at")
            )
            or row["queued_at"] < trace_at.timestamp()
            or row["finished_at"] < row["queued_at"]
        ):
            raise ValueError("Gateway receipt identity or timing is invalid")
        receipts.append((path, row))
    receipts.sort(key=lambda item: item[1]["queued_at"])
    queue_s, finished = 0, trace_at.timestamp()
    for index, (_, row) in enumerate(receipts):
        if row["queued_at"] < finished - 0.01:
            raise ValueError("Gateway queue intervals overlap")
        finished = row["finished_at"]
        if index == len(receipts) - 1:
            if (
                row.get("status") != "rejected_before_execution"
                or row.get("error_type") != "CancelledError"
                or row.get("reason")
                != "Operation ended while waiting for gateway capacity"
                or type(row.get("attempts", 0)) is not int
                or row.get("attempts", 0) != 0
            ):
                raise ValueError(
                    "Gateway termination does not prove cancellation while queued"
                )
            queue_s += finished - row["queued_at"]
        else:
            if (
                row.get("status") != "completed"
                or type(row.get("attempts")) is not int
                or row["attempts"] != 1
                or row.get("actual_model") != config["roles"]["actor"]
                or row.get("stream_timeout_s") != config["actor_timeout_s"]
                or not all(
                    _finite_nonnegative(row.get(k)) for k in ("queue_s", "elapsed_s")
                )
                or abs(finished - row["queued_at"] - row["queue_s"] - row["elapsed_s"])
                > 0.02
            ):
                raise ValueError(
                    "Gateway completed request timing or budget is unproven"
                )
            queue_s += row["queue_s"]
    fraction = queue_s / config["actor_timeout_s"]
    # STARTED precedes container/service setup; duration_s covers the later Process execution.
    if (
        fraction < 0.9
        or queue_s > execution["duration_s"] + 1
        or finished - receipts[0][1]["queued_at"] > execution["duration_s"] + 1
    ):
        raise ValueError(
            "Gateway waiting did not consume at least 90% of the native Actor budget"
        )
    return {
        "channel": channel.name,
        "queue_s": queue_s,
        "fraction_native_budget": fraction,
        "receipts": {
            str(path.relative_to(campaign)): file_hash(path) for path, _ in receipts
        },
    }


def _failure(config, slot, actor):
    process = slot / "jiuwenswarm/process"
    if {p.name for p in process.glob("*.request.json")} != {"turn-1.request.json"}:
        raise ValueError(
            "Infrastructure recovery requires one terminated first execution"
        )
    request, execution = (
        read(process / f"turn-1.{name}.json") for name in ("request", "execution")
    )
    budget, duration = config["actor_timeout_s"], execution.get("duration_s")
    if (
        not _finite_nonnegative(budget)
        or budget <= 0
        or not isinstance(request.get("request_id"), str)
        or not request["request_id"]
        or request.get("session_id") is not None
        or request.get("timeout_seconds") != budget
        or execution.get("requested_timeout_s") != budget
        or execution.get("host_timeout_s") != budget + 15
        or not _finite_nonnegative(duration)
        or duration <= 0
        or execution.get("signals")
        or "cleanup_error" in execution
    ):
        raise ValueError("Infrastructure termination or unchanged budget is unproven")
    text = (process / "turn-1.jsonl").read_text()
    results = {p.name for p in process.glob("*.result.json")}
    if results:
        if results != {"turn-1.result.json"}:
            raise ValueError("Native infrastructure terminal is ambiguous")
        result = read(process / "turn-1.result.json")
        _, terminal = _validate_record_stream(
            text, request, execution.get("returncode")
        )
        error, session = result.get("error") or {}, result.get("session_id")
        expected = (
            r"\[181001\] model call failed, reason: openAI API async stream error: "
            r"InternalServerError: Error code: (?:500|502|503|504) - "
            r"\{'detail': 'Upstream request failed; see campaign transport receipt'\}"
        )
        if (
            terminal != result
            or result.get("status") != "failed"
            or result.get("exit_code") != 1
            or error.get("code") != "RUNTIME_ERROR"
            or not re.fullmatch(expected, error.get("message", ""))
            or error["message"] not in actor.get("reason", "")
            or actor.get("error_type") != "RuntimeError"
            or not isinstance(session, str)
            or not session
            or Path(session).name != session
            or session in {".", ".."}
            or actor.get("session") != session
            or execution.get("status") != "exited"
            or execution.get("returncode") != 1
            or duration >= budget
        ):
            raise ValueError("Native upstream server failure is unproven")
        archive = slot / "sessions"
        if not archive.is_dir() or {p.name for p in archive.iterdir()} != {"1"}:
            raise ValueError("Failed native session archive is absent or ambiguous")
        session_root = archive / "1/data/agent/sessions" / session
        for path in (session_root / "metadata.json", session_root / "history.jsonl"):
            _plain_file(path, slot)
        if (
            read(session_root / "metadata.json").get("session_id") != session
            or not (session_root / "history.jsonl").read_text().strip()
        ):
            raise ValueError("Failed native session identity or history is unproven")
        return "native_upstream_server_failure", session, None, duration
    elapsed = execution.get("timeout_elapsed_s")
    if (
        actor.get("error_type") != "TimeoutError"
        or actor.get("session") is not None
        or (slot / "sessions").exists()
        or execution.get("status") != "timed_out"
        or execution.get("error") != "TimeoutError"
        or execution.get("timeout_source") != "host"
        or execution.get("returncode") != 124
        or not _finite_nonnegative(elapsed)
        or not budget + 15 <= elapsed <= budget + 16
        or duration < elapsed
    ):
        raise ValueError("Host queue timeout termination is unproven")
    rows = [json.loads(line) for line in text.splitlines()]
    if any(isinstance(row, dict) and row.get("type") == "result" for row in rows):
        rows, _ = _native_timeout_terminal(text, request, execution, budget)
    sessions = set()
    for number, row in enumerate(rows):
        if (
            not isinstance(row, dict)
            or row.get("type") != "event"
            or type(row.get("sequence")) is not int
            or row["sequence"] != number
            or row.get("request_id") != request["request_id"]
        ):
            raise ValueError(
                "Queue timeout event stream identity or sequence is invalid"
            )
        session = row.get("session_id")
        if session is not None:
            if not isinstance(session, str) or not session:
                raise ValueError("Queue timeout event session is invalid")
            sessions.add(session)
    if not rows or len(sessions) > 1:
        raise ValueError("Queue timeout execution evidence is absent or ambiguous")
    return (
        "gateway_queue_host_timeout",
        next(iter(sessions), None),
        _gateway_evidence(config, slot, execution),
        duration,
    )


def propose(config, task_id, stage_name, *, archived=None):
    """Inspect three closed failures and their full lineage without consuming an approval."""
    root, original = _locations(config, task_id, stage_name)
    state = root / "state.json"
    if archived is None and (
        root.joinpath("result.json").exists()
        or state.exists()
        and read(state).get("status") == "completed"
    ):
        raise ValueError("Completed tasks cannot receive fresh Actor executions")
    prior = claim_path(root, stage_name, UPSTREAM_POLICY)
    authorization = prior / "authorization.json"
    _plain_file(authorization, root)
    previous = _approved(read(authorization))
    if previous["policy"] != UPSTREAM_POLICY or previous != propose_transport_retry(
        config, task_id, stage_name, archived=prior, policy=UPSTREAM_POLICY
    ):
        raise ValueError("Prior Actor failures or consumed authorization changed")
    slot = prior / "actor"
    files = _inventory(slot)
    stage_path = (
        archived / "previous_stage.json"
        if archived
        else root / stage_name / "stage.json"
    )
    failure_path = (
        archived / "previous_failure.json"
        if archived
        else root / stage_name / "failure.json"
    )
    history = (
        archived / "previous_attempt_failure.json"
        if archived
        else root / stage_name / "failures/3.json"
    )
    original_history = root / stage_name / "failures/3.json"
    for path in (stage_path, failure_path, history, original_history):
        _plain_file(path, root)
    stage, failure = read(stage_path), read(failure_path)
    actor, started = read(slot / "FAILURE.json"), read(slot / "STARTED.json")
    identity = previous["actor_identity"]
    if (
        stage.get("status") != "started"
        or type(stage.get("attempt")) is not int
        or stage["attempt"] != 3
        or previous["original_attempt"] != 2
        or stage.get("identity") != previous["stage_identity"]
        or any(
            failure.get(k) != stage.get(k)
            for k in ("attempt", "identity", "implementation")
        )
        or actor.get("identity") != identity
        or started.get("identity") != identity
        or failure.get("error_type") != actor.get("error_type")
        or failure.get("reason") != actor.get("reason")
        or any(
            file_hash(path) != file_hash(failure_path)
            for path in (history, original_history)
        )
        or any(
            (slot / name).exists()
            for name in ("RESULT.json", "actor_state.json", "snapshots")
        )
    ):
        raise ValueError(
            "Third Actor failure identity, history, or absence of delivery is unproven"
        )
    classification, session, gateway, duration = _failure(config, slot, actor)
    costs = [
        previous["first_failed_execution_s"],
        previous["native_failed_execution_s"],
        duration,
    ]
    return {
        "policy": INFRASTRUCTURE_POLICY,
        "campaign": str(Path(config["output"]).resolve()),
        "checkpoint_identity": config.get(
            "checkpoint_config_identity", config["identity"]
        ),
        "task_id": task_id,
        "stage": stage_name,
        "original_attempt": 3,
        "stage_identity": stage["identity"],
        "actor_identity": identity,
        "model": config["roles"]["actor"],
        "actor_protocol_identity": previous["actor_protocol_identity"],
        "actor_timeout_s": config["actor_timeout_s"],
        "max_additional_attempts": 1,
        "failed_runtime_directory": str(slot),
        "original_runtime_directory": str(original),
        "prior_authorization": str(authorization),
        "prior_authorization_sha256": file_hash(authorization),
        "failed_executions_s": costs,
        "failed_execution_s": sum(costs),
        "observed_session_id": session,
        "native_session_id": session
        if classification == "native_upstream_server_failure"
        else None,
        "source_files": files,
        "failure_classification": classification,
        "gateway_evidence": gateway,
        "recovery_mode": "fresh_actor_after_proven_infrastructure_failure",
        "stage_sha256": file_hash(stage_path),
        "failure_sha256": file_hash(failure_path),
        "failure_history_sha256": file_hash(history),
    }


def recover(manager, key, stage, document, proposal):
    root, original = _locations(manager.config, *key)
    claim = claim_path(root, key[1], INFRASTRUCTURE_POLICY)
    if claim.exists() or claim.is_symlink() or claim.parent.is_symlink():
        raise ValueError(
            "Infrastructure retry authorization was already consumed or path is linked"
        )
    if (
        read(root / key[1] / "stage.json") != stage
        or propose(manager.config, *key) != proposal
    ):
        raise ValueError("Infrastructure retry source evidence changed")
    claim.parent.mkdir(exist_ok=True)
    claim.mkdir()
    archive_authorization(root, claim, key[1], stage, document)
    if propose(manager.config, *key, archived=claim) != proposal:
        raise ValueError("Infrastructure retry evidence changed while archiving")
    manager.active[key] = (original, claim / "actor", proposal)
    return True


def result_fields(slot, proposal, execution_s):
    receipt = slot.parent / "authorization.json"
    return {
        "actor_retry": {
            "policy": INFRASTRUCTURE_POLICY,
            "additional_attempts": 3,
            "this_authorization_attempts": 1,
            "authorization": str(receipt),
            "authorization_sha256": file_hash(receipt),
            "failed_runtime_directory": proposal["failed_runtime_directory"],
            "original_runtime_directory": proposal["original_runtime_directory"],
            "actor_timeout_s_per_attempt": proposal["actor_timeout_s"],
            "failed_executions_s": proposal["failed_executions_s"],
            "failed_execution_s": proposal["failed_execution_s"],
            "successful_execution_s": execution_s,
            "total_execution_s": proposal["failed_execution_s"] + execution_s,
            "failed_duration_includes_cleanup": True,
            "recovery_mode": proposal["recovery_mode"],
            "failure_classification": proposal["failure_classification"],
            "retained_native_session_id": proposal["native_session_id"],
            "observed_failed_session_id": proposal["observed_session_id"],
        }
    }


def verify(row):
    runtime = Path(row["runtime_directory"])
    receipt = runtime.parent / "authorization.json"
    _plain_file(receipt, runtime.parent)
    proposal = _approved(read(receipt))
    config = {
        "output": proposal["campaign"],
        "task_ids": [proposal["task_id"]],
        "identity": proposal["checkpoint_identity"],
        "actor_timeout_s": proposal["actor_timeout_s"],
        "roles": {"actor": proposal["model"]},
        "actor_backend": "jiuwenswarm",
        "generation_protocol": "three_b_fixed_baselines_v1",
    }
    root, _ = _locations(config, proposal["task_id"], proposal["stage"])
    claim = claim_path(root, proposal["stage"], INFRASTRUCTURE_POLICY)
    if (
        proposal["policy"] != INFRASTRUCTURE_POLICY
        or runtime != claim / "actor"
        or claim.is_symlink()
        or claim.parent.is_symlink()
        or row.get("identity") != proposal["actor_identity"]
        or propose(config, proposal["task_id"], proposal["stage"], archived=claim)
        != proposal
    ):
        raise ValueError("Infrastructure retry lineage or evidence changed")
    elapsed = row.get("actor_execution_s")
    if (
        not _finite_nonnegative(elapsed)
        or row["actor_retry"]
        != result_fields(runtime, proposal, elapsed)["actor_retry"]
    ):
        raise ValueError("Infrastructure retry cost or attempt count changed")
