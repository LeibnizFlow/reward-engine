"""One separately authorized native transport retry, retaining both prior failures."""

import re
from pathlib import Path

from reward_engine.adapters.jiuwenswarm.process import _validate_record_stream
from reward_engine.adapters.workbuddy.io import file_hash, read
from reward_engine.core.outcome.construction import fingerprint

from .actor_retry import (
    INFRASTRUCTURE_POLICY,
    POLICY,
    TRANSPORT_POLICY,
    UPSTREAM_POLICY,
    _approved,
    _finite_nonnegative,
    _inventory,
    _locations,
    _plain_file,
    archive_authorization,
)
from .actor_retry import (
    _proposal as timeout_proposal,
)


def claim_path(root, stage, policy=TRANSPORT_POLICY):
    if policy == INFRASTRUCTURE_POLICY:
        return (
            root
            / "actor_infrastructure_retries"
            / fingerprint({"stage": stage, "policy": policy})
        )
    if policy not in {TRANSPORT_POLICY, UPSTREAM_POLICY}:
        raise ValueError("Unknown native transport retry policy")
    return (
        root
        / "actor_transport_retries"
        / fingerprint({"stage": stage, "policy": policy})
    )


def propose_transport_retry(
    config, task_id, stage_name, *, archived=None, policy=TRANSPORT_POLICY
):
    """Inspect a consumed first retry; never infer approval or manufacture a session."""
    if policy == INFRASTRUCTURE_POLICY:
        from .actor_infrastructure_retry import propose

        return propose(config, task_id, stage_name, archived=archived)
    if policy not in {TRANSPORT_POLICY, UPSTREAM_POLICY}:
        raise ValueError("Unknown native transport retry policy")
    root, original = _locations(config, task_id, stage_name)
    prior = root / "actor_retries" / fingerprint({"stage": stage_name})
    authorization = prior / "authorization.json"
    _plain_file(authorization, root)
    previous = _approved(read(authorization))
    if previous["policy"] != POLICY or previous != timeout_proposal(
        config, task_id, stage_name, archived=prior
    ):
        raise ValueError("Prior Actor retry evidence or authorization changed")
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
        else root / stage_name / "failures/2.json"
    )
    for path in (stage_path, failure_path, history):
        _plain_file(path, root)
    stage, failure = read(stage_path), read(failure_path)
    actor = read(slot / "FAILURE.json")
    identity = previous["actor_identity"]
    if (
        stage.get("status") != "started"
        or stage.get("attempt") != 2
        or previous["original_attempt"] != 1
        or stage.get("identity") != previous["stage_identity"]
        or any(
            failure.get(k) != stage.get(k)
            for k in ("attempt", "identity", "implementation")
        )
        or failure.get("error_type") != "RuntimeError"
        or actor.get("error_type") != "RuntimeError"
        or actor.get("identity") != identity
        or read(slot / "STARTED.json").get("identity") != identity
        or failure.get("reason") != actor.get("reason")
        or file_hash(history) != file_hash(failure_path)
        or any(
            (slot / name).exists()
            for name in ("RESULT.json", "actor_state.json", "snapshots")
        )
        or policy == TRANSPORT_POLICY
        and (slot / "sessions").exists()
    ):
        raise ValueError(
            "Native Actor failure is unbound, already delivered, or has recoverable session state"
        )
    process = slot / "jiuwenswarm/process"
    if {p.name for p in process.glob("*.request.json")} != {"turn-1.request.json"} or {
        p.name for p in process.glob("*.result.json")
    } != {"turn-1.result.json"}:
        raise ValueError("Transport recovery requires one failed first execution")
    request, execution, result = (
        read(process / f"turn-1.{name}.json")
        for name in ("request", "execution", "result")
    )
    _, terminal = _validate_record_stream(
        (process / "turn-1.jsonl").read_text(), request, execution.get("returncode")
    )
    budget, duration = config["actor_timeout_s"], execution.get("duration_s")
    model = config["roles"]["actor"]
    error = result.get("error") or {}
    expected = (
        r"\[181001\] model call failed, reason: LLM stream timeout: stream frame timeout: "
        r"stage=first_chunk, timeout=[0-9.]+s, chunk_count=0, "
        r"first_chunk_elapsed=[0-9.]+s, total_elapsed=[0-9.]+s, model="
        + re.escape(model)
    )
    if policy == UPSTREAM_POLICY:
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
        or not isinstance(result.get("session_id"), str)
        or not result["session_id"]
        or actor.get("session") not in (None, result["session_id"])
        or request.get("session_id") is not None
        or request.get("timeout_seconds") != budget
        or execution.get("requested_timeout_s") != budget
        or execution.get("status") != "exited"
        or execution.get("returncode") != 1
        or execution.get("signals")
        or "cleanup_error" in execution
        or not _finite_nonnegative(duration)
        or not 0 < duration < budget
    ):
        raise ValueError("Native transport failure or unchanged budget is unproven")
    if policy == UPSTREAM_POLICY:
        session = result["session_id"]
        if Path(session).name != session or session in {".", ".."}:
            raise ValueError("Retained native session identity is not a plain name")
        archive_root = slot / "sessions"
        if not archive_root.is_dir() or {
            path.name for path in archive_root.iterdir()
        } != {"1"}:
            raise ValueError("Retained native execution archive is ambiguous")
        session_root = slot / "sessions/1/data/agent/sessions" / session
        for path in (session_root / "metadata.json", session_root / "history.jsonl"):
            _plain_file(path, slot)
        if (
            actor.get("session") != session
            or read(session_root / "metadata.json").get("session_id") != session
            or not (session_root / "history.jsonl").read_text().strip()
            or execution.get("host_timeout_s") != budget + 15
        ):
            raise ValueError(
                "Retained native session or unchanged host budget is unproven"
            )
    original_history = root / stage_name / "failures/2.json"
    _plain_file(original_history, root)
    if file_hash(original_history) != file_hash(failure_path):
        raise ValueError("Native Actor failure history changed")
    return {
        "policy": policy,
        "campaign": str(Path(config["output"]).resolve()),
        "checkpoint_identity": config.get(
            "checkpoint_config_identity", config["identity"]
        ),
        "task_id": task_id,
        "stage": stage_name,
        "original_attempt": 2,
        "stage_identity": stage["identity"],
        "actor_identity": identity,
        "model": model,
        "actor_protocol_identity": read(prior / "previous_stage.json")[
            "implementation"
        ],
        "actor_timeout_s": budget,
        "max_additional_attempts": 1,
        "failed_runtime_directory": str(slot),
        "original_runtime_directory": str(original),
        "prior_authorization": str(authorization),
        "prior_authorization_sha256": file_hash(authorization),
        "first_failed_execution_s": previous["failed_execution_s"],
        "native_failed_execution_s": duration,
        "failed_execution_s": previous["failed_execution_s"] + duration,
        "native_session_id": result["session_id"],
        "source_files": files,
        "stage_sha256": file_hash(stage_path),
        "failure_sha256": file_hash(failure_path),
        "failure_history_sha256": file_hash(history),
        **(
            {"recovery_mode": "fresh_actor_with_retained_failed_session"}
            if policy == UPSTREAM_POLICY
            else {}
        ),
    }


def recover(manager, key, stage, document, proposal):
    if proposal["policy"] == INFRASTRUCTURE_POLICY:
        from .actor_infrastructure_retry import recover as recover_infrastructure

        return recover_infrastructure(manager, key, stage, document, proposal)
    root, original = _locations(manager.config, *key)
    policy = proposal["policy"]
    claim = claim_path(root, key[1], policy)
    if claim.exists() or claim.is_symlink() or claim.parent.is_symlink():
        raise ValueError(
            "Native transport retry authorization was already consumed or path is linked"
        )
    if (
        read(root / key[1] / "stage.json") != stage
        or propose_transport_retry(manager.config, *key, policy=policy) != proposal
    ):
        raise ValueError("Native transport retry source evidence changed")
    claim.mkdir(parents=True)
    archive_authorization(root, claim, key[1], stage, document)
    manager.active[key] = (original, claim / "actor", proposal)
    return True


def result_fields(slot, proposal, execution_s):
    if proposal["policy"] == INFRASTRUCTURE_POLICY:
        from .actor_infrastructure_retry import result_fields as infrastructure_fields

        return infrastructure_fields(slot, proposal, execution_s)
    receipt = slot.parent / "authorization.json"
    return {
        "actor_retry": {
            "policy": proposal["policy"],
            "additional_attempts": 2,
            "this_authorization_attempts": 1,
            "authorization": str(receipt),
            "authorization_sha256": file_hash(receipt),
            "failed_runtime_directory": proposal["failed_runtime_directory"],
            "original_runtime_directory": proposal["original_runtime_directory"],
            "actor_timeout_s_per_attempt": proposal["actor_timeout_s"],
            "first_failed_execution_s": proposal["first_failed_execution_s"],
            "native_failed_execution_s": proposal["native_failed_execution_s"],
            "failed_execution_s": proposal["failed_execution_s"],
            "successful_execution_s": execution_s,
            "total_execution_s": proposal["failed_execution_s"] + execution_s,
            "failed_duration_includes_cleanup": True,
            **(
                {
                    "recovery_mode": proposal["recovery_mode"],
                    "retained_native_session_id": proposal["native_session_id"],
                }
                if proposal["policy"] == UPSTREAM_POLICY
                else {}
            ),
        }
    }


def verify(row):
    runtime = Path(row["runtime_directory"])
    receipt = runtime.parent / "authorization.json"
    _plain_file(receipt, runtime.parent)
    proposal = _approved(read(receipt))
    if proposal["policy"] == INFRASTRUCTURE_POLICY:
        from .actor_infrastructure_retry import verify as verify_infrastructure

        return verify_infrastructure(row)
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
    policy = proposal["policy"]
    claim = claim_path(root, proposal["stage"], policy)
    if (
        runtime != claim / "actor"
        or claim.is_symlink()
        or claim.parent.is_symlink()
        or row.get("identity") != proposal["actor_identity"]
        or propose_transport_retry(
            config,
            proposal["task_id"],
            proposal["stage"],
            archived=claim,
            policy=policy,
        )
        != proposal
    ):
        raise ValueError("Native transport retry lineage or evidence changed")
    elapsed = row.get("actor_execution_s")
    if (
        not _finite_nonnegative(elapsed)
        or row["actor_retry"]
        != result_fields(runtime, proposal, elapsed)["actor_retry"]
    ):
        raise ValueError("Native transport retry cost or attempt count changed")
