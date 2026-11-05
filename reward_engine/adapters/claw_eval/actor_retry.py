"""Explicit, evidence-bound authorization for one terminated Actor retry."""

import json
import math
import re
import shutil
from pathlib import Path

from reward_engine.adapters.workbuddy.io import file_hash, read, write
from reward_engine.core.outcome.construction import fingerprint

POLICY = "one_terminated_actor_retry_v1"
TRANSPORT_POLICY = "one_native_transport_actor_retry_v1"
UPSTREAM_POLICY = "one_native_upstream_actor_retry_v1"
INFRASTRUCTURE_POLICY = "one_terminated_infrastructure_actor_retry_v1"


def _locations(config, task_id, stage_name):
    if (
        config.get("actor_backend") != "jiuwenswarm"
        or config.get("generation_protocol") != "three_b_fixed_baselines_v1"
        or task_id not in config["task_ids"]
    ):
        raise ValueError(
            "Actor retry requires the frozen JiuwenSwarm three-B experiment"
        )
    campaign = Path(config["output"]).resolve()
    root = campaign / "tasks" / task_id
    if re.fullmatch(r"seed_[0-2]", stage_name):
        slot = root / "seeds" / str(int(stage_name[-1]) + 1)
    elif re.fullmatch(r"round_[1-3]/B[1-3]", stage_name):
        round_name, arm = stage_name.split("/")
        slot = root / round_name / "actors" / arm
    else:
        raise ValueError("Unknown Actor stage")
    for path in (root.parent, root, slot.parent.parent, slot.parent, slot):
        if path.is_symlink() or not path.resolve().is_relative_to(campaign):
            raise ValueError("Actor retry path escapes the experiment")
    return root, slot


def _inventory(root):
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("Actor evidence directory is missing or linked")
    hashes = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError("Actor evidence contains linked or special files")
        if path.is_file():
            hashes[str(path.relative_to(root))] = file_hash(path)
    return hashes


def _finite_nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _plain_file(path, root):
    if (
        not path.is_file()
        or path.is_symlink()
        or not path.resolve().is_relative_to(root.resolve())
        or any(parent.is_symlink() for parent in path.parents if parent != root.parent)
    ):
        raise ValueError(
            "Actor retry evidence file is missing, linked or outside the task"
        )


def propose_actor_retry(config, task_id, stage_name):
    """Inspect only; produce no authorization and change no experiment files."""
    return _proposal(config, task_id, stage_name)


def _native_timeout_terminal(text, request, execution, budget):
    from reward_engine.adapters.jiuwenswarm.process import (
        ProcessProtocolError,
        _validate_record_stream,
    )

    try:
        events, terminal = _validate_record_stream(
            text, request, execution["returncode"]
        )
    except ProcessProtocolError as error:
        raise ValueError("Native timeout stream is invalid or ambiguous") from error
    elapsed = execution.get("timeout_elapsed_s")
    if (
        not events
        or terminal.get("status") != "timed_out"
        or terminal.get("exit_code") != 124
        or terminal.get("error")
        != {
            "code": "TIMEOUT",
            "message": "Command timed out.",
            "retryable": False,
            "details": {},
        }
        or not terminal.get("session_id")
        or execution.get("requested_timeout_s") != budget
        or execution.get("host_timeout_s") != budget + 15
        or not _finite_nonnegative(elapsed)
        or not budget + 15 <= elapsed <= budget + 16
        or execution["duration_s"] < elapsed
    ):
        raise ValueError("Native timeout termination or unchanged deadline is unproven")
    return events, {
        "status": terminal["status"],
        "exit_code": terminal["exit_code"],
        "sequence": terminal["sequence"],
    }


def _proposal(config, task_id, stage_name, *, archived=None):
    root, slot = _locations(config, task_id, stage_name)
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
    _plain_file(stage_path, root)
    _plain_file(failure_path, root)
    stage, failure = read(stage_path), read(failure_path)
    attempt = stage.get("attempt")
    if (
        stage.get("status") != "started"
        or type(attempt) is not int
        or attempt < 1
        or failure.get("attempt") != attempt
        or failure.get("identity") != stage.get("identity")
        or failure.get("error_type") != "TimeoutError"
    ):
        raise ValueError("Actor stage is not a bound failed timeout")
    history = (
        archived / "previous_attempt_failure.json"
        if archived
        else root / stage_name / "failures" / f"{attempt}.json"
    )
    _plain_file(history, root)
    if not history.is_file() or file_hash(history) != file_hash(failure_path):
        raise ValueError("Original Actor failure history is missing or changed")
    source_files = _inventory(slot)
    actor_failure, started = read(slot / "FAILURE.json"), read(slot / "STARTED.json")
    identity = started.get("identity")
    if (
        not isinstance(identity, str)
        or not identity
        or actor_failure.get("identity") != identity
        or actor_failure.get("error_type") != "TimeoutError"
        or actor_failure.get("session") is not None
        or any(
            (slot / name).exists()
            for name in ("RESULT.json", "actor_state.json", "sessions", "snapshots")
        )
    ):
        raise ValueError("Actor has ambiguous identity or a prior delivery/session")
    process = slot / "jiuwenswarm/process"
    if {p.name for p in process.glob("*.request.json")} != {
        "turn-1.request.json"
    } or list(process.glob("*.result.json")):
        raise ValueError(
            "Only a failed first execution without a terminal result is eligible"
        )
    request = read(process / "turn-1.request.json")
    execution = read(process / "turn-1.execution.json")
    budget = config["actor_timeout_s"]
    if (
        not _finite_nonnegative(budget)
        or budget <= 0
        or not isinstance(request.get("request_id"), str)
        or not request["request_id"]
        or request.get("timeout_seconds") != budget
        or request.get("session_id") is not None
        or execution.get("status") != "timed_out"
        or execution.get("error") != "TimeoutError"
        or execution.get("timeout_source", "host") != "host"
        or type(execution.get("returncode")) is not int
        or execution["returncode"] == 0
        or "cleanup_error" in execution
        or not _finite_nonnegative(execution.get("duration_s"))
        or execution["duration_s"] < budget
    ):
        raise ValueError("Actor termination or unchanged execution budget is unproven")
    sessions = set()
    count = 0
    text = (process / "turn-1.jsonl").read_text()
    try:
        rows = [json.loads(line) for line in text.splitlines()]
    except ValueError as error:
        raise ValueError("Actor event stream is truncated or invalid") from error
    native_timeout = None
    if any(isinstance(row, dict) and row.get("type") == "result" for row in rows):
        rows, native_timeout = _native_timeout_terminal(
            text, request, execution, budget
        )
    for row in rows:
        if (
            not isinstance(row, dict)
            or row.get("type") != "event"
            or type(row.get("sequence")) is not int
            or row["sequence"] != count
            or row.get("request_id") != request.get("request_id")
        ):
            raise ValueError(
                "Actor stream has a terminal, changed identity or invalid sequence"
            )
        if row.get("session_id") is not None:
            if not isinstance(row["session_id"], str) or not row["session_id"]:
                raise ValueError("Actor event session is invalid")
            sessions.add(row["session_id"])
        count += 1
    if not count or len(sessions) > 1:
        raise ValueError("Actor execution evidence is absent or ambiguous")
    return {
        "policy": POLICY,
        "campaign": str(Path(config["output"]).resolve()),
        "checkpoint_identity": config.get(
            "checkpoint_config_identity", config["identity"]
        ),
        "task_id": task_id,
        "stage": stage_name,
        "original_attempt": attempt,
        "stage_identity": stage["identity"],
        "actor_identity": identity,
        "actor_timeout_s": budget,
        "max_additional_attempts": 1,
        "failed_execution_s": execution["duration_s"],
        "observed_session_id": next(iter(sessions), None),
        "stage_sha256": file_hash(stage_path),
        "failure_sha256": file_hash(failure_path),
        "failure_history_sha256": file_hash(history),
        "source_files": source_files,
        **(
            {"native_timeout_terminal": native_timeout}
            if native_timeout is not None
            else {}
        ),
    }


def _approved(document):
    if not isinstance(document, dict):
        raise ValueError("Explicit matching Actor retry approval is required")
    proposal = document.get("proposal")
    approval = document.get("approval")
    if (
        not isinstance(proposal, dict)
        or not isinstance(approval, dict)
        or approval.get("decision") != "approve"
        or approval.get("proposal_sha256") != fingerprint(proposal)
        or proposal.get("policy")
        not in {POLICY, TRANSPORT_POLICY, UPSTREAM_POLICY, INFRASTRUCTURE_POLICY}
        or type(proposal.get("max_additional_attempts")) is not int
        or proposal["max_additional_attempts"] != 1
    ):
        raise ValueError("Explicit matching Actor retry approval is required")
    return proposal


def archive_authorization(root, claim, stage_name, stage, document):
    """Retain the same authorization and prior failure bytes for every retry policy."""
    sources = (
        (root / stage_name / "stage.json", "previous_stage.json"),
        (root / stage_name / "failure.json", "previous_failure.json"),
        (
            root / stage_name / "failures" / f"{stage['attempt']}.json",
            "previous_attempt_failure.json",
        ),
    )
    for source, _ in sources:
        _plain_file(source, root)
    write(claim / "authorization.json", document)
    for source, destination in sources:
        shutil.copy2(source, claim / destination)


class ActorRetries:
    """Used only when an operator supplies approved proposals at launch."""

    def __init__(self, config, documents):
        self.config, self.entries, self.active = config, {}, {}
        for document in documents:
            proposal = _approved(document)
            key = (proposal["task_id"], proposal["stage"])
            _locations(config, *key)
            if (
                key in self.entries
                or proposal["campaign"] != str(Path(config["output"]).resolve())
                or proposal["checkpoint_identity"]
                != config.get("checkpoint_config_identity", config["identity"])
                or proposal["actor_timeout_s"] != config["actor_timeout_s"]
            ):
                raise ValueError(
                    "Actor retry authorization changed experiment identity or budget"
                )
            self.entries[key] = document

    def recover(self, task_id, stage_name, stage):
        key = (task_id, stage_name)
        if key not in self.entries:
            return False
        document = self.entries[key]
        proposal = _approved(document)
        if proposal["policy"] in {
            TRANSPORT_POLICY,
            UPSTREAM_POLICY,
            INFRASTRUCTURE_POLICY,
        }:
            from .actor_transport_retry import recover

            return recover(self, key, stage, document, proposal)
        root, old_slot = _locations(self.config, *key)
        claim = root / "actor_retries" / fingerprint({"stage": stage_name})
        if claim.exists() or claim.is_symlink():
            self._recover_unstarted(root, claim, key, stage, document, proposal)
            self.active[key] = (old_slot, claim / "actor", proposal)
            return True
        if (
            read(root / stage_name / "stage.json") != stage
            or propose_actor_retry(self.config, *key) != proposal
        ):
            raise ValueError("Actor retry source evidence or stage identity changed")
        if claim.parent.is_symlink():
            raise ValueError("Actor retry archive is linked")
        claim.parent.mkdir(exist_ok=True)
        claim.mkdir()
        archive_authorization(root, claim, stage_name, stage, document)
        self.active[key] = (old_slot, claim / "actor", proposal)
        return True

    def _recover_unstarted(self, root, claim, key, stage, document, proposal):
        """An identity preflight failure used no Actor execution allowance."""
        required = {
            "authorization.json",
            "previous_stage.json",
            "previous_failure.json",
            "previous_attempt_failure.json",
        }
        if (
            claim.is_symlink()
            or claim.parent.is_symlink()
            or not claim.is_dir()
            or {p.name for p in claim.iterdir()} != required
        ):
            raise ValueError("Actor retry authorization was already consumed")
        for name in required:
            _plain_file(claim / name, root)
        number = proposal["original_attempt"] + 1
        failure_path = root / key[1] / "failure.json"
        history = root / key[1] / "failures" / f"{number}.json"
        original = root / key[1] / "failures" / f"{proposal['original_attempt']}.json"
        for path in (root / key[1] / "stage.json", failure_path, history, original):
            if not path.exists():
                raise ValueError("Actor retry authorization was already consumed")
            _plain_file(path, root)
        failure = read(failure_path)
        if (
            read(claim / "authorization.json") != document
            or stage != read(root / key[1] / "stage.json")
            or stage.get("status") != "started"
            or stage.get("attempt") != number
            or stage.get("identity") != proposal["stage_identity"]
            or failure.get("identity") != proposal["stage_identity"]
            or failure.get("attempt") != number
            or failure.get("error_type") != "ValueError"
            or failure.get("reason") != "Actor retry input identity changed"
            or file_hash(history) != file_hash(failure_path)
            or file_hash(original) != proposal["failure_history_sha256"]
            or _proposal(self.config, *key, archived=claim) != proposal
        ):
            raise ValueError(
                "Actor retry authorization was already consumed or evidence changed"
            )

    def protocol_identity(self, output, default):
        """Preserve the original pre-explicit-protocol binding across code revisions."""
        if self.config.get("actor_protocol_identity") is not None:
            return default
        for _, slot, proposal in self.active.values():
            if Path(output).resolve() == slot.resolve():
                path = slot.parent / "previous_stage.json"
                if file_hash(path) != proposal["stage_sha256"]:
                    raise ValueError("Actor retry archived stage evidence changed")
                if proposal["policy"] in {
                    TRANSPORT_POLICY,
                    UPSTREAM_POLICY,
                    INFRASTRUCTURE_POLICY,
                }:
                    return proposal["actor_protocol_identity"]
                return read(path).get("implementation", default)
        return default

    def slot(self, task_id, stage_name, original):
        selected = self.active.get((task_id, stage_name))
        if selected is None:
            return original
        if Path(original).resolve() != selected[0].resolve():
            raise ValueError("Actor retry logical slot changed")
        return selected[1]

    def validate_slot(self, output, identity):
        for _, slot, proposal in self.active.values():
            if (
                Path(output).resolve() == slot.resolve()
                and identity != proposal["actor_identity"]
            ):
                raise ValueError("Actor retry input identity changed")

    def result_fields(self, output, execution_s):
        for old_slot, slot, proposal in self.active.values():
            if Path(output).resolve() != slot.resolve():
                continue
            if not _finite_nonnegative(execution_s):
                raise ValueError("Actor retry execution cost is invalid")
            if proposal["policy"] in {
                TRANSPORT_POLICY,
                UPSTREAM_POLICY,
                INFRASTRUCTURE_POLICY,
            }:
                from .actor_transport_retry import result_fields

                return result_fields(slot, proposal, execution_s)
            receipt = slot.parent / "authorization.json"
            return {
                "actor_retry": {
                    "policy": POLICY,
                    "additional_attempts": 1,
                    "authorization": str(receipt),
                    "authorization_sha256": file_hash(receipt),
                    "failed_runtime_directory": str(old_slot),
                    "actor_timeout_s_per_attempt": proposal["actor_timeout_s"],
                    "failed_execution_s": proposal["failed_execution_s"],
                    "successful_execution_s": execution_s,
                    "total_execution_s": proposal["failed_execution_s"] + execution_s,
                    "failed_duration_includes_cleanup": True,
                }
            }
        return {}


def verify_actor_retry(row):
    """Verify retry lineage when a completed Actor is reused across launches."""
    proof = row.get("actor_retry")
    if proof is None:
        return
    if not isinstance(proof, dict):
        raise ValueError("Actor retry proof must be an object")
    if proof.get("policy") in {
        TRANSPORT_POLICY,
        UPSTREAM_POLICY,
        INFRASTRUCTURE_POLICY,
    }:
        from .actor_transport_retry import verify

        verify(row)
        return
    runtime = Path(row["runtime_directory"])
    receipt = runtime.parent / "authorization.json"
    _plain_file(receipt, runtime.parent)
    if proof.get("authorization") != str(receipt) or proof.get(
        "authorization_sha256"
    ) != file_hash(receipt):
        raise ValueError("Actor retry authorization receipt changed")
    proposal = _approved(read(receipt))
    config = {
        "output": proposal["campaign"],
        "task_ids": [proposal["task_id"]],
        "actor_backend": "jiuwenswarm",
        "generation_protocol": "three_b_fixed_baselines_v1",
    }
    root, original = _locations(config, proposal["task_id"], proposal["stage"])
    claim = root / "actor_retries" / fingerprint({"stage": proposal["stage"]})
    if "native_timeout_terminal" in proposal:
        config["identity"] = proposal["checkpoint_identity"]
        config["actor_timeout_s"] = proposal["actor_timeout_s"]
        if (
            _proposal(config, proposal["task_id"], proposal["stage"], archived=claim)
            != proposal
        ):
            raise ValueError("Native timeout retry lineage or evidence changed")
    if (
        runtime != claim / "actor"
        or claim.is_symlink()
        or claim.parent.is_symlink()
        or row.get("identity") != proposal["actor_identity"]
        or _inventory(original) != proposal["source_files"]
    ):
        raise ValueError("Actor retry source identity, path or evidence changed")
    for filename, key in (
        ("previous_stage.json", "stage_sha256"),
        ("previous_failure.json", "failure_sha256"),
        ("previous_attempt_failure.json", "failure_history_sha256"),
    ):
        _plain_file(claim / filename, root)
        if file_hash(claim / filename) != proposal[key]:
            raise ValueError("Actor retry archived stage evidence changed")
    history = (
        root / proposal["stage"] / "failures" / f"{proposal['original_attempt']}.json"
    )
    _plain_file(history, root)
    if file_hash(history) != proposal["failure_history_sha256"]:
        raise ValueError("Original Actor failure receipt changed")
    elapsed = row.get("actor_execution_s")
    if (
        not _finite_nonnegative(elapsed)
        or proof.get("policy") != POLICY
        or type(proof.get("additional_attempts")) is not int
        or proof["additional_attempts"] != 1
        or proof.get("failed_runtime_directory") != str(original)
        or proof.get("actor_timeout_s_per_attempt") != proposal["actor_timeout_s"]
        or proof.get("failed_execution_s") != proposal["failed_execution_s"]
        or proof.get("successful_execution_s") != elapsed
        or proof.get("total_execution_s") != proposal["failed_execution_s"] + elapsed
        or proof.get("failed_duration_includes_cleanup") is not True
    ):
        raise ValueError("Actor retry execution cost or attempt count changed")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare an unapproved Actor retry proposal; never launches an Actor"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    proposal = propose_actor_retry(read(args.config), args.task_id, args.stage)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump({"proposal": proposal, "approval": None}, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
