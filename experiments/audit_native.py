"""Audit native for the existing public entry point."""

import hashlib
import json
from pathlib import Path


def native_turn(r, slot, benchmark, cid, session, number):
    from experiments.audit_receipts import candidate, require, retained_workspace

    if benchmark == "claw_eval":
        folder, pattern = slot, f"actor-{number}-*.jsonl"
        expected = slot / f"actor-{number}-0.jsonl"
    else:
        folder = slot / "harbor" / cid / "agent"
        pattern = f"outcome-turn-{number}*.jsonl"
        expected = folder / f"outcome-turn-{number}.jsonl"
    # Read the mandatory first log even when glob currently finds nothing.
    r.read(expected, lines=True)
    paths = sorted(r.path(folder).glob(pattern))
    require(1 <= len(paths) <= 3, "Invalid native continuation count")
    if benchmark == "claw_eval":
        names = [f"actor-{number}-{i}.jsonl" for i in range(len(paths))]
    else:
        names = [f"outcome-turn-{number}.jsonl"] + [
            f"outcome-turn-{number}-continuation-{i}.jsonl"
            for i in range(1, len(paths))
        ]
    require({p.name for p in paths} == set(names), "Noncontiguous native continuations")
    last = None
    for index, name in enumerate(names):
        events = r.read(folder / name, lines=True)
        require(
            events and all(isinstance(e, dict) for e in events),
            "Empty native event log",
        )
        require(
            not any(e.get("type") == "error" for e in events),
            "Native Actor error event",
        )
        sessions = {
            e.get("sessionID") or e.get("session_id")
            for e in events
            if e.get("sessionID") or e.get("session_id")
        }
        require(sessions == {session}, "Missing or changed native Actor session")
        finishes = [e for e in events if e.get("type") == "step_finish"]
        require(finishes, "No native terminal receipt")
        reason = finishes[-1].get("part", {}).get("reason")
        require(
            reason == ("stop" if index == len(names) - 1 else "length"),
            "Incomplete native Actor turn",
        )
        last = events
    parts = [e.get("part", {}) for e in last if e.get("type") == "text"]
    require(parts, "No native final text")
    message_id = parts[-1].get("messageID")
    reply = "\n".join(
        p.get("text", "") for p in parts if p.get("messageID") == message_id
    )
    snapshot = (
        slot
        / "snapshots"
        / (str(number) if benchmark == "claw_eval" else f"delivery-{number}")
        / "candidate.json"
    )
    observed = r.read(snapshot)
    retained_workspace(r, snapshot, observed)
    require(
        candidate(observed) == cid and observed["artifact"]["final_response"] == reply,
        "Native text/snapshot candidate mismatch",
    )
    return observed


def native_actor(r, slot, benchmark, task_id):
    from experiments.audit_receipts import candidate, require, text

    failure = r.optional(slot / "FAILURE.json")
    require(failure is None, "Native Actor failure receipt present")
    row = r.read(slot / "RESULT.json")
    require(row.get("status") == "completed", "Native Actor did not complete")
    r.closed = True
    require(
        not row.get("reused_from") and not row.get("recovered_source_identity"),
        "Reused Actor is not a fresh validation sample",
    )
    cid = candidate(row["candidate"])
    state = r.read(slot / "actor_state.json")
    require(
        text(row.get("session")) and state.get("session") == row["session"],
        "Actor session/state mismatch",
    )
    turns = state.get("turn")
    require(type(turns) is int and 1 <= turns <= 3, "Invalid Actor turn count")
    snapshots = [
        native_turn(r, slot, benchmark, cid, row["session"], n)
        for n in range(1, turns + 1)
    ]
    require(
        snapshots[-1] == row["candidate"], "Native final candidate differs from RESULT"
    )
    identity_key = "identity" if benchmark == "claw_eval" else "spec_identity"
    require(text(row.get(identity_key)), "Missing native slot identity")
    if benchmark == "claw_eval":
        require(
            r.read(slot / "STARTED.json").get("identity") == row["identity"],
            "Actor start identity mismatch",
        )
        r.bound_path(row.get("runtime_directory"), slot)
        r.bound_path(row.get("native_grader_trial"), slot)
        trace = r.read(slot / "trace.jsonl", lines=True)
        starts = [e for e in trace if e.get("type") == "trace_start"]
        ends = [e for e in trace if e.get("type") == "trace_end"]
        require(len(starts) == len(ends) == 1, "Native trace not completed")
        require(
            starts[0].get("task_id") == task_id
            and starts[0].get("trace_id") == ends[0].get("trace_id") == cid,
            "Native trace identity mismatch",
        )
        require(
            isinstance(r.read(slot / "environment-snapshot.json"), dict),
            "Missing native environment snapshot",
        )
    else:
        spec = r.read(slot / "spec.json")
        require(
            spec.get("spec_identity") == row["spec_identity"]
            and spec.get("candidate_id") == cid
            and spec["task"]["task_id"] == task_id,
            "Native spec/result mismatch",
        )
        require(
            spec.get("grade") is (slot.parent.name == "actors"),
            "Seed/arm grading scope mismatch",
        )
        require(not spec.get("source_spec"), "Replayed native spec")
        actor = r.read(slot / "ACTOR_RESULT.json")
        require(
            actor.get("status") == "completed"
            and actor.get("candidate") == row["candidate"]
            and actor.get("session") == row["session"],
            "Native Actor/worker result mismatch",
        )
    r.facts.update(candidate_id=cid, session=row["session"], turns=turns)
    r.facts["protocol_identity" if benchmark == "claw_eval" else "slot_identity"] = row[
        identity_key
    ]
    return row


def official(r, slot, benchmark, task_id, identity):
    from experiments.audit_receipts import candidate, finite, require, text

    row = r.read(slot / "OFFICIAL.private.json")
    # Preserve every native field even when it describes failure or a low score.
    r.facts["raw_native_output"] = row
    require(
        row.get("status") == ("graded" if benchmark == "claw_eval" else "scored"),
        "Native grading did not succeed",
    )
    r.closed = True
    actor = r.read(slot / "RESULT.json")
    require(actor.get("status") == "completed", "Score without completed Actor")
    cid = candidate(actor["candidate"])
    if benchmark == "claw_eval":
        require(
            row.get("protocol") == "native-claw-eval-custom-judge",
            "Artifact-only/non-native grading is out of scope",
        )
        require(
            row.get("task_id") == task_id and row.get("trace_id") == cid,
            "Score belongs to another native trial",
        )
        require(
            finite(row.get("task_score"))
            and isinstance(row.get("scores"), dict)
            and row["scores"],
            "Missing native numeric scores",
        )
        require(
            all(finite(v) for v in row["scores"].values()),
            "Invalid native score component",
        )
        binding = row["binding"]
        require(binding.get("identity") == identity, "Score config binding mismatch")
        for key, name in (
            ("trace", "trace.jsonl"),
            ("snapshot", "environment-snapshot.json"),
        ):
            require(
                binding.get(key) == hashlib.sha256(r.raw(slot / name)).hexdigest(),
                "Score artifact hash mismatch",
            )
        require(not row.get("snapshot_error_keys"), "Native snapshot contains errors")
        directory = row.get("record_directory")
        require(text(directory), "Native private grader receipt missing")
        private = Path(directory)
        if private.is_absolute():
            require(
                private.is_relative_to(r.root / slot / "official.private"),
                "Private grader belongs to another slot",
            )
            private = private.relative_to(r.root)
        else:
            require(
                private.is_relative_to(slot / "official.private"),
                "Private grader belongs to another slot",
            )
        original = r.read(private / "result.private.json")
        require(
            all(row.get(k) == v for k, v in original.items())
            and original.get("status") == "graded"
            and original.get("scores") == row["scores"],
            "Published/native grader output mismatch",
        )
    else:
        trial_path = slot / "harbor" / cid
        r.bound_path(row.get("trial"), trial_path)
        r.bound_path(row.get("native_score"), trial_path / "verifier/score.json")
        detail = r.read(trial_path / "verifier/score.json")
        trial = r.read(slot / "TRIAL_RESULT.private.json")
        r.facts["raw_native_score"] = detail
        require(
            trial.get("trial_name") == cid and not trial.get("exception_info"),
            "Native trial failed or identity mismatch",
        )
        rewards = trial.get("verifier_result", {}).get("rewards")
        require(
            isinstance(rewards, dict) and rewards == row.get("rewards"),
            "Native reward receipt mismatch",
        )
        score = row.get("overall")
        require(
            finite(score)
            and score
            == row.get("raw_overall")
            == rewards.get("overall")
            == detail.get("overall"),
            "Native overall score missing or inconsistent",
        )
        require(
            not row.get("exception") and not row.get("scoring_errors"),
            "Native scoring exception",
        )
        from experiments.workbuddy.scoring import native_contract, scoring_errors

        require(
            detail.get("test_status") == row.get("native_test_status"),
            "Native test status receipt mismatch",
        )
        spec = r.read(slot / "spec.json")
        task_path = Path(spec["task_path"])
        if task_path.is_absolute():
            require(
                task_path.is_relative_to(r.root),
                "Native authored task outside retained run",
            )
            task_path = task_path.relative_to(r.root)
        contract = native_contract(
            task_id,
            r.raw(task_path / "task.toml"),
            r.raw(task_path / "tests/judge.yaml"),
        )
        errors = scoring_errors(detail, contract=contract, rewards=rewards)
        r.facts["native_scoring_validation_errors"] = errors
        require(
            not errors,
            "Native grading invalid: " + json.dumps(errors, ensure_ascii=False),
        )


def _final_candidate_source(r, slot, benchmark):
    from experiments.audit_receipts import candidate, require, retained_workspace, text

    slot = Path(slot)
    require(benchmark in {"claw_eval", "workbuddy"}, "Unknown native benchmark")
    native = r.read(slot / "RESULT.json")
    require(native.get("status") == "completed", "Requires a completed fresh Actor")
    r.closed = True
    observed = native["candidate"]
    candidate(observed)
    state = r.read(slot / "actor_state.json")
    turn = state["turn"]
    require(
        type(turn) is int
        and 1 <= turn <= 3
        and text(native.get("session"))
        and state.get("session") == native["session"],
        "Source has no bound final Actor turn",
    )
    snapshot = (
        slot
        / "snapshots"
        / (str(turn) if benchmark == "claw_eval" else f"delivery-{turn}")
        / "candidate.json"
    )
    require(
        r.read(snapshot) == observed, "Source is not exact fresh-run final artifact"
    )
    retained_workspace(r, snapshot, observed)
    return snapshot


def final_candidate_source(directory, slot, benchmark):
    """Return the bound final candidate's run-relative Path, without writing.

    ``slot`` is run-relative, e.g. ``tasks/T/round_1/actors/B``. Validate the
    completed RESULT, final actor_state session/turn, and exact snapshot equality.
    ``receipt_digest`` is the canonical digest for the three required hashes.
    Also verify retained inventory/workspace byte hashes. Native event/trace
    validation is performed separately by ``audit_run``.
    """
    from experiments.audit_receipts import Receipts, require

    root = Path(directory).resolve()
    require(not Path(slot).is_absolute(), "slot must be run-relative")
    return _final_candidate_source(Receipts(root, closed=True), Path(slot), benchmark)
