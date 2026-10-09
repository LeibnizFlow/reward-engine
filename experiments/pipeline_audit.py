"""Read-only, deterministic receipt audit for experiments.validate_pipeline.

Run ``python -m experiments.pipeline_audit RUN_DIRECTORY``. JSON goes to stdout;
Exit codes are 0 (required coverage complete), 1 (fail), 2 (pending), and
3 (valid receipts, required revision branch untested). No models, graders, subprocesses,
clock, or outcome edits are involved. Missing/incomplete files are pending while
their producer is open, but fail after its completion receipt. Explicit invalid
or failed receipts fail immediately, including during a live run.

This verifies retained execution evidence, not its authenticity, the semantic
correctness of model judgments, improvement, or official leaderboard equivalence.

An incomplete delivered measurement can be supplemented without editing it:
``tasks/T/round_N/B/outcome_estimate.full.supplemental.json`` (also A/C) must
contain schema="pipeline-supplemental-measurement-v1", source=<run-relative
native final snapshots/.../candidate.json>, pack_source=<run-relative pack.json>,
artifact_sha256, candidate_sha256, pack_sha256, measurement=<full harness result>,
and provenance={kind: "supplemental_full_measurement", producer: <nonempty>,
created_at: <nonempty>}. Hashes use receipt_digest below, not a normalized score
or a reconstructed artifact. Optional input_binding uses workbuddy.io.digest on
{task: effective_task or checkpoint.task, candidate: delivery.candidate,
pack: effective_pack or round pack rebased to a changed effective_task using
TaskSequence.rebase}; pack_sha256 binds this same resolved pack. An explicit
effective_pack is used as supplied. source_hashes must include the absolute
original delivery.json, pack.json, checkpoint.json paths and exact byte SHA-256.
Optional code_hashes are retained as declared code provenance, not authenticated
execution evidence. The supplemental file is read, never generated here.

Original construction/delivery and supplemental full measurements also require
the harness-recorded measurement_input_schema/measurement_input_sha256, verified
against build_context(task, [native_candidate]) and measurement_context-normalized
inputs. Envelope hashes cannot supply a missing execution fingerprint. Retained
snapshot workspace bytes and inventory are required even when JSON agrees.
Every B turn is bound independently. Targeted reviews require recorded
review_input provenance for the overview, original prior measurement, budget,
pre-measurement overview and inquiry plan; no historical binding is backfilled.
Full review turns require their own complete, input-bound harness measurement.
"""

import argparse
import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path


class IncompleteReceipt(Exception):
    """A producer has not published a readable receipt yet."""


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def text(value):
    return isinstance(value, str) and bool(value.strip())


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def receipt_digest(value):
    """SHA-256 of UTF-8 canonical JSON: sorted keys, compact, no ASCII escaping."""
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def strings(value):
    return isinstance(value, list) and bool(value) and all(map(text, value))


def ids(rows, key):
    require(isinstance(rows, list) and rows, f"Nonempty {key} rows required")
    values = [row[key] for row in rows]
    require(all(map(text, values)), f"Nonempty {key} identities required")
    require(len(set(values)) == len(values), f"Duplicate {key} identities")
    return values


def candidate(row):
    require(text(row.get("candidate_id")), "Candidate identity missing")
    artifact = row["artifact"]
    require(text(artifact.get("final_response")), "No nonempty native final response")
    return row["candidate_id"]


def retained_workspace(r, snapshot, observed):
    """Verify retained native bytes, not merely the extracted candidate JSON."""
    from reward_engine.adapters.workbuddy.evidence import EXCLUDED

    directory = snapshot.parent
    for parent in (directory, *directory.parents):
        require(not (r.root / parent).is_symlink(), "Linked native snapshot directory")
    inventory = r.read(directory / "inventory.json")
    require(isinstance(inventory, list), "Native inventory must be a list")
    workspace = r.path(directory / "workspace")
    require(
        workspace.is_dir() and not workspace.is_symlink(),
        "Native retained workspace missing or linked",
    )

    def relative_path(name):
        require(text(name), "Empty inventory path")
        path = Path(name)
        require(
            not path.is_absolute()
            and name == path.as_posix()
            and path.parts
            and not {".", ".."}.intersection(path.parts),
            "Unsafe native inventory path",
        )
        return path

    names = []
    for entry in inventory:
        path = relative_path(entry["path"])
        require(not set(path.parts) & EXCLUDED, "Excluded file in native inventory")
        require(entry["path"] not in names, "Duplicate native inventory path")
        names.append(entry["path"])
        require(
            type(entry.get("bytes")) is int
            and entry["bytes"] >= 0
            and type(entry.get("changed")) is bool,
            "Invalid native inventory metadata",
        )
        for parent in (path, *path.parents):
            require(
                not (workspace / parent).is_symlink(), "Linked native inventory path"
            )
        raw = r.raw(directory / "workspace" / path)
        require(
            len(raw) == entry["bytes"]
            and hashlib.sha256(raw).hexdigest() == entry.get("sha256"),
            "Retained native workspace bytes differ from inventory: " + entry["path"],
        )
    actual = {
        path.relative_to(workspace).as_posix()
        for path in workspace.rglob("*")
        if not set(path.relative_to(workspace).parts) & EXCLUDED
        and path.is_file()
        and not path.is_symlink()
    }
    require(
        actual == set(names), "Retained workspace files differ from native inventory"
    )
    require(
        observed["artifact"].get("changed_files")
        == [row for row in inventory if row["changed"]],
        "Candidate changed-file hashes differ from native inventory",
    )
    environment = observed["environment"]
    require(
        environment.get("file_count") == len(inventory),
        "Candidate file count differs from native inventory",
    )
    deleted = environment.get("deleted_files")
    require(
        isinstance(deleted, list) and len(set(deleted)) == len(deleted),
        "Invalid native deleted-file inventory",
    )
    for name in deleted:
        path = relative_path(name)
        require(
            name not in actual
            and not (workspace / path).exists()
            and not (workspace / path).is_symlink(),
            "Recorded deleted file is still present",
        )


class Receipts:
    def __init__(self, root, closed=False):
        self.root, self.closed = root, closed
        self.paths, self.facts = [], {}

    def path(self, relative):
        path = self.root / relative
        require(
            path.resolve().is_relative_to(self.root), "Evidence escapes run directory"
        )
        name = path.relative_to(self.root).as_posix()
        if name not in self.paths:
            self.paths.append(name)
        return path

    def raw(self, relative):
        try:
            return self.path(relative).read_bytes()
        except FileNotFoundError as error:
            raise IncompleteReceipt(f"Missing receipt: {relative}") from error

    def read(self, relative, *, lines=False):
        try:
            raw = self.raw(relative).decode("utf-8")
            return (
                [json.loads(line) for line in raw.splitlines() if line.strip()]
                if lines
                else json.loads(raw)
            )
        except (UnicodeError, json.JSONDecodeError) as error:
            raise IncompleteReceipt(
                f"Incomplete or malformed JSON: {relative}"
            ) from error

    def optional(self, relative):
        path = self.path(relative)
        return self.read(relative) if path.exists() else None

    def bound_path(self, recorded, expected):
        require(text(recorded), "Missing native evidence path")
        observed = Path(recorded)
        if not observed.is_absolute():
            observed = self.root / observed
        require(
            observed.resolve() == self.path(expected).resolve(),
            "Native path belongs to another slot",
        )


def criteria(rows, *, require_anchors=False):
    """Jury defines standards; builder adds fail/partial/full rubric anchors."""
    require(
        isinstance(rows, list)
        and 1 <= len(rows) <= 20
        and all(isinstance(row, dict) for row in rows),
        "Jury criteria must be 1..20 objects",
    )
    values = ids(rows, "id")
    for row in rows:
        require(
            all(text(row.get(k)) for k in ("standard", "measurement_question")),
            "Empty Jury criterion",
        )
        require(row.get("kind") in {"hard", "soft"}, "Invalid criterion kind")
        require(
            row.get("origin") in {"task", "user", "domain", "observed_gap"},
            "Invalid criterion origin",
        )
        require(strings(row.get("evidence_refs")), "Ungrounded Jury criterion")
        if row["kind"] == "hard":
            require(
                row["origin"] in {"task", "user"},
                "Domain preferences cannot become hard Task requirements",
            )
            require(
                row["origin"] != "task" or "task" in row["evidence_refs"],
                "A hard Task requirement must cite the Task",
            )
        if "priority" in row:
            require(
                isinstance(row["priority"], str)
                and row["priority"] in {"critical", "important", "minor"},
                "Invalid criterion priority",
            )
        if "priority_reason" in row:
            require(text(row["priority_reason"]), "Empty criterion priority_reason")
        if "subrequirements" in row:
            require(
                isinstance(row["subrequirements"], list)
                and all(text(part) for part in row["subrequirements"]),
                "Criterion subrequirements must be a list of nonempty strings",
            )
        if require_anchors:
            anchors = row.get("anchors")
            require(
                isinstance(anchors, dict)
                and all(text(anchors.get(key)) for key in ("fail", "partial", "full")),
                "Builder rubric needs explicit fail/partial/full anchors",
            )
    return values


def batch(row, minimum=1, expected=None):
    items = row["items"]
    ids(items, "job_id")
    succeeded = sum(item.get("status") == "ok" for item in items)
    require(
        row.get("total") == len(items) and row.get("succeeded") == succeeded,
        "Inconsistent Map receipt",
    )
    require(succeeded >= minimum, "No sufficient successful native Map outputs")
    require(
        succeeded == len(items) and row.get("complete") is True,
        "Map contains failed/inactive seats",
    )
    require(expected is None or len(items) == expected, "Missing expected Map seats")


def check_result(row, field):
    """Validate a retained verdict without coercing or repairing model output."""
    require(
        isinstance(row, dict),
        f"Malformed {field}: expected an object, got {type(row).__name__}",
    )
    require(
        isinstance(row.get("status"), str)
        and row["status"] in {"pass", "fail", "unknown"}
        and text(row.get("reason")),
        f"Malformed {field}: expected pass/fail/unknown status and nonempty reason",
    )
    require(
        "score" not in row,
        f"Malformed {field}: Outcome checks do not return absolute scores",
    )
    require(not row.get("error"), f"Checker execution failed at {field}")
    gap = row.get("gap")
    require(
        gap is None or isinstance(gap, str), f"Malformed {field}.gap: expected text"
    )
    if row["status"] != "unknown":
        require(
            strings(row.get("evidence_refs")),
            f"Known measurement without evidence at {field}",
        )
    if row["status"] == "fail":
        require(text(gap), f"Product failure without a concrete gap at {field}")


def measurement_results(row, pack, *, full=True):
    require(isinstance(row, dict), "Malformed measurement: expected an object")
    require(
        row.get("pack_id") == pack["pack_id"], "Measurement belongs to another pack"
    )
    results = row.get("results")
    require(
        isinstance(results, list)
        and results
        and all(isinstance(result, dict) for result in results),
        "Malformed measurement.results: expected nonempty verdict objects",
    )
    tests = {t["test_id"]: t["criterion_id"] for t in pack["tests"]}
    measured = ids(results, "test_id")
    require(set(measured) <= set(tests), "Unknown measured test")
    if full:
        require(set(measured) == set(tests), "Full construction audit omitted tests")
    for result in results:
        require(
            result.get("criterion_id") == tests[result["test_id"]],
            "Measurement criterion mismatch",
        )
        field = f"measurement.results[{result['test_id']}]"
        check_result(result, field)
        if "inquiry_result" in result:
            check_result(result["inquiry_result"], field + ".inquiry_result")
    require(
        any(r["status"] != "unknown" for r in results),
        "Measurements entirely unknown; no validated check execution",
    )
    facts = {
        s: sum(r["status"] == s for r in results) for s in ("pass", "fail", "unknown")
    }
    diagnostics = [
        {"test_id": row["test_id"], "diagnostic": row["inquiry_result_diagnostic"]}
        for row in results
        if "inquiry_result_diagnostic" in row
    ]
    if diagnostics:
        facts["inquiry_result_diagnostics"] = diagnostics
    return facts


def measurement_execution(
    row, pack, selected, *, context=None, evidence=None, previous=None
):
    """Check execution ledgers and only independently reconstructable reuse."""
    results = row.get("results")
    require(
        isinstance(results, list) and all(isinstance(r, dict) for r in results),
        "Malformed measurement execution results",
    )
    result_ids = [r.get("test_id") for r in results]
    require(
        all(text(key) for key in result_ids)
        and len(result_ids) == len(set(result_ids))
        and set(result_ids) == set(selected),
        "Measurement selected-check/result coverage mismatch",
    )
    tests = {test["test_id"]: test for test in pack["tests"]}
    require(set(selected) <= set(tests), "Unknown selected measurement test")
    reused = []
    for result in results:
        require(
            "reused_unchanged_input" not in result
            or type(result["reused_unchanged_input"]) is bool,
            "Malformed measurement reuse flag",
        )
        if not result.get("reused_unchanged_input"):
            continue
        require(
            previous is not None and context is not None and evidence is not None,
            "Measurement reuse has no verified prior execution inputs",
        )
        old_context, old_evidence, old_measurement = previous
        prior = {r["test_id"]: r for r in old_measurement["results"]}.get(
            result["test_id"]
        )
        require(
            prior is not None
            and prior.get("status") == "pass"
            and not prior.get("error"),
            "Measurement reuse requires a prior passing check without error",
        )
        require(
            not context.get("measurement_inquiries", {}).get(result["test_id"]),
            "Measurement reuse forbidden with an active inquiry",
        )
        require(
            result == {**prior, "reused_unchanged_input": True},
            "Reused measurement changed the retained prior result",
        )
        # This builds the shipped registry only; it neither runs a checker nor
        # imports arbitrary recorded code to invent a dependency contract.
        from reward_engine.core.outcome.harness import defaults

        registry = defaults(None)
        test = tests[result["test_id"]]
        require(
            registry.dependencies.get(test["checker"]) is not None
            and not test.get("execution_recipe"),
            "Measurement reuse dependency contract unverified for this checker",
        )
        require(
            registry.inputs(test, context, evidence)
            == registry.inputs(test, old_context, old_evidence),
            "Measurement reuse dependency inputs changed",
        )
        reused.append(result["test_id"])
    submitted = set(result_ids) - set(reused)
    execution = row.get("execution_summary")
    require(isinstance(execution, dict), "Missing measurement execution summary")
    expected = {
        "selected_checks": len(selected),
        "submitted_checks": len(submitted),
        "reused_checks": len(reused),
    }
    for name, usage_key in (
        ("reported_model_requests", "model_requests"),
        ("reported_tool_calls", "tool_calls"),
    ):
        values = [
            r.get("execution_usage", {}).get(usage_key, 0)
            for r in results
            if not r.get("reused_unchanged_input")
        ]
        require(
            all(type(value) is int and value >= 0 for value in values),
            "Malformed measurement execution usage",
        )
        expected[name] = sum(values)
    require(
        all(
            type(execution.get(key)) is int and execution[key] == value
            for key, value in expected.items()
        ),
        "Measurement execution summary/result mismatch",
    )
    executed_batch = row.get("batch")
    require(isinstance(executed_batch, dict), "Missing measurement execution batch")
    if submitted:
        batch(executed_batch, expected=len(submitted))
        require(
            {r["job_id"] for r in executed_batch["items"]} == submitted,
            "Measurement executed batch/check identity mismatch",
        )
        require(
            not any(r.get("error_type") for r in executed_batch["items"]),
            "Measurement batch contains execution errors",
        )
    else:
        require(
            executed_batch
            == {"complete": True, "total": 0, "succeeded": 0, "items": []},
            "Empty measurement execution batch inconsistent",
        )


def measurement(row, pack, *, full=True, context=None, evidence=None, previous=None):
    facts = measurement_results(row, pack, full=full)
    selected = (
        [test["test_id"] for test in pack["tests"]]
        if full
        else [r["test_id"] for r in row["results"]]
    )
    measurement_execution(
        row, pack, selected, context=context, evidence=evidence, previous=previous
    )
    return facts


def measurement_binding_schema(row):
    from reward_engine.core.outcome.harness import MEASUREMENT_INPUT_SCHEMA

    require(
        row.get("measurement_input_schema") == MEASUREMENT_INPUT_SCHEMA
        and text(row.get("measurement_input_sha256")),
        "Unbound measurement: missing recorded measurement input fingerprint",
    )


def measurement_binding(row, pack, task, observed):
    """Recompute expected inputs; never add missing evidence to a receipt."""
    from reward_engine.core.outcome.context import build_context
    from reward_engine.core.outcome.harness import measurement_input_fingerprint

    measurement_binding_schema(row)
    context, evidence = build_context(task, [observed])
    expected = measurement_input_fingerprint(pack, context, evidence)
    require(
        row["measurement_input_sha256"] == expected,
        "Measurement input fingerprint does not match actual task/candidate/pack",
    )
    return expected


def native_turn(r, slot, benchmark, cid, session, number):
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


def review_measurement(row, pack):
    require(row.get("pack_id") == pack["pack_id"], "Review pack mismatch")
    require(not row.get("checks_unavailable_only"), "Only unavailable checks in review")
    overview = row.get("delivery_review")
    if overview is not None:
        require(
            not overview.get("error")
            and overview.get("status")
            in {"target_satisfied", "needs_improvement", "needs_verification"},
            "Review model failed or unknown",
        )
        require(text(overview.get("reason")), "Review has no conclusion")
        claims = overview["claim_checks"]
        require(isinstance(claims, list), "Missing reviewed claims")
        for claim in claims:
            require(
                text(claim.get("claim"))
                and text(claim.get("reason"))
                and text(claim.get("claim_ref"))
                and claim["claim_ref"].startswith("product_1:"),
                "Ungrounded review claim",
            )
            require(
                claim.get("assessment")
                in {"supported", "needs_verification", "not_required"},
                "Invalid claim assessment",
            )
        for gap in overview["gaps"]:
            require(
                text(gap.get("gap")) and strings(gap.get("evidence_refs")),
                "Review gap has no evidence",
            )
        batch(row["review_batch"])
    if row.get("results"):
        measurement_results(row, pack, full=False)
    else:
        require(
            overview is not None
            and (
                overview["claim_checks"]
                or overview["gaps"]
                or overview.get("improvements")
            ),
            "No actual B review measurement",
        )


def _final_candidate_source(r, slot, benchmark):
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
    root = Path(directory).resolve()
    require(not Path(slot).is_absolute(), "slot must be run-relative")
    return _final_candidate_source(Receipts(root, closed=True), Path(slot), benchmark)


def delivered_measurement(r, folder, arm, benchmark):
    """A targeted B review never substitutes for the complete delivered Pack."""
    pack_path = folder / "pack.json"
    delivery_path = folder / arm / "delivery.json"
    delivery = r.read(delivery_path)
    task = delivery.get("effective_task")
    round_pack = r.read(pack_path)
    pack = delivery.get("effective_pack") or round_pack
    if (
        task
        and pack is round_pack
        and task != r.read(folder / "checkpoint.json")["task"]
    ):
        from reward_engine.core.outcome.task_sequence import TaskSequence

        # Match both the core loop and the supplemental producer. A changed
        # public task invalidates the old pack scope even without an override.
        pack = TaskSequence(task).rebase(round_pack)
    primary = r.read(folder / arm / "outcome_estimate.json")
    r.facts.update(provenance="original_run", primary_full_pack_measurement=False)
    primary_error = None
    try:
        counts = measurement(primary, pack)
        measurement_binding_schema(primary)
        task = task or r.read(folder / "checkpoint.json")["task"]
        snapshot = _final_candidate_source(r, folder / "actors" / arm, benchmark)
        observed = r.read(snapshot)
        require(
            delivery.get("candidate") == observed,
            "Primary measurement delivery differs from native candidate",
        )
        binding = measurement_binding(primary, pack, task, observed)
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        primary_error = str(error)
        r.facts["primary_measurement_issue"] = primary_error
    else:
        r.facts["primary_full_pack_measurement"] = True
        r.facts["original_measurements"] = counts
        r.facts["original_measurement_input_sha256"] = binding
    # Inspect a supplement even when the original is complete: a failed recorded
    # follow-up must not be silently discarded in favour of a passing receipt.
    supplemental = r.optional(folder / arm / "outcome_estimate.full.supplemental.json")
    if supplemental is None:
        require(
            primary_error is None,
            "Incomplete/failed delivered pack measurement: " + str(primary_error),
        )
        r.facts["measurements"] = counts
        return
    r.closed = True
    require(
        supplemental.get("schema") == "pipeline-supplemental-measurement-v1",
        "Unknown supplemental measurement schema",
    )
    provenance = supplemental["provenance"]
    r.facts["supplemental_provenance"] = provenance
    require(
        provenance.get("kind") == "supplemental_full_measurement"
        and text(provenance.get("producer"))
        and text(provenance.get("created_at")),
        "Missing explicit supplemental provenance",
    )
    slot = folder / "actors" / arm
    snapshot = _final_candidate_source(r, slot, benchmark)
    observed = r.read(snapshot)
    require(
        delivery.get("candidate") == observed,
        "Supplemental delivery is not the native candidate",
    )
    require(
        supplemental.get("source") == snapshot.as_posix()
        and supplemental.get("pack_source") == pack_path.as_posix(),
        "Supplemental source/pack paths do not name this delivery",
    )
    for key, value in (
        ("artifact_sha256", observed["artifact"]),
        ("candidate_sha256", observed),
        ("pack_sha256", pack),
    ):
        require(
            supplemental.get(key) == receipt_digest(value),
            "Supplemental artifact/candidate/pack hash mismatch: " + key,
        )
    if "input_binding" in supplemental:
        task = task or r.read(folder / "checkpoint.json")["task"]
        # This optional hash follows workbuddy.io.digest's spaced JSON encoding,
        # unlike the explicitly compact required receipt_digest hashes.
        inputs = {"task": task, "candidate": observed, "pack": pack}
        expected = hashlib.sha256(
            json.dumps(inputs, sort_keys=True, ensure_ascii=False, default=str).encode()
        ).hexdigest()
        require(
            supplemental["input_binding"] == expected,
            "Supplemental whole-input binding mismatch",
        )
        r.facts["input_binding"] = supplemental["input_binding"]
    if "source_hashes" in supplemental:
        hashes = supplemental["source_hashes"]
        required = {
            str(r.root / path)
            for path in (delivery_path, pack_path, folder / "checkpoint.json")
        }
        require(
            isinstance(hashes, dict) and required <= set(hashes),
            "Supplemental source hashes omit required original receipts",
        )
        for path, digest in hashes.items():
            absolute = Path(path)
            require(
                absolute.is_absolute() and absolute.is_relative_to(r.root),
                "Supplemental source hash path escapes run",
            )
            require(
                hashlib.sha256(r.raw(absolute.relative_to(r.root))).hexdigest()
                == digest,
                "Supplemental source file hash mismatch",
            )
        r.facts["source_hashes"] = hashes
    if "code_hashes" in supplemental:
        hashes = supplemental["code_hashes"]
        r.facts["code_hashes"] = hashes
        require(
            isinstance(hashes, dict) and hashes, "Empty supplemental code provenance"
        )
        require(
            all(
                text(path)
                and isinstance(digest, str)
                and len(digest) == 64
                and all(ch in "0123456789abcdef" for ch in digest)
                for path, digest in hashes.items()
            ),
            "Malformed supplemental code hashes",
        )
    counts = measurement(supplemental["measurement"], pack)
    task = task or r.read(folder / "checkpoint.json")["task"]
    binding = measurement_binding(supplemental["measurement"], pack, task, observed)
    r.facts.update(
        provenance="supplemental",
        binding={
            k: supplemental[k]
            for k in (
                "source",
                "pack_source",
                "artifact_sha256",
                "candidate_sha256",
                "pack_sha256",
            )
        },
        measurements=counts,
        supplemental_measurement_input_sha256=binding,
    )


def review_binding(measured, pack, task, observed, previous):
    """Verify recorded execution inputs, including targeted questions and history."""
    from reward_engine.core.outcome.construction import fingerprint
    from reward_engine.core.outcome.context import build_context
    from reward_engine.core.outcome.delivery_triage import (
        REVIEW_INPUT_SCHEMA,
        measurement_plan,
        prepare_review_inputs,
        review_input_fingerprint,
        review_prior_rows,
        unresolved,
        validate_review,
    )
    from reward_engine.core.outcome.harness import (
        measurement_context,
        measurement_input_fingerprint,
    )
    from reward_engine.core.outcome.improvement import validate as validate_improvements

    measurement_binding_schema(measured)
    context, evidence = build_context(task, [observed])
    context, evidence = measurement_context(pack, context, evidence)
    facts = {"measurement_input_sha256": measured["measurement_input_sha256"]}
    if measured.get("review_mode") == "targeted":
        recorded = measured.get("review_input")
        require(
            isinstance(recorded, dict),
            "Missing targeted review execution-input provenance",
        )
        require(
            recorded.get("schema") == REVIEW_INPUT_SCHEMA,
            "Unknown targeted review input schema",
        )
        require(
            type(recorded.get("max_tests")) is int and recorded["max_tests"] == 3,
            "Targeted review budget differs from native controller",
        )
        require(
            recorded.get("draft_checks") == [],
            "Draft review inputs have no bound native draft sources in legacy pipeline",
        )
        require(
            recorded.get("previous_measurement_sha256")
            == (fingerprint(previous[2]) if previous else None),
            "Targeted review prior-measurement binding mismatch",
        )
        expected = review_input_fingerprint(
            pack, context, evidence, previous=previous, max_tests=3, draft_checks=[]
        )
        require(
            recorded.get("sha256") == expected,
            "Targeted overview input fingerprint mismatch",
        )
        overview = recorded.get("overview_before_measurement")
        require(
            isinstance(overview, dict) and not overview.get("error"),
            "Missing/failed bound pre-measurement overview",
        )
        reviewed_context, reviewed_evidence, _ = prepare_review_inputs(
            pack, context, evidence
        )
        validated_overview = validate_improvements(
            validate_review(
                deepcopy(overview),
                reviewed_evidence,
                {t["test_id"] for t in pack["tests"]},
            ),
            reviewed_evidence,
        )
        require(
            validated_overview == overview,
            "Bound overview differs from production-validated review fields",
        )
        plan = measurement_plan(pack, overview, previous=previous, max_tests=3)
        require(
            recorded.get("measurement_plan") == plan,
            "Targeted recorded measurement plan mismatch",
        )
        result_ids = [row["test_id"] for row in measured["results"]]
        require(
            len(result_ids) == len(set(result_ids))
            and set(result_ids) == set(plan["selected_tests"]),
            "Targeted selected-check coverage mismatch",
        )
        require(
            measured.get("deferred_tests") == plan["deferred_tests"],
            "Targeted deferred-check receipt mismatch",
        )
        pending = [
            row
            for row in review_prior_rows(previous).values()
            if unresolved(row) and row["test_id"] not in result_ids
        ]
        require(
            measured.get("unresolved_checks") == pending,
            "Targeted unresolved prior checks mismatch",
        )
        actual_context = {
            **reviewed_context,
            "measurement_inquiries": plan["measurement_inquiries"],
        }
        measurement_execution(
            measured,
            pack,
            plan["selected_tests"],
            context=actual_context,
            evidence=reviewed_evidence,
            previous=previous,
        )
        expected_measurement = measurement_input_fingerprint(
            pack, actual_context, reviewed_evidence
        )
        require(
            measured["measurement_input_sha256"] == expected_measurement,
            "Targeted harness input fingerprint mismatch (including inquiries)",
        )
        for result in measured["results"]:
            require(
                result.get("measurement_inquiry")
                == plan["measurement_inquiries"].get(result["test_id"]),
                "Targeted recorded inquiry differs from executed plan",
            )
        require(
            pack.get("feedback_policy") != "evidence_admitted",
            "Hardened review provenance outside legacy pipeline scope",
        )
        clear = (
            overview["status"] in {"target_satisfied", "needs_verification"}
            and not plan["deferred_tests"]
            and not pending
            and not any(
                overview.get(k) for k in ("gaps", "full_score_gaps", "improvements")
            )
            and not any(unresolved(row) for row in measured["results"])
        )
        require(
            measured.get("delivery_clear") is clear
            and measured.get("review_status_before_measurement") == overview["status"],
            "Bound overview stopping receipt mismatch",
        )
        final_overview = deepcopy(overview)
        outstanding = [row for row in measured["results"] + pending if unresolved(row)]
        if clear:
            final_overview["status"] = "target_satisfied"
        elif any(
            (row["status"] == "fail" and not row.get("error"))
            or (
                row.get("inquiry_result", {}).get("status") == "fail"
                and not row["inquiry_result"].get("error")
            )
            for row in outstanding
        ):
            final_overview["status"] = "needs_improvement"
        elif final_overview["status"] == "target_satisfied":
            final_overview["status"] = "needs_verification"
        require(
            measured.get("delivery_review") == final_overview,
            "Final review differs from bound overview/measurements",
        )
        facts["review_input_sha256"] = expected
        facts["measurement_plan"] = plan
    else:
        require(
            measured.get("review_mode") in (None, "full")
            and "delivery_review" not in measured
            and "review_input" not in measured,
            "Unrecognized review mode/provenance",
        )
        measurement(
            measured, pack, context=context, evidence=evidence, previous=previous
        )
        measurement_binding(measured, pack, task, observed)
    return facts, (context, evidence, measured)


def review(r, slot, pack, benchmark):
    """Validate every review; count continuation/change only when actually observed."""
    from reward_engine.core.outcome.context import build_context
    from reward_engine.core.outcome.harness import measurement_context
    from reward_engine.core.outcome.task_progress import TaskProgress

    row = r.read(slot / "RESULT.json")
    require(row.get("status") == "completed", "B Actor failed")
    r.closed = True
    r.facts["review_status"] = row.get("review_status")
    require(
        row.get("review_status")
        in {
            "satisfied_under_estimator",
            "full_score_target_satisfied",
            "budget_exhausted_unresolved",
        },
        "B review inactive, unavailable, or failed",
    )
    state = r.read(slot / "actor_state.json")
    turns = state.get("turn")
    require(
        type(turns) is int and 1 <= turns <= 3,
        "Review requires one to three native delivery turns",
    )
    session, cid = row["session"], candidate(row["candidate"])
    require(
        text(session) and state.get("session") == session, "Review session mismatch"
    )
    deliveries = []
    progress = None
    previous = None
    for n in range(1, turns + 1):
        receipt = r.read(slot / "checks" / f"delivery_{n}.json")
        observed = native_turn(r, slot, benchmark, cid, session, n)
        require(
            receipt.get("candidate") == observed,
            "Review candidate not the native turn snapshot",
        )
        measured = receipt["measurement"]
        review_measurement(measured, pack)
        if "completeness_review" in measured:
            check_result(measured["completeness_review"], "completeness_review")
        recorded_progress = r.read(slot / "checks" / f"task_progress_{n}.json")
        if progress is None:
            task = {
                "task_id": recorded_progress["task_id"],
                "prompt": recorded_progress["task_prompt"],
            }
            require(
                task["task_id"] == pack["task_id"]
                and task["prompt"] == pack["task_prompt"],
                "Review progress Task/Pack mismatch",
            )
            progress = TaskProgress(task, pack)
        binding, previous = review_binding(measured, pack, task, observed, previous)
        r.facts.setdefault("review_bindings", []).append({"turn": n, **binding})
        context, evidence = build_context(task, [observed])
        context, _ = measurement_context(pack, context, evidence)
        progress.begin(context, n, "checks/delivery.json")
        progress.finish(measured)

        # Adapter source prefixes differ; actual check paths are read above.
        def without_sources(items):
            return [{k: v for k, v in item.items() if k != "source"} for item in items]

        require(
            without_sources(recorded_progress["items"])
            == without_sources(progress.state["items"])
            and all(
                recorded_progress.get(key) == progress.state[key]
                for key in (
                    "task_id",
                    "task_prompt",
                    "pack_id",
                    "user_updates",
                    "check_number",
                    "outcome_coverage",
                )
            ),
            "Task-progress receipt disagrees with current candidate/review measurement",
        )
        require(
            type(recorded_progress.get("max_revisions")) is int
            and recorded_progress["max_revisions"] == 2
            and type(recorded_progress.get("revisions_requested")) is int
            and recorded_progress["revisions_requested"] == (n if n < turns else n - 1),
            "Review progress revision budget/count mismatch",
        )
        clear = measured.get("delivery_clear", measured.get("ready_under_estimator"))
        require(type(clear) is bool, "Review has no boolean stopping measurement")
        if clear and measured.get("review_mode") == "targeted":
            overview = measured["delivery_review"]
            require(
                overview["status"] == "target_satisfied"
                and not any(
                    measured.get(k) for k in ("deferred_tests", "unresolved_checks")
                )
                and not any(
                    overview.get(k) for k in ("gaps", "full_score_gaps", "improvements")
                )
                and all(
                    result["status"] == "pass"
                    and result.get("inquiry_result", {}).get("status", "pass") == "pass"
                    for result in measured["results"]
                ),
                "Targeted clear decision contradicts remaining measured review findings",
            )
        for result in measured.get("results", []):
            if "inquiry_result_diagnostic" in result:
                r.facts.setdefault("inquiry_result_diagnostics", []).append(
                    {
                        "turn": n,
                        "test_id": result["test_id"],
                        "diagnostic": result["inquiry_result_diagnostic"],
                    }
                )
        satisfied = clear and progress.verified
        expected_status = "revision_requested" if n < turns else row["review_status"]
        require(
            recorded_progress.get("status") == expected_status,
            "Review progress status mismatch",
        )
        if n < turns:
            require(
                not satisfied and n <= 2,
                "Feedback contradicts satisfied review or revision budget",
            )
            feedback = r.read(slot / "checks" / f"feedback_{n}.json")
            require(text(feedback.get("message")), "Empty B feedback")
            overview = measured.get("delivery_review", {})
            require(
                any(
                    overview.get(k)
                    for k in ("gaps", "verify", "full_score_gaps", "improvements")
                )
                or any(x.get("status") == "fail" for x in measured.get("results", [])),
                "Feedback has no measured findings or verification requests",
            )
        else:
            require(
                not (r.root / slot / "checks" / f"feedback_{n}.json").exists(),
                "Terminal review has unresolved continuation feedback",
            )
            if expected_status == "budget_exhausted_unresolved":
                require(n == 3 and not satisfied, "Unsupported budget-exhausted stop")
            else:
                require(
                    satisfied,
                    "Satisfaction stop unsupported by measurement/task progress",
                )
                require(
                    (expected_status == "full_score_target_satisfied")
                    == (measured.get("review_mode") == "targeted"),
                    "Satisfaction status does not match review mode",
                )
            require(
                r.read(slot / "checks/task_progress.json") == recorded_progress
                and row.get("task_progress") == recorded_progress,
                "Final review/task-progress mismatch",
            )
        deliveries.append(receipt)
    require(
        deliveries[-1]["candidate"] == row["candidate"]
        and deliveries[-1]["measurement"] == row.get("final_measurement"),
        "Final B result/recheck mismatch",
    )
    changed = sum(
        d["candidate"]["artifact"] != deliveries[i - 1]["candidate"]["artifact"]
        for i, d in enumerate(deliveries[1:], 1)
    )
    r.facts.update(
        revisions=turns - 1,
        changed_artifact_revisions=changed,
        continuation_exercised=turns > 1,
        changed_artifact_revision_exercised=changed > 0,
        revision_coverage="exercised" if changed else "not_exercised",
        satisfied_under_estimator=satisfied,
        same_session=session,
        improvement_validated=False,
    )


def audit_run(directory):
    """Return a JSON-serializable report; never write into the run directory."""
    root = Path(directory).resolve()
    checks = []

    def peek(path):
        try:
            value = json.loads((root / path).read_text())
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError):
            return {}

    status = peek("validation_status.json")
    config = peek("config.json")
    tid, benchmark = (
        config.get("validation_task_id"),
        config.get("validation_benchmark"),
    )
    terminal = status.get("status") in {
        "execution_completed",
        "completed",
        "failed",
        "cancelled",
    }

    def check(name, validator, *, closed=False):
        receipts = Receipts(root, terminal or closed)
        try:
            validator(receipts)
            verdict, reason = "pass", "Required receipts present and consistent"
        except IncompleteReceipt as error:
            verdict, reason = ("fail" if receipts.closed else "pending"), str(error)
        except (
            ValueError,
            TypeError,
            KeyError,
            IndexError,
            AttributeError,
            OSError,
        ) as error:
            verdict, reason = "fail", f"{type(error).__name__}: {error}"
        result = {
            "stage": name,
            "status": verdict,
            "reason": reason,
            "evidence": receipts.paths,
            "facts": receipts.facts,
        }
        checks.append(result)
        return result

    def protocol(r):
        c = r.read("config.json")
        require(
            c.get("validation_benchmark") in {"claw_eval", "workbuddy"},
            "Unsupported validation benchmark",
        )
        require(
            text(c.get("validation_task_id"))
            and Path(c["validation_task_id"]).name == c["validation_task_id"]
            and c["validation_task_id"] not in {".", ".."},
            "Invalid task identity",
        )
        require(
            c.get("rounds") == 3
            and c.get("full_audit") is True
            and c.get("quality_resampling") is False,
            "Requires three full-audit rounds without quality resampling",
        )
        require(text(c.get("identity")), "Missing configuration identity")
        require(
            not any(
                c.get(k)
                for k in (
                    "seed_sources",
                    "reused_slots",
                    "reuse_first_round_from",
                    "recover_from",
                )
            ),
            "Fresh-run protocol excludes historical samples",
        )
        require(
            c.get("selection_policy", "legacy") == "legacy",
            "Independent pool selection is not this validate_pipeline protocol",
        )
        manifest = r.read("manifest.json")
        require(
            manifest.get("config_identity") == c["identity"]
            and manifest.get("actor_slots") == 12
            and manifest.get("grading_slots") == 9,
            "Validation manifest mismatch",
        )
        p = r.read("validation_protocol.json")
        require(
            p.get("full_harness_audit") is True
            and p.get("historical_samples_reused") is False
            and p.get("quality_resampling") is False,
            "Validation protocol missing or inactive",
        )

    check("protocol", protocol)

    def execution(r):
        row = r.read("validation_status.json")
        require(
            row.get("status") not in {"failed", "cancelled"},
            "Validation process failed",
        )
        if row.get("status") == "running":
            raise IncompleteReceipt(
                "Validation execution is still running; stage checks remain independent"
            )
        require(
            row.get("status") == "execution_completed"
            and row.get("completed_rounds") == 3,
            "Process did not complete three rounds",
        )

    check("execution", execution)
    if (
        not text(tid)
        or Path(tid).name != tid
        or tid in {".", ".."}
        or benchmark not in {"claw_eval", "workbuddy"}
    ):
        return _report(root, benchmark, tid, checks)
    task = Path("tasks") / tid
    final = peek(task / "result.private.json")
    state = peek(task / "state.json")
    finished = (
        final.get("status") in {"completed", "failed"}
        or bool(final.get("construction_failure"))
        or state.get("status") in {"failed", "cancelled"}
    )

    def task_failures(r):
        state = r.optional(task / "state.json")
        require(
            not state or state.get("status") not in {"failed", "cancelled"},
            "Task failure receipt present",
        )
        result = r.optional(task / "result.private.json")
        require(
            not result
            or (
                result.get("status") != "failed"
                and not result.get("construction_failure")
            ),
            "Final task construction/execution failure",
        )
        if state is None and result is None:
            raise IncompleteReceipt("Task has not published state/result evidence")
        if not result or result.get("status") != "completed":
            raise IncompleteReceipt(
                "Task has not published completed execution evidence"
            )

    check("task_execution", task_failures, closed=finished)
    slots = [task / "seeds" / str(n) for n in range(1, 4)]
    slots += [
        task / f"round_{n}" / "actors" / arm for n in range(1, 4) for arm in "ABC"
    ]
    for slot in slots:
        label = str(slot.relative_to(task))
        check(
            label + "/native_actor",
            lambda r, slot=slot: native_actor(r, slot, benchmark, tid),
            closed=finished,
        )
        if slot.parent.name == "actors":
            check(
                label + "/native_grade",
                lambda r, slot=slot: official(
                    r, slot, benchmark, tid, config.get("identity")
                ),
                closed=finished,
            )

    def distinct(r):
        rows = [r.read(slot / "RESULT.json") for slot in slots]
        for row in rows:
            require(row.get("status") == "completed", "Incomplete seed/Actor")
        identities = (
            [candidate(row["candidate"]) for row in rows],
            [row["session"] for row in rows],
        )
        # Claw identity hashes protocol inputs, not native execution identity:
        # fresh seeds and unguided A executions legitimately share it.
        if benchmark == "workbuddy":
            identities += ([row["spec_identity"] for row in rows],)
        for values in identities:
            require(
                all(map(text, values)) and len(set(values)) == 12,
                "Seeds and nine Actors must have twelve distinct native identities/sessions",
            )
        r.facts.update(seeds=3, actors=9, distinct_sessions=12)

    check("distinct_native_samples", distinct, closed=finished)
    for number in range(1, 4):
        folder = task / f"round_{number}"
        summary = peek(folder / "summary.private.json")
        closed = finished or summary.get("status") in {"completed", "partial", "failed"}

        def stage(name, validator, number=number, closed=closed):
            return check(f"round_{number}/{name}", validator, closed=closed)

        def construction_failure(r, folder=folder):
            row = r.read(folder / "construction_failure.json")
            r.facts["failure"] = row
            require(False, "Construction failed; see recorded stage and error")

        if (root / folder / "construction_failure.json").exists():
            stage("construction_failure", construction_failure)

        def construction(r, folder=folder):
            row = r.read(folder / "jury.json")
            r.facts["criteria"] = len(criteria(row["criteria"]))
            batch(r.read(folder / "jury_map.json"), expected=3)

        stage("jury", construction)

        def builder(r, folder=folder):
            pack = r.read(folder / "pack.json")
            require(
                text(pack.get("pack_id"))
                and pack.get("task_id") == tid
                and pack.get("status") == "callable",
                "Builder pack inactive or unbound",
            )
            criterion_ids = criteria(pack["criteria"], require_anchors=True)
            require(
                set(criterion_ids)
                == set(criteria(r.read(folder / "jury.json")["criteria"])),
                "Builder/Jury criterion mismatch",
            )
            tests = pack["tests"]
            ids(tests, "test_id")
            require(
                {t["criterion_id"] for t in tests} == set(criterion_ids),
                "Builder tests omit criteria",
            )
            require(
                all(
                    t.get("binding_status") == "callable" and text(t.get("checker"))
                    for t in tests
                ),
                "Builder tests not callable",
            )
            smoke = r.read(folder / "construction_smoke.json")
            require(
                smoke.get("status") == "receipt_path_checked"
                and smoke.get("bindings") == {t["test_id"]: "callable" for t in tests},
                "Harness smoke only inspected bindings",
            )
            receipt = smoke.get("receipt") or {}
            require(
                receipt.get("status") == "pass"
                and not receipt.get("error")
                and text(receipt.get("reason"))
                and strings(receipt.get("evidence_refs")),
                "Missing executed parser smoke",
            )
            r.facts.update(
                tests=len(tests),
                scope="Synthetic parser infrastructure only; semantic tests audited separately",
            )

        stage("builder_and_smoke", builder)

        def hypothesis(r, folder=folder):
            row = r.read(folder / "hypothesis.json")
            mode = (
                "unavailable"
                if row.get("availability") == "unavailable"
                else row.get("status")
            )
            r.facts.update(mode=mode, proposal_projected=mode == "propose")
            require(
                mode in {"keep", "propose"},
                "Unavailable hypothesis is not semantic keep",
            )
            require(
                not row.get("error") and not row.get("error_type"),
                "Failed hypothesis output",
            )
            batch(r.read(folder / "hypothesis_map.json"), expected=3)
            if mode == "propose":
                require(
                    all(
                        text(row.get(k))
                        for k in (
                            "method",
                            "target",
                            "expected_improvement",
                            "verification",
                        )
                    )
                    and strings(row.get("evidence_refs")),
                    "Empty or unsupported hypothesis proposal",
                )
            else:
                require(
                    not row.get("clarifications"),
                    "Keep cannot contain proposed clarifications",
                )
            protocol = r.read(folder / "arm_protocol.json")
            require(
                protocol.get("id") == "legacy_abcd"
                and protocol.get("hypothesis_applied") is (mode == "propose"),
                "Hypothesis treatment receipt mismatch",
            )

        stage("hypothesis", hypothesis)

        def measurements(r, folder=folder, number=number):
            pack = r.read(folder / "pack.json")
            rows = r.read(folder / "construction_checks.json")
            require(
                isinstance(rows, list) and len(rows) == number * 3,
                "Construction did not measure the complete incoming pool",
            )
            # Validate shape/binding presence before awaiting a live checkpoint.
            r.facts["measurements"] = [measurement(row, pack) for row in rows]
            for row in rows:
                measurement_binding_schema(row)
            measured_task = r.read(folder / "checkpoint.json")["task"]
            incoming = slots[:3] + [
                task / f"round_{n}" / "actors" / arm
                for n in range(1, number)
                for arm in "ABC"
            ]
            bindings = []
            for row, slot in zip(rows, incoming, strict=True):
                snapshot = _final_candidate_source(r, slot, benchmark)
                observed = r.read(snapshot)
                bindings.append(measurement_binding(row, pack, measured_task, observed))
            r.facts["measurement_input_sha256"] = bindings

        stage("construction_measurements", measurements)
        stage(
            "B_review_execution",
            lambda r, folder=folder: review(
                r, folder / "actors/B", r.read(folder / "pack.json"), benchmark
            ),
        )
        for arm in "ABC":
            stage(
                f"{arm}/delivered_pack_measurement",
                lambda r, folder=folder, arm=arm: delivered_measurement(
                    r, folder, arm, benchmark
                ),
            )

        def selection(r, folder=folder, number=number):
            rank = r.read(folder / "ranking.json")
            require(
                rank.get("status") == "compared",
                "Ranking unavailable/deferred; fallback A is not ranking success",
            )
            groups, unknown = rank["groups"], rank["unknown"]
            require(
                isinstance(groups, list)
                and groups
                and all(strings(g) for g in groups)
                and isinstance(unknown, list),
                "No actual ranking groups",
            )
            ordered = [p for group in groups for p in group]
            all_products = ordered + unknown
            expected = {f"product_{i}" for i in range(1, 3 * (number + 1) + 1)}
            require(
                len(all_products) == len(expected) and set(all_products) == expected,
                "Ranking duplicates/omits actual pool products",
            )
            require(text(rank.get("reason")), "Ranking lacks reasoning")
            observations = rank["observations"]
            require(
                set(ids(observations, "product_id")) == expected,
                "Ranking lacks product observations",
            )
            check_refs = {}
            if any(
                ref.startswith("check:")
                for observation in observations
                for ref in observation.get("evidence_refs", [])
                if isinstance(ref, str)
            ):
                bundles = r.read(folder / "check_sources.json")
                check_refs = {
                    f"check:{i}:{j}": bundle["product_id"]
                    for i, bundle in enumerate(bundles)
                    for j, _ in enumerate(bundle["checks"])
                }
            require(
                all(
                    text(o.get("reason"))
                    and strings(o.get("evidence_refs"))
                    and any(
                        ref.startswith(o["product_id"] + ":")
                        or check_refs.get(ref) == o["product_id"]
                        for ref in o["evidence_refs"]
                    )
                    for o in observations
                ),
                "Ranking lacks product evidence",
            )
            batch(r.read(folder / "ranking_map.json"), minimum=2, expected=3)
            chosen = r.read(folder / "selected_delivery.json")
            arm = chosen.get("arm")
            require(
                arm in ("A", "B", "C")
                and chosen.get("round") == number
                and text(chosen.get("reason")),
                "Empty selection",
            )
            product_id = f"product_{3 * number + 'ABC'.index(arm) + 1}"
            require(
                chosen.get("product_id") == product_id and product_id in ordered,
                "Selection is not an actually ranked current Actor",
            )
            require(
                chosen.get("source") == f"round_{number}/{arm}/delivery.json",
                "Selection source mismatch",
            )
            delivery = r.read(folder / arm / "delivery.json")
            require(
                delivery.get("candidate")
                == r.read(folder / "actors" / arm / "RESULT.json").get("candidate"),
                "Selection candidate differs from native Actor",
            )
            candidate(delivery["candidate"])
            r.facts.update(
                arm=arm,
                product_id=product_id,
                scope="Jury relative preference, not native score improvement",
            )

        stage("ranking_and_selection", selection)

        def persistence(r, folder=folder, number=number):
            checkpoint = r.read(folder / "checkpoint.json")
            memory = r.read(folder / "memory.json")
            memory_ids = ids(memory, "memory_id")
            require(
                checkpoint.get("completed_round") == number
                and checkpoint.get("task", {}).get("task_id") == tid,
                "Checkpoint round/task mismatch",
            )
            require(checkpoint.get("memory") == memory, "Memory/checkpoint mismatch")
            for entry in memory:
                require(
                    entry.get("task_id") == tid
                    and text(entry.get("key"))
                    and entry.get("value", {}).get("source"),
                    "Empty/unbound persistent memory",
                )
                source = entry["value"]["source"]
                sources = source.values() if isinstance(source, dict) else [source]
                for path in sources:
                    require(text(path), "Invalid memory evidence source")
                    p = Path(path.split("#", 1)[0])
                    if not p.is_absolute():
                        p = task / p
                    else:
                        require(
                            p.is_relative_to(root / task),
                            "Memory source belongs to another task",
                        )
                        p = p.relative_to(root)
                    require(bool(r.read(p)), "Empty persistent memory source")
            expected_slots = slots[:3] + [
                task / f"round_{n}" / "actors" / arm
                for n in range(1, number + 1)
                for arm in "ABC"
            ]
            pool = [r.read(s / "RESULT.json")["candidate"] for s in expected_slots]
            require(
                checkpoint.get("pool") == pool,
                "Checkpoint lost or substituted native pool entries",
            )
            require(
                checkpoint.get("previous_pack") == r.read(folder / "pack.json"),
                "Checkpoint pack mismatch",
            )
            require(
                isinstance(checkpoint.get("expectation_cache"), dict),
                "Missing persistent expectation cache",
            )
            require(
                isinstance(checkpoint.get("checks"), list) and checkpoint["checks"],
                "Missing persistent check evidence",
            )
            decisions = checkpoint["decisions"]
            require(
                len(decisions) == number
                and [d.get("round") for d in decisions] == list(range(1, number + 1)),
                "Decision history lost across rounds",
            )
            context = r.read(folder / "context.json")
            require(
                len(context["products"]) == 3 * number,
                "Next round did not receive accumulated products",
            )
            for index, product in enumerate(context["products"]):
                require(
                    product.get("product_id") == f"product_{index + 1}"
                    and product.get("artifact", {}).get("final_response")
                    == pool[index]["artifact"]["final_response"],
                    "Next round context substituted a prior product",
                )
            if number > 1:
                prior = r.read(task / f"round_{number - 1}" / "checkpoint.json")
                prior_ids = ids(prior["memory"], "memory_id")
                require(
                    set(prior_ids) <= set(memory_ids),
                    "Persistent memories lost across rounds",
                )
                require(
                    checkpoint["decisions"][:-1] == prior["decisions"],
                    "Prior decisions changed across checkpoints",
                )
                retrieved = context.get("memories")
                require(
                    isinstance(retrieved, list) and retrieved,
                    "No cross-round memory retrieval evidence",
                )
                prior_by_id = {m["memory_id"]: m for m in prior["memory"]}
                for row in retrieved:
                    old = prior_by_id[row["memory_id"]]
                    require(
                        row.get("value") == old["value"]
                        and row.get("key") == old["key"]
                        and finite(row.get("relevance"))
                        and row["relevance"] > 0,
                        "Retrieved memory not from prior checkpoint",
                    )
                require(
                    context.get("checks"), "Prior check evidence not carried forward"
                )
            r.facts.update(memory_entries=len(memory), pool_size=len(pool))

        stage("memory_and_checkpoint", persistence)

        def completed(r, folder=folder, number=number):
            row = r.read(folder / "summary.private.json")
            require(
                row.get("status") == "completed"
                and row.get("round") == number
                and row.get("pool_size") == 3 * (number + 1),
                "Round not completed",
            )
            r.closed = True
            require(
                set(row["arms"]) == set("ABC") and set(row["official"]) == set("ABC"),
                "Round did not complete A/B/C",
            )
            method = r.read(folder / "hypothesis.json")
            applied = (
                method.get("status") == "propose"
                and method.get("availability") != "unavailable"
            )
            for arm in "ABC":
                delivery = r.read(folder / arm / "delivery.json")
                native = r.read(folder / "actors" / arm / "RESULT.json")
                require(
                    row["arms"][arm] == delivery
                    and delivery.get("status") == "completed"
                    and delivery.get("candidate") == native.get("candidate"),
                    "Round/native delivery mismatch",
                )
                require(
                    delivery.get("hypothesis_applied") is (arm == "C" and applied),
                    "Completed Actor hypothesis treatment mismatch",
                )
                require(
                    row["official"][arm]
                    == r.read(folder / "actors" / arm / "OFFICIAL.private.json"),
                    "Round/private score mismatch",
                )
                # Measurement completeness is audited independently, including
                # explicitly bound supplements; do not rewrite this summary.
                r.read(folder / arm / "outcome_estimate.json")
            require(
                row.get("ranking") == r.read(folder / "ranking.json")
                and row.get("selection") == r.read(folder / "selected_delivery.json"),
                "Round ranking/selection mismatch",
            )

        stage("completed", completed)

    def completion(r):
        result = r.read(task / "result.private.json")
        require(
            not result.get("construction_failure"),
            "Run ended with a construction failure",
        )
        require(
            result.get("status") == "completed" and result.get("completed_rounds") == 3,
            "Requires three completed rounds",
        )
        r.closed = True
        summaries = [
            r.read(task / f"round_{n}" / "summary.private.json") for n in range(1, 4)
        ]
        require(
            result.get("rounds") == summaries,
            "Final run rounds differ from durable round receipts",
        )
        require(
            result.get("selected_delivery") == summaries[-1]["selection"]
            and result.get("selected_delivery"),
            "Missing final selection",
        )
        require(
            result.get("memory") == r.read(task / "round_3/memory.json"),
            "Final memory not persisted",
        )

    check("three_completed_rounds", completion, closed=finished)
    return _report(root, benchmark, tid, checks)


def _report(root, benchmark, task_id, checks):
    verdict = (
        "fail"
        if any(c["status"] == "fail" for c in checks)
        else "pending"
        if any(c["status"] == "pending" for c in checks)
        else "pass"
    )

    def passed(suffix):
        return sum(c["status"] == "pass" for c in checks if c["stage"].endswith(suffix))

    by_stage = {c["stage"]: c for c in checks}

    def successful(name):
        return by_stage.get(name, {}).get("status") == "pass"

    valid_reviews = [
        n
        for n in range(1, 4)
        if successful(f"round_{n}/B_review_execution")
        and successful(f"round_{n}/actors/B/native_actor")
    ]
    continuation_rounds = [
        n
        for n in valid_reviews
        if by_stage[f"round_{n}/B_review_execution"]["facts"].get(
            "continuation_exercised"
        )
        is True
    ]
    revision_rounds = [
        n
        for n in valid_reviews
        if by_stage[f"round_{n}/B_review_execution"]["facts"].get(
            "changed_artifact_revision_exercised"
        )
        is True
    ]
    coverage_complete = verdict == "pass" and bool(revision_rounds)
    coverage_status = (
        "failed"
        if verdict == "fail"
        else "pending"
        if verdict == "pending"
        else "complete"
        if coverage_complete
        else "incomplete"
    )
    return {
        "schema": "pipeline-receipt-audit-v2",
        "run_directory": str(root),
        "benchmark": benchmark,
        "task_id": task_id,
        "status": verdict,
        "coverage": {
            "status": coverage_status,
            "required_positive_branches": ["same_session_changed_artifact_revision"],
            "unobserved_required_branches": []
            if revision_rounds
            else ["same_session_changed_artifact_revision"],
            "same_session_continuation_exercised_rounds": continuation_rounds,
            "changed_artifact_revision_exercised_rounds": revision_rounds,
            "criterion_source": ".aris/traces/experiment-audit/2026-10-10_run01/003-repaired-source.response.md#addendum-003-a",
        },
        "counts": {
            "completed_rounds": passed("/completed"),
            "native_seeds": sum(
                c["status"] == "pass"
                for c in checks
                if c["stage"].startswith("seeds/")
                and c["stage"].endswith("/native_actor")
            ),
            "native_actors": sum(
                c["status"] == "pass"
                for c in checks
                if c["stage"].startswith("round_")
                and c["stage"].endswith("/native_actor")
            ),
            "native_graded_outputs": passed("/native_grade"),
            "validated_review_rounds": len(valid_reviews),
            "validated_continuation_rounds": len(continuation_rounds),
            "validated_revision_rounds": len(revision_rounds),
            "full_delivered_pack_measurements": passed("/delivered_pack_measurement"),
            "supplemental_delivered_pack_measurements": sum(
                c["status"] == "pass" and c["facts"].get("provenance") == "supplemental"
                for c in checks
                if c["stage"].endswith("/delivered_pack_measurement")
            ),
        },
        "claims": {
            "full_pipeline_receipts_validated": verdict == "pass",
            # Retain v1's stronger all-three meaning; never silently map it to any().
            "same_session_revision_validated": len(revision_rounds) == 3,
            "b_reviews_validated_all_rounds": len(valid_reviews) == 3,
            "same_session_revision_exercised_at_least_once": bool(revision_rounds),
            "same_session_revision_exercised_all_rounds": len(revision_rounds) == 3,
            "required_module_coverage_complete": coverage_complete,
            "original_run_full_delivered_measurements": sum(
                c["status"] == "pass"
                and c["facts"].get("primary_full_pack_measurement") is True
                for c in checks
                if c["stage"].endswith("/delivered_pack_measurement")
            )
            == 9,
            "hypothesis_proposals_exercised": sum(
                successful(f"round_{n}/hypothesis")
                and by_stage[f"round_{n}/hypothesis"]["facts"].get("mode") == "propose"
                and successful(f"round_{n}/actors/C/native_actor")
                and successful(f"round_{n}/completed")
                for n in range(1, 4)
            ),
            "semantic_keep_rounds": sum(
                c["status"] == "pass" and c["facts"].get("mode") == "keep"
                for c in checks
                if c["stage"].endswith("/hypothesis")
            ),
            "quality_improvement_validated": False,
            "official_leaderboard_equivalence_validated": False,
        },
        "scope": {
            "required": "Three fresh seeds; three completed legacy A/B/C rounds; nine distinct native Actors and native grades; executed construction, review/revision, ranking and persistent memory receipts.",
            "smoke": "Synthetic parser receipt only. Callable bindings do not establish semantic checker correctness.",
            "native_artifacts": "Native candidate/session IDs identify executions; Claw protocol identities may repeat. Retained snapshot files must match inventory paths, sizes, byte hashes and candidate file metadata.",
            "measurements": "Full construction/delivery receipts require a recorded harness input fingerprint matching the actual task, candidate and complete pack under measurement_context normalization. Measured product fail is valid evidence, not checker failure. Unknown remains unknown; entirely unknown or error-only stages cannot validate.",
            "supplemental_measurements": "Separately recorded full delivered-pack measurements require explicit provenance, verified retained native workspace bytes, exact final snapshot and pack paths, artifact/candidate/pack SHA-256 bindings, and a matching harness-recorded measurement input fingerprint. Optional whole-input and source hashes are recomputed; code hashes remain declared provenance. Supplemental passes preserve original measurement issues and cannot establish original execution completeness.",
            "hypotheses": "Semantic keep validates a completed decision only; it does not exercise proposal application. Unavailable is a failure, never keep.",
            "revision": "All three B reviews must be valid, including supported no-change stops. Required module coverage needs at least one grounded-feedback, same-session native continuation, changed-artifact and valid-recheck branch. Unchanged continuations count separately. same_session_revision_validated retains its v1 all-three meaning. No quality-improvement claim.",
            "receipt_status": "status and full_pipeline_receipts_validated describe receipt integrity/execution only. Required positive-branch coverage is separate: consult coverage.status and required_module_coverage_complete. Three valid no-change reviews leave revision untested and coverage incomplete.",
            "scores": "Native outputs are preserved without normalization, averaging, threshold-based acceptance, or cross-benchmark comparison.",
            "not_covered": [
                "Optional/adversarial construction and recovery branches",
                "Causal improvement or semantic correctness of Jury/checker judgments",
                "Frozen code integrity, receipt authenticity, or absence of undisclosed historical inputs",
                "Official leaderboard equivalence",
            ],
            "live": "Read-only sequential snapshot, not atomic across files. Rerun as receipts arrive; missing/partial JSON stays pending only while its producer remains open.",
        },
        "checks": checks,
        "coordination": [
            {
                "stage": c["stage"],
                "status": c["status"],
                "reason": c["reason"],
                "evidence": c["evidence"],
            }
            for c in checks
            if c["status"] != "pass"
        ]
        + (
            [
                {
                    "stage": "required_module_coverage",
                    "status": coverage_status,
                    "reason": "No validated changed-artifact same-session revision observed; required branch remains untested. Do not force a revision.",
                    "evidence": sorted(
                        {
                            path
                            for n in range(1, 4)
                            for path in by_stage.get(
                                f"round_{n}/B_review_execution", {}
                            ).get("evidence", [])
                        }
                    ),
                }
            ]
            if not revision_rounds
            else []
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args(argv)
    report = audit_run(args.run_directory)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if (
        report["status"] == "pass"
        and not report["claims"]["required_module_coverage_complete"]
    ):
        return 3
    return {"pass": 0, "fail": 1, "pending": 2}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
