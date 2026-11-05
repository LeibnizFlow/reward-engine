"""Read-only, deterministic receipt audit for tests.validation.validate_pipeline.

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

# Preserve the established module import surface.
import hashlib as hashlib
import json
import math as math
from copy import deepcopy as deepcopy
from pathlib import Path

from experiments.audit_measurement import batch as batch
from experiments.audit_measurement import check_result as check_result
from experiments.audit_measurement import criteria as criteria
from experiments.audit_measurement import measurement as measurement
from experiments.audit_measurement import measurement_binding as measurement_binding
from experiments.audit_measurement import (
    measurement_binding_schema as measurement_binding_schema,
)
from experiments.audit_measurement import measurement_execution as measurement_execution
from experiments.audit_measurement import measurement_results as measurement_results
from experiments.audit_native import _final_candidate_source as _final_candidate_source
from experiments.audit_native import final_candidate_source as final_candidate_source
from experiments.audit_native import native_actor as native_actor
from experiments.audit_native import native_turn as native_turn
from experiments.audit_native import official as official
from experiments.audit_receipts import IncompleteReceipt as IncompleteReceipt
from experiments.audit_receipts import Receipts as Receipts
from experiments.audit_receipts import candidate as candidate
from experiments.audit_receipts import finite as finite
from experiments.audit_receipts import ids as ids
from experiments.audit_receipts import receipt_digest as receipt_digest
from experiments.audit_receipts import require as require
from experiments.audit_receipts import retained_workspace as retained_workspace
from experiments.audit_receipts import strings as strings
from experiments.audit_receipts import text as text
from experiments.audit_review import delivered_measurement as delivered_measurement
from experiments.audit_review import review as review
from experiments.audit_review import review_binding as review_binding
from experiments.audit_review import review_measurement as review_measurement


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
            "criterion_source": "experiments/pipeline_audit.py::_report",
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
