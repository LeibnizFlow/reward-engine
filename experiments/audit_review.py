"""Audit review for the existing public entry point."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path


def review_measurement(row, pack):
    from experiments.audit_measurement import batch, measurement_results
    from experiments.audit_receipts import require, strings, text

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


def delivered_measurement(r, folder, arm, benchmark):
    """A targeted B review never substitutes for the complete delivered Pack."""
    from experiments.audit_measurement import (
        measurement,
        measurement_binding,
        measurement_binding_schema,
    )
    from experiments.audit_native import _final_candidate_source
    from experiments.audit_receipts import receipt_digest, require, text

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
    from experiments.audit_measurement import (
        measurement,
        measurement_binding,
        measurement_binding_schema,
        measurement_execution,
    )
    from experiments.audit_receipts import require
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
    from experiments.audit_measurement import check_result
    from experiments.audit_native import native_turn
    from experiments.audit_receipts import candidate, require, text
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
