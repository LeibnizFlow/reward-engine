"""Audited recovery hooks, copied into each historical WorkBuddy code snapshot.

This module deliberately imports the frozen implementation only at runtime. It
does not replace the historical Actor, learning algorithm, or selection policy.
"""

import argparse
import asyncio
import inspect
import os
import shutil
import sys
import time
from contextvars import ContextVar
from copy import deepcopy
from pathlib import Path

from reward_engine.adapters.workbuddy.io import digest, file_hash, read, write

ACTIVE_TASK = ContextVar("office_archive_task", default=None)
RECOVERY_PHASES = ContextVar("office_archive_phases", default=None)


def snapshot_partial_phases(root):
    """Keep accepted stage outputs before a retry can overwrite its journal."""
    phases = {}
    for phase in root.glob("round_*"):
        if (phase / "checkpoint.json").exists() or not (phase / "jury.json").exists():
            continue
        if not (phase / "context.json").exists():
            continue
        destination = root / "recovery_states" / str(time.time_ns()) / phase.name
        destination.mkdir(parents=True)
        for path in phase.glob("*.json"):
            shutil.copy2(path, destination / path.name)
        write(
            destination / "HASHES.json",
            {p.name: file_hash(p) for p in destination.glob("*.json")},
        )
        phases[phase.name] = destination
    return phases


async def builder_evidence_request(ask, call, mode, context, evidence, *args, **kwargs):
    """One format correction for nonexistent handles; never supply product data."""

    class CheckedCall:
        corrected = False

        def __getattr__(self, name):
            return getattr(call, name)

        async def __call__(self, request):
            if mode == "outcome_builder" and isinstance(request.get("repair"), dict):
                request = deepcopy(request)
                request["repair"]["field_issues"] = builder_repair_issues(request)
            result = await call(request)

            def completed_contract(value):
                if (
                    mode == "outcome_builder"
                    and isinstance(value, dict)
                    and value.get("rubrics")
                    and isinstance(value.get("tests"), list)
                    and value["tests"]
                    and "need_evidence" in value
                ):
                    # A final contract plus an evidence-request envelope is ambiguous.
                    # Preserve the complete contract; compile_contract still validates it.
                    value = deepcopy(value)
                    value.pop("need_evidence")
                return value

            result = completed_contract(result)
            refs = result.get("need_evidence") if isinstance(result, dict) else None
            if refs is None:
                return result
            valid = (
                isinstance(refs, list)
                and refs
                and all(isinstance(ref, str) and ref in evidence for ref in refs)
            )
            if valid or self.corrected:
                return result
            self.corrected = True
            correction = deepcopy(request)
            correction["evidence_access"] = (
                "This is the final bounded correction. Return the complete requested contract, "
                "without a need_evidence field. No unavailable handle can be fetched. "
                "Future product bindings are placeholders for measurement, not compilation inputs."
            )
            correction["response_recovery"] = {
                "attempt": 1,
                "field": "need_evidence",
                "rejected_request": refs,
                "available_evidence_refs": sorted(evidence),
                "instruction": "The requested handles are not observations in this compilation context. "
                "Cite only available handles, without inventing observations. "
                "Current-product contract handles bind a future product at measurement time; "
                "their contents cannot be read while compiling. Return the requested stage contract "
                "using the same accepted criteria, or request only evidence listed as available. "
                "Do not invent or infer any unavailable product contents.",
            }
            if hasattr(call, "output"):
                write(
                    Path(call.output)
                    / "evidence_recovery"
                    / (digest(request) + ".json"),
                    correction["response_recovery"],
                )
            return completed_contract(await call(correction))

    return await ask(CheckedCall(), mode, context, evidence, *args, **kwargs)


def builder_repair_issues(request):
    """Report independent test-shape errors even if rubric coverage failed first."""
    output = request.get("repair", {}).get("original_output", {})
    issues = []
    if not isinstance(output, dict):
        return issues
    tools = request.get("measurement_tools") or {}
    tests = output.get("tests", [])
    if not isinstance(tests, list):
        return [
            {
                "path": ["tests"],
                "reason": "Tests must be a list of complete executable test objects.",
            }
        ]
    seen = set()
    for i, test in enumerate(tests):
        if not isinstance(test, dict):
            continue
        for field in ("test_id", "criterion_id", "checker", "question"):
            if not isinstance(test.get(field), str) or not test[field].strip():
                issues.append(
                    {
                        "path": ["tests", i, field],
                        "reason": "Required nonempty text field.",
                    }
                )
        identifier = test.get("test_id")
        if isinstance(identifier, str):
            if identifier in seen:
                issues.append(
                    {
                        "path": ["tests", i, "test_id"],
                        "reason": "Test IDs must be unique.",
                    }
                )
            seen.add(identifier)
        checker = test.get("checker")
        if (
            test.get("probes")
            and checker != "evidence.review"
            and isinstance(checker, str)
            and checker not in tools
        ):
            issues.append(
                {
                    "path": ["tests", i],
                    "test_id": identifier,
                    "reason": "Inline probes require evidence.review (or a catalog measurement tool "
                    "that code wraps). json.parse checks JSON syntax only; it cannot judge "
                    "counts, schema completeness, uniqueness or consistency. Choose a "
                    "checker matching the intended measurement and correct every affected test. "
                    "Do not discard substantive checks to repair their representation.",
                }
            )
    return issues


def evidence_contract_issues(draft, context, evidence):
    """Give exact failing paths and available handles, without changing judgments."""
    issues = []
    if not isinstance(draft, dict):
        return issues
    criteria = draft.get("criteria")
    if not isinstance(criteria, list) or not 1 <= len(criteria) <= 20:
        issues.append(
            {
                "path": ["criteria"],
                "count": len(criteria) if isinstance(criteria, list) else None,
                "reason": "Publish 1 to 20 criteria. Merge overlapping criteria while preserving "
                "supported requirements, explicit subrequirements and coverage mappings; "
                "do not drop obligations to satisfy the size bound.",
            }
        )
    for section in ("criteria", "gaps"):
        rows = draft.get(section, [])
        seen_ids = set()
        for i, row in enumerate(rows if isinstance(rows, list) else []):
            if not isinstance(row, dict):
                continue
            id_field = "gap_id" if section == "gaps" else "id"
            identifier = row.get(id_field)
            if isinstance(identifier, str):
                if identifier in seen_ids:
                    issues.append(
                        {
                            "path": [section, i, id_field],
                            "value": identifier,
                            "reason": "Duplicate identifier. Use a unique ID for distinct content, "
                            "or merge genuinely duplicate content while preserving its support.",
                        }
                    )
                seen_ids.add(identifier)
            if (
                section == "criteria"
                and row.get("kind") == "hard"
                and row.get("origin") not in {"task", "user"}
            ):
                issues.append(
                    {
                        "path": ["criteria", i],
                        "criterion_id": identifier,
                        "reason": "This hard criterion declares domain/observed-gap provenance. "
                        "A hard requirement needs actual task/user authority. Correct kind "
                        "or origin against the cited evidence; do not invent authority "
                        "merely to pass validation.",
                    }
                )
            for key in ("evidence_refs", "expected_refs"):
                refs = row.get(key, [])
                if (
                    not isinstance(refs, list)
                    or (key == "evidence_refs" and not refs)
                    or any(not isinstance(r, str) or r not in evidence for r in refs)
                ):
                    issues.append(
                        {
                            "path": [section, i, key],
                            "value": refs,
                            "reason": "Use exact existing evidence handles only. A missing observation "
                            "does not establish a product defect.",
                        }
                    )
            field = (
                "expected_refs"
                if section == "gaps" and row.get("expected_basis") == "task"
                else "evidence_refs"
                if section == "criteria"
                and row.get("kind") == "hard"
                and row.get("origin") == "task"
                else None
            )
            if field and "task" not in (row.get(field) or []):
                issues.append(
                    {
                        "path": [section, i, field],
                        "reason": "The declared task provenance must cite task. Confirm the actual "
                        "source; never relabel a domain preference as a task obligation.",
                    }
                )
            if section != "gaps":
                continue
            targets = row.get("product_ids")
            allowed = [p["product_id"] for p in context.get("products", [])]
            if (
                not isinstance(targets, list)
                or not targets
                or any(p not in allowed for p in targets)
            ):
                issues.append(
                    {
                        "path": ["gaps", i, "product_ids"],
                        "value": targets,
                        "allowed_product_ids": allowed,
                        "reason": "A published gap must identify existing affected products supported "
                        "by actual observations. Withhold a product accusation when no affected "
                        "product is supported; an open question is not an observed defect.",
                    }
                )
            refs = row.get("evidence_refs", [])
            refs = (
                [r for r in refs if isinstance(r, str) and r in evidence]
                if isinstance(refs, list)
                else []
            )
            for product in targets if isinstance(targets, list) else []:
                available = [
                    r
                    for r, v in evidence.items()
                    if r.startswith(str(product) + ":")
                    or (
                        r.startswith("check:")
                        and isinstance(v, dict)
                        and v.get("product_id") == product
                    )
                ]
                if not set(refs).intersection(available):
                    issues.append(
                        {
                            "path": ["gaps", i],
                            "product_id": product,
                            "available_product_refs": available,
                            "reason": "This affected product has no cited observation. Inspect its "
                            "evidence and cite support, or withhold the unsupported accusation. "
                            "Another product or task text is not evidence of this observation.",
                        }
                    )
    if issues:
        issues.append(
            {
                "available_evidence_refs": sorted(evidence),
                "reason": "Handle inventory only; existence alone does not support a claim. "
                "Replace the enclosing row when a missing field must be added.",
            }
        )
    return issues


def normalize_task_citations(draft):
    """Make already-declared task provenance explicit, without changing its origin."""
    result = deepcopy(draft)
    if not isinstance(result, dict):
        return result
    for section, field in [("criteria", "evidence_refs"), ("gaps", "expected_refs")]:
        rows = result.get(section, [])
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict):
                continue
            declared = (
                row.get("expected_basis") == "task"
                if section == "gaps"
                else row.get("kind") == "hard" and row.get("origin") == "task"
            )
            refs = row.get(field, [])
            if declared and isinstance(refs, list) and "task" not in refs:
                row[field] = [*refs, "task"]
    return result


def normalize_existing_citations(draft, evidence):
    """Remove nonexistent handles; remaining evidence must still pass every validator."""
    result = normalize_task_citations(draft) if "task" in evidence else deepcopy(draft)
    if not isinstance(result, dict):
        return result
    for section in ("criteria", "gaps"):
        rows = result.get(section, [])
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict):
                continue
            for field in ("evidence_refs", "expected_refs"):
                refs = row.get(field)
                if isinstance(refs, list) and all(isinstance(r, str) for r in refs):
                    row[field] = [r for r in refs if r in evidence]
    return result


def repair_targets(draft, issues):
    """Expose original row indices prominently; never renumber a filtered issue list."""
    targets = []
    for issue in issues:
        path = issue.get("path")
        if not isinstance(path, list) or not path:
            continue
        path = path[:2]
        if any(t["path"] == path for t in targets):
            continue
        value = draft
        try:
            for key in path:
                value = value[key]
        except (KeyError, IndexError, TypeError):
            continue
        targets.append({"path": path, "current_value": deepcopy(value)})
    return targets


def validate_repair_targets(reply, targets):
    """A targeted edit must actually address its reported rows, not list-local indices."""
    if not isinstance(reply, dict) or "edits" not in reply:
        return reply
    edits = reply["edits"]
    if not isinstance(edits, list) or not edits:
        raise ValueError("Return nonempty explicit edits for the reported invalid rows")
    paths = [
        e.get("path")
        for e in edits
        if isinstance(e, dict) and isinstance(e.get("path"), list)
    ]
    missed = [
        t["path"]
        for t in targets
        if not any(
            path
            and path[: min(len(path), len(t["path"]))]
            == t["path"][: min(len(path), len(t["path"]))]
            for path in paths
        )
    ]
    if missed:
        raise ValueError(
            f"Edits do not address reported original paths {missed!r}. "
            "Use indices from invalid_draft, not indices within repair_targets. "
            "Correct all reported rows without inventing evidence or changing valid judgments."
        )
    return reply


def publication_issues(draft, expectations=()):
    """Describe invalid published clauses and dispositions without rewriting model judgments.

    This mirrors the frozen validators (jury.reconcile_coverage and
    jury_drafts.resolve_parts) so a bounded contract repair receives precise,
    existing paths for coverage/subrequirement disposition errors. It never
    changes a quality decision, only the representational contract.
    """
    issues = []
    for i, criterion in enumerate(draft.get("criteria", [])):
        if not isinstance(criterion, dict):
            continue
        for key in ["id", "standard", "measurement_question"]:
            if key in criterion and (
                not isinstance(criterion[key], str) or not criterion[key].strip()
            ):
                issues.append(
                    {
                        "path": ["criteria", i, key],
                        "value": criterion[key],
                        "reason": "This existing field must contain nonempty explicit text. "
                        "Correct the representation against the supplied task and evidence.",
                    }
                )
        if "subrequirements" not in criterion:
            continue
        parts = criterion["subrequirements"]
        if not isinstance(parts, list) or any(
            not isinstance(p, str) or not p.strip() for p in parts
        ):
            issues.append(
                {
                    "path": ["criteria", i, "subrequirements"],
                    "value": parts,
                    "reason": "Published subrequirements must be a list of nonempty text strings. "
                    "Provide explicit supported wording, not part IDs, objects, or null. "
                    "Use an existing path and a nonempty reason for each repair; preserve meaning.",
                }
            )
    criteria = draft.get("criteria", [])
    have_ids = bool(criteria) and all(
        isinstance(c, dict) and isinstance(c.get("id"), str) and c.get("id")
        for c in criteria
    )
    valid = {c["id"] for c in criteria} if have_ids else set()
    expectations = [e for e in (expectations or []) if isinstance(e, dict)]
    by_expect = {e.get("expectation_id"): e for e in expectations}
    coverage = draft.get("coverage")
    if not isinstance(coverage, list):
        coverage = []
    for i, row in enumerate(coverage):
        if not isinstance(row, dict):
            continue
        ids, status, reason = (
            row.get("criterion_ids"),
            row.get("status"),
            row.get("reason"),
        )
        parent_bad = (
            status not in {"mapped", "deferred", "excluded"}
            or not isinstance(reason, str)
            or not reason.strip()
            or not isinstance(ids, list)
            or any(not isinstance(x, str) or (valid and x not in valid) for x in ids)
            or (status == "mapped") != bool(ids)
        )
        if parent_bad:
            issues.append(
                {
                    "path": ["coverage", i],
                    "value": {
                        k: row.get(k) for k in ("status", "criterion_ids", "reason")
                    },
                    "allowed_status": ["mapped", "deferred", "excluded"],
                    "allowed_criterion_ids": sorted(valid),
                    "reason": "Each coverage row needs status mapped/deferred/excluded, a nonempty "
                    "reason, only existing final criterion IDs, and criterion_ids set exactly "
                    "when mapped. Defer an unsupported expectation instead of mapping it.",
                }
            )
        expected = by_expect.get(row.get("expectation_id"))
        expected_parts = (
            expected.get("subrequirements", []) if isinstance(expected, dict) else []
        )
        parts = row.get("subrequirements")
        if not expected_parts or parts is None:
            continue
        allowed_parts = [
            p["part_id"]
            for p in expected_parts
            if isinstance(p, dict) and p.get("part_id")
        ]
        if not isinstance(parts, list) or any(not isinstance(p, dict) for p in parts):
            issues.append(
                {
                    "path": ["coverage", i, "subrequirements"],
                    "value": parts,
                    "allowed_part_ids": allowed_parts,
                    "reason": "Subrequirement decisions must be a list of objects, one per supplied part_id.",
                }
            )
            continue
        for j, part in enumerate(parts):
            if not isinstance(part, dict):
                continue
            pids, pstatus, preason = (
                part.get("criterion_ids"),
                part.get("status"),
                part.get("reason"),
            )
            part_bad = (
                pstatus not in {"mapped", "deferred", "excluded"}
                or not isinstance(preason, str)
                or not preason.strip()
                or not isinstance(pids, list)
                or any(
                    not isinstance(x, str) or (valid and x not in valid) for x in pids
                )
                or (pstatus == "mapped") != bool(pids)
            )
            if pstatus == "mapped" and (
                not isinstance(part.get("retained_text"), str)
                or not part["retained_text"].strip()
            ):
                part_bad = True
            if part_bad:
                issues.append(
                    {
                        "path": ["coverage", i, "subrequirements", j],
                        "value": {
                            k: part.get(k)
                            for k in (
                                "status",
                                "criterion_ids",
                                "reason",
                                "retained_text",
                            )
                        },
                        "allowed_status": ["mapped", "deferred", "excluded"],
                        "allowed_criterion_ids": sorted(valid),
                        "reason": "Each part needs status mapped/deferred/excluded, a nonempty reason, "
                        "only existing final criterion IDs, criterion_ids exactly when mapped, "
                        "and retained_text quoting the final clause when mapped. Defer a part "
                        "with no supported retained clause; never invent one.",
                    }
                )
    return issues


def resolve_source(config, value):
    path = Path(value)
    for old, new in sorted(
        config.get("source_locations", {}).items(), key=lambda x: -len(x[0])
    ):
        if path.is_relative_to(old):
            return Path(new) / path.relative_to(old)
    return path


def verify_retained(config, relative, *, task=None, prompt=None, pack=None, grade=None):
    source = config["retained_slots"][relative]
    output = Path(config["output"]) / relative
    for name, expected in source["hashes"].items():
        if file_hash(output / name) != expected:
            raise ValueError(f"Retained slot changed: {relative}/{name}")
    spec, row = read(output / "spec.json"), read(output / "RESULT.json")
    if (
        row.get("status") != "completed"
        or row.get("spec_identity") != spec["spec_identity"]
    ):
        raise ValueError("Retained Actor has no complete original binding")
    if task is not None and (
        spec["task"] != task
        or spec["prompt"] != prompt
        or spec.get("pack") != pack
        or spec["grade"] != grade
    ):
        raise ValueError(
            f"Recovery would change original generation inputs: {relative}"
        )
    if row["candidate"]["candidate_id"] != spec["candidate_id"]:
        raise ValueError("Retained candidate identity differs")
    return row


def completed_rounds(root, task, seeds):
    """Restore old rounds including the later, derived hypothesis annotation."""
    pool, history, checkpoint = list(seeds), [], None
    for number in range(1, 4):
        phase = root / f"round_{number}"
        if not (phase / "checkpoint.json").exists():
            break
        summary = read(phase / "summary.private.json")
        if summary.get("status") != "completed":
            break
        candidate = read(phase / "checkpoint.json")
        if candidate["task"] != task or candidate["completed_round"] != number:
            raise ValueError("Checkpoint task or round differs")
        for arm in "ABC":
            row = read(phase / f"actors/{arm}/RESULT.json")
            expected = deepcopy(row)
            saved = summary["arms"][arm]
            if "hypothesis_applied" in saved:
                from reward_engine.core.outcome.arms import arm_spec, has_hypothesis

                protocol, method = (
                    read(phase / "arm_protocol.json"),
                    read(phase / "hypothesis.json"),
                )
                applied = arm_spec(protocol["id"], arm)[
                    "hypothesis"
                ] and has_hypothesis(method)
                if (
                    summary["arm_protocol"] != protocol["id"]
                    or protocol["arms"][arm] != arm_spec(protocol["id"], arm)
                    or protocol["hypothesis_applied"] is not has_hypothesis(method)
                ):
                    raise ValueError("Checkpoint treatment differs")
                expected["hypothesis_applied"] = applied
            if expected != saved or row.get("status") != "completed":
                raise ValueError("Checkpoint product differs from retained Actor")
            if summary["official"][arm] != read(
                phase / f"actors/{arm}/OFFICIAL.private.json"
            ):
                raise ValueError("Checkpoint grading receipt differs")
            pool.append(row["candidate"])
        if candidate["pool"] != pool:
            raise ValueError("Checkpoint pool differs")
        for key, filename in [
            ("memory", "memory.json"),
            ("checks", "check_sources.json"),
            ("previous_pack", "pack.json"),
        ]:
            if candidate[key] != read(phase / filename):
                raise ValueError(f"Checkpoint {key} differs")
        history.append(summary)
        checkpoint = candidate
    return history, checkpoint


def rankings_complete(root):
    for number in range(1, 4):
        path = root / f"round_{number}/summary.private.json"
        if (
            not path.exists()
            or read(path).get("ranking", {}).get("status") != "compared"
        ):
            return False
    return True


def incomplete_rounds(root):
    """Completed rounds whose terminal ranking is unavailable, in round order."""
    missing = []
    for number in range(1, 4):
        path = root / f"round_{number}/summary.private.json"
        if not path.exists():
            continue
        summary = read(path)
        if (
            summary.get("status") == "completed"
            and summary.get("ranking", {}).get("status") != "compared"
        ):
            missing.append(number)
    return missing


def _sync_result_private(root, number, ranking, selection):
    path = root / "result.private.json"
    if not path.exists():
        return
    result = read(path)
    for row in result.get("rounds", []):
        if isinstance(row, dict) and row.get("round") == number:
            row["ranking"] = ranking
            row["selection"] = selection
    if number == result.get("completed_rounds"):
        result["selected_delivery"] = selection
    write(path, result)


async def rank_for_recovery(
    rank, context, evidence, members, reducer, *, policy, record, request_budget_bytes
):
    """Honor the ranking API of the source batch, including its legacy defaults."""
    options = {"policy": policy, "record": record}
    if "reduce_max_bytes" in inspect.signature(rank).parameters:
        options["reduce_max_bytes"] = request_budget_bytes
    return await rank(context, evidence, members, reducer, **options)


async def recover_missing_rankings(config, task_id):
    """Recompute only unavailable rankings from the round's retained public inputs.

    Uses the frozen build_context/jury.rank with the round's own pool, checks and
    role outputs; official scores are never read. Original artifacts are copied to
    recovery_states before any update, and a failure leaves the round untouched.
    """
    from reward_engine.adapters.workbuddy.models import Role
    from reward_engine.core.batch import BatchPolicy
    from reward_engine.core.outcome import jury
    from reward_engine.core.outcome.context import build_context
    from reward_engine.core.outcome.loop import select_delivery

    root = Path(config["output"]) / "tasks" / task_id
    task = {
        "task_id": task_id,
        "prompt": (Path(config["tasks_root"]) / task_id / "instruction.md").read_text(),
    }
    outcomes = []
    for number in incomplete_rounds(root):
        phase = root / f"round_{number}"
        checkpoint = read(phase / "checkpoint.json")
        if (
            checkpoint.get("task") != task
            or checkpoint.get("completed_round") != number
        ):
            raise ValueError("Round checkpoint identity differs")
        checks = read(phase / "check_sources.json")
        context, evidence = build_context(
            task, checkpoint["pool"], skills=(), checks=checks
        )

        def record(name, value, phase=phase):
            write(phase / (name + ".json"), value)

        members = {
            str(i): Role(
                config,
                "jury_member_" + str(i),
                phase / "terminal" / "models" / ("jury_member_" + str(i)),
            )
            for i in range(1, 4)
        }
        reducer = Role(
            config, "jury_reducer", phase / "terminal" / "models" / "jury_reducer"
        )
        ranking = await rank_for_recovery(
            jury.rank,
            context,
            evidence,
            members,
            reducer,
            policy=BatchPolicy(**config["batch"]),
            record=record,
            request_budget_bytes=Role.request_budget_bytes,
        )
        if ranking.get("status") != "compared":
            raise ValueError("Recovered ranking is still unavailable")
        arm_products = {
            arm: f"product_{3 * number + offset}"
            for offset, arm in enumerate("ABC", start=1)
        }
        measurements = {
            arm: read(phase / arm / "outcome_estimate.json") for arm in "ABC"
        }
        selection = select_delivery(ranking, arm_products, measurements)
        selection.update(
            round=number,
            product_id=arm_products.get(selection["arm"]),
            source=f"round_{number}/{selection['arm']}/delivery.json"
            if selection["arm"]
            else None,
        )
        # Preserve the original round journal before any mutation.
        backup = root / "recovery_states" / str(time.time_ns()) / phase.name
        backup.mkdir(parents=True)
        for name in (
            "summary.private.json",
            "ranking.json",
            "ranking_map.json",
            "selected_delivery.json",
            "loop_decision.json",
        ):
            if (phase / name).exists():
                shutil.copy2(phase / name, backup / name)
        original = read(phase / "summary.private.json")
        write(
            backup / "RANKING_RECOVERY.json",
            {
                "task": task_id,
                "round": number,
                "time": time.time(),
                "method": "frozen build_context + jury.rank on retained pool/checks/roles",
                "official_scores_used": False,
                "original_ranking": original.get("ranking"),
                "recovered_ranking": ranking,
            },
        )
        summary = read(phase / "summary.private.json")
        summary["ranking"] = ranking
        summary["selection"] = selection
        write(phase / "summary.private.json", summary)
        write(phase / "selected_delivery.json", selection)
        _sync_result_private(root, number, ranking, selection)
        outcomes.append({"round": number, "arm": selection.get("arm")})
    return outcomes


def audit(config, *, require_complete=False):
    from experiments.workbuddy.experiment import verify
    from experiments.workbuddy.scoring import scoring_errors

    verify(config)
    root = Path(config["output"])
    for path in root.glob("tasks/*/recovery_states/*/round_*/HASHES.json"):
        for name, expected in read(path).items():
            if file_hash(path.parent / name) != expected:
                raise ValueError(
                    f"Recovery stage snapshot changed: {path.parent}/{name}"
                )
    for source, files in config.get("construction_hashes", {}).items():
        for name, expected in files.items():
            if file_hash(Path(source) / name) != expected:
                raise ValueError(f"Retained construction changed: {source}/{name}")
    for relative in config.get("retained_slots", {}):
        verify_retained(config, relative)
    rows, tasks = [], {}
    for task_id in config["task_ids"]:
        task_root = root / "tasks" / task_id
        task = {
            "task_id": task_id,
            "prompt": (
                Path(config["tasks_root"]) / task_id / "instruction.md"
            ).read_text(),
        }
        seeds = [
            read(p)
            for i in range(1, 4)
            if (p := task_root / f"seeds/{i}/RESULT.json").exists()
        ]
        history, checkpoint = (
            completed_rounds(task_root, task, [s["candidate"] for s in seeds])
            if (len(seeds) == 3 and all(s.get("status") == "completed" for s in seeds))
            else ([], None)
        )
        actors = sum(
            read(p).get("status") == "completed"
            for p in task_root.glob("round_*/actors/*/RESULT.json")
        )
        valid = 0
        for number in range(1, 4):
            for arm in "ABC":
                slot = task_root / f"round_{number}/actors/{arm}"
                receipt = (
                    read(slot / "OFFICIAL.private.json")
                    if (slot / "OFFICIAL.private.json").exists()
                    else {}
                )
                native = (
                    resolve_source(config, receipt["native_score"])
                    if receipt.get("native_score")
                    else None
                )
                detail = read(native) if native and native.is_file() else {}
                errors = scoring_errors(detail)
                actor = (
                    read(slot / "RESULT.json")
                    if (slot / "RESULT.json").exists()
                    else {}
                )
                clean = (
                    receipt.get("status") == "scored"
                    and actor.get("status") == "completed"
                    and actor.get("official_verifier_observed") is False
                    and bool(detail)
                    and not errors
                    and not receipt.get("exception")
                    and detail.get("overall") == receipt.get("overall")
                    and detail.get("test_status") not in {"judge_error", "build_error"}
                )
                valid += int(clean)
                rows.append(
                    {
                        "task_id": task_id,
                        "round": number,
                        "arm": arm,
                        "status": "scored"
                        if clean
                        else receipt.get("status", "not_run"),
                        "overall": receipt.get("overall") if clean else None,
                        "raw_overall": receipt.get(
                            "raw_overall", receipt.get("overall")
                        ),
                        "judge_errors": len(errors),
                        "reused": str(slot.relative_to(root))
                        in config.get("retained_slots", {}),
                        "native_score": str(native) if native else None,
                    }
                )
        final = (
            read(task_root / "result.private.json")
            if (task_root / "result.private.json").exists()
            else {}
        )
        selected = final.get("selected_delivery") or {}
        complete = (
            len(history) == 3
            and actors == 9
            and valid == 9
            and final.get("status") == "completed"
            and final.get("completed_rounds") == 3
            and final.get("rounds") == history
            and final.get("memory") == checkpoint["memory"]
            and selected == history[-1]["selection"]
            and all(r["ranking"].get("status") == "compared" for r in history)
        )
        tasks[task_id] = {
            "completed_rounds": len(history),
            "completed_seeds": len(seeds),
            "completed_actors": actors,
            "valid_scores": valid,
            "complete": complete,
            "selected_arm": selected.get("arm") if complete else None,
        }
        if require_complete and not complete:
            raise ValueError(f"Task has not passed final acceptance: {task_id}")
    return {
        "tasks": tasks,
        "scores": rows,
        "valid_scores": sum(t["valid_scores"] for t in tasks.values()),
        "complete_tasks": sum(t["complete"] for t in tasks.values()),
        "restored_rounds": sum(t["completed_rounds"] for t in tasks.values()),
    }


def install(config):
    """Bind original products explicitly; reuse construction only on exact inputs."""
    from experiments.workbuddy import experiment, report, resume
    from reward_engine.core.outcome import builder, hypotheses, jury
    from reward_engine.core.outcome.memory import Memory

    root = Path(config["output"])
    resume.completed_rounds = completed_rounds
    original_launch = experiment.launch

    async def launch(config, task, output, *, prompt="", pack=None, grade=True):
        relative = str(output.relative_to(root))
        if relative in config.get("retained_slots", {}):
            row = verify_retained(
                config, relative, task=task, prompt=prompt, pack=pack, grade=grade
            )
            write(
                output / "REUSE.json",
                {
                    **config["retained_slots"][relative],
                    "new_binding": digest(
                        {
                            "campaign": config["identity"],
                            "slot": relative,
                            "original": row["spec_identity"],
                        }
                    ),
                    "original_spec_identity": row["spec_identity"],
                },
            )
            return deepcopy(row)
        # Only incomplete attempts are moved; complete products are never redrawn.
        for attempt in range(3):
            if output.exists() and not (output / "RESULT.json").exists():
                raise ValueError(
                    f"Interrupted Actor requires inspection before retry: {output}"
                )
            row = await original_launch(
                config, task, output, prompt=prompt, pack=pack, grade=grade
            )
            if row.get("status") == "completed":
                return row
            if attempt == 2:
                return row
            destination = root / "failed_attempts" / relative / str(time.time_ns())
            destination.parent.mkdir(parents=True, exist_ok=True)
            output.rename(destination)
            write(
                destination / "RETRY.json",
                {
                    "reason": "Incomplete Actor; no successful product or score selected",
                    "next_slot": str(output),
                    "attempt": attempt + 1,
                    "time": time.time(),
                },
            )
        return row

    experiment.launch = launch
    old_one_task = experiment.one_task

    async def one_task(config, task_id):
        token = ACTIVE_TASK.set(task_id)
        phases = RECOVERY_PHASES.set(snapshot_partial_phases(root / "tasks" / task_id))
        try:
            result = await old_one_task(config, task_id)
            if result.get("status") == "completed" and not rankings_complete(
                root / "tasks" / task_id
            ):
                task_root = root / "tasks" / task_id
                try:
                    outcomes = await recover_missing_rankings(config, task_id)
                except Exception as error:
                    return {
                        **result,
                        "status": "partial",
                        "reason": "Ranking recovery failed: " + str(error),
                    }
                if not rankings_complete(task_root):
                    return {
                        **result,
                        "status": "partial",
                        "reason": "Terminal ranking requires recovery",
                    }
                write(
                    task_root / "ranking_recovery.json",
                    {
                        "rounds": outcomes,
                        "time": time.time(),
                        "official_scores_used": False,
                    },
                )
            return result
        finally:
            RECOVERY_PHASES.reset(phases)
            ACTIVE_TASK.reset(token)

    experiment.one_task = one_task

    def saved_phase(role):
        phase = Path(role.output).parent.parent
        relative = str(phase.relative_to(root))
        source = config.get("construction_sources", {}).get(relative)
        return Path(source) if source else (RECOVERY_PHASES.get() or {}).get(phase.name)

    old_jury, old_builder, old_hypothesis = (
        jury.propose,
        builder.build,
        hypotheses.propose,
    )
    original_builder_ask = builder.ask

    async def checked_builder_ask(*args, **kwargs):
        return await builder_evidence_request(original_builder_ask, *args, **kwargs)

    builder.ask = checked_builder_ask
    # Ranking map/reduce outputs get the same bounded contract correction as the
    # Jury reduce: a single representational redo that is recorded in the role
    # request. It never changes a quality judgment, only the required structure
    # (for example unknown must list product IDs, not sentences).
    if not getattr(jury, "_recovery_ranking_patch", False):
        original_ask = jury.ask

        def normalize_repair_reply(reply):
            """Unwrap a repair envelope and strip a repeated invalid_draft path prefix.

            The model sometimes returns edits wrapped in 'response' or addresses the
            path as if from the request envelope (validation_repair.invalid_draft.*).
            Both are pure representation; the values and reasons are unchanged.
            """
            if isinstance(reply, list):
                return {"edits": reply}
            if not isinstance(reply, dict):
                return reply
            if not isinstance(reply.get("edits"), list):
                for key in ("response", "repair", "result", "edits"):
                    value = reply.get(key)
                    if isinstance(value, list):
                        reply = {"edits": value}
                        break
                    if isinstance(value, dict) and isinstance(value.get("edits"), list):
                        reply = value
                        break
            if isinstance(reply.get("edits"), list):
                fixed = deepcopy(reply)
                for edit in fixed["edits"]:
                    if isinstance(edit, dict) and isinstance(edit.get("path"), list):
                        path = edit["path"]
                        if (
                            len(path) >= 3
                            and path[0] == "validation_repair"
                            and path[1] == "invalid_draft"
                        ):
                            edit["path"] = path[2:]
                return fixed
            return reply

        async def ranking_contract_ask(
            call, mode, context, evidence, instruction, response, **kwargs
        ):
            # Legacy frozen context.ask predates contract retries/validation keywords.
            # Its caller still validates the returned ranking with the frozen validator.
            if "max_contract_retries" not in inspect.signature(original_ask).parameters:
                return await builder_evidence_request(
                    original_ask,
                    call,
                    mode,
                    context,
                    evidence,
                    instruction,
                    response,
                    **kwargs,
                )
            # Malformed JSON is a transport/representation failure, not an accepted
            # judgment. Reuse the existing one-redo contract mechanism for all Jury roles.
            kwargs.setdefault("max_contract_retries", 1)
            if mode == "outcome_jury_repair":
                extra = deepcopy(kwargs.get("extra") or {})
                repair = extra.get("validation_repair", {})
                critical = evidence_contract_issues(
                    normalize_existing_citations(repair.get("invalid_draft"), evidence),
                    context,
                    evidence,
                )
                repair["field_issues"] = critical + repair.get("field_issues", [])
                targets = repair_targets(repair.get("invalid_draft"), critical)
                repair["repair_targets"] = targets
                kwargs["validate"] = lambda value: validate_repair_targets(
                    normalize_repair_reply(value), targets
                )
                kwargs.setdefault("max_contract_retries", 1)
                instruction += (
                    " Resolve the primary validation reason and the evidence/criteria-count issues FIRST. "
                    "For an oversized definition, replace the entire criteria array with at most 20 "
                    "merged criteria retaining explicit supported subrequirements, and update coverage. "
                    "For each gap, justify every affected product with its actual evidence, or remove "
                    "the unsupported product allegation. Do not merely repair cosmetic fields while "
                    "leaving the reported blocking errors unchanged. Object subrequirements carrying "
                    "text are normalized by code; do not spend edits on those representations. "
                    "repair_targets provides the exact ORIGINAL array paths and current rows. "
                    "Do not renumber these paths starting at zero. Address every listed target. "
                    "Preserve the existing gap_id when editing a gap."
                )
                if targets:
                    response = deepcopy(response)
                    response["edits"][0]["path"] = targets[0]["path"]
                kwargs["extra"] = extra
            if (
                mode in {"outcome_ranking_map", "outcome_ranking_reduce"}
                and kwargs.get("validate") is None
            ):
                kwargs["validate"] = lambda value: jury.validate_ranking(
                    value, context, evidence
                )
                kwargs.setdefault("max_contract_retries", 1)
            result = await builder_evidence_request(
                original_ask,
                call,
                mode,
                context,
                evidence,
                instruction,
                response,
                **kwargs,
            )
            if mode == "outcome_jury_repair":
                result = normalize_repair_reply(result)
            return result

        jury.ask = ranking_contract_ask
        jury._recovery_ranking_patch = True
    # The reducer can legitimately need more than the historical 64-edit cap and may
    # include a few edits whose path no longer exists after earlier edits. Apply the
    # valid edits in order, record skips, and let the frozen validator/repair loop
    # decide whether the result is acceptable. No model value or judgment is changed.
    if not getattr(jury, "_recovery_edit_patch", False):
        original_apply_edits = jury.apply_edits

        def _child(target, key):
            if isinstance(target, dict) and isinstance(key, str) and key in target:
                return target[key]
            if isinstance(target, list) and type(key) is int and 0 <= key < len(target):
                return target[key]
            raise KeyError(key)

        def tolerant_apply_edits(draft, reply):
            edits = reply.get("edits") if isinstance(reply, dict) else None
            if not isinstance(edits, list) or not 1 <= len(edits) <= 4096:
                return original_apply_edits(draft, reply)
            result = deepcopy(draft)
            skipped = []
            for index, edit in enumerate(edits):
                if not isinstance(edit, dict) or edit.get("op") not in {
                    "replace",
                    "remove",
                }:
                    skipped.append(
                        {
                            "index": index,
                            "reason": "operation must be replace or remove",
                        }
                    )
                    continue
                path = edit.get("path")
                if (
                    not isinstance(path, list)
                    or not path
                    or path[0] not in {"criteria", "gaps", "coverage", "uncertainties"}
                    or any(type(k) not in {str, int} for k in path)
                    or not isinstance(edit.get("reason"), str)
                    or not edit["reason"].strip()
                ):
                    skipped.append(
                        {
                            "index": index,
                            "reason": "invalid path or missing reason",
                            "path": path,
                        }
                    )
                    continue
                try:
                    target = result
                    for key in path[:-1]:
                        target = _child(target, key)
                    key = path[-1]
                    _child(target, key)
                except (KeyError, IndexError):
                    skipped.append(
                        {"index": index, "reason": "path does not exist", "path": path}
                    )
                    continue
                if edit["op"] == "remove":
                    del target[key]
                elif "value" in edit:
                    target[key] = deepcopy(edit["value"])
                else:
                    skipped.append(
                        {"index": index, "reason": "replacement has no value"}
                    )
            if skipped:
                write(
                    Path(config["output"])
                    / "jury_edit_skips"
                    / (str(time.time_ns()) + ".json"),
                    {"total_edits": len(edits), "skipped": skipped},
                )
            return result

        jury.apply_edits = tolerant_apply_edits
        jury._recovery_edit_patch = True
    # Pure representation normalization for the most common Reduce contract errors:
    # criteria subrequirements carried as objects, unknown/duplicate part IDs, and
    # coverage rows whose status/criterion_ids are inconsistent. Values, meanings
    # and quality judgments are preserved; only the published shape is corrected.
    if not getattr(jury, "_recovery_normalize_patch", False):
        original_normalize = jury.normalize_reduced
        original_reconcile = jury.reconcile_coverage

        def clean_definition(result):
            if not isinstance(result, dict):
                return result
            result = deepcopy(result)
            rows = result.get("criteria")
            for row in rows if isinstance(rows, list) else []:
                if not isinstance(row, dict) or "subrequirements" not in row:
                    continue
                parts = row.get("subrequirements")
                cleaned = []
                if isinstance(parts, list):
                    for part in parts:
                        if isinstance(part, str) and part.strip():
                            text = part.strip()
                        elif isinstance(part, dict):
                            text = next(
                                (
                                    part[k]
                                    for k in (
                                        "text",
                                        "standard",
                                        "requirement",
                                        "clause",
                                    )
                                    if isinstance(part.get(k), str) and part[k].strip()
                                ),
                                None,
                            )
                        else:
                            text = None
                        if text and text not in cleaned:
                            cleaned.append(text)
                if cleaned:
                    row["subrequirements"] = cleaned
                else:
                    row.pop("subrequirements", None)
            return result

        def clean_coverage(output, expectations):
            if not isinstance(output, dict):
                return output
            output = deepcopy(output)
            valid = {
                c["id"]
                for c in output.get("criteria", [])
                if isinstance(c, dict) and isinstance(c.get("id"), str) and c.get("id")
            }
            known = {
                e.get("expectation_id")
                for e in (expectations or [])
                if isinstance(e, dict)
            }

            def clean_ids(ids):
                if not isinstance(ids, list):
                    return []
                return list(
                    dict.fromkeys(
                        i
                        for i in ids
                        if isinstance(i, str) and (not valid or i in valid)
                    )
                )

            coverage, seen = [], set()
            for row in (
                output.get("coverage", [])
                if isinstance(output.get("coverage"), list)
                else []
            ):
                if not isinstance(row, dict):
                    continue
                key = row.get("expectation_id")
                if known and key not in known:
                    continue
                if key in seen:
                    continue
                seen.add(key)
                ids = clean_ids(row.get("criterion_ids"))
                status = row.get("status")
                if status not in {"mapped", "deferred", "excluded"}:
                    status = "deferred"
                if status == "mapped" and not ids:
                    status = "deferred"
                row["status"] = status
                row["criterion_ids"] = ids if status == "mapped" else []
                if not isinstance(row.get("reason"), str) or not row["reason"].strip():
                    row["reason"] = "Normalized: no supported mapping was published."
                expected = next(
                    (
                        e
                        for e in (expectations or [])
                        if isinstance(e, dict) and e.get("expectation_id") == key
                    ),
                    None,
                )
                parts = row.get("subrequirements")
                if not isinstance(expected, dict) or not isinstance(parts, list):
                    coverage.append(row)
                    continue
                allowed = [
                    p["part_id"]
                    for p in expected.get("subrequirements", [])
                    if isinstance(p, dict) and p.get("part_id")
                ]
                part_seen, cleaned = set(), []
                for part in parts:
                    if not isinstance(part, dict):
                        continue
                    part = deepcopy(part)
                    pid = part.get("part_id")
                    if pid not in allowed or pid in part_seen:
                        continue
                    part_seen.add(pid)
                    pids = clean_ids(part.get("criterion_ids"))
                    pstatus = part.get("status")
                    if pstatus not in {"mapped", "deferred", "excluded"}:
                        pstatus = "deferred"
                    if pstatus == "mapped" and (
                        not pids
                        or not isinstance(part.get("retained_text"), str)
                        or not part["retained_text"].strip()
                    ):
                        pstatus = "deferred"
                    part["status"] = pstatus
                    part["criterion_ids"] = pids if pstatus == "mapped" else []
                    if (
                        not isinstance(part.get("reason"), str)
                        or not part["reason"].strip()
                    ):
                        part["reason"] = (
                            "Normalized: no explicit support was published."
                        )
                    cleaned.append(part)
                row["subrequirements"] = cleaned
                coverage.append(row)
            output["coverage"] = coverage
            return output

        def patched_normalize(result, context, evidence):
            result = clean_definition(original_normalize(result, context, evidence))
            return normalize_existing_citations(result, evidence)

        def patched_reconcile(expectations, output):
            return original_reconcile(
                expectations, clean_coverage(output, expectations)
            )

        jury.normalize_reduced = patched_normalize
        jury.reconcile_coverage = patched_reconcile
        jury._recovery_normalize_patch = True
    if hasattr(jury, "diagnostics"):
        old_diagnostics = jury.diagnostics
        jury.diagnostics = lambda draft, expectations: (
            publication_issues(clean_definition(draft), expectations)
            + old_diagnostics(clean_definition(draft), expectations)
        )

    async def propose(context, evidence, members, reducer, **kwargs):
        source = saved_phase(reducer)
        if source and (source / "jury.json").exists():
            if read(source / "context.json") != context:
                raise ValueError(
                    "Retained Jury input differs; refusing to mix construction"
                )
            reduction = source / "jury_reduce_checkpoint.json"
            if reduction.exists() and read(reduction)["evidence"] != evidence:
                raise ValueError("Retained Jury evidence differs")
            value = read(source / "jury.json")
            jury.validate(value, evidence, context)
            checkpoint = source / "checkpoint.json"
            if checkpoint.exists() and kwargs.get("expectation_cache") is not None:
                kwargs["expectation_cache"].update(
                    read(checkpoint)["expectation_cache"]
                )
            kwargs.get("record", lambda *_: None)("jury", value)
            return deepcopy(value)
        return await old_jury(context, evidence, members, reducer, **kwargs)

    async def build(context, evidence, definition, registry, designer, **kwargs):
        source = saved_phase(designer)
        if source and (source / "pack.json").exists():
            if (
                read(source / "learning_context.json") != context
                or read(source / "jury.json") != definition
            ):
                raise ValueError("Retained Builder input differs")
            value = read(source / "pack.json")
            kwargs.get("record", lambda *_: None)("pack", value)
            return deepcopy(value)
        return await old_builder(
            context, evidence, definition, registry, designer, **kwargs
        )

    async def hypothesis(context, evidence, definition, members, reducer, **kwargs):
        source = saved_phase(reducer)
        if source and (source / "hypothesis.json").exists():
            if (
                read(source / "learning_context.json") != context
                or read(source / "jury.json") != definition
            ):
                raise ValueError("Retained Hypothesis input differs")
            value = read(source / "hypothesis.json")
            kwargs.get("record", lambda *_: None)("hypothesis", value)
            return deepcopy(value)
        return await old_hypothesis(
            context, evidence, definition, members, reducer, **kwargs
        )

    jury.propose, builder.build, hypotheses.propose = propose, build, hypothesis
    old_observe = Memory.observe_jury

    def observe(self, task, definition, *, source):
        old_observe(self, task, definition, source=source)
        phase = f"tasks/{task['task_id']}/{Path(source).parent}"
        original = config.get("construction_sources", {}).get(phase)
        original = (
            Path(original)
            if original
            else (RECOVERY_PHASES.get() or {}).get(Path(source).parent.name)
        )
        if original and (Path(original) / "memory.json").exists():
            original_rows = read(Path(original) / "memory.json")
            by_key = {
                r["key"]: r for r in original_rows if r["value"].get("source") == source
            }
            for row in self.rows:
                if row["key"] in by_key and row["value"].get("source") == source:
                    expected = by_key[row["key"]]
                    if {k: v for k, v in row.items() if k != "memory_id"} != {
                        k: v for k, v in expected.items() if k != "memory_id"
                    }:
                        raise ValueError("Retained Jury memory content differs")
                    row["memory_id"] = expected["memory_id"]
        elif original and (Path(original) / "learning_context.json").exists():
            # An incomplete phase has no final memory dump. Its retrieved rows
            # still bind the exact context already consumed by Builder/Hypothesis.
            by_key = {
                r["key"]: r
                for r in read(Path(original) / "learning_context.json")["memories"]
            }
            for row in self.rows:
                if row["key"] in by_key and row["value"].get("source") == source:
                    expected = by_key[row["key"]]
                    if row["value"] != expected["value"]:
                        raise ValueError("Retained partial Jury memory content differs")
                    row["memory_id"] = expected["memory_id"]

    Memory.observe_jury = observe

    def frozen_report(output):
        result = audit(config)
        write(Path(output) / "acceptance.json", result)
        return result

    report.report = frozen_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["check", "run", "audit"])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = read(args.output / "config.json")
    code = Path(config["code"]).resolve()
    if Path(__file__).resolve().parents[2] != code:
        os.chdir(code)
        os.execve(
            sys.executable,
            [
                sys.executable,
                "-B",
                "-m",
                "experiments.workbuddy.archive_runtime",
                args.action,
                "--output",
                config["output"],
            ],
            {**os.environ, "PYTHONPATH": str(code)},
        )
    install(config)
    if args.action == "run":
        from experiments.workbuddy.experiment import campaign

        asyncio.run(campaign(config))
    result = audit(config, require_complete=args.action == "audit")
    write(args.output / "acceptance.json", result)
    print({k: v for k, v in result.items() if k not in {"scores", "tasks"}}, flush=True)


if __name__ == "__main__":
    main()
