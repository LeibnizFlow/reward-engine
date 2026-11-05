"""Bounded independent audits of one evolving Jury definition, not Actor reruns.

Original proposals remain an audit ledger. Only the current clauses and unresolved
clauses enter the next selection pool; coverage follows explicit rewrite edges.
"""

from copy import deepcopy

from reward_engine.core.batch import BatchPolicy, collect_batch
from reward_engine.core.outcome.context import ResponseContractError, ask, require_refs
from reward_engine.core.outcome.jury_drafts import (
    expectations,
    selection_inputs,
    selector_conflicts,
)
from reward_engine.core.outcome.jury_prompts import (
    CONSTRUCTION_PROMPT_VERSION,
    OUTCOME_JURY_AUDIT_PROMPT,
)


def definition_view(result):
    audit_fields = {
        "part_revisions",
        "source_expectation_ids",
        "draft_validation_issues",
    }
    return {
        "criteria": [
            {k: deepcopy(v) for k, v in c.items() if k not in audit_fields}
            for c in result["criteria"]
        ],
        "gaps": deepcopy(result.get("gaps", [])),
        "uncertainties": deepcopy(result.get("uncertainties", [])),
    }


def check_retired_clauses(raw, inputs):
    """Reject silent literal resurrection; this does not prove semantic correctness."""
    selected = selection_inputs(inputs)
    parts = {
        p["selection_id"]: p["text"]
        for e in selected["independent_expectations"]
        + selected["proposal_expectations"]
        for p in e.get("subrequirements", [])
    }
    decisions = raw.get("part_decisions")
    if not isinstance(decisions, list):
        raise ValueError(
            "Return a part_decisions list; omitted selectors remain deferred"
        )
    conflicts = selector_conflicts(decisions, parts)
    if conflicts:
        failure = ResponseContractError("part_decisions", "Unknown or duplicate active clause selector")
        failure.details.update(selector_conflicts=conflicts, allowed_selection_ids=list(parts),
                               instruction="Resolve every listed selector collision or unknown reference "
                               "at its original draft path. Do not invent IDs or silently drop requirements.")
        raise failure
    seen, retained, withdrawn, reconsidered = set(), set(), set(), set()
    for decision in decisions:
        if not isinstance(decision, dict) or not isinstance(
            decision.get("parts"), list
        ):
            raise ValueError("Each clause decision must identify its source parts")
        for key in decision["parts"]:
            if not isinstance(key, str) or key not in parts or key in seen:
                raise ValueError("Unknown or duplicate active clause selector")
            seen.add(key)
            old = parts[key]
            if decision.get("action") == "retain":
                retained.add(old)
            elif decision.get("action") == "rewrite":
                text = decision.get("text")
                if isinstance(text, str):
                    retained.add(text)
                    if text != old:
                        withdrawn.add(old)
                    if decision.get("evidence_refs"):
                        reconsidered.add(text)
            elif decision.get("action") in {"exclude", "defer"}:
                withdrawn.add(old)
    final = {
        text
        for c in raw.get("criteria", [])
        if isinstance(c, dict)
        for text in [
            c.get("standard"),
            *c.get("subrequirements", []),
            *(
                part.get("text")
                for part in c.get("clauses", [])
                if isinstance(part, dict)
            ),
        ]
        if isinstance(text, str)
    }
    conflicts = final & (withdrawn - retained)
    if conflicts:
        failure = ResponseContractError("part_decisions", "A withdrawn clause remains in the published criteria")
        rows = []
        for text in sorted(conflicts):
            paths = []
            for i, criterion in enumerate(raw.get("criteria", [])):
                if not isinstance(criterion, dict):
                    continue
                if criterion.get("standard") == text:
                    paths.append(["criteria", i, "standard"])
                paths.extend(["criteria", i, "subrequirements", j]
                             for j, value in enumerate(criterion.get("subrequirements", [])) if value == text)
                paths.extend(["criteria", i, "clauses", j, "text"]
                             for j, value in enumerate(criterion.get("clauses", []))
                             if isinstance(value, dict) and value.get("text") == text)
            rows.append({"text": text, "criterion_paths": paths,
                         "decision_indices": [i for i, d in enumerate(decisions)
                                              if d.get("action") != "retain" and any(parts[p] == text for p in d["parts"])]})
        indices = sorted({i for row in rows for i in row["decision_indices"]})
        locations = []
        for row in rows:
            for path in row["criterion_paths"]:
                criterion = raw["criteria"][path[1]]
                location = {"path": path, "criterion_id": criterion.get("id"), "text": row["text"]}
                if path[2] == "clauses":
                    location["clause_id"] = criterion["clauses"][path[3]].get("id")
                locations.append(location)
        failure.details.update(
            conflicts=rows,
            criterion_locations=locations,
            conflicting_decisions=[{
                "decision_index": i, "decision": deepcopy(decisions[i]),
                "source_texts": {p: parts[p] for p in decisions[i]["parts"]},
            } for i in indices],
            instruction="Resolve the contradiction between these published clause locations and "
            "their decisions. Indices are zero-based. Match both the path and stable criterion/clause "
            "IDs in the repair base before editing. conflicting_decisions shows the validated "
            "normalization: binding_resolution records any supplied retain that explicitly merged "
            "different texts into one named clause. That merge withdraws other original texts; "
            "the unnormalized repair base still has the supplied action. If distinct clauses are "
            "intended to remain, author separate consistent bindings; otherwise author consistent "
            "criteria and decisions against the Task. These diagnostics do not choose which "
            "requirements to retain or remove.",
        )
        raise failure
    if (final | retained) & (set(inputs.get("_retired_clauses", [])) - reconsidered):
        raise ValueError(
            "A superseded historical clause cannot return without an explicit evidenced rewrite"
        )


def enrich_coverage(expected, coverage, number):
    original = {
        p["part_id"]: p["text"] for e in expected for p in e.get("subrequirements", [])
    }
    rows = deepcopy(coverage)
    for row in rows:
        for part in row.get("subrequirements", []):
            part["current_text"] = (
                part.get("retained_text")
                or part.get("current_text")
                or original[part["part_id"]]
            )
            part.setdefault("lineage", []).append(
                {
                    "round": number,
                    "status": part["status"],
                    "current_text": part["current_text"],
                    "criterion_ids": deepcopy(part["criterion_ids"]),
                    **({"code_deferred": True} if part.get("code_deferred") else {}),
                }
            )
        row["current_standard"] = "\n".join(
            p["current_text"]
            for p in row.get("subrequirements", [])
            if p["status"] != "excluded"
        )
        row["current_why"] = (
            "Current clauses tracked through explicit construction decisions; not established defects."
        )
        row["current_verification"] = (
            "Verify current clauses against the original Task, scoped user requirements and actual evidence."
        )
    return rows


def pending_expectations(expected, coverage):
    by_id = {r["expectation_id"]: r for r in coverage}
    result = []
    for row in expected:
        state = by_id[row["expectation_id"]]
        pending = [
            p for p in state.get("subrequirements", []) if p["status"] == "deferred"
        ]
        if state["status"] == "deferred" and not pending:
            # A hard clause excluded by a model remains an unresolved obligation.
            pending = [
                p for p in state.get("subrequirements", []) if p["status"] == "excluded"
            ]
        if not pending:
            continue
        # Preserve open-ended semantic scope, but never replay superseded
        # wording or audit explanations as live unresolved requirements.
        historical_fields = {
            "standard",
            "subrequirements",
            "measurement_question",
            "why",
            "verification",
            "reason",
            "rationale",
            "part_revisions",
            "source_expectation_ids",
            "draft_validation_issues",
        }
        current = {k: deepcopy(v) for k, v in row.items() if k not in historical_fields}
        current.update(
            proposal_phase="unresolved_previous_clause",
            standard="\n".join(p["current_text"] for p in pending),
            why=state.get("current_why")
            or "Unresolved current clauses; not established defects.",
            measurement_question=state.get("current_verification")
            or "Resolve these current clauses against the Task and actual evidence.",
            subrequirements=[
                {
                    "part_id": p["part_id"],
                    "text": p["current_text"],
                    **({"code_deferred": True} if p.get("code_deferred") else {}),
                }
                for p in pending
            ],
        )
        result.append(current)
    return result


def project_coverage(expected, previous, active, reduced, number):
    """Follow each original part through current selectors, never compare paraphrases."""
    from reward_engine.core.outcome.jury import reconcile_coverage

    active_parts = {
        (e["id"], p["text"]): p["part_id"]
        for e in active
        for p in e.get("subrequirements", [])
    }
    decisions = {
        p["part_id"]: p
        for r in reduced["coverage"]
        for p in r.get("subrequirements", [])
    }
    rows = deepcopy(previous)
    for row in rows:
        for part in row.get("subrequirements", []):
            if part["status"] == "mapped":
                targets = [
                    active_parts[(c, part["current_text"])]
                    for c in part["criterion_ids"]
                    if (c, part["current_text"]) in active_parts
                ]
                if len(set(targets)) != 1:
                    raise ValueError(
                        "Published clause lineage no longer resolves to one current clause"
                    )
                target = targets[0]
            else:
                target = part["part_id"]
                if target not in decisions:
                    continue  # Explicitly excluded history is not a selectable source.
            decision = decisions[target]
            latest = decision.get("retained_text") or part["current_text"]
            part.update(
                status=decision["status"],
                criterion_ids=deepcopy(decision["criterion_ids"]),
                reason=decision["reason"],
                current_text=latest,
            )
            part.pop("code_deferred", None)
            if decision.get("code_deferred"):
                part["code_deferred"] = True
            part.pop("retained_text", None)
            if part["status"] == "mapped":
                part["retained_text"] = latest
            part["lineage"].append(
                {
                    "round": number,
                    "source_part_id": target,
                    "status": part["status"],
                    "current_text": latest,
                    "criterion_ids": deepcopy(part["criterion_ids"]),
                    **({"code_deferred": True} if part.get("code_deferred") else {}),
                }
            )
        parts = row.get("subrequirements", [])
        ids = list(dict.fromkeys(c for p in parts for c in p["criterion_ids"]))
        status = (
            "deferred"
            if any(p["status"] == "deferred" for p in parts)
            else "mapped"
            if ids
            else "excluded"
        )
        row.update(
            status=status,
            criterion_ids=ids if status == "mapped" else [],
            reason="Projected through explicit construction clause decisions",
            current_standard="\n".join(
                p["current_text"] for p in parts if p["status"] != "excluded"
            ),
        )
    return reconcile_coverage(
        expected, {"criteria": reduced["criteria"], "coverage": rows}
    )


def retired_texts(expected, coverage):
    original = {
        p["part_id"]: p["text"] for e in expected for p in e.get("subrequirements", [])
    }
    retired = set()
    for row in coverage:
        for part in row.get("subrequirements", []):
            history = [
                original[part["part_id"]],
                *[h["current_text"] for h in part.get("lineage", [])],
            ]
            retired.update(text for text in history if text != part["current_text"])
            if part["status"] == "excluded":
                retired.add(part["current_text"])
    return sorted(retired)


async def construct(
    context,
    evidence,
    members,
    reducer,
    *,
    rounds,
    policy=None,
    expectation_cache=None,
    reduce_max_bytes=160000,
    record=lambda *_: None,
    revision_protocol=False,
):
    from reward_engine.core.outcome import jury
    from reward_engine.core.outcome.construction import fingerprint

    if len(members) != 3:
        raise ValueError(
            "Jury construction requires exactly three independent member seats"
        )
    policy = policy or BatchPolicy()
    snapshots = []
    identity = {
        "task": context["task"],
        "context_hash": fingerprint(context),
        "evidence_hash": fingerprint(evidence),
        "prompt_version": CONSTRUCTION_PROMPT_VERSION,
    }

    def save_round(number):
        return lambda name, value: record(
            f"construction/jury/round_{number}/{name}", value
        )

    first_record = save_round(1)
    first_record("input", {**identity, "construction_rounds": rounds})
    try:
        current = await jury._propose_once(
            context,
            evidence,
            members,
            reducer,
            policy=policy,
            expectation_cache=expectation_cache,
            reduce_max_bytes=reduce_max_bytes,
            record=first_record,
            strict=True,
            task_only=True,
            protocol_version=CONSTRUCTION_PROMPT_VERSION,
            revision_protocol=revision_protocol,
        )
        independent = deepcopy(current["independent_expectations"])
        observed = deepcopy(current["proposal_expectations"])
        expected = independent + observed
        coverage = enrich_coverage(expected, current["coverage"], 1)
        first_record("output", current)
        snapshots.append({"round": 1, "output_hash": fingerprint(current)})
        for number in range(2, rounds + 1):
            save = save_round(number)
            previous = definition_view(current)
            pending = pending_expectations(expected, coverage)
            active = expectations(
                previous["criteria"], f"jury_round_{number - 1}", "previous_synthesis"
            )
            save(
                "input",
                {
                    **identity,
                    "previous_hash": fingerprint(previous),
                    "pending_hash": fingerprint(pending),
                    "previous_synthesis": previous,
                    "unresolved_expectations": pending,
                },
            )
            criterion_ids = {c["id"] for c in previous["criteria"]}

            def validate_audit(row, criterion_ids=criterion_ids):
                jury.validate(row, evidence, context)
                audits = row.get("criterion_audit")
                if (
                    not isinstance(audits, list)
                    or len(audits) != len(criterion_ids)
                    or any(not isinstance(a, dict) for a in audits)
                    or {a.get("criterion_id") for a in audits} != criterion_ids
                ):
                    raise ValueError("Audit every current criterion exactly once")
                for audit in audits:
                    if audit.get("assessment") not in {
                        "keep",
                        "revise",
                        "uncertain",
                    } or not audit.get("reason"):
                        raise ValueError(
                            "Each criterion audit needs an assessment and reason"
                        )
                    require_refs(audit.get("evidence_refs"), evidence)
                return row

            async def audit(
                key,
                call,
                *,
                number=number,
                previous=previous,
                pending=pending,
                save=save,
                validate_audit=validate_audit,
            ):
                row = await ask(
                    call,
                    "outcome_jury_map",
                    context,
                    evidence,
                    OUTCOME_JURY_AUDIT_PROMPT,
                    {
                        **jury.SCHEMA,
                        "criterion_audit": [
                            {
                                "criterion_id": "each supplied current ID",
                                "assessment": "keep/revise/uncertain",
                                "reason": "specific assessment against the Task",
                                "evidence_refs": ["actual supplied refs"],
                            }
                        ],
                    },
                    extra={
                        "construction_round": number,
                        "previous_synthesis": previous,
                        "unresolved_expectations": pending,
                        "prompt_version": CONSTRUCTION_PROMPT_VERSION,
                    },
                    validate=validate_audit,
                    max_contract_retries=1,
                    max_evidence_rounds=2,
                    close_evidence=True,
                )
                save(f"audits/{key}", row)
                return row

            batch = await collect_batch(
                {
                    key: lambda k=key, c=call, audit=audit: audit(k, c)
                    for key, call in members.items()
                },
                policy,
            )
            save("map", batch.summary())
            if len([r for r in batch.items if r.status == "ok"]) != 3:
                raise RuntimeError(
                    "All three Jury audits must finish; incomplete rounds are not consensus"
                )
            additions, proposals = [], []
            for row in batch.items:
                clauses = expectations(
                    row.value["criteria"],
                    f"jury_r{number}:{row.job_id}",
                    "refinement_audit",
                )
                additions.extend(clauses)
                proposals.append(
                    {k: deepcopy(v) for k, v in row.value.items() if k != "criteria"}
                    | {"criterion_refs": [c["expectation_id"] for c in clauses]}
                )
            inputs = {
                "proposals": proposals,
                "independent_expectations": active + pending,
                "proposal_expectations": additions,
                "construction_round": number,
                "_retired_clauses": retired_texts(expected, coverage),
            }
            if revision_protocol:
                from reward_engine.core.outcome.jury_revision import VERSION

                inputs["revision_protocol"] = VERSION
            reduced = await jury.reduce_proposals(
                context,
                evidence,
                inputs,
                reducer,
                reduce_max_bytes=reduce_max_bytes,
                record=save,
                strict=True,
                protocol_version=CONSTRUCTION_PROMPT_VERSION,
            )
            coverage = project_coverage(expected, coverage, active, reduced, number)
            addition_ids = {e["expectation_id"] for e in additions}
            coverage.extend(
                enrich_coverage(
                    additions,
                    [
                        r
                        for r in reduced["coverage"]
                        if r["expectation_id"] in addition_ids
                    ],
                    number,
                )
            )
            observed.extend(deepcopy(additions))
            expected = independent + observed
            current = {
                **reduced,
                "independent_expectations": independent,
                "proposal_expectations": observed,
                "coverage": coverage,
            }
            save("output", current)
            snapshots.append({"round": number, "output_hash": fingerprint(current)})
    except Exception as error:
        record(
            "construction/jury/status",
            {
                **identity,
                "status": "unavailable",
                "completed_rounds": len(snapshots),
                "rounds": snapshots,
                "error_type": type(error).__name__,
                "reason": str(error),
            },
        )
        raise
    current.update(
        coverage=coverage,
        construction={
            "rounds": rounds,
            "completed_rounds": len(snapshots),
            "status": "completed",
            "snapshots": snapshots,
            "boundary": "Fixed independent audits, not proof of correctness or improvement",
        },
    )
    record("construction/jury/status", current["construction"])
    record("jury", current)
    return current
