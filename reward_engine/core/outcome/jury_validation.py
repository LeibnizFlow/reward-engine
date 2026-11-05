"""Jury validation for the existing public entry point."""

from copy import deepcopy

from reward_engine.core.outcome.context import (
    ResponseContractError,
    require_refs,
)
from reward_engine.core.outcome.jury_drafts import (
    resolve_parts,
)


def clause_schema(enabled):
    from reward_engine.core.outcome.jury import SCHEMA

    schema = deepcopy(SCHEMA)
    if not enabled:
        return schema, ""
    from reward_engine.core.outcome.clauses import INSTRUCTION
    from reward_engine.core.outcome.clauses import SCHEMA as CLAUSE_SCHEMA

    schema["criteria"][0].pop("subrequirements", None)
    schema["criteria"][0]["clauses"] = [deepcopy(CLAUSE_SCHEMA)]
    return schema, INSTRUCTION


class ReduceValidationError(ValueError):
    """A model reply failed publication; successful Maps remain reusable."""

    def __init__(self, error, raw):
        super().__init__(str(error))
        self.raw = deepcopy(raw)


def product_ref(ref, product, evidence):
    value = evidence.get(ref)
    return ref.startswith(product + ":") or (
        ref.startswith("check:")
        and isinstance(value, dict)
        and value.get("product_id") == product
    )


def validate(result, evidence, context=None, *, clause_contract=False):
    """Without context only non-user criteria can be checked; scoped claims need it."""
    if not isinstance(result, dict):
        raise ValueError("Outcome definition must be an object")
    rows = result.get("criteria")
    if (
        not isinstance(rows, list)
        or not rows
        or len(rows) > 20
        or any(not isinstance(r, dict) for r in rows)
    ):
        raise ResponseContractError(
            "criteria",
            "Need a small nonempty Outcome definition in the exact top-level criteria field: "
            "a list of 1..20 criterion objects; preserve the supplied draft content when fixing its wrapper",
        )
    from reward_engine.core.outcome.clauses import normalize_criteria

    result = {
        **deepcopy(result),
        "criteria": normalize_criteria(
            rows, context, evidence, required=clause_contract
        ),
    }
    rows = result["criteria"]
    # Surface wire-shape and reference errors in the same bounded redo. Draft
    # input clauses have IDs, but this output contract requires plain strings.
    # Never infer provenance or silently discard fields to repair a response.
    structure_issues = []
    field_repairs = []
    metadata_issue_count = 0
    for index, row in enumerate(rows):
        invalid_metadata = False
        for field, allowed in (
            ("kind", ("hard", "soft")),
            ("origin", ("task", "user", "domain", "observed_gap")),
        ):
            if row.get(field) not in allowed:
                invalid_metadata = True
                metadata_issue_count += 1
                structure_issues.append(
                    f"criteria[{index}].{field}: actual={row.get(field)!r}; "
                    f"allowed: {', '.join(allowed)}. "
                    + (
                        "kind is obligation strength, NOT priority; important belongs in priority."
                        if field == "kind"
                        else "origin identifies actual Task/user support or a soft domain proposal; "
                        "shared_instruction is not Task evidence. Do not invent provenance."
                    )
                )
        if row.get("kind") == "hard" and row.get("origin") in {
            "domain",
            "observed_gap",
        }:
            invalid_metadata = True
            metadata_issue_count += 1
            structure_issues.append(
                f"criteria[{index}]: Domain preferences cannot become hard Task requirements; "
                "domain/observed_gap origin requires kind=soft. Keep actual provenance; do not relabel it task."
            )
        if invalid_metadata:
            field_repairs.append({"id": row.get("id"), "fields": ["kind", "origin"]})
        if "subrequirements" not in row:
            continue
        parts = row["subrequirements"]
        field = f"criteria[{index}].subrequirements"
        if not isinstance(parts, list):
            structure_issues.append(
                f"{field}: actual={parts!r}; required: list of nonempty strings"
            )
            continue
        for part_index, part in enumerate(parts):
            if not isinstance(part, str) or not part.strip():
                structure_issues.append(
                    f"{field}[{part_index}]: actual={part!r}; required: nonempty string, "
                    "not an object (do not copy input part_id/text records)"
                )
    gaps = result.get("gaps", [])
    for index, row in enumerate(gaps if isinstance(gaps, list) else []):
        if isinstance(row, dict) and row.get("expected_basis") not in (
            "task",
            "user",
            "domain",
        ):
            structure_issues.append(
                f"gaps[{index}].expected_basis: actual={row.get('expected_basis')!r}; "
                "allowed: exactly one of 'task', 'user', 'domain'; "
                "put explanations in expected/verification, not this enum"
            )

    def check_reference_fields(value, path=""):
        if isinstance(value, dict):
            for key, item in value.items():
                field = f"{path}.{key}" if path else key
                if key not in {"evidence_refs", "expected_refs"}:
                    check_reference_fields(item, field)
                elif isinstance(item, list):
                    for index, ref in enumerate(item):
                        if not isinstance(ref, str) or ref not in evidence:
                            structure_issues.append(
                                f"{field}[{index}]: actual={ref!r}; required: an exact "
                                "supplied evidence handle; do not invent or change its namespace"
                            )
                elif item or key == "evidence_refs":
                    structure_issues.append(
                        f"{field}: actual={item!r}; required: a list of exact supplied evidence handles"
                    )
        elif isinstance(value, list):
            for index, item in enumerate(value):
                check_reference_fields(item, f"{path}[{index}]")

    check_reference_fields(result)
    if structure_issues:
        error = ResponseContractError(
            "response", "Response structure errors:\n" + "\n".join(structure_issues)
        )
        if (
            field_repairs
            and len(structure_issues) == metadata_issue_count
            and all(isinstance(r.get("id"), str) and r["id"] for r in rows)
            and len({r["id"] for r in rows}) == len(rows)
        ):
            error.details["criterion_field_repairs"] = field_repairs
        raise error
    ids = set()
    scoped_users = {
        p["product_id"]: {r["ref"] for r in p["user_updates"]}
        for p in (context or {}).get("products", [])
    }
    user_refs = set().union(*scoped_users.values()) if scoped_users else set()
    for row in rows:
        if "priority" in row and (
            not isinstance(row["priority"], str)
            or row["priority"] not in {"critical", "important", "minor"}
        ):
            raise ValueError(
                "Criterion priority must be critical, important or minor; it is not a gate"
            )
        if "priority_reason" in row and (
            not isinstance(row["priority_reason"], str)
            or not row["priority_reason"].strip()
        ):
            raise ValueError(
                "Explain the criterion priority in terms of the requested outcome"
            )
        for key in ("id", "standard", "measurement_question"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(
                    f"Incomplete Outcome criterion: {key} must be nonempty text"
                )
        if row["id"] in ids:
            raise ValueError(f"Duplicate criterion id: {row['id']!r}")
        for field, allowed in (
            ("kind", ("hard", "soft")),
            ("origin", ("task", "user", "domain", "observed_gap")),
        ):
            if row.get(field) not in allowed:
                raise ValueError(
                    f"Criterion {row['id']!r}.{field}={row.get(field)!r}; allowed: {', '.join(allowed)}"
                )
        ids.add(row["id"])
        if "subrequirements" in row and (
            not isinstance(row["subrequirements"], list)
            or any(
                not isinstance(p, str) or not p.strip() for p in row["subrequirements"]
            )
        ):
            raise ValueError("Published subrequirements must be explicit text")
        require_refs(row.get("evidence_refs"), evidence)
        if row["kind"] == "hard" and row["origin"] not in {"task", "user"}:
            raise ValueError("Domain preferences cannot become hard Task requirements")
        if (
            row["kind"] == "hard"
            and row["origin"] == "task"
            and "task" not in row["evidence_refs"]
        ):
            raise ValueError("A hard Task requirement must cite the Task")
        if row["origin"] == "user" and context is None:
            raise ValueError("User criterion validation requires scoped context")
        if row["origin"] == "user" and not set(row["evidence_refs"]) & user_refs:
            raise ValueError("User criterion must cite an actual user statement")
    gap_ids = set()
    products = {p["product_id"] for p in (context or {}).get("products", [])}
    gaps = result.get("gaps", [])
    if not isinstance(gaps, list) or any(not isinstance(r, dict) for r in gaps):
        raise ValueError("Gaps must be a list of objects")
    if not products:
        # Pre-product questions are useful plans, not observed defects. Preserve
        # their exact content outside the gap list; never invent a product ID.
        pending = [
            row
            for row in gaps
            if row.get("product_ids") == [] and row.get("status") == "needs_check"
        ]
        if pending:
            result.setdefault("unobserved_gap_hypotheses", []).extend(deepcopy(pending))
            gaps = [row for row in gaps if row not in pending]
            result["gaps"] = gaps
    for row in gaps:
        if any(
            not isinstance(row.get(k), str) or not row[k].strip()
            for k in ("gap_id", "expected", "observed", "gap", "verification")
        ):
            raise ValueError(
                "Gap needs expectation, observation, discrepancy and verification"
            )
        if row["gap_id"] in gap_ids:
            raise ValueError("Duplicate gap id")
        gap_ids.add(row["gap_id"])
        targets = row.get("product_ids")
        if not isinstance(targets, list) or not targets or not set(targets) <= products:
            raise ValueError("Gap must identify observed products")
        refs = require_refs(row.get("evidence_refs"), evidence)
        if row.get("expected_basis") not in {"task", "user", "domain"}:
            raise ValueError("Expectation needs provenance")
        expected_refs = row.get("expected_refs", [])
        if expected_refs:
            require_refs(expected_refs, evidence)
        if row["expected_basis"] == "task" and "task" not in expected_refs:
            raise ValueError("Task expectation must refer to the Task")
        if row["expected_basis"] == "user" and not set(expected_refs) & user_refs:
            raise ValueError(
                "User expectation requires an actual user statement, not Actor narration"
            )
        for product in targets:
            if row["expected_basis"] == "user" and not set(
                expected_refs
            ) & scoped_users.get(product, set()):
                raise ValueError(
                    "Do not transfer one product's user requirements to another product"
                )
            if not any(product_ref(r, product, evidence) for r in refs):
                raise ValueError("Gap must cite evidence for every affected product")
        if row.get("status") not in {"needs_check", "check_supported"}:
            raise ValueError("Gap is a suspicion unless a real check supports it")
        if row["status"] == "check_supported":
            for product in targets:
                if not any(
                    r.startswith("check:")
                    and evidence[r].get("product_id") == product
                    and evidence[r].get("status") == "fail"
                    and not evidence[r].get("error")
                    for r in refs
                ):
                    raise ValueError("Votes cannot replace a recorded failed check")
    return result


def validate_proposal(result, evidence, context, *, canonical=False):
    """A draft cannot manufacture an executed check; keep unsupported gaps provisional."""
    if canonical:
        return validate(result, evidence, context)
    result = deepcopy(result)
    changes = []
    for gap in result.get("gaps", []):
        if gap.get("status") != "check_supported":
            continue
        products = gap.get("product_ids", [])
        refs = gap.get("evidence_refs", [])
        supported = bool(products) and all(
            any(
                isinstance(ref, str)
                and ref.startswith("check:")
                and isinstance(evidence.get(ref), dict)
                and evidence[ref].get("product_id") == product
                and evidence[ref].get("status") == "fail"
                and not evidence[ref].get("error")
                for ref in refs
            )
            for product in products
        )
        if not supported:
            changes.append(
                {
                    "gap_id": gap.get("gap_id"),
                    "claimed_status": "check_supported",
                    "admitted_status": "needs_check",
                    "reason": "No supplied failed-check receipt for every target",
                }
            )
            gap["status"] = "needs_check"
    if changes:
        result["gap_authority_corrections"] = changes
    return validate(result, evidence, context)


def validate_reduced_standards(
    result, evidence, context, *, clause_contract=False, record=lambda *_: None
):
    """Keep invalid proposed criteria and observations outside adopted standards.

    Never repair provenance by guessing a reference. A rejected criterion makes
    its covered requirement deferred; a rejected observation stays diagnostic.
    """
    raw_criteria = result.get("criteria", [])
    if not isinstance(raw_criteria, list):
        return validate(result, evidence, context, clause_contract=clause_contract)
    output = None
    rejected_criteria = []
    for raw in raw_criteria:
        try:
            if not isinstance(raw, dict):
                raise ValueError("Criterion must be an object")
            candidate = validate(
                {
                    **(result if output is None else output),
                    "criteria": [raw] if output is None else [*output["criteria"], raw],
                    "gaps": [],
                },
                evidence,
                context,
                clause_contract=clause_contract,
            )
        except ValueError as error:
            rejected_criteria.append(
                {
                    "id": raw.get("id") if isinstance(raw, dict) else None,
                    "reason": str(error),
                    "raw": deepcopy(raw),
                }
            )
        else:
            output = candidate
    if output is None:
        if not rejected_criteria or any(
            row["reason"] != "User criterion must cite an actual user statement"
            for row in rejected_criteria
        ):
            return validate(result, evidence, context, clause_contract=clause_contract)
        # No provenance-valid draft survived. Adopt only this explicit Task
        # baseline, while retaining every rejected draft as diagnostic evidence.
        output = validate(
            {
                **result,
                "criteria": [
                    {
                        "id": "task_outcome_requirement",
                        "standard": "The delivery satisfies the requested task outcome.",
                        "kind": "hard",
                        "origin": "task",
                        "evidence_refs": ["task"],
                        "measurement_question": "Compare the delivery against the task request.",
                    }
                ],
                "gaps": [],
            },
            evidence,
            context,
            clause_contract=clause_contract,
        )
    if rejected_criteria:
        rejected_ids = {
            row["id"] for row in rejected_criteria if isinstance(row["id"], str)
        }

        def references_rejected(value):
            return isinstance(value, str) and any(
                value == rejected or rejected.startswith(value + "_")
                for rejected in rejected_ids
            )

        for coverage in output.get("coverage", []):
            if not isinstance(coverage, dict) or not any(
                references_rejected(value)
                for value in coverage.get("criterion_ids", [])
            ):
                continue
            coverage.update(
                status="deferred",
                criterion_ids=[],
                reason="Deferred because a mapped criterion failed provenance validation.",
            )
            for part in coverage.get("subrequirements", []):
                if isinstance(part, dict):
                    part.update(
                        status="deferred",
                        criterion_ids=[],
                        reason="Deferred because a mapped criterion failed provenance validation.",
                    )
                    part.pop("retained_text", None)
        record(
            "jury_criterion_validation",
            {
                "rejected": rejected_criteria,
                "boundary": "Unvalidated criteria are retained for diagnosis only; not adopted requirements.",
            },
        )
        output["criterion_validation_issues"] = [
            {key: row[key] for key in ("id", "reason")} for row in rejected_criteria
        ]
    raw_gaps = result.get("gaps", [])
    rejected = []
    if not isinstance(raw_gaps, list):
        rejected.append(
            {
                "gap_id": None,
                "reason": "Gaps must be a list of objects",
                "raw": deepcopy(raw_gaps),
            }
        )
        raw_gaps = []
    for raw in raw_gaps:
        try:
            if not isinstance(raw, dict):
                raise ValueError("Gap must be an object")
            candidate = validate(
                {**output, "gaps": [*output["gaps"], raw]},
                evidence,
                context,
                clause_contract=clause_contract,
            )
        except ValueError as error:
            rejected.append(
                {
                    "gap_id": raw.get("gap_id") if isinstance(raw, dict) else None,
                    "reason": str(error),
                    "raw": deepcopy(raw),
                }
            )
        else:
            output = candidate
    if rejected:
        record(
            "jury_gap_validation",
            {
                "rejected": rejected,
                "boundary": "Unvalidated observations retained for diagnosis only; not adopted gaps or requirements.",
            },
        )
        output["gap_validation_issues"] = [
            {key: row[key] for key in ("gap_id", "reason")} for row in rejected
        ]
    return output


def reconcile_coverage(expectations, output):
    """Missing Reduce rows become visible unknowns, never silent deletion or pass."""
    valid = {c["id"] for c in output["criteria"]}
    criteria = {c["id"]: c for c in output["criteria"]}
    supplied = output.get("coverage", [])
    keys = {e["expectation_id"] for e in expectations}
    if not isinstance(supplied, list) or any(not isinstance(r, dict) for r in supplied):
        raise ValueError("Coverage must be a list")

    def canonical_id(value):
        if not isinstance(value, str) or value in valid:
            return value
        matches = [
            criterion_id
            for criterion_id in valid
            if criterion_id.startswith(value + "_")
        ]
        return matches[0] if len(matches) == 1 else value

    normalized = []
    for raw in supplied:
        row = deepcopy(raw)
        if isinstance(row.get("criterion_ids"), list):
            row["criterion_ids"] = [
                canonical_id(value) for value in row["criterion_ids"]
            ]
        if isinstance(row.get("subrequirements"), list):
            for part in row["subrequirements"]:
                if isinstance(part, dict) and isinstance(
                    part.get("criterion_ids"), list
                ):
                    part["criterion_ids"] = [
                        canonical_id(value) for value in part["criterion_ids"]
                    ]
        normalized.append(row)
    supplied = normalized
    if len({r.get("expectation_id") for r in supplied}) != len(supplied):
        raise ValueError("Duplicate expectation coverage")
    if any(r.get("expectation_id") not in keys for r in supplied):
        raise ValueError("Unknown expectation in coverage")
    lookup = {r["expectation_id"]: r for r in supplied}
    rows = []
    for expected in expectations:
        key = expected["expectation_id"]
        row = deepcopy(
            lookup.get(
                key,
                {
                    "expectation_id": key,
                    "criterion_ids": [],
                    "status": "deferred",
                    "reason": "Reduce omitted this independent expectation.",
                },
            )
        )
        ids = row.get("criterion_ids")
        if (
            not isinstance(ids, list)
            or not set(ids) <= valid
            or not row.get("reason")
            or row.get("status") not in {"mapped", "deferred", "excluded"}
            or (row["status"] == "mapped") != bool(ids)
        ):
            raise ValueError("Invalid outcome coverage mapping")
        if (
            row["status"] == "excluded"
            and expected.get("kind") == "hard"
            and not expected.get("draft_validation_issues")
        ):
            row.update(
                status="deferred",
                reason="Hard expectation requires resolution: " + row["reason"],
            )
        if expected.get("subrequirements"):
            row["subrequirements"] = resolve_parts(expected, row, criteria)
            mapped_ids = {i for p in row["subrequirements"] for i in p["criterion_ids"]}
            if (
                any(p["status"] == "deferred" for p in row["subrequirements"])
                or (row["status"] == "mapped" and mapped_ids != set(ids))
                or (row["status"] == "excluded" and mapped_ids)
            ):
                row.update(
                    status="deferred",
                    criterion_ids=[],
                    reason="Important subrequirements unresolved or inconsistent with parent mapping; see per-part decisions.",
                )
        rows.append(row)
    return rows
