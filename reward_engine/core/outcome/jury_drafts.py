"""Keep substantive drafts and their details; only Reduce publishes standards."""

import json
from copy import deepcopy

from reward_engine.core.outcome.context import ResponseContractError, require_refs


def selection_inputs(inputs):
    """Short selectors locate original clauses; never ask the model to retype them."""
    view = deepcopy(inputs)
    index = 0
    for group in ("independent_expectations", "proposal_expectations"):
        view[group] = [deepcopy(row) for row in inputs[group]]
    for row in view["independent_expectations"] + view["proposal_expectations"]:
        row["subrequirements"] = [
            deepcopy(part) for part in row.get("subrequirements", [])
        ]
        for part in row["subrequirements"]:
            part["selection_id"] = f"p{index}"
            index += 1
    return view


def compact_selection_inputs(inputs):
    """Coalesce only byte-identical clauses with identical parent semantics.

    No casefold, whitespace deletion or embedding: code/formula case and scope
    matter. All original part IDs remain addressable in the expansion ledger.
    """
    view = selection_inputs(inputs)
    view.pop(
        "_retired_clauses", None
    )  # Historical text is an admission ledger, not a live proposal.
    if inputs.get("revision_protocol"):
        from reward_engine.core.outcome.jury_revision import attach_aliases

        view = attach_aliases(view)
    seen, aliases = {}, view.pop("selection_aliases", {})
    for row in view["independent_expectations"] + view["proposal_expectations"]:
        scope = {
            k: v
            for k, v in row.items()
            if k not in {"id", "expectation_id", "proposal_phase", "subrequirements"}
        }
        scope["source_clauses"] = [
            part["text"] for part in row.get("subrequirements", [])
        ]
        kept = []
        for part in row.get("subrequirements", []):
            key = json.dumps([scope, part["text"]], sort_keys=True, ensure_ascii=False)
            selector = part["selection_id"]
            if key in seen:
                aliases.setdefault(seen[key], []).append(selector)
            else:
                seen[key] = selector
                kept.append(part)
        row["subrequirements"] = kept
    if aliases:
        # Flatten chains formed when a revised part also duplicates another seat.
        def children(key):
            return [
                child
                for direct in aliases.get(key, [])
                for child in [direct, *children(direct)]
            ]

        nested = {a for values in aliases.values() for a in values}
        aliases = {
            key: list(dict.fromkeys(children(key)))
            for key in aliases
            if key not in nested
        }
        view["selection_aliases"] = aliases
    return view


def apply_part_decisions(raw, inputs, evidence, *, mark_omissions=False):
    """Expand grouped retain/rewrite decisions into the existing coverage contract.

    Code preserves selected wording, not its truth. Rewrites remain explicit
    model judgments with reasons/evidence; omitted clauses stay deferred.
    """
    if "part_decisions" not in raw:
        return raw  # Historical saved replies use the original coverage format.
    result = deepcopy(raw)
    if "coverage" in result:
        raise ValueError("Use part_decisions or historical coverage, not both")
    source = selection_inputs(inputs)
    expected = source["independent_expectations"] + source["proposal_expectations"]
    catalog = {
        p["selection_id"]: (row, p)
        for row in expected
        for p in row.get("subrequirements", [])
    }
    # Only replies to the compact protocol opt in; historical saved decisions
    # retain their original per-part meaning.
    aliases = (
        compact_selection_inputs(inputs).get("selection_aliases", {})
        if raw.get("selection_protocol") == "exact-alias-v1"
        else {}
    )
    criteria = {c["id"]: c for c in result["criteria"]}
    decisions = result.pop("part_decisions")
    if not isinstance(decisions, list):
        raise ValueError("Part decisions must be a list")
    chosen, binding_errors = {}, []
    for decision in decisions:
        parts, action = decision.get("parts"), decision.get("action")
        if isinstance(parts, list) and all(isinstance(p, str) for p in parts):
            parts = [expanded for p in parts for expanded in [p, *aliases.get(p, [])]]
        if (
            not isinstance(parts, list)
            or not parts
            or any(not isinstance(p, str) or p not in catalog for p in parts)
            or len(set(parts)) != len(parts)
            or set(parts) & chosen.keys()
            or action not in {"retain", "rewrite", "defer", "exclude"}
        ):
            raise ValueError("Unknown, duplicate or invalid part decision")
        target = decision.get("criterion_id")
        reason = decision.get("reason") or (
            "Original proposed clause retained unchanged" if action == "retain" else ""
        )
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Rewriting or withholding a clause needs a reason")
        if action in {"retain", "rewrite"} and target not in criteria:
            raise ValueError("Retained clause needs a real criterion")
        if action == "rewrite":
            if (
                not isinstance(decision.get("text"), str)
                or not decision["text"].strip()
            ):
                raise ValueError("Rewrite needs replacement text")
            require_refs(decision.get("evidence_refs"), evidence)
            criteria[target].setdefault("part_revisions", []).append(deepcopy(decision))
        for selector in parts:
            _, part = catalog[selector]
            row = {
                "part_id": part["part_id"],
                "status": "mapped"
                if action in {"retain", "rewrite"}
                else {"defer": "deferred", "exclude": "excluded"}[action],
                "criterion_ids": [target] if action in {"retain", "rewrite"} else [],
                "reason": reason,
            }
            if action in {"retain", "rewrite"}:
                text = part["text"] if action == "retain" else decision["text"]
                if "clauses" in criteria[target]:
                    matches = [
                        c
                        for c in criteria[target]["clauses"]
                        if c.get("id") == decision.get("clause_id")
                    ]
                    if len(matches) != 1 or matches[0].get("text") != text:
                        detail = {
                            "selection_id": selector,
                            "criterion_id": target,
                            "clause_id": decision.get("clause_id"),
                            "action": action,
                            "expected_text": text[:1000],
                            "actual_text": str(matches[0].get("text", ""))[:1000]
                            if len(matches) == 1
                            else None,
                        }
                        binding_errors.append(detail)
                        chosen[selector] = (
                            row  # Still detect duplicate decisions; no invalid result is returned.
                        )
                        continue
                    lineage = matches[0].setdefault("source_part_ids", [])
                    if part["part_id"] not in lineage:
                        lineage.append(part["part_id"])
                    row["clause_id"] = matches[0]["id"]
                clauses = criteria[target].setdefault("subrequirements", [])
                if text not in clauses:
                    clauses.append(text)
                row["retained_text"] = text
            chosen[selector] = row
    if binding_errors:
        diagnostics = {
            "count": len(binding_errors),
            "selection_ids": [e["selection_id"] for e in binding_errors],
            "examples": binding_errors[:8],
            "examples_limit": 8,
        }
        raise ValueError(
            "Clause decision must bind the exact final clause ID and text: "
            + json.dumps(diagnostics, ensure_ascii=False)
            + ". Fix all listed bindings together. retain preserves each original part verbatim; "
            "differently worded parts cannot share one retained clause. Use separate exact clauses, "
            "or an explicit rewrite with matching final text, reason and evidence. "
            "Do not weaken the Task requirement."
        )
    for criterion in criteria.values():
        if "clauses" in criterion:
            criterion["subrequirements"] = [c["text"] for c in criterion["clauses"]]
    coverage = []
    for parent in expected:
        rows = [
            chosen.get(
                p["selection_id"],
                {
                    "part_id": p["part_id"],
                    "status": "deferred",
                    "criterion_ids": [],
                    "reason": "No Reduce decision for this clause",
                    **({"code_deferred": True} if mark_omissions else {}),
                },
            )
            for p in parent.get("subrequirements", [])
        ]
        ids = list(dict.fromkeys(i for row in rows for i in row["criterion_ids"]))
        status = (
            "deferred"
            if not rows or any(r["status"] == "deferred" for r in rows)
            else "mapped"
            if ids
            else "excluded"
        )
        coverage.append(
            {
                "expectation_id": parent["expectation_id"],
                "status": status,
                "criterion_ids": ids if status == "mapped" else [],
                "reason": "Derived from recorded clause decisions",
                "subrequirements": rows,
            }
        )
    result["coverage"] = coverage
    return result


def normalize_part_bindings(raw, inputs, evidence):
    """Canonicalize explicit, reasoned merge claims; never certify equivalence.

    Structured final clauses own their wording. A model sometimes labels a
    reasoned paraphrase/merge `retain`; represent it as its actual proposed
    rewrite while preserving the original texts and label in the audit ledger.
    Missing IDs, reasons or supporting handles still require model correction.
    """
    value = deepcopy(raw)
    source = selection_inputs(inputs)
    parts = {
        p["selection_id"]: p["text"]
        for row in source["independent_expectations"] + source["proposal_expectations"]
        for p in row.get("subrequirements", [])
    }
    criteria = {c["id"]: c for c in value.get("criteria", [])}
    for decision in value.get("part_decisions", []):
        if decision.get("action") != "retain" or not isinstance(
            decision.get("parts"), list
        ):
            continue
        criterion = criteria.get(decision.get("criterion_id"), {})
        matches = [
            c
            for c in criterion.get("clauses", [])
            if c.get("id") == decision.get("clause_id")
        ]
        if len(matches) != 1 or any(p not in parts for p in decision["parts"]):
            continue  # Strict binding validation reports the malformed identity.
        clause = matches[0]
        if all(parts[p] == clause.get("text") for p in decision["parts"]):
            continue
        if (
            not isinstance(decision.get("reason"), str)
            or not decision["reason"].strip()
        ):
            continue  # An unexplained wording change is not an explicit merge.
        require_refs(clause.get("evidence_refs"), evidence)
        decision["binding_resolution"] = {
            "provided_action": "retain",
            "resolved_action": "rewrite",
            "original_texts": {p: parts[p] for p in decision["parts"]},
            "semantic_equivalence_verified": False,
            "reason": "Model supplied an explicit final clause ID and merge rationale; final clause is canonical",
        }
        decision.update(
            action="rewrite",
            text=clause["text"],
            evidence_refs=deepcopy(clause["evidence_refs"]),
        )
    _project_rewritten_summaries(value, parts, evidence)
    return value


def _project_rewritten_summaries(value, parts, evidence):
    """Refresh a stale parent projection from explicitly selected final clauses.

    This is exact lineage resolution, not a semantic similarity exemption. An
    excluded clause, an unrelated summary, or stale normative clause is never
    silently removed. Original prose remains in the returned audit metadata.
    """
    for criterion in value.get("criteria", []):
        clauses = criterion.get("clauses", [])
        if not clauses or not all(
            isinstance(c, dict) and isinstance(c.get("text"), str) and c["text"].strip()
            for c in clauses
        ):
            continue
        old = criterion.get("standard")
        if (
            not isinstance(old, str)
            or not old.strip()
            or old in [c["text"] for c in clauses]
        ):
            continue  # An explicit stale clause must still pass the retirement guard.
        for decision in value.get("part_decisions", []):
            if (
                decision.get("action") != "rewrite"
                or decision.get("criterion_id") != criterion.get("id")
                or not isinstance(decision.get("parts"), list)
                or old
                not in [parts.get(p) for p in decision["parts"] if isinstance(p, str)]
                or not isinstance(decision.get("reason"), str)
                or not decision["reason"].strip()
            ):
                continue
            matches = [c for c in clauses if c.get("id") == decision.get("clause_id")]
            if len(matches) != 1 or matches[0]["text"] != decision.get("text"):
                continue
            require_refs(decision.get("evidence_refs"), evidence)
            criterion["standard"] = "\n".join(c["text"] for c in clauses)
            criterion.setdefault("summary_projections", []).append(
                {
                    "original_standard": old,
                    "source_clause_ids": [c.get("id") for c in clauses],
                    "reason": "Parent wording was explicitly rewritten into this criterion; project all final clauses",
                    "semantic_equivalence_verified": False,
                }
            )
            break


def normalize_reduced(result, context, evidence):
    """Resolve the current Task alias; never let model reasoning impersonate a check."""
    result = deepcopy(result)
    notes = []
    alias = "task:" + context["task"]["task_id"]

    def refs(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"evidence_refs", "expected_refs"} and isinstance(item, list):
                    if alias in item and alias not in evidence and "task" in evidence:
                        value[key] = ["task" if r == alias else r for r in item]
                        notes.append({"source_ref": alias, "resolved_ref": "task"})
                else:
                    refs(item)
        elif isinstance(value, list):
            for item in value:
                refs(item)

    refs(result)
    for gap in (
        result.get("gaps", []) if isinstance(result.get("gaps", []), list) else []
    ):
        if not isinstance(gap, dict) or gap.get("status") != "check_supported":
            continue
        cited = gap.get("evidence_refs", [])
        products = gap.get("product_ids", [])
        if not isinstance(cited, list) or not isinstance(products, list):
            continue  # Structural errors still go through normal validation.
        supported = products and all(
            any(
                isinstance(ref, str)
                and ref.startswith("check:")
                and isinstance(evidence.get(ref), dict)
                and evidence[ref].get("product_id") == product
                and evidence[ref].get("status") == "fail"
                and not evidence[ref].get("error")
                for ref in cited
            )
            for product in products
        )
        if not supported:
            gap["status"] = "needs_check"
            notes.append(
                {
                    "gap_id": gap.get("gap_id"),
                    "original_status": "check_supported",
                    "reason": "No cited failed execution for every affected product; model inference only.",
                }
            )
    if notes:
        result["admission_notes"] = notes
    return result


def reduction_inputs(proposals):
    """Send each annotated criterion once; member metadata refers to its IDs.

    No summarization or truncation: standards, subrequirements, provenance and
    field diagnostics remain in the two canonical expectation lists.
    """
    initial, observed, members = [], [], []
    for proposal in proposals:
        before = proposal["independent_expectations"]
        after = proposal["proposal_expectations"]
        initial.extend(deepcopy(before))
        observed.extend(deepcopy(after))
        member = {
            k: deepcopy(v)
            for k, v in proposal.items()
            if k
            not in {
                "criteria",
                "initial_draft",
                "independent_expectations",
                "proposal_expectations",
            }
        }
        member["criterion_refs"] = [r["expectation_id"] for r in after]
        member["initial_draft"] = {
            k: deepcopy(v)
            for k, v in proposal["initial_draft"].items()
            if k != "criteria"
        }
        member["initial_draft"]["criterion_refs"] = [
            r["expectation_id"] for r in before
        ]
        members.append(member)
    result = {
        "proposals": members,
        "independent_expectations": initial,
        "proposal_expectations": observed,
    }
    if any(p.get("revision_protocol") for p in proposals):
        from reward_engine.core.outcome.jury_revision import VERSION

        result["revision_protocol"] = VERSION
    return result


def keep_draft(raw, evidence, context, validator):
    rows = raw.get("criteria") if isinstance(raw, dict) else None
    if not isinstance(rows, list) or not 1 <= len(rows) <= 20:
        raise ResponseContractError(
            "criteria",
            "No substantive criterion proposal: return the exact top-level criteria field "
            "as a list of 1..20 criterion objects, preserving the proposed requirements. "
            "An assessment, score or differently named wrapper is not an Outcome definition.",
        )
    rows = deepcopy(rows)
    for row in rows:
        if not isinstance(row, dict):
            continue
        clauses = row.get("clauses")
        if (
            not row.get("standard")
            and isinstance(clauses, list)
            and clauses
            and all(
                isinstance(c, dict)
                and isinstance(c.get("text"), str)
                and c["text"].strip()
                for c in clauses
            )
        ):
            # Clauses already contain the proposed definition. This is a
            # draft-only text projection, not a new model claim or authority.
            row["standard"] = "\n".join(c["text"] for c in clauses)
            row["draft_field_projections"] = ["standard_from_existing_clause_texts"]
    usable = [
        r
        for r in rows
        if isinstance(r, dict)
        and isinstance(r.get("standard"), str)
        and r["standard"].strip()
    ]
    if not usable:
        raise ResponseContractError(
            "criteria",
            "No substantive criterion proposal: each usable criterion needs nonempty standard "
            "text or explicit clauses with nonempty text. Correct the supplied draft; do not "
            "invent a requirement just to fill the schema.",
        )
    draft = deepcopy(raw)
    draft["criteria"] = deepcopy(usable)
    if len(usable) != len(rows):
        draft["unusable_criteria"] = [deepcopy(r) for r in rows if r not in usable]
    for row in draft["criteria"]:
        try:
            validator({"criteria": [row]}, evidence, context)
        except (ValueError, TypeError, KeyError) as error:
            row["draft_validation_issues"] = [str(error)]
        # Preserve individual proposed clauses for Reduce even when provenance
        # validation failed before it could derive the legacy projection.
        clauses = row.get("clauses")
        if (
            isinstance(clauses, list)
            and clauses
            and all(
                isinstance(c, dict)
                and isinstance(c.get("text"), str)
                and c["text"].strip()
                for c in clauses
            )
        ):
            row["subrequirements"] = [c["text"] for c in clauses]
    try:
        validator(draft, evidence, context)
    except (ValueError, TypeError, KeyError) as error:
        draft["draft_validation_issues"] = [str(error)]
    if any(r.get("draft_field_projections") for r in draft["criteria"]):
        draft.setdefault("draft_validation_issues", []).append(
            "A missing parent standard was projected from existing clauses; Reduce must publish a complete definition"
        )
    return draft


def expectations(rows, prefix, phase):
    """Record atomic details from both phases; legacy prose is retained as one item.

    IDs locate raw proposals, not true facts. A malformed optional detail list
    falls back to the original prose, rather than deleting the entire member.
    """
    result = []
    for index, row in enumerate(rows):
        key = f"{prefix}:{index}"
        parts = row.get("subrequirements")
        valid = (
            isinstance(parts, list)
            and bool(parts)
            and all(isinstance(p, str) and p.strip() for p in parts)
        )
        if not valid:
            parts = [row["standard"]]
        result.append(
            {
                **deepcopy(row),
                "expectation_id": key,
                "proposal_phase": phase,
                "subrequirements": [
                    {"part_id": f"{key}/{i}", "text": p} for i, p in enumerate(parts)
                ],
            }
        )
    return result


def resolve_parts(expected, coverage, criteria):
    """Every detail gets a disposition. A broad parent mapping is insufficient.

    retained_text must locate an explicit clause in a final criterion. Code checks
    traceability, not semantic equivalence; the recorded Reduce owns that judgment.
    """
    parts = expected.get("subrequirements", [])
    raw = coverage.get("subrequirements", [])
    if not isinstance(raw, list) or any(not isinstance(r, dict) for r in raw):
        raise ValueError("Subrequirement decisions must be a list")
    keys = {p["part_id"] for p in parts}
    if len({r.get("part_id") for r in raw}) != len(raw) or any(
        r.get("part_id") not in keys for r in raw
    ):
        raise ValueError("Unknown or duplicate subrequirement decision")
    by_id = {r["part_id"]: r for r in raw}
    resolved = []
    for part in parts:
        row = deepcopy(
            by_id.get(
                part["part_id"],
                {
                    "part_id": part["part_id"],
                    "status": "deferred",
                    "criterion_ids": [],
                    "reason": "Reduce did not explicitly address this subrequirement.",
                },
            )
        )
        ids = row.get("criterion_ids", [])
        if (
            row.get("status") not in {"mapped", "deferred", "excluded"}
            or not isinstance(row.get("reason"), str)
            or not row["reason"].strip()
            or not isinstance(ids, list)
            or any(not isinstance(i, str) or i not in criteria for i in ids)
            or (row["status"] == "mapped") != bool(ids)
        ):
            raise ValueError("Invalid subrequirement disposition")
        if row["status"] == "mapped":
            text = row.get("retained_text")
            if (
                not isinstance(text, str)
                or not text.strip()
                or not any(
                    text == criteria[i].get("standard")
                    or text == criteria[i].get("measurement_question")
                    or text in criteria[i].get("subrequirements", [])
                    for i in ids
                )
            ):
                row.update(
                    status="deferred",
                    criterion_ids=[],
                    reason="Mapped detail has no explicit retained clause in the final criterion.",
                )
        row["proposed_text"] = part["text"]
        resolved.append(row)
    return resolved
