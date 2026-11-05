"""Materialize a member's revised draft; revisions remain proposals for Reduce."""

from copy import deepcopy

from reward_engine.core.outcome.construction import fingerprint
from reward_engine.core.outcome.context import ResponseContractError, require_refs

VERSION = "outcome-member-revision/2026-09-15-v1"
SCHEMA = {
    "source_parts": ["original subrequirement part_id; group unchanged parts"],
    "action": "retain/rewrite/merge/delete/defer/add",
    "targets": [{"criterion_id": "revised criterion ID", "part_index": 0}],
    "reason": "required for changes, deletions, deferrals and additions",
    "evidence_refs": ["existing refs supporting a change; not proof of correctness"],
}
INSTRUCTION = (
    " Produce ONE complete revised proposal based on your independent_expectations. "
    'An unchanged entire criterion may be {"inherit_from":"expectation_id"}; code copies it exactly. '
    "Otherwise provide the complete revised criterion, with a unique id. "
    "When providing clauses, omit the redundant subrequirements field, or copy the clause "
    "text list exactly in the same order; never paraphrase it or copy annotated expectation parts. "
    "Also return revision_decisions connecting original subrequirement part_ids to revised "
    "criterion_id and zero-based part_index (clauses order, or subrequirements order). "
    "Group unchanged parts with action=retain and ordered one-to-one targets. "
    "Retain preserves every parent field, including why and priority_reason, as well as the exact "
    "part text and the presence or absence of each field. Preserve why exactly when present; "
    "if the original has no why, omit it when retaining rather than inventing a new rationale. "
    "Adding or changing why requires rewrite with reason and evidence. For an unchanged entire criterion "
    "prefer inherit_from without any overrides. "
    "Use rewrite or merge for changed text OR changed parent meaning, conditions, alternatives, "
    "origin, priority, evidence or verification; include reason and supporting evidence_refs. "
    "Use delete/defer with empty targets and a reason; deletion also requires evidence. "
    "New parts use add, source_parts=[], targets and a source-grounded reason. "
    "A deletion is only your proposal: Reduce must review its original wording. "
    "Omitted original parts remain unresolved, never implicitly removed or approved. "
    "Do not weaken an obligation because every product omitted it. Do not promote a domain "
    "suggestion to a hard Task requirement. Preserve gaps, uncertainty and scoped evidence. "
)


def criterion_body(row):
    return {
        k: deepcopy(v)
        for k, v in row.items()
        if k not in {"expectation_id", "proposal_phase", "subrequirements"}
    } | {"subrequirements": [p["text"] for p in row["subrequirements"]]}


def meaning(row, index):
    """Exact equality permits reuse; similar prose never proves equivalence."""
    ignored = {
        "id",
        "expectation_id",
        "proposal_phase",
        "subrequirements",
        "clauses",
        "draft_validation_issues",
        "draft_field_projections",
        "clause_protocol",
    }
    result = {k: deepcopy(v) for k, v in row.items() if k not in ignored}
    if row.get("clauses"):
        result["clause"] = {
            k: deepcopy(v) for k, v in row["clauses"][index].items() if k != "id"
        }
    part = row["subrequirements"][index]
    result["text"] = part["text"] if isinstance(part, dict) else part
    return result


def nested_differences(before, after):
    """Exact JSON field differences, with presence distinct from null values."""
    missing = object()
    rows = []

    def visit(left, right, path):
        if left is not missing and right is not missing and left == right:
            return
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(left.keys() | right.keys()):
                visit(left.get(key, missing), right.get(key, missing), [*path, key])
        elif isinstance(left, list) and isinstance(right, list):
            for index in range(max(len(left), len(right))):
                visit(left[index] if index < len(left) else missing,
                      right[index] if index < len(right) else missing, [*path, index])
        else:
            rows.append({"path": path, "source_present": left is not missing,
                         "target_present": right is not missing,
                         **({"source_value": deepcopy(left)} if left is not missing else {}),
                         **({"target_value": deepcopy(right)} if right is not missing else {})})

    visit(before, after, [])
    return rows


def retain_differences(parts, keys, source, targets, evidence=None):
    """Describe rejected equality only; never construct a replacement proposal."""
    mismatches = []
    for part, key in zip(parts, keys, strict=False):
        before, after = meaning(*source[part]), meaning(*targets[key])
        if before == after:
            continue
        fields = []
        for name in sorted(before.keys() | after.keys()):
            if name in before and name in after and before[name] == after[name]:
                continue
            fields.append(
                {
                    "field": name,
                    "source_present": name in before,
                    "target_present": name in after,
                    **(
                        {"source_value": deepcopy(before[name])}
                        if name in before
                        else {}
                    ),
                    **(
                        {"target_value": deepcopy(after[name])} if name in after else {}
                    ),
                    **(
                        {"nested_differences": nested_differences(before[name], after[name])}
                        if name in before and name in after
                        and isinstance(before[name], (dict, list))
                        and isinstance(after[name], (dict, list)) else {}
                    ),
                }
            )
        invalid_quotes = []
        if evidence is not None and isinstance(before.get("clause"), dict):
            from reward_engine.core.outcome.clauses import quote_occurs

            for i, quote in enumerate(before["clause"].get("source_quotes", [])):
                if isinstance(quote, dict) and isinstance(quote.get("ref"), str) and (
                    quote["ref"] not in evidence or not quote_occurs(quote.get("quote"), evidence[quote["ref"]])
                ):
                    invalid_quotes.append({
                        "quote_index": i, "source_ref": quote["ref"],
                        "provided_quote": deepcopy(quote.get("quote")),
                        "instruction": "The original draft quote is not verbatim evidence. Do not restore it. "
                        "Correcting provenance requires an explicit rewrite with reason and supporting "
                        "evidence_refs, even if requirement text is unchanged. Split a grouped retain "
                        "decision where necessary; preserve unaffected parts and never invent a quotation.",
                    })
        mismatches.append(
            {
                "source_part_id": part,
                "target": {"criterion_id": key[0], "part_index": key[1]},
                "field_differences": fields,
                **({"invalid_source_quotes": invalid_quotes} if invalid_quotes else {}),
            }
        )
    return mismatches


def reference_preflight(raw, initial):
    """Find impossible target references even when inheritance cannot expand.

    Include every possible inherited target to avoid guessing which expectation
    an invalid reference meant. These are diagnostics, never a repaired draft or
    proof that a target is present in the final materialized proposal.
    """
    possible, known = set(), {}
    originals = {row["expectation_id"]: row for row in initial}
    inherited_names = {row["id"] for row in initial}
    complete = []
    for item in raw["criteria"]:
        if not isinstance(item, dict):
            continue
        item = deepcopy(item)
        if "inherit_from" in item:
            ref = item["inherit_from"]
            if set(item) != {"inherit_from"} or not isinstance(ref, str) or ref not in originals:
                continue
            item = criterion_body(originals[ref])
        if not isinstance(item.get("id"), str) or not item["id"]:
            continue
        if "clauses" in item:
            clauses = item["clauses"]
            if not isinstance(clauses, list) or not clauses or any(
                not isinstance(c, dict) or not isinstance(c.get("text"), str) or not c["text"].strip()
                for c in clauses
            ):
                continue
            projection = [c["text"] for c in clauses]
            if "subrequirements" in item and item["subrequirements"] != projection:
                continue
            item["subrequirements"] = projection
        parts = item.get("subrequirements") or [item.get("standard", "")]
        if not isinstance(parts, list) or any(not isinstance(p, str) or not p.strip() for p in parts):
            continue
        item["subrequirements"] = parts
        complete.append(item)
    names = [row["id"] for row in complete]
    for row in complete:
        if names.count(row["id"]) == 1:
            known.update({(row["id"], i): (row, i) for i in range(len(row["subrequirements"]))})
    for row in [*initial, *raw["criteria"]]:
        if not isinstance(row, dict) or "inherit_from" in row:
            continue
        name = row.get("id")
        if not isinstance(name, str) or not name:
            continue
        parts = row.get("clauses", row.get("subrequirements"))
        if isinstance(parts, list) and parts:
            possible.update((name, index) for index in range(len(parts)))
        elif isinstance(row.get("standard"), str) and row["standard"].strip():
            possible.add((name, 0))
    invalid, retain_errors = [], []
    source = {p["part_id"]: (row, i) for row in initial
              for i, p in enumerate(row["subrequirements"])}
    decisions = raw.get("revision_decisions", [])
    for decision_index, decision in enumerate(decisions if isinstance(decisions, list) else []):
        if not isinstance(decision, dict) or not isinstance(decision.get("targets"), list):
            continue
        for target_index, target in enumerate(decision["targets"]):
            if (not isinstance(target, dict)
                    or not isinstance(target.get("criterion_id"), str)
                    or type(target.get("part_index")) is not int
                    or target["part_index"] < 0
                    or ((target["criterion_id"], target["part_index"]) not in possible
                        and not (target["criterion_id"] in inherited_names
                                 and target["criterion_id"] not in names))):
                invalid.append({"decision_index": decision_index, "target_index": target_index,
                                "path": ["revision_decisions", decision_index, "targets", target_index],
                                "actual": deepcopy(target)})
        parts = decision.get("source_parts")
        if decision.get("action") != "retain" or not isinstance(parts, list) or any(
            not isinstance(p, str) or p not in source for p in parts
        ):
            continue
        dest = decision["targets"]
        if any(not isinstance(t, dict) or not isinstance(t.get("criterion_id"), str)
               or type(t.get("part_index")) is not int
               or (t["criterion_id"], t["part_index"]) not in known for t in dest):
            continue  # The meaning of unresolved inheritance is not guessed.
        keys = [(t["criterion_id"], t["part_index"]) for t in dest]
        mismatches = retain_differences(parts, keys, source, known)
        if len(parts) != len(keys) or mismatches:
            retain_errors.append({"decision_index": decision_index,
                                  "path": ["revision_decisions", decision_index],
                                  "source_count": len(parts), "target_count": len(keys),
                                  "retain_mismatches": mismatches})
    return {
        "invalid_targets": invalid,
        "retain_errors": retain_errors,
        "possible_targets": [{"criterion_id": name, "part_index": index}
                             for name, index in sorted(possible)],
        "instruction": "This is read-only diagnostic metadata, not part of the repair base. "
        "Never edit reference_preflight, invalid_targets or retain_errors. The path in each item "
        "addresses the actual rejected draft. Correct invalid_targets and retain_errors alongside "
        "invalid inheritance. Invalid target references match "
        "neither a complete proposed criterion nor any possible inherited criterion. "
        "possible_targets includes ALL independent expectations as alternatives, not adopted "
        "criteria; unresolved inherited criteria may be explicitly rewritten with different part "
        "counts. Final targets must exist in your actual corrected proposal. Retain errors compare "
        "only independently resolvable complete criteria: preserve exact parent/part meaning, or "
        "author an explicit rewrite with reason and supplied evidence for actual changes. Use exact "
        "criterion IDs and zero-based part indices. No reference or decision has been rewritten.",
    }


def materialize(raw, initial, evidence, *, canonical=False):
    """Expand inheritance, validate explicit lineage and expose ALL omissions.

    Historical complete-criteria replies can omit the new ledger. They retain
    their drafts, but every original part then remains pending for Reduce.
    """

    def error(message, **details):
        failure = ResponseContractError("revision_decisions", message)
        failure.details.update(details)
        raise failure

    if not isinstance(raw, dict) or not isinstance(raw.get("criteria"), list):
        error("Revised proposal requires criteria")
    result = deepcopy(raw)
    if canonical:
        from reward_engine.core.outcome.source_projection import (
            meaning as compare_meaning,
        )
    else:
        compare_meaning = meaning
    originals = {r["expectation_id"]: r for r in initial}
    invalid_inheritance = [
        {"criterion_index": index, "actual": deepcopy(item)}
        for index, item in enumerate(result["criteria"])
        if isinstance(item, dict) and "inherit_from" in item
        and (set(item) != {"inherit_from"}
             or not isinstance(item["inherit_from"], str)
             or item["inherit_from"] not in originals)
    ]
    if invalid_inheritance:
        error(
            "inherit_from must name one existing expectation without overrides",
            **invalid_inheritance[0],
            invalid_inheritance=invalid_inheritance,
            allowed_inheritance=[{"inherit_from": key} for key in originals],
            reference_preflight=reference_preflight(result, initial),
            instruction="Correct every listed invalid_inheritance location at criteria[criterion_index]. "
            "These errors concern criteria[].inherit_from, not revision_decisions[].targets. "
            "Also address reference_preflight.invalid_targets and retain_errors at their draft paths "
            "in the same response; reference_preflight itself is read-only diagnostic metadata. "
            "Choose an exact independent_expectations[].expectation_id, not its criterion id. "
            "An inherited criterion contains ONLY inherit_from: omit id and every other field. "
            "For changes, author a complete criterion and an explicit rewrite with evidence instead. "
            "A complete criterion is a proposal, not an expectation record: do not copy expectation_id, "
            "proposal_phase or part_id objects. If subrequirements is included, it must be a list of "
            "strings exactly equal to the clause text list, not the expectation's annotated parts. "
            "Target criterion_id always uses the criterion's id; inherit_from alone uses expectation_id. "
            "These alternatives identify legal references; they do not choose a requirement for you.",
        )
    mismatches = []
    for index, item in enumerate(result["criteria"]):
        if not isinstance(item, dict) or "inherit_from" in item:
            continue
        clauses = item.get("clauses")
        if (isinstance(clauses, list) and clauses
                and all(isinstance(c, dict) and isinstance(c.get("text"), str) for c in clauses)
                and "subrequirements" in item
                and item["subrequirements"] != [c["text"] for c in clauses]):
            mismatches.append({
                "path": ["criteria", index, "subrequirements"],
                "clauses_path": ["criteria", index, "clauses"],
                "actual": deepcopy(item["subrequirements"]),
                "clause_texts": [c["text"] for c in clauses],
            })
    if mismatches:
        # Inspect the remaining ledger against the declared clauses so an early
        # redundant-text error does not hide independent retain/provenance errors.
        # This temporary projection is never returned or used as a repaired draft.
        projected = deepcopy(raw)
        for mismatch in mismatches:
            projected["criteria"][mismatch["path"][1]].pop("subrequirements")
        downstream = {}
        try:
            materialize(projected, initial, evidence, canonical=canonical)
        except ResponseContractError as failure:
            downstream = deepcopy(failure.details)
        error("Revised subrequirements must equal clauses text",
              subrequirement_mismatches=mismatches,
              downstream_diagnostics=downstream,
              instruction="Each listed path is an existing redundant text projection in the rejected draft. "
              "If the clauses are your intended requirements, explicitly remove that subrequirements "
              "field or replace it with the exact clause_texts list. Otherwise author the corrected "
              "clauses and matching text projection together, with grounded revision decisions. "
              "Changing only revision_decisions does not correct these criteria fields. "
              "Address every listed mismatch and downstream_diagnostics. The latter only inspects "
              "the ledger against declared clause texts; it does not select clauses over conflicting "
              "subrequirements. No edit or semantic choice has been applied.")
    criteria = []
    inherited = []
    for item in result["criteria"]:
        if not isinstance(item, dict):
            error("Each revised criterion must be an object")
        if "inherit_from" in item:
            ref = item["inherit_from"]
            item = criterion_body(originals[ref])
            inherited.append((ref, item["id"]))
        if "clauses" in item:
            clauses = item["clauses"]
            if (
                not isinstance(clauses, list)
                or not clauses
                or any(
                    not isinstance(c, dict) or not isinstance(c.get("text"), str)
                    for c in clauses
                )
            ):
                error("Revised clauses require complete text")
            projection = [c["text"] for c in clauses]
            if "subrequirements" in item and item["subrequirements"] != projection:
                error("Revised subrequirements must equal clauses text")
            item["subrequirements"] = projection
        if (
            not isinstance(item.get("subrequirements"), list)
            or not item["subrequirements"]
        ):
            item["subrequirements"] = [item.get("standard", "")]
        if any(
            not isinstance(p, str) or not p.strip() for p in item["subrequirements"]
        ):
            error("Revised subrequirements require nonempty strings")
        criteria.append(item)
    ids = [c.get("id") for c in criteria]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        error("Revised criterion IDs must be nonempty and unique")
    source = {
        p["part_id"]: (r, index)
        for r in initial
        for index, p in enumerate(r["subrequirements"])
    }
    targets = {
        (c["id"], i): (c, i) for c in criteria for i in range(len(c["subrequirements"]))
    }
    decisions = result.get("revision_decisions", [])
    if not isinstance(decisions, list):
        error("revision_decisions must be a list")
    decisions = deepcopy(decisions)
    declared = {
        p
        for d in decisions
        if isinstance(d, dict) and isinstance(d.get("source_parts"), list)
        for p in d["source_parts"]
        if isinstance(p, str)
    }
    for ref, target in inherited:
        for index, part in enumerate(originals[ref]["subrequirements"]):
            if part["part_id"] not in declared:
                decisions.append(
                    {
                        "source_parts": [part["part_id"]],
                        "action": "retain",
                        "targets": [{"criterion_id": target, "part_index": index}],
                    }
                )
    used, addressed_targets = set(), set()
    def validate_decision(decision_index, d):
        if not isinstance(d, dict):
            error("Revision decisions must be objects")
        parts, action, dest = d.get("source_parts"), d.get("action"), d.get("targets")
        if (
            not isinstance(parts, list)
            or any(not isinstance(p, str) or p not in source for p in parts)
            or len(set(parts)) != len(parts)
            or used.intersection(parts)
            or action not in {"retain", "rewrite", "merge", "delete", "defer", "add"}
            or not isinstance(dest, list)
        ):
            error("Unknown, duplicate or invalid revision source/action/targets",
                  decision_index=decision_index, actual=deepcopy(d),
                  allowed_source_parts=list(source), already_used_source_parts=sorted(used),
                  allowed_actions=["retain", "rewrite", "merge", "delete", "defer", "add"],
                  instruction="Use exact source part_id handles, not criterion IDs or expectation IDs. "
                  "Each source part may occur once across decisions. Use a list for targets.")
        if (action == "add") != (not parts):
            error("Only additions have empty source_parts")
        if bool(dest) != (action not in {"delete", "defer"}):
            error("Retain/rewrite/merge/add require targets; delete/defer require none")
        keys, invalid_targets = [], []
        for target_index, t in enumerate(dest):
            if (
                not isinstance(t, dict)
                or not isinstance(t.get("criterion_id"), str)
                or type(t.get("part_index")) is not int
            ):
                invalid_targets.append({"target_index": target_index, "actual": deepcopy(t),
                                        "reason": "Target requires criterion_id and integer part_index"})
                continue
            key = (t["criterion_id"], t["part_index"])
            if key not in targets or key in keys:
                invalid_targets.append({"target_index": target_index, "actual": deepcopy(t),
                                        "reason": "Unknown or duplicate revised target"})
            keys.append(key)
        if invalid_targets:
            error(invalid_targets[0]["reason"], decision_index=decision_index,
                  actual=deepcopy(invalid_targets[0]["actual"]), invalid_targets=invalid_targets,
                  allowed_targets=[{"criterion_id": k, "part_index": i} for k, i in targets],
                  already_used_targets=[{"criterion_id": k, "part_index": i} for k, i in keys
                                        if (k, i) in targets],
                  instruction="Correct every listed invalid_targets location. Targets name revised "
                  "criterion IDs and zero-based subrequirement indices. Do not use expectation IDs, "
                  "source part IDs or one-based indices as targets.")
        if action == "retain":
            if len(parts) != len(keys) or any(
                compare_meaning(*source[p]) != compare_meaning(*targets[k])
                for p, k in zip(parts, keys, strict=True)
            ):
                failure = ResponseContractError(
                    "revision_decisions",
                    "Retain must preserve wording AND scope/priority/provenance/verification; use rewrite with evidence",
                )
                failure.details.update(
                    decision_index=decision_index,
                    source_count=len(parts),
                    target_count=len(keys),
                    retain_mismatches=retain_differences(parts, keys, source, targets, evidence),
                    instruction="Compare these exact source/target fields. text denotes the selected "
                    "subrequirement text; clause denotes its structured clause fields. "
                    "nested_differences paths are relative to the named field; for clause, locate the "
                    "target criterion's clauses[part_index]. Literal source quotes must match exactly, "
                    "including small wording differences. These paths are diagnostics, not edit roots. Presence matters: "
                    "a missing field is not an unchanged value. Retain needs ordered one-to-one targets "
                    "and exact original fields, including why. For an unchanged entire criterion, use "
                    "inherit_from alone; otherwise explicitly author the complete preserved fields or "
                    "use rewrite/merge with reason and evidence for actual changes. These diagnostics "
                    "do not apply edits, supply defaults, or authorize weakening a requirement.",
                )
                raise failure
        else:
            if not isinstance(d.get("reason"), str) or not d["reason"].strip():
                error("Changed or withheld clauses require an explicit reason")
            if action != "defer":
                require_refs(d.get("evidence_refs"), evidence)
        return parts, keys

    failures = []
    for decision_index, decision in enumerate(decisions):
        try:
            parts, keys = validate_decision(decision_index, decision)
        except ValueError as failure:
            if not isinstance(failure, ResponseContractError):
                failure = ResponseContractError("revision_decisions", str(failure))
            failure.details.setdefault("decision_index", decision_index)
            failures.append(failure)
            # Diagnose subsequent duplicate declarations too. Nothing from an
            # invalid ledger can be returned or adopted by this function.
            if isinstance(decision, dict) and isinstance(decision.get("source_parts"), list):
                used.update(p for p in decision["source_parts"] if isinstance(p, str) and p in source)
            continue
        used.update(parts)
        addressed_targets.update(keys)
    if failures:
        from reward_engine.core.outcome.clauses import invalid_quote_diagnostics

        first = failures[0]
        quote_errors = invalid_quote_diagnostics(raw["criteria"], evidence)
        if quote_errors:
            first.details["draft_quote_errors"] = quote_errors
        first.details["decision_errors"] = [deepcopy(f.details) for f in failures]
        first.details["instruction"] = (
            "Correct every listed decision_errors entry using model-authored changes; "
            "Also correct any draft_quote_errors at their exact draft paths. "
            "all original validation rules still apply. " + first.details.get("instruction", "")
        )
        raise first
    for part in source:
        if part in used:
            continue
        decisions.append(
            {
                "source_parts": [part],
                "action": "defer",
                "targets": [],
                "reason": "Member omitted a disposition; original requirement remains pending",
                "code_deferred": True,
            }
        )
    # A missing add declaration never licenses the proposed obligation. Its
    # complete text still reaches Reduce with explicit unknown provenance.
    unclaimed = [
        {"criterion_id": k, "part_index": i}
        for k, i in targets
        if (k, i) not in addressed_targets
    ]
    result.update(
        criteria=criteria,
        revision_protocol=VERSION,
        revision_decisions=decisions,
        unclaimed_additions=unclaimed,
        revision_source_sha256=fingerprint(initial),
    )
    return result


def attach_aliases(view):
    """Reuse only member-validated identical source/target parts, preserving IDs."""
    before = {r["expectation_id"]: r for r in view["independent_expectations"]}
    after = {r["expectation_id"]: r for r in view["proposal_expectations"]}
    parts = {
        p["part_id"]: (deepcopy(r), i)
        for r in before.values()
        for i, p in enumerate(r["subrequirements"])
    }
    aliases = view.setdefault("selection_aliases", {})
    for member in view["proposals"]:
        if member.get("revision_protocol") != VERSION:
            continue
        targets = {
            r["id"]: r
            for ref in member["criterion_refs"]
            if (r := after.get(ref)) is not None
        }
        for d in member.get("revision_decisions", []):
            if d["action"] != "retain":
                continue
            for source_id, dest in zip(d["source_parts"], d["targets"], strict=True):
                original = parts.get(source_id)
                revised = targets[dest["criterion_id"]]
                target = revised["subrequirements"][dest["part_index"]]
                old = original[0]["subrequirements"][original[1]] if original else None
                if original and meaning(*original) != meaning(
                    revised, dest["part_index"]
                ):
                    raise ValueError(
                        "Retained revision no longer matches its canonical source"
                    )
                # Existing exact-alias compaction may already have removed a part.
                if not old or "selection_id" not in target:
                    continue
                a, b = target["selection_id"], old["selection_id"]
                if a == b:
                    continue
                aliases.setdefault(a, []).extend([b, *aliases.pop(b, [])])
                for r in before.values():
                    r["subrequirements"] = [
                        p for p in r["subrequirements"] if p["part_id"] != source_id
                    ]
    if not aliases:
        view.pop("selection_aliases", None)
    return view
