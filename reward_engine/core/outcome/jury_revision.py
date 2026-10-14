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
    "Also return revision_decisions connecting original subrequirement part_ids to revised "
    "criterion_id and zero-based part_index (clauses order, or subrequirements order). "
    "Group unchanged parts with action=retain and ordered one-to-one targets. "
    "Retain preserves every parent field, including why and priority_reason, as well as the exact "
    "part text. Do not omit why when re-emitting a criterion; for an unchanged entire criterion "
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


def retain_differences(parts, keys, source, targets):
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
                }
            )
        mismatches.append(
            {
                "source_part_id": part,
                "target": {"criterion_id": key[0], "part_index": key[1]},
                "field_differences": fields,
            }
        )
    return mismatches


def materialize(raw, initial, evidence):
    """Expand inheritance, validate explicit lineage and expose ALL omissions.

    Historical complete-criteria replies can omit the new ledger. They retain
    their drafts, but every original part then remains pending for Reduce.
    """

    def error(message):
        raise ResponseContractError("revision_decisions", message)

    if not isinstance(raw, dict) or not isinstance(raw.get("criteria"), list):
        error("Revised proposal requires criteria")
    result = deepcopy(raw)
    originals = {r["expectation_id"]: r for r in initial}
    criteria = []
    inherited = []
    for item in result["criteria"]:
        if not isinstance(item, dict):
            error("Each revised criterion must be an object")
        if "inherit_from" in item:
            ref = item["inherit_from"]
            if (
                set(item) != {"inherit_from"}
                or not isinstance(ref, str)
                or ref not in originals
            ):
                error(
                    "inherit_from must name one existing expectation without overrides"
                )
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
    for decision_index, d in enumerate(decisions):
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
            error("Unknown, duplicate or invalid revision source/action/targets")
        if (action == "add") != (not parts):
            error("Only additions have empty source_parts")
        if bool(dest) != (action not in {"delete", "defer"}):
            error("Retain/rewrite/merge/add require targets; delete/defer require none")
        keys = []
        for t in dest:
            if (
                not isinstance(t, dict)
                or not isinstance(t.get("criterion_id"), str)
                or type(t.get("part_index")) is not int
            ):
                error("Target requires criterion_id and integer part_index")
            key = (t["criterion_id"], t["part_index"])
            if key not in targets or key in keys:
                error("Unknown or duplicate revised target")
            keys.append(key)
        if action == "retain":
            if len(parts) != len(keys) or any(
                meaning(*source[p]) != meaning(*targets[k])
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
                    retain_mismatches=retain_differences(parts, keys, source, targets),
                    instruction="Compare these exact source/target fields. text denotes the selected "
                    "subrequirement text; clause denotes its structured clause fields. Presence matters: "
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
        used.update(parts)
        addressed_targets.update(keys)
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
