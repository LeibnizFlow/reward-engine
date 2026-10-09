"""Small, audited protocol edits; every reconstructed definition is revalidated."""

from collections import Counter
from copy import deepcopy


def diagnostics(draft, expectations):
    issues = []
    criteria = draft.get("criteria", [])
    criterion_ids = [row["id"] for row in criteria
                     if isinstance(row, dict) and isinstance(row.get("id"), str) and row["id"].strip()]
    identities_complete = (bool(criteria) and len(criterion_ids) == len(criteria)
                           and len(set(criterion_ids)) == len(criterion_ids))
    for i, criterion in enumerate(draft.get("criteria", [])):
        if not isinstance(criterion, dict):
            continue
        invalid_fields = [key for key in ("id", "standard", "measurement_question")
                          if not isinstance(criterion.get(key), str) or not criterion[key].strip()]
        if invalid_fields:
            issues.append({"path": ["criteria", i], "invalid_required_fields": invalid_fields,
                "required_text_fields": ["id", "standard", "measurement_question"],
                "reason": "Required criterion fields must be nonempty text; the identity field is id, "
                          "not criterion_id. To add or rename a missing field, replace this existing "
                          "parent criterion object, preserving all grounded requirement text and other "
                          "fields. Correct malformed subrequirements in that same parent replacement. "
                          "Keep existing coverage references pending revalidation; an incomplete parsed "
                          "ID set does not mean the draft has no supported criteria."})
        if criterion.get("evidence_refs") == []:
            issues.append({"path": ["criteria", i, "evidence_refs"],
                "reason": "A published criterion requires nonempty available evidence handles. "
                          "Cite its actual Task/user/source grounding without inventing authority; "
                          "an unsupported criterion must remain unresolved. This empty criterion "
                          "field is distinct from a gap's product-specific references."})
        if "subrequirements" not in criterion:
            continue
        parts = criterion["subrequirements"]
        if not isinstance(parts, list) or any(not isinstance(p, str) or not p.strip() for p in parts):
            issues.append({"path": ["criteria", i, "subrequirements"],
                "reason": "Published criterion subrequirements must be a list of nonempty strings, "
                          "not objects. Preserve the full requirement text. Replace this list in one edit. "
                          "This rule applies to criteria, NOT coverage: coverage decisions remain objects "
                          "with part_id, criterion_ids and retained_text."})
    for i, gap in enumerate(draft.get("gaps", [])):
        if isinstance(gap, dict) and (not isinstance(gap.get("expected_basis"), str)
                or gap["expected_basis"] not in {"task", "user", "domain"}):
            issues.append({"path": ["gaps", i, "expected_basis"], "value": gap.get("expected_basis"),
                "allowed_values": ["task", "user", "domain"],
                "reason": "Expectation provenance is its actual task/user/domain basis, not observed_gap. "
                          "A failed check is an observation, not authority for a new task requirement."})
    expected = {row["expectation_id"]: row for row in expectations}
    coverage = draft.get("coverage", [])
    if not isinstance(coverage, list):
        return issues + [{"path": ["coverage"], "reason": "Coverage must be a list"}]
    seen = set()
    for i, row in enumerate(coverage):
        if not isinstance(row, dict):
            issues.append({"path": ["coverage", i], "reason": "Coverage row must be an object"})
            continue
        key = row.get("expectation_id")
        if not isinstance(key, str) or key not in expected or key in seen:
            issues.append({"path": ["coverage", i], "reason": "Unknown or duplicate expectation ID",
                           "value": key, "allowed_ids": list(expected)})
            continue
        seen.add(key)
        parts = row.get("subrequirements", [])
        allowed = [p["part_id"] for p in expected[key].get("subrequirements", [])]
        if not isinstance(parts, list) or any(not isinstance(p, dict) for p in parts):
            issues.append({"path": ["coverage", i, "subrequirements"],
                           "reason": "Decisions must be a list of objects", "allowed_ids": allowed})
            continue
        invalid_targets = [j for j, part in enumerate(parts)
                           if isinstance(part.get("criterion_ids"), list)
                           and any(not isinstance(cid, str) or cid not in criterion_ids
                                   for cid in part["criterion_ids"])]
        if identities_complete and invalid_targets:
            issues.append({"path": ["coverage", i, "subrequirements"],
                "invalid_target_indices": invalid_targets, "allowed_criterion_ids": criterion_ids,
                "reason": "Decision criterion_ids must refer to final criterion IDs, not part IDs. "
                          "Repair the existing decisions together in one list replacement, preserving "
                          "part_id, status, reason and retained_text. Do not invent supporting mappings."})
        counts = Counter(p.get("part_id") for p in parts if isinstance(p.get("part_id"), str))
        for j, part in enumerate(parts):
            key = part.get("part_id")
            if not isinstance(key, str) or key not in allowed or counts[key] > 1:
                issues.append({"path": ["coverage", i, "subrequirements", j], "value": key,
                    "reason": "Unknown or duplicate subrequirement ID; use only supplied IDs. "
                              "Removing an unsupported decision keeps the actual requirement visible as deferred.",
                    "allowed_ids": allowed})
                continue
            status, targets = part.get("status"), part.get("criterion_ids", [])
            fields = []
            if not isinstance(status, str) or status not in {"mapped", "deferred", "excluded"}:
                fields.append("status must be mapped, deferred, or excluded")
            if not isinstance(part.get("reason"), str) or not part["reason"].strip():
                fields.append("reason must be nonempty explanatory text")
            if not isinstance(targets, list):
                fields.append("criterion_ids must be a list")
            elif (status == "mapped") != bool(targets):
                fields.append("mapped requires criterion_ids; deferred/excluded require an empty list")
            if fields:
                issues.append({"path": ["coverage", i, "subrequirements", j],
                    "invalid_fields": fields, "allowed_statuses": ["mapped", "deferred", "excluded"],
                    "reason": "Correct this part's disposition against the existing evidence and final criteria. "
                              "Unresolved coverage remains deferred; do not invent a supported mapping."})
    return issues


def apply_edits(draft, reply):
    """Apply exact existing paths atomically; never guess a replacement ID or source."""
    edits = reply.get("edits")
    if not isinstance(edits, list) or not 1 <= len(edits) <= 64:
        raise ValueError("Repair needs one to 64 explicit edits")
    result = deepcopy(draft)
    for edit in edits:
        if not isinstance(edit, dict) or edit.get("op") not in {"replace", "remove"}:
            raise ValueError("Repair operation must be replace or remove")
        path = edit.get("path")
        if (not isinstance(path, list) or not path or path[0] not in {"criteria", "gaps", "coverage", "uncertainties"}
                or any(type(k) not in {str, int} for k in path)
                or not isinstance(edit.get("reason"), str) or not edit["reason"].strip()):
            raise ValueError("Repair needs an existing public definition path and reason")
        target = result
        for key in path[:-1]:
            target = child(target, key)
        key = path[-1]
        child(target, key)  # Reject nonexistent keys, negative indices and bool indices.
        if edit["op"] == "remove":
            del target[key]
        else:
            if "value" not in edit:
                raise ValueError("Replacement needs an explicit value")
            target[key] = deepcopy(edit["value"])
    return result


def child(target, key):
    if isinstance(target, dict) and isinstance(key, str) and key in target:
        return target[key]
    if isinstance(target, list) and type(key) is int and 0 <= key < len(target):
        return target[key]
    raise ValueError("Repair path must already exist")
