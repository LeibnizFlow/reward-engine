"""A single visible selector namespace; expansion remains in local validation."""

from copy import deepcopy


def project(request):
    aliases = request.get("selection_aliases", {})
    if request.get("mode") != "outcome_jury_reduce" or not aliases:
        return request
    owners = {alias: key for key, values in aliases.items() for alias in values}
    catalog = list(dict.fromkeys(
        part["selection_id"]
        for group in ("independent_expectations", "proposal_expectations")
        for row in request.get(group, [])
        for part in row.get("subrequirements", [])
        if isinstance(part.get("selection_id"), str) and part["selection_id"] not in owners
    ))
    result = deepcopy(request)
    result.pop("selection_aliases")
    result["selection_catalog"] = catalog

    def diagnostic_view(value):
        if isinstance(value, list):
            return [diagnostic_view(child) for child in value]
        if not isinstance(value, dict):
            return deepcopy(value)
        return {key: deepcopy(catalog) if key == "allowed_selection_ids" else diagnostic_view(child)
                for key, child in value.items() if key != "selection_aliases"}

    recovery = result.get("response_recovery")
    if isinstance(recovery, dict):
        # Preserve the exact rejected draft and its hash. Only read-only
        # diagnostic option lists change; historical IDs remain visible there
        # as invalid occurrences, never as additional selectable alternatives.
        recovery["details"] = diagnostic_view(recovery.get("details", {}))
        base = recovery.get("rejected_response", {})
        occurrences = {}
        decisions = base.get("part_decisions", []) if isinstance(base, dict) else []
        for i, decision in enumerate(decisions if isinstance(decisions, list) else []):
            parts = decision.get("parts", []) if isinstance(decision, dict) else []
            for j, selector in enumerate(parts if isinstance(parts, list) else []):
                if not isinstance(selector, str):
                    continue
                canonical = owners.get(selector, selector)
                if canonical in catalog:
                    occurrences.setdefault(canonical, []).append({
                        "path": ["part_decisions", i, "parts", j], "provided_id": selector})
        conflicts = [{"selection_id": key, "occurrences": rows}
                     for key, rows in occurrences.items()
                     if len(rows) > 1 or any(row["provided_id"] != key for row in rows)]
        if conflicts:
            result["selection_reference_conflicts"] = conflicts
    result["instruction"] = result.get("instruction", "") + (
        " selection_catalog is the complete, non-overlapping list of selectable IDs. "
        "Use only these IDs in part_decisions.parts, each at most once across the whole draft. "
        "The engine keeps the exact-duplicate expansion ledger locally and preserves all original "
        "source coverage. Historical IDs in the rejected draft or error diagnostics are not extra "
        "choices. selection_reference_conflicts groups the exact draft positions that consume the "
        "same selectable ID. Correct all such positions together using your own edits and reasons; "
        "do not merely replace a historical ID with a canonical ID that is already selected. "
        "Resolve any conflicting dispositions using the Task and evidence. These diagnostics do "
        "not edit, deduplicate, approve a merge, or authorize dropping a requirement. All original "
        "coverage, provenance, clause binding and duplicate checks still apply."
    )
    if isinstance(recovery, dict):
        recovery["instruction"] = result["instruction"]
    return result
