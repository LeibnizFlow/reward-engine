"""Locate construction repair work in the model's original compact draft."""

import re
from collections import defaultdict
from copy import deepcopy

INSTRUCTION = (
    "Repair the supplied rejected draft in this same operation. Preserve unaffected analysis, "
    "requirements, evidence and decisions; change only what is needed to resolve the diagnostics "
    "and their dependent source bindings. Return the COMPLETE compact JSON object. "
    "Count TOP-LEVEL criteria before returning: at most 20, including every inherit_from entry. "
    "Complete the required net reduction in this reply; merely lowering the count is insufficient. "
    "If grouping is necessary, author coherent "
    "requirements with multiple clauses, preserving every meaningful obligation, condition and "
    "alternative. Do not truncate the list or silently discard coverage to meet the limit. "
    "Regrouping inherited requirements requires complete authored criteria. If a parent field "
    "such as standard, kind or priority changes, explicitly rewrite the affected source clauses "
    "with evidence; do not label them retain. Use merge for a clause combining multiple sources. "
    "Each canonical source needs exactly one decision, including sources consumed implicitly by "
    "inherit_from. Bind an existing source to the clause that retains or rewrites it; add is only "
    "for genuinely new content. Excluding a source as duplicate while republishing its text as "
    "an unbound add does not preserve lineage. Resolve those bindings yourself against the Task "
    "and evidence. Do not return internal part_decisions, IDs, patches or field corrections."
    " Check every source_quotes entry: copy its quote verbatim from its cited source and "
    "explicitly include that ref in the SAME clause's evidence_refs. Parent refs do not suffice. "
    "Do not paraphrase a quote or remove a valid obligation just to clear its citation error."
)


def worklist(raw, mode, inputs, details):
    """Describe exact occurrences, without choosing corrections or modifying the draft."""
    if mode not in {"outcome_expectations", "outcome_jury_map", "outcome_jury_reduce"}:
        return None
    if not isinstance(raw, dict) or not isinstance(raw.get("criteria"), list):
        return None
    from reward_engine.core.outcome.authored_json import source_catalog

    projected = isinstance(inputs.get("source_requirements"), dict)
    catalog = inputs.get("source_catalog", {}) if projected else source_catalog(mode, inputs)
    occurrences = defaultdict(list)
    published = defaultdict(list)
    groups = inputs.get("independent_expectations", []) + inputs.get("proposal_expectations", [])
    originals = inputs["source_requirements"] if projected else {
        r["expectation_id"]: r for r in groups}

    def record(row, path):
        parts = row.get("source_parts", [])
        if isinstance(parts, list):
            for part in parts:
                if isinstance(part, str):
                    occurrences[part].append({"path": path, "action": row.get("action")})

    for i, row in enumerate(raw["criteria"]):
        if not isinstance(row, dict):
            continue
        path = ["criteria", i]
        ref = row.get("inherit_from")
        if isinstance(ref, str) and ref in originals:
            bindings = inputs.get("inheritance_bindings", {}).get(ref)
            if bindings is None:
                bindings = [{"selection_id": p.get("selection_id", p.get("part_id"))}
                            for p in originals[ref].get("subrequirements", [])]
                if projected and not bindings:
                    bindings = [{"selection_id": part} for part, source in catalog.items()
                                if source["expectation_id"] == ref]
            for binding in bindings:
                part = binding["selection_id"]
                if part in catalog:
                    occurrences[part].append({"path": path + ["inherit_from"], "action": "retain"})
            for clause in originals[ref].get("clauses", []):
                if isinstance(clause.get("text"), str):
                    published[clause["text"]].append(path + ["inherit_from"])
        clauses = row.get("clauses", [])
        for j, clause in enumerate(clauses if isinstance(clauses, list) else []):
            if isinstance(clause, dict):
                record(clause, path + ["clauses", j])
                if isinstance(clause.get("text"), str):
                    published[clause["text"]].append(path + ["clauses", j, "text"])
    dispositions = raw.get("dispositions", [])
    for i, row in enumerate(dispositions if isinstance(dispositions, list) else []):
        if isinstance(row, dict):
            record(row, ["dispositions", i])

    conflicts, citation_checks, retain_checks = [], [], []

    def field_value(field):
        value = raw
        for key in re.findall(r"[^.\[\]]+", field):
            if isinstance(value, dict):
                value = value.get(key)
            elif isinstance(value, list) and key.isdigit() and int(key) < len(value):
                value = value[int(key)]
            else:
                return None
        return value
    # The legacy validator describes expanded part_decisions. Translate only
    # literal source identities/text back to locations in the submitted draft.
    pending = [details]
    while pending:
        detail = pending.pop()
        if not isinstance(detail, dict):
            continue
        pending.extend(detail.get("validation_errors", []))
        for issue in detail.get("issues", []):
            if issue.get("code") in {"QUOTE_REFERENCE", "QUOTE_EXACT"}:
                row = {key: deepcopy(issue[key]) for key in (
                    "code", "field", "actual", "reason", "quote_field", "inherited_from"
                ) if key in issue}
                row["quoted_ref"] = issue.get("quoted_ref")
                if row["quoted_ref"] is None and issue.get("code") == "QUOTE_EXACT":
                    row["quoted_ref"] = field_value(issue["field"].rsplit(".", 1)[0] + ".ref")
                row["required_action"] = (
                    "Explicitly cite quoted_ref in the same clause's evidence_refs. "
                    "Preserve its grounded obligation and quote; parent refs do not suffice."
                    if issue["code"] == "QUOTE_REFERENCE" else
                    "Read the cited source and author an exact verbatim quotation supporting the "
                    "intended obligation. Do not treat a paraphrase as a quote or invent source text."
                )
                if row not in citation_checks:
                    citation_checks.append(row)
            if issue.get("source_part") and issue.get("changed_fields"):
                row = {key: deepcopy(issue[key]) for key in (
                    "field", "source_part", "changed_fields", "resolution_choices"
                ) if key in issue}
                if row not in retain_checks:
                    retain_checks.append(row)
        for decision in detail.get("conflicting_decisions", []):
            for source, text in decision.get("source_texts", {}).items():
                row = {"source": source, "source_locations": occurrences.get(source, []),
                       "published_locations": published.get(text, [])}
                if row["published_locations"] and row not in conflicts:
                    conflicts.append(row)
    count = len(raw["criteria"])
    inherited = sum(isinstance(row, dict) and isinstance(row.get("inherit_from"), str)
                    for row in raw["criteria"])
    return {
        "criteria_count": len(raw["criteria"]), "criteria_limit": 20,
        "criteria_budget": {
            "counted_top_level_entries": count,
            "inherited_entries_included": inherited,
            "available_slots": max(0, 20 - count),
            "minimum_net_reduction": max(0, count - 20),
            "instruction": (
                f"The draft currently uses {count} of 20 top-level slots, including {inherited} "
                "inherited entries. The corrected reply must finish with at most 20 entries. "
                "Consolidate enough related entries into complete multi-clause requirements "
                "in this reply; keep every meaningful obligation, condition and source decision. "
                "Fixing coverage by adding or splitting entries still consumes this same budget."
            ),
        },
        "missing_sources": sorted(set(catalog) - occurrences.keys()),
        "duplicate_sources": {k: v for k, v in occurrences.items() if len(v) > 1},
        "unknown_sources": sorted(occurrences.keys() - set(catalog)),
        "withdrawn_publication_conflicts": conflicts,
        "citation_checks": citation_checks,
        "retain_checks": retain_checks,
        "location_basis": "Paths address rejected_response, not internal expanded criteria or part_decisions",
    }
