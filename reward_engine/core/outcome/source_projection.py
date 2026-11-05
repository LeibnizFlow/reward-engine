"""Lossless, literal source directory for explicitly opted-in role requests."""

from copy import deepcopy

VERSION = "canonical_sources_v1"


def enabled(call):
    return getattr(call, "role_input_projection", None) == VERSION


def semantic_requirement(row):
    """Exclude identifiers and only byte-proven text projections from equality."""
    value = deepcopy(row)
    for key in (
        "id",
        "expectation_id",
        "proposal_phase",
        "draft_validation_issues",
        "draft_field_projections",
        "clause_protocol",
    ):
        value.pop(key, None)
    clauses = value.get("clauses", [])
    texts = [c.get("text") for c in clauses if isinstance(c, dict)]
    if (
        clauses
        and len(texts) == len(clauses)
        and all(isinstance(t, str) for t in texts)
    ):
        if value.get("standard") == "\n".join(texts):
            value.pop("standard")
        parts = value.get("subrequirements")
        if parts == texts or (
            isinstance(parts, list)
            and all(
                isinstance(p, dict)
                and set(p) <= {"part_id", "selection_id", "text"}
                and isinstance(p.get("part_id"), str)
                and p["part_id"].rsplit("/", 1)[-1].isdigit()
                and int(p["part_id"].rsplit("/", 1)[-1]) < len(texts)
                and p.get("text") == texts[int(p["part_id"].rsplit("/", 1)[-1])]
                for p in parts
            )
        ):
            value.pop("subrequirements", None)
    for clause in clauses:
        if isinstance(clause, dict):
            clause.pop("id", None)
    return value


def meaning(row, index):
    value = semantic_requirement(row)
    clauses = value.pop("clauses", [])
    return {**value, "clause": clauses[index]}


def project(request, *, policy=None):
    """The audit is local; every omitted string has an exact visible counterpart."""
    from reward_engine.core.outcome.construction import fingerprint

    result = deepcopy(request)
    sources, omitted = {}, []
    from reward_engine.core.outcome.measurement_requests import POLICY

    local_bindings = {}
    if policy == POLICY:
        # This exact-equivalence map is an internal compilation binding. It is
        # not a second set of sources requiring model-authored dispositions.
        if "selection_aliases" in result:
            local_bindings["selection_aliases"] = result.pop("selection_aliases")
        result["role_contract_policy"] = POLICY
        result["instruction"] += (
            " Only keys in source_catalog are actionable source_parts. Each must be covered exactly once. "
            "Historical proposal decisions are audit context, not additional coverage obligations. "
            "Never add dispositions for historical aliases or sources already covered by inherit_from."
            " Plan at most 20 final criteria before writing them. Related obligations can be separate "
            "clauses in one coherent criterion; preserve their conditions and alternatives. "
            "If a final clause comes from an existing source, bind that source with retain, rewrite "
            "or merge. Do not exclude it and republish its text as an unbound add."
        )
    for group in ("independent_expectations", "proposal_expectations"):
        if group not in request:
            continue
        refs = []
        for row in request[group]:
            key = row["expectation_id"]
            item = deepcopy(row)
            clauses = item.get("clauses", [])
            texts = [c.get("text") for c in clauses]
            if texts and all(isinstance(t, str) for t in texts):
                if item.get("standard") == "\n".join(texts):
                    omitted.append(
                        {
                            "requirement": key,
                            "field": "standard",
                            "rule": "join clauses[].text with newline",
                        }
                    )
                    del item["standard"]
                for i, part in enumerate(item.get("subrequirements", [])):
                    index = part["part_id"].rsplit("/", 1)[-1]
                    if (
                        index.isdigit()
                        and int(index) < len(texts)
                        and part.get("text") == texts[int(index)]
                    ):
                        del part["text"]
                        part["clause_index"] = int(index)
                        omitted.append(
                            {
                                "requirement": key,
                                "field": f"subrequirements[{i}].text",
                                "clause_index": int(index),
                            }
                        )
            if key in sources and sources[key] != item:
                raise ValueError("Conflicting source requirement identity")
            sources[key] = item
            refs.append(key)
        result[group] = refs
    if sources:
        result["source_requirements"] = sources
        result["instruction"] += (
            " source_requirements is the single literal requirement directory. "
            "independent_expectations and proposal_expectations contain its exact keys. "
            "subrequirements[].clause_index points to that requirement's clauses array (zero based). "
            "source_catalog maps allowed source_parts to these requirements. "
            "An omitted standard is exactly clauses[].text joined with newline. "
            "Use inherit_from only for an unchanged entire requirement."
            " When supplied, inheritance_bindings lists the canonical source_parts consumed by "
            "each inherit_from requirement and their clause_index. Do not also defer, exclude, "
            "or bind those same sources elsewhere."
        )
    result["role_input_projection"] = VERSION
    audit = {
        "version": VERSION,
        "original_sources": {
            g: deepcopy(request[g])
            for g in ("independent_expectations", "proposal_expectations")
            if g in request
        },
        "omitted": omitted,
        **({"local_bindings": local_bindings, "contract_policy": policy} if policy == POLICY else {}),
        "original_sha256": fingerprint(request),
        "projected_sha256": fingerprint(result),
    }
    return result, audit
