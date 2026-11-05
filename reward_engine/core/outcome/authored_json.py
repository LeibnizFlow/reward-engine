"""Model-authored complete drafts; deterministic projections never decide semantics."""

import json
import re
from copy import deepcopy

VERSION = "model_authored_json_v2"
RECOVERY = "strict_bounded_v2"
LIMITS = {
    "requests": 8,
    "contract_repairs": 3,
    "syntax_repairs": 1,
    "evidence_rounds": 2,
    "timeout_s": 1800,
}

JSON_SYNTAX_INSTRUCTION = (
    'Use valid JSON string escaping in every field: escape embedded double quotes as \\", '
    'backslashes as \\\\, and line breaks as \\n. '
    'Example of a string containing quoted text: '
    + json.dumps({"text": 'The note says "hello".'}, ensure_ascii=False)
    + '. Preserve the intended text while encoding it correctly. Return one complete JSON object; '
    'do not return a fragment or describe the correction.'
)


def syntax_diagnostics(error):
    """Expose parser coordinates and a bounded exact excerpt, without repairing text."""
    start = max(0, error.pos - 80)
    return {
        "reason": error.msg,
        "syntax": {
            "offset": error.pos,
            "line": error.lineno,
            "column": error.colno,
            "coordinate_basis": "rejected_raw_response; zero-based Unicode character offset, one-based line and column",
            "location_kind": "parser_reported",
            "context_start": start,
            "context": error.doc[start:error.pos + 80],
        },
    }


def enabled(call):
    return getattr(call, "role_output_protocol", None) == VERSION


def decode(text):
    """Accept one JSON document or one complete JSON fence, never extract prose."""
    original = text
    match = re.fullmatch(r"\s*```(?:json)?\s*\n([\s\S]*?)\n```\s*", text)
    if match:
        text = match[1]

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise json.JSONDecodeError("Duplicate object key: " + key, original, 0)
            result[key] = value
        return result

    def constant(value):
        raise json.JSONDecodeError("Non-JSON constant: " + value, original, 0)

    value = json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    if not isinstance(value, dict):
        raise json.JSONDecodeError("Expected one JSON object", original, 0)
    return value


def reject(issues):
    from reward_engine.core.outcome.context import ResponseContractError

    error = ResponseContractError("authored_json", "Complete draft failed validation")
    error.details["issues"] = issues
    raise error


INSTRUCTION = (
    "Return ONE complete JSON object matching response. Author semantic content, not edits. "
    "clauses[].text is the only requirement text: omit standard, subrequirements and internal IDs. "
    "Code derives these fields. Each criterion AND each clause must explicitly supply nonempty evidence_refs. "
    "Preserve the Task scope, source support, conditions and alternatives. "
    'For an unchanged WHOLE requirement use only {"inherit_from":"expectation_id"}. '
    "Otherwise write the full requirement and give each clause source_parts, action and reason. "
    "source_parts selects requirement clause IDs from source_catalog; evidence_refs selects evidence handles. "
    "Never put task, product or document evidence handles in source_parts. For add use an empty source_parts "
    "list and cite support in evidence_refs. A final clause linked to an old requirement uses retain/rewrite/merge. "
    "retain preserves exact text AND parent/clause meaning; use rewrite for any change, merge for "
    "combining sources, add with source_parts=[] for a new requirement. "
    "Adding evidence, paraphrasing a rationale or changing any extra parent field "
    "is a change for this contract. For an entirely unchanged requirement, inherit_from avoids "
    "copying its fields; otherwise explicitly declare the changes. Evidence must be explicitly "
    "supplied by you. Every source must be addressed exactly once, including withheld sources in "
    "dispositions. Never repeat final text in a separate ledger. Unknown, duplicate or omitted "
    "sources are errors. These declarations are proposals, not proof of truth. "
)


def contract(mode, schema, *, canonical=False):
    """One compact representation for independent, member and merged drafts."""
    from reward_engine.core.outcome.clauses import QUOTE_REFERENCE_RULE
    from reward_engine.core.outcome.clauses import SCHEMA as clause_schema

    value = deepcopy(schema)
    if mode not in {"outcome_expectations", "outcome_jury_map", "outcome_jury_reduce"}:
        return value
    criterion = value["criteria"][0]
    for key in ("id", "standard", "subrequirements"):
        criterion.pop(key, None)
    clause = deepcopy(clause_schema)
    clause.pop("id")
    if canonical:
        clause["evidence_refs"] = [clause["evidence_refs"][0] + "; " + QUOTE_REFERENCE_RULE]
    if mode != "outcome_expectations":
        clause.update(
            source_parts=[
                "exact requirement clause IDs from source_catalog; never evidence handles"
            ],
            action="retain/rewrite/merge/add",
            reason="source-grounded decision",
        )
        value["dispositions"] = [
            {
                "source_parts": ["exact source handles"],
                "action": "delete/defer"
                if mode == "outcome_jury_map"
                else "exclude/defer",
                "reason": "why withheld",
                "evidence_refs": ["supporting refs"],
            }
        ]
    criterion["why"] = "importance to the requested Task outcome"
    criterion["clauses"] = [clause]
    if canonical:
        value["criteria"] = {"minItems": 1, "maxItems": 20,
                             "items": {"oneOf": ([{"inherit_from": "exact expectation_id; no other fields"}, criterion]
                                                   if mode != "outcome_expectations" else [criterion])}}
    value.pop("revision_decisions", None)
    value.pop("part_decisions", None)
    value.pop("selection_protocol", None)
    return value


def instruction(mode, original):
    from reward_engine.core.outcome.clauses import INSTRUCTION as clause_instruction
    from reward_engine.core.outcome.jury_prompts import (
        OUTCOME_EXPECTATIONS_PROMPT,
        OUTCOME_JURY_MAP_PROMPT,
        OUTCOME_VALUE_INSTRUCTION,
    )

    if mode == "outcome_expectations":
        return OUTCOME_EXPECTATIONS_PROMPT + clause_instruction + INSTRUCTION
    if mode == "outcome_jury_map":
        return OUTCOME_JURY_MAP_PROMPT + clause_instruction + INSTRUCTION
    if mode == "outcome_jury_reduce":
        return (
            "Synthesize and audit all supplied unvalidated drafts against Task, scoped user amendments "
            "and evidence. Preserve important omissions and professional quality goals. Resolve false "
            "prescriptions and distinguish illustrative methods from obligations. Retain all meaningful "
            "conditions/alternatives. Explicitly exclude irrelevant suggestions or defer uncertainty. "
            "Do not weaken requirements because products omitted them. Gaps are suspected unless an "
            "actual failed execution supports them. No private grader criteria or absolute scores. "
            "Use canonical selection_id values; selection_aliases are exact duplicates expanded by code. "
            + OUTCOME_VALUE_INSTRUCTION
            + clause_instruction
            + INSTRUCTION
        )
    return original


def expand(raw, mode, inputs, evidence, *, canonical=False, context=None):
    """Compile only declared relationships. All original validators still run."""
    if mode not in {"outcome_expectations", "outcome_jury_map", "outcome_jury_reduce"}:
        return raw
    from reward_engine.core.outcome.context import require_refs
    from reward_engine.core.outcome.jury_revision import criterion_body, meaning

    value = deepcopy(raw)
    from reward_engine.core.outcome.draft_preflight import inspect

    issues = inspect(raw, evidence, context) if canonical else []

    def issue(field, reason, **details):
        issues.append({"field": field, "reason": reason, **details})
        if canonical:
            actual = raw
            for key in re.findall(r"[^.\[\]]+", field):
                if isinstance(actual, dict):
                    actual = actual.get(key)
                elif isinstance(actual, list) and key.isdigit() and int(key) < len(actual):
                    actual = actual[int(key)]
                else:
                    actual = None
                    break
            issues[-1].update(code="SOURCE_CONTRACT", rule=reason, actual=deepcopy(actual))
            if field.startswith("criteria["):
                issues[-1]["requirement_location"] = field.split(".", 1)[0]

    if not isinstance(value.get("criteria"), list) or not value["criteria"]:
        if canonical:
            value["criteria"] = []  # Continue independent gap and source checks; never publish.
        else:
            reject([{"field": "criteria", "reason": "Expected nonempty criteria list directly at the JSON root",
                     "actual_root_fields": sorted(value), "actual_type": type(value.get("criteria")).__name__}])
    for key in (
        "revision_decisions",
        "part_decisions",
        "coverage",
        "selection_protocol",
    ):
        if key in value:
            issue(key, "Derived internal field; submit the compact draft")
    merge = mode == "outcome_jury_reduce"
    initial = inputs.get("independent_expectations", [])
    sources = initial + (inputs.get("proposal_expectations", []) if merge else [])
    originals = {r["expectation_id"]: r for r in sources}
    catalog = {
        p["selection_id"] if merge else p["part_id"]: (r, i, p)
        for r in sources
        for i, p in enumerate(r["subrequirements"])
    }
    aliases = inputs.get("selection_aliases", {}) if merge else {}
    hidden = {p for rows in aliases.values() for p in rows}
    expected = set(catalog) - hidden
    used, decisions, criteria = set(), [], []
    used_at = {}
    reserved = {r["id"] for r in sources}

    def fresh_id(index):
        key = f"authored_{index + 1}"
        while key in reserved:
            key = "_" + key
        reserved.add(key)
        return key

    def lineage(row, field, criterion=None, clause=None, index=None):
        parts, action = row.get("source_parts"), row.get("action")
        allowed = (
            {"retain", "rewrite", "merge", "add"}
            if clause is not None
            else {"exclude", "defer"}
            if merge
            else {"delete", "defer"}
        )
        if not isinstance(parts, list) or any(not isinstance(p, str) for p in parts):
            issue(field, "source_parts must be a list of exact source handles")
            return
        if action not in allowed or (not parts) != (action == "add"):
            issue(field, "Invalid action or empty source_parts", action=action,
                  allowed_actions=sorted(allowed), source_parts=parts,
                  rule="add requires []; all other actions require explicit source parts")
            return
        if len(set(parts)) != len(parts) or set(parts) & used or set(parts) - expected:
            issue(field + ".source_parts" if canonical else field, "Unknown, duplicate or noncanonical source reference",
                  unknown_sources=sorted(set(parts) - expected - hidden),
                  noncanonical_aliases={p: {"canonical_source": target,
                                            "already_covered": used_at.get(target),
                                            "rule": "Use the canonical source exactly once; resolve conflicting actions yourself"}
                                       for target, values in aliases.items() for p in parts if p in values},
                  repeated_in_clause=sorted({p for p in parts if parts.count(p) > 1}),
                  already_covered={p: used_at[p] for p in parts if p in used_at},
                  unconflicted_sources=sorted(p for p in set(parts) & expected - used
                                             if parts.count(p) == 1),
                  coverage_rule="Unconflicted sources in this group still require one explicit decision. "
                  "Removing a conflicting source does not remove the other sources' coverage obligations. "
                  "Reconsider each decision against the Task and evidence; return a complete draft.")
        used.update(parts)
        for part in parts:
            used_at.setdefault(part, field)
        if not isinstance(row.get("reason"), str) or not row["reason"].strip():
            issue(field, "Explicit decision reason required")
        if action not in {"retain", "defer"}:
            try:
                require_refs(row.get("evidence_refs"), evidence)
            except ValueError as error:
                issue(field, str(error))
        if action == "retain" and clause is not None:
            for part in parts:
                if part not in catalog:
                    continue
                parent, old_index, old = catalog[part]
                # Compacted Reduce arrays can omit duplicate parts: locate the
                # clause by its original part_id index, never by similar text.
                source_index = (
                    int(old["part_id"].rsplit("/", 1)[1]) if merge else old_index
                )
                if canonical:
                    from reward_engine.core.outcome.source_projection import (
                        meaning as source_meaning,
                    )

                    old_meaning = source_meaning(parent, source_index)
                    new_meaning = source_meaning(criterion, index)
                    equal = old_meaning == new_meaning
                elif merge:
                    # Retain must preserve every semantic parent field, including
                    # optional fields authored by the model. Only identifiers,
                    # projections and validation annotations are derived here.
                    derived = {
                        "id", "expectation_id", "proposal_phase", "standard",
                        "subrequirements", "clauses", "draft_validation_issues",
                        "draft_field_projections", "clause_protocol",
                    }
                    before = {
                        k: v
                        for k, v in parent.items()
                        if k not in derived
                    }
                    after = {k: v for k, v in criterion.items() if k not in derived}
                    old_clause = {
                        k: v
                        for k, v in parent["clauses"][source_index].items()
                        if k not in {"id", "source_part_ids"}
                    }
                    new_clause = {k: v for k, v in clause.items() if k != "id"}
                    equal = (
                        before == after
                        and old_clause == new_clause
                        and old["text"] == clause["text"]
                    )
                    old_meaning = {"parent": before, "clause": old_clause}
                    new_meaning = {"parent": after, "clause": new_clause}
                else:
                    old_meaning, new_meaning = meaning(parent, old_index), meaning(criterion, index)
                    equal = old_meaning == new_meaning
                if not equal:
                    issue(
                        field,
                        "retain changed text or meaning; explicitly rewrite with evidence",
                        source_part=part,
                        changed_fields=differences(old_meaning, new_meaning),
                        resolution_choices=[
                            {"when": "Keep the ENTIRE original requirement and every field unchanged",
                             "criterion": {"inherit_from": parent["expectation_id"]}},
                            {"when": "Change any text, parent field, evidence or clause",
                             "instruction": "Submit the complete revised requirement; explicitly use rewrite "
                             "(or merge when combining sources), with your reason and supporting evidence. "
                             "Choose the intended semantics yourself."},
                        ],
                    )
        if merge:
            if action == "add":
                return  # Explicit addition is retained in the authored audit.
            decision = {
                "parts": parts,
                "action": "rewrite" if action == "merge" else action,
                "reason": row.get("reason"),
                "evidence_refs": row.get("evidence_refs", []),
            }
            if clause is not None:
                decision.update(criterion_id=criterion["id"], clause_id=clause["id"])
                if action in {"rewrite", "merge"}:
                    decision["text"] = clause["text"]
        else:
            decision = {
                "source_parts": parts,
                "action": action,
                "reason": row.get("reason"),
                "evidence_refs": row.get("evidence_refs", []),
                "targets": [{"criterion_id": criterion["id"], "part_index": index}]
                if clause is not None
                else [],
            }
        decisions.append(decision)

    for i, item in enumerate(value["criteria"]):
        field = f"criteria[{i}]"
        if not isinstance(item, dict):
            issue(field, "Expected criterion object")
            continue
        if "inherit_from" in item:
            ref = item["inherit_from"]
            if (
                mode == "outcome_expectations"
                or set(item) != {"inherit_from"}
                or not isinstance(ref, str)
                or ref not in originals
            ):
                issue(
                    field,
                    "inherit_from must name one supplied expectation without overrides",
                )
                continue
            parent = originals[ref]
            criterion = criterion_body(parent)
            # Seat-local IDs are generated independently and can collide.
            # Renaming only the internal target ID preserves the exact inherited
            # requirement; explicit source handles still detect duplicate use.
            if any(row["id"] == criterion["id"] for row in criteria):
                criterion["id"] = fresh_id(i)
            if merge:
                # Compacted records do not contain hidden duplicate parts; copy
                # complete clauses and derive the internal text projection.
                criterion["subrequirements"] = [c["text"] for c in criterion["clauses"]]
            if canonical:
                inherited_issues = inspect({"criteria": [criterion]}, evidence, context)
                for item_issue in inherited_issues:
                    item_issue["field"] = item_issue["field"].replace("criteria[0]", field, 1)
                    item_issue["requirement_id"] = ref
                    item_issue["inherited_source_valid"] = False
                    item_issue["inheritance_rule"] = (
                        "inherit_from copies the entire supplied source, including this defect. "
                        "Overrides on inherit_from are forbidden. To correct the source, submit a "
                        "complete authored criterion with explicit source bindings and rewrite or "
                        "merge decisions for changed fields; retain requires exact original meaning. "
                        "For Reduce, use the canonical source_parts in inheritance_bindings or "
                        "source_catalog. Return the complete draft; inherited source validity is "
                        "not guaranteed by its presence in the historical inputs."
                    )
                issues.extend(inherited_issues)
            criteria.append(criterion)
            inherited = inputs.get("inheritance_bindings", {}).get(ref) if merge else None
            if inherited is None:
                inherited = [{"selection_id": p["selection_id"] if merge else p["part_id"],
                              "clause_index": int(p["part_id"].rsplit("/", 1)[1]) if merge else j}
                             for j, p in enumerate(parent["subrequirements"])]
            for part in inherited:
                key = part["selection_id"]
                if key in hidden:
                    continue
                if key in used:
                    issue(field, "Duplicate inherited source", source_part=key, already_covered=used_at[key])
                used.add(key)
                used_at.setdefault(key, field)
                idx = part["clause_index"]
                decisions.append(
                    {
                        "parts": [key],
                        "action": "retain",
                        "criterion_id": criterion["id"],
                        "clause_id": criterion["clauses"][idx]["id"],
                    }
                    if merge
                    else {
                        "source_parts": [key],
                        "action": "retain",
                        "targets": [{"criterion_id": criterion["id"], "part_index": idx}],
                    }
                )
            continue
        if any(k in item for k in ("id", "standard", "subrequirements")):
            issue(field, "Omit derived id, standard and subrequirements")
        clauses = item.get("clauses")
        if (
            not isinstance(clauses, list)
            or not clauses
            or any(
                not isinstance(c, dict)
                or not isinstance(c.get("text"), str)
                or not c["text"].strip()
                for c in clauses
            )
        ):
            issue(field, "Provide nonempty clauses with text")
            continue
        criterion = {
            k: deepcopy(v)
            for k, v in item.items()
            if k not in {"clauses", "id", "standard", "subrequirements"}
        }
        criterion.update(
            id=fresh_id(i),
            standard="\n".join(c["text"] for c in clauses),
            subrequirements=[c["text"] for c in clauses],
            clauses=[],
        )
        for j, clause in enumerate(clauses):
            if "id" in clause:
                issue(f"{field}.clauses[{j}]", "Omit derived clause id")
            criterion["clauses"].append(
                {
                    k: deepcopy(v)
                    for k, v in clause.items()
                    if k not in {"id", "source_parts", "action", "reason"}
                }
                | {"id": f"clause_{j + 1}"}
            )
        criteria.append(criterion)
        if mode != "outcome_expectations":
            for j, clause in enumerate(clauses):
                lineage(
                    clause,
                    f"{field}.clauses[{j}]",
                    criterion,
                    criterion["clauses"][j],
                    j,
                )
    dispositions = value.pop("dispositions", [])
    if not isinstance(dispositions, list):
        issue("dispositions", "Expected list")
    else:
        for i, row in enumerate(dispositions):
            if not isinstance(row, dict):
                issue(f"dispositions[{i}]", "Expected object")
            else:
                lineage(row, f"dispositions[{i}]")
    if mode != "outcome_expectations" and expected - used:
        issue(
            "source_parts",
            "Missing dispositions: " + ", ".join(sorted(expected - used)),
            missing_sources=sorted(expected - used),
        )
    if len({c["id"] for c in criteria}) != len(criteria):
        issue("criteria", "Duplicate inherited criterion")
    # Aggregate independent citation failures before the existing semantic
    # validators; no reference is supplied, moved or inferred here.
    for i, criterion in enumerate(criteria):
        for field, row in [(f"criteria[{i}]", criterion), *[
            (f"criteria[{i}].clauses[{j}]", clause)
            for j, clause in enumerate(criterion.get("clauses", []))
        ]]:
            try:
                require_refs(row.get("evidence_refs"), evidence)
            except (ValueError, TypeError) as error:
                issue(field + ".evidence_refs", str(error))
    if issues:
        if canonical:
            for issue_row in issues:
                issue_row.setdefault("code", "SOURCE_CONTRACT")
                issue_row.setdefault("rule", issue_row["reason"])
                issue_row.setdefault("actual", None)
        reject(issues)
    value["criteria"] = criteria
    if mode != "outcome_expectations":
        value["part_decisions" if merge else "revision_decisions"] = decisions
    if merge:
        value["selection_protocol"] = "exact-alias-v1"
    # Audit is external to criteria, so it cannot change retain equality.
    value["authored_json"] = {"protocol": VERSION, "draft": deepcopy(raw),
                              **({"input_projection": "canonical_sources_v1"} if canonical else {})}
    return value


def source_catalog(mode, inputs):
    """Expose exact requirement handles separately from evidence handles."""
    if mode not in {"outcome_jury_map", "outcome_jury_reduce"}:
        return {}
    groups = ["independent_expectations"] + (["proposal_expectations"] if mode == "outcome_jury_reduce" else [])
    hidden = {key for values in inputs.get("selection_aliases", {}).values() for key in values}
    result = {}
    for group in groups:
        for row in inputs.get(group, []):
            for part in row["subrequirements"]:
                key = part.get("selection_id") if mode == "outcome_jury_reduce" else part["part_id"]
                if key not in hidden:
                    result[key] = {"expectation_id": row["expectation_id"], "part_id": part["part_id"]}
    return result


def diagnostics(value):
    """Keep validation facts, excluding legacy field-edit instructions."""
    if isinstance(value, dict):
        return {key: diagnostics(item) for key, item in value.items()
                if key not in {"repair_instruction", "recovery_instruction",
                               "field_correction_reply", "response_unwrap_reply"}}
    if isinstance(value, list):
        return [diagnostics(item) for item in value]
    return deepcopy(value)


def differences(before, after, prefix=""):
    """Describe exact mismatches without proposing replacement semantics."""
    from reward_engine.core.outcome.jury_revision import nested_differences

    return [prefix + ".".join(map(str, row["path"])) for row in nested_differences(before, after)]
