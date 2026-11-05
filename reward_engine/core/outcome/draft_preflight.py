"""Independent, non-mutating draft diagnostics; publication still validates fully."""

from copy import deepcopy


def inspect(raw, evidence, context):
    from reward_engine.core.outcome.clauses import (
        QUOTE_REFERENCE_RULE,
        USES,
        quote_occurs,
    )
    from reward_engine.core.outcome.context import require_refs
    from reward_engine.core.outcome.jury import product_ref
    from reward_engine.core.outcome.task_inputs import refs as task_refs

    issues = []
    context = context or {}
    products = {
        p["product_id"]: {u["ref"] for u in p.get("user_updates", [])}
        for p in context.get("products", [])
    }
    users = set().union(*products.values()) if products else set()

    def add(code, field, actual, rule, **details):
        issues.append(
            {
                "code": code,
                "field": field,
                "actual": deepcopy(actual),
                "rule": rule,
                "reason": rule,
                **details,
            }
        )

    def refs(value, field, **details):
        try:
            require_refs(value, evidence)
        except (ValueError, TypeError, KeyError) as error:
            add("EVIDENCE_REFS", field, value, str(error), **details)
            return []
        return value

    def enum(row, key, allowed, field, **details):
        if not isinstance(row.get(key), str) or row[key] not in allowed:
            add(
                "ENUM",
                field + "." + key,
                row.get(key),
                "Allowed values: " + ", ".join(sorted(allowed)),
                **details,
            )

    criteria = raw.get("criteria")
    if not isinstance(criteria, list):
        add(
            "STRUCTURE",
            "criteria",
            criteria,
            "Expected 1–20 requirement objects",
            unchecked="Requirement semantics not checked because criteria is not a list",
        )
        criteria = []
    elif not 1 <= len(criteria) <= 20:
        add("CRITERIA_COUNT", "criteria", len(criteria), "Require 1–20 requirements")
    for i, row in enumerate(criteria):
        path = f"criteria[{i}]"
        if not isinstance(row, dict):
            add(
                "STRUCTURE",
                path,
                row,
                "Expected requirement object",
                unchecked="Requirement semantics not checked because row is not an object",
            )
            continue
        if "inherit_from" in row:
            continue  # Exact inheritance and source existence are checked by expand.
        identity = {"requirement_id": row.get("id", path)}
        enum(row, "kind", {"hard", "soft"}, path, **identity)
        enum(
            row, "origin", {"task", "user", "domain", "observed_gap"}, path, **identity
        )
        if "priority" in row:
            enum(row, "priority", {"critical", "important", "minor"}, path, **identity)
        if "priority_reason" in row and (
            not isinstance(row["priority_reason"], str)
            or not row["priority_reason"].strip()
        ):
            add(
                "REQUIRED_TEXT",
                path + ".priority_reason",
                row["priority_reason"],
                "Nonempty priority rationale required",
                **identity,
            )
        for key in ("why", "measurement_question"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                add(
                    "REQUIRED_TEXT",
                    path + "." + key,
                    row.get(key),
                    "Nonempty text required",
                    **identity,
                )
        cited = refs(row.get("evidence_refs"), path + ".evidence_refs", **identity)
        if row.get("kind") == "hard" and row.get("origin") not in {"task", "user"}:
            add(
                "HARD_ORIGIN",
                path + ".origin",
                row.get("origin"),
                "Hard requirements require task/user origin",
                **identity,
            )
        if (
            row.get("kind") == "hard"
            and row.get("origin") == "task"
            and "task" not in cited
        ):
            add(
                "TASK_REFERENCE",
                path + ".evidence_refs",
                row.get("evidence_refs"),
                "Hard task requirement must cite task",
                **identity,
            )
        if row.get("origin") == "user" and not users.intersection(cited):
            add(
                "USER_REFERENCE",
                path + ".evidence_refs",
                row.get("evidence_refs"),
                "User requirement must cite an actual scoped user statement",
                **identity,
            )
        clauses = row.get("clauses")
        if not isinstance(clauses, list) or not clauses:
            add(
                "STRUCTURE",
                path + ".clauses",
                clauses,
                "Nonempty clause list required",
                unchecked="Clause semantics not checked because clauses is invalid",
                **identity,
            )
            continue
        for j, clause in enumerate(clauses):
            field = f"{path}.clauses[{j}]"
            if not isinstance(clause, dict):
                add(
                    "STRUCTURE",
                    field,
                    clause,
                    "Expected clause object",
                    unchecked="Clause semantics not checked because clause is not an object",
                    **identity,
                )
                continue
            enum(clause, "use", USES, field, **identity)
            cited = refs(
                clause.get("evidence_refs"), field + ".evidence_refs", **identity
            )
            conditions = clause.get("conditions")
            if not isinstance(conditions, list) or any(
                not isinstance(c, str) or not c.strip() for c in conditions
            ):
                add(
                    "STRUCTURE",
                    field + ".conditions",
                    conditions,
                    "Explicit list of nonempty conditions required",
                    **identity,
                )
            if not isinstance(clause.get("alternative_group"), str):
                add(
                    "STRUCTURE",
                    field + ".alternative_group",
                    clause.get("alternative_group"),
                    "Explicit string required",
                    **identity,
                )
            quotes = clause.get("source_quotes")
            if not isinstance(quotes, list):
                add(
                    "STRUCTURE",
                    field + ".source_quotes",
                    quotes,
                    "Explicit list required",
                    **identity,
                )
                continue
            if clause.get("use") == "task_constraint" and not quotes:
                add(
                    "SOURCE_QUOTE",
                    field + ".source_quotes",
                    quotes,
                    "Task constraint requires exact Task/scoped user/input quote",
                    **identity,
                )
            allowed = (
                {"task", *users, *task_refs(context)}
                if clause.get("use") == "task_constraint"
                else set(evidence)
            )
            for k, quote in enumerate(quotes):
                loc = f"{field}.source_quotes[{k}]"
                if not isinstance(quote, dict):
                    add(
                        "STRUCTURE",
                        loc,
                        quote,
                        "Quote requires ref and quote fields",
                        **identity,
                    )
                    continue
                ref = quote.get("ref")
                if not isinstance(ref, str) or ref not in allowed:
                    add(
                        "QUOTE_PROVENANCE",
                        loc + ".ref",
                        ref,
                        "Quote must cite an allowed supplied source",
                        **identity,
                    )
                elif not quote_occurs(quote.get("quote"), evidence.get(ref)):
                    add(
                        "QUOTE_EXACT",
                        loc + ".quote",
                        quote.get("quote"),
                        "Quote must occur verbatim in cited evidence",
                        **identity,
                    )
                if ref not in cited:
                    add(
                        "QUOTE_REFERENCE",
                        field + ".evidence_refs",
                        clause.get("evidence_refs"),
                        QUOTE_REFERENCE_RULE + " Code will not add evidence.",
                        quoted_ref=deepcopy(ref),
                        quote_field=loc + ".ref",
                        **identity,
                    )
    gaps = raw.get("gaps", [])
    if not isinstance(gaps, list):
        add(
            "STRUCTURE",
            "gaps",
            gaps,
            "Expected list",
            unchecked="Gap semantics not checked because gaps is not a list",
        )
        return issues
    seen = set()
    for i, row in enumerate(gaps):
        path = f"gaps[{i}]"
        if not isinstance(row, dict):
            add(
                "STRUCTURE",
                path,
                row,
                "Expected gap object",
                unchecked="Gap semantics not checked because gap is not an object",
            )
            continue
        identity = {"gap_id": row.get("gap_id")}
        for key in ("gap_id", "expected", "observed", "gap", "verification"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                add(
                    "REQUIRED_TEXT",
                    path + "." + key,
                    row.get(key),
                    "Nonempty text required",
                    **identity,
                )
        key = row.get("gap_id")
        if isinstance(key, str):
            if key in seen:
                add(
                    "DUPLICATE_GAP",
                    path + ".gap_id",
                    key,
                    "Unique gap ID required",
                    **identity,
                )
            seen.add(key)
        enum(row, "expected_basis", {"task", "user", "domain"}, path, **identity)
        enum(row, "status", {"needs_check", "check_supported"}, path, **identity)
        cited = refs(row.get("evidence_refs"), path + ".evidence_refs", **identity)
        expected = row.get("expected_refs", [])
        expected = (
            refs(expected, path + ".expected_refs", **identity)
            if expected != []
            else []
        )
        basis = row.get("expected_basis")
        if basis == "task" and "task" not in expected:
            add(
                "GAP_TASK_REFERENCE",
                path + ".expected_refs",
                row.get("expected_refs"),
                'expected_basis="task" requires task in expected_refs',
                **identity,
            )
        if basis == "user" and not users.intersection(expected):
            add(
                "GAP_USER_REFERENCE",
                path + ".expected_refs",
                row.get("expected_refs"),
                "User expectation requires a scoped user reference",
                **identity,
            )
        targets = row.get("product_ids")
        if (
            not isinstance(targets, list)
            or any(not isinstance(p, str) or p not in products for p in targets)
            or (not targets and (products or row.get("status") != "needs_check"))
        ):
            add(
                "GAP_PRODUCTS",
                path + ".product_ids",
                targets,
                "Identify observed products; empty only for pre-product needs_check hypotheses",
                **identity,
            )
            continue
        for product in targets:
            if basis == "user" and not products[product].intersection(expected):
                add(
                    "GAP_USER_SCOPE",
                    path + ".expected_refs",
                    expected,
                    "User expectation must apply to " + product,
                    **identity,
                )
            if not any(product_ref(r, product, evidence) for r in cited):
                add(
                    "GAP_PRODUCT_EVIDENCE",
                    path + ".evidence_refs",
                    cited,
                    "Evidence required for " + product,
                    **identity,
                )
            if row.get("status") == "check_supported" and not any(
                r.startswith("check:")
                and isinstance(evidence[r], dict)
                and evidence[r].get("product_id") == product
                and evidence[r].get("status") == "fail"
                and not evidence[r].get("error")
                for r in cited
            ):
                add(
                    "GAP_FAILED_CHECK",
                    path + ".status",
                    row.get("status"),
                    "check_supported requires a real failed execution receipt for "
                    + product,
                    **identity,
                )
    return issues
