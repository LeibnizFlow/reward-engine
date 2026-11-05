"""Audit measurement for the existing public entry point."""


def criteria(rows, *, require_anchors=False):
    """Jury defines standards; builder adds fail/partial/full rubric anchors."""
    from experiments.audit_receipts import ids, require, strings, text

    require(
        isinstance(rows, list)
        and 1 <= len(rows) <= 20
        and all(isinstance(row, dict) for row in rows),
        "Jury criteria must be 1..20 objects",
    )
    values = ids(rows, "id")
    for row in rows:
        require(
            all(text(row.get(k)) for k in ("standard", "measurement_question")),
            "Empty Jury criterion",
        )
        require(row.get("kind") in {"hard", "soft"}, "Invalid criterion kind")
        require(
            row.get("origin") in {"task", "user", "domain", "observed_gap"},
            "Invalid criterion origin",
        )
        require(strings(row.get("evidence_refs")), "Ungrounded Jury criterion")
        if row["kind"] == "hard":
            require(
                row["origin"] in {"task", "user"},
                "Domain preferences cannot become hard Task requirements",
            )
            require(
                row["origin"] != "task" or "task" in row["evidence_refs"],
                "A hard Task requirement must cite the Task",
            )
        if "priority" in row:
            require(
                isinstance(row["priority"], str)
                and row["priority"] in {"critical", "important", "minor"},
                "Invalid criterion priority",
            )
        if "priority_reason" in row:
            require(text(row["priority_reason"]), "Empty criterion priority_reason")
        if "subrequirements" in row:
            require(
                isinstance(row["subrequirements"], list)
                and all(text(part) for part in row["subrequirements"]),
                "Criterion subrequirements must be a list of nonempty strings",
            )
        if require_anchors:
            anchors = row.get("anchors")
            require(
                isinstance(anchors, dict)
                and all(text(anchors.get(key)) for key in ("fail", "partial", "full")),
                "Builder rubric needs explicit fail/partial/full anchors",
            )
    return values


def batch(row, minimum=1, expected=None):
    from experiments.audit_receipts import ids, require

    items = row["items"]
    ids(items, "job_id")
    succeeded = sum(item.get("status") == "ok" for item in items)
    require(
        row.get("total") == len(items) and row.get("succeeded") == succeeded,
        "Inconsistent Map receipt",
    )
    require(succeeded >= minimum, "No sufficient successful native Map outputs")
    require(
        succeeded == len(items) and row.get("complete") is True,
        "Map contains failed/inactive seats",
    )
    require(expected is None or len(items) == expected, "Missing expected Map seats")


def check_result(row, field):
    """Validate a retained verdict without coercing or repairing model output."""
    from experiments.audit_receipts import require, strings, text

    require(
        isinstance(row, dict),
        f"Malformed {field}: expected an object, got {type(row).__name__}",
    )
    require(
        isinstance(row.get("status"), str)
        and row["status"] in {"pass", "fail", "unknown"}
        and text(row.get("reason")),
        f"Malformed {field}: expected pass/fail/unknown status and nonempty reason",
    )
    require(
        "score" not in row,
        f"Malformed {field}: Outcome checks do not return absolute scores",
    )
    require(not row.get("error"), f"Checker execution failed at {field}")
    gap = row.get("gap")
    require(
        gap is None or isinstance(gap, str), f"Malformed {field}.gap: expected text"
    )
    if row["status"] != "unknown":
        require(
            strings(row.get("evidence_refs")),
            f"Known measurement without evidence at {field}",
        )
    if row["status"] == "fail":
        require(text(gap), f"Product failure without a concrete gap at {field}")


def measurement_results(row, pack, *, full=True):
    from experiments.audit_receipts import ids, require

    require(isinstance(row, dict), "Malformed measurement: expected an object")
    require(
        row.get("pack_id") == pack["pack_id"], "Measurement belongs to another pack"
    )
    results = row.get("results")
    require(
        isinstance(results, list)
        and results
        and all(isinstance(result, dict) for result in results),
        "Malformed measurement.results: expected nonempty verdict objects",
    )
    tests = {t["test_id"]: t["criterion_id"] for t in pack["tests"]}
    measured = ids(results, "test_id")
    require(set(measured) <= set(tests), "Unknown measured test")
    if full:
        require(set(measured) == set(tests), "Full construction audit omitted tests")
    for result in results:
        require(
            result.get("criterion_id") == tests[result["test_id"]],
            "Measurement criterion mismatch",
        )
        field = f"measurement.results[{result['test_id']}]"
        check_result(result, field)
        if "inquiry_result" in result:
            check_result(result["inquiry_result"], field + ".inquiry_result")
    require(
        any(r["status"] != "unknown" for r in results),
        "Measurements entirely unknown; no validated check execution",
    )
    facts = {
        s: sum(r["status"] == s for r in results) for s in ("pass", "fail", "unknown")
    }
    diagnostics = [
        {"test_id": row["test_id"], "diagnostic": row["inquiry_result_diagnostic"]}
        for row in results
        if "inquiry_result_diagnostic" in row
    ]
    if diagnostics:
        facts["inquiry_result_diagnostics"] = diagnostics
    return facts


def measurement_execution(
    row, pack, selected, *, context=None, evidence=None, previous=None
):
    """Check execution ledgers and only independently reconstructable reuse."""
    from experiments.audit_receipts import require, text

    results = row.get("results")
    require(
        isinstance(results, list) and all(isinstance(r, dict) for r in results),
        "Malformed measurement execution results",
    )
    result_ids = [r.get("test_id") for r in results]
    require(
        all(text(key) for key in result_ids)
        and len(result_ids) == len(set(result_ids))
        and set(result_ids) == set(selected),
        "Measurement selected-check/result coverage mismatch",
    )
    tests = {test["test_id"]: test for test in pack["tests"]}
    require(set(selected) <= set(tests), "Unknown selected measurement test")
    reused = []
    for result in results:
        require(
            "reused_unchanged_input" not in result
            or type(result["reused_unchanged_input"]) is bool,
            "Malformed measurement reuse flag",
        )
        if not result.get("reused_unchanged_input"):
            continue
        require(
            previous is not None and context is not None and evidence is not None,
            "Measurement reuse has no verified prior execution inputs",
        )
        old_context, old_evidence, old_measurement = previous
        prior = {r["test_id"]: r for r in old_measurement["results"]}.get(
            result["test_id"]
        )
        require(
            prior is not None
            and prior.get("status") == "pass"
            and not prior.get("error"),
            "Measurement reuse requires a prior passing check without error",
        )
        require(
            not context.get("measurement_inquiries", {}).get(result["test_id"]),
            "Measurement reuse forbidden with an active inquiry",
        )
        require(
            result == {**prior, "reused_unchanged_input": True},
            "Reused measurement changed the retained prior result",
        )
        # This builds the shipped registry only; it neither runs a checker nor
        # imports arbitrary recorded code to invent a dependency contract.
        from reward_engine.core.outcome.harness import defaults

        registry = defaults(None)
        test = tests[result["test_id"]]
        require(
            registry.dependencies.get(test["checker"]) is not None
            and not test.get("execution_recipe"),
            "Measurement reuse dependency contract unverified for this checker",
        )
        require(
            registry.inputs(test, context, evidence)
            == registry.inputs(test, old_context, old_evidence),
            "Measurement reuse dependency inputs changed",
        )
        reused.append(result["test_id"])
    submitted = set(result_ids) - set(reused)
    execution = row.get("execution_summary")
    require(isinstance(execution, dict), "Missing measurement execution summary")
    expected = {
        "selected_checks": len(selected),
        "submitted_checks": len(submitted),
        "reused_checks": len(reused),
    }
    for name, usage_key in (
        ("reported_model_requests", "model_requests"),
        ("reported_tool_calls", "tool_calls"),
    ):
        values = [
            r.get("execution_usage", {}).get(usage_key, 0)
            for r in results
            if not r.get("reused_unchanged_input")
        ]
        require(
            all(type(value) is int and value >= 0 for value in values),
            "Malformed measurement execution usage",
        )
        expected[name] = sum(values)
    require(
        all(
            type(execution.get(key)) is int and execution[key] == value
            for key, value in expected.items()
        ),
        "Measurement execution summary/result mismatch",
    )
    executed_batch = row.get("batch")
    require(isinstance(executed_batch, dict), "Missing measurement execution batch")
    if submitted:
        batch(executed_batch, expected=len(submitted))
        require(
            {r["job_id"] for r in executed_batch["items"]} == submitted,
            "Measurement executed batch/check identity mismatch",
        )
        require(
            not any(r.get("error_type") for r in executed_batch["items"]),
            "Measurement batch contains execution errors",
        )
    else:
        require(
            executed_batch
            == {"complete": True, "total": 0, "succeeded": 0, "items": []},
            "Empty measurement execution batch inconsistent",
        )


def measurement(row, pack, *, full=True, context=None, evidence=None, previous=None):
    facts = measurement_results(row, pack, full=full)
    selected = (
        [test["test_id"] for test in pack["tests"]]
        if full
        else [r["test_id"] for r in row["results"]]
    )
    measurement_execution(
        row, pack, selected, context=context, evidence=evidence, previous=previous
    )
    return facts


def measurement_binding_schema(row):
    from experiments.audit_receipts import require, text
    from reward_engine.core.outcome.harness import MEASUREMENT_INPUT_SCHEMA

    require(
        row.get("measurement_input_schema") == MEASUREMENT_INPUT_SCHEMA
        and text(row.get("measurement_input_sha256")),
        "Unbound measurement: missing recorded measurement input fingerprint",
    )


def measurement_binding(row, pack, task, observed):
    """Recompute expected inputs; never add missing evidence to a receipt."""
    from experiments.audit_receipts import require
    from reward_engine.core.outcome.context import build_context
    from reward_engine.core.outcome.harness import measurement_input_fingerprint

    measurement_binding_schema(row)
    context, evidence = build_context(task, [observed])
    expected = measurement_input_fingerprint(pack, context, evidence)
    require(
        row["measurement_input_sha256"] == expected,
        "Measurement input fingerprint does not match actual task/candidate/pack",
    )
    return expected
