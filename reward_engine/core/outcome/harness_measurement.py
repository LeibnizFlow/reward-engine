"""Harness measurement for the existing public entry point."""

from copy import deepcopy

from reward_engine.core.batch import BatchPolicy, collect_batch
from reward_engine.core.outcome.construction import fingerprint
from reward_engine.core.outcome.coverage import complete, expectations, report


def measurement_context(pack, context, evidence):
    """Carry published task-scoped sources into fresh delivery measurements."""
    context, evidence = deepcopy(context), deepcopy(evidence)
    if pack.get("public_delivery_contract"):
        from reward_engine.core.outcome.public_delivery import (
            attach_context,
            mandatory_tests,
        )

        mandatory_tests(pack)
        context, evidence = attach_context(
            pack["public_delivery_contract"], context, evidence
        )
    elif context["task"].get("public_delivery_contract"):
        raise ValueError("Public delivery contract missing from published Pack")
    for validation in pack.get("validation", {}).get("records", []):
        if not validation.get("cases"):
            continue
        ref = "reference:validation:" + validation["validation_id"]
        evidence[ref] = deepcopy(validation)
        docs = context.setdefault("reference_documents", [])
        if not any(document["ref"] == ref for document in docs):
            docs.append(
                {
                    "ref": ref,
                    "title": "Scoped counterexample evidence for "
                    + validation["target_id"],
                    "preview_only": True,
                    "preview": validation["status"]
                    + "; read cases and applicability before relying",
                }
            )
    for ref, source in pack.get("measurement_sources", {}).items():
        if ref in evidence and evidence[ref] != source:
            raise ValueError(
                "Published measurement source conflicts with current evidence"
            )
        evidence[ref] = deepcopy(source)
        if ref.startswith("task_input:"):
            from reward_engine.core.outcome.task_inputs import attach

            context, evidence = attach(context, evidence, [source])
        docs = context.setdefault("reference_documents", [])
        if not any(d["ref"] == ref for d in docs):
            docs.append(
                {
                    "ref": ref,
                    "url": source.get("url"),
                    "title": source.get("title"),
                    "preview_only": True,
                }
            )
    for prior in pack.get("measurement_priors", []):
        if prior["ref"] in evidence and evidence[prior["ref"]] != prior["content"]:
            raise ValueError("Published prior conflicts with current evidence")
        if prior["ref"] not in evidence:
            evidence[prior["ref"]] = deepcopy(prior["content"])
            context.setdefault("skills", []).append(deepcopy(prior))
    context["available_evidence_refs"] = list(evidence)
    return context, evidence


def _bound_measurement_inputs(pack, context, evidence):
    from reward_engine.core.outcome.harness import MEASUREMENT_INPUT_SCHEMA

    context, evidence = measurement_context(pack, context, evidence)
    input_sha256 = fingerprint(
        {
            "schema": MEASUREMENT_INPUT_SCHEMA,
            "pack": pack,
            "context": context,
            "evidence": evidence,
        }
    )
    return context, evidence, input_sha256


def measurement_input_fingerprint(pack, context, evidence):
    """Hash exact normalized inputs without executing checks or mutating callers.

    Reconstruct from build_context(actual_task, [native_candidate]) and the actual
    Pack. Already measurement_context-normalized inputs produce the same hash.
    Preserve any additional measured context (such as inquiries) when rebuilding.
    This binds input identity, not selected-check coverage, correctness or scores.
    Missing historical bindings remain unbound; never synthesize one onto a receipt.
    """
    return _bound_measurement_inputs(pack, context, evidence)[2]


async def measure(
    pack,
    registry,
    context,
    evidence,
    *,
    policy=None,
    test_ids=None,
    previous=None,
    completed=None,
):
    from reward_engine.core.outcome.harness import MEASUREMENT_INPUT_SCHEMA, unknown

    pack = deepcopy(pack)
    if (
        context["task"]["task_id"] != pack["task_id"]
        or context["task"]["prompt"] != pack["task_prompt"]
    ):
        raise ValueError("Cannot measure another Task with this Pack")
    if pack.get("reward_protocol"):
        from reward_engine.core.outcome.public_facts import install, validate_binding
        from reward_engine.core.outcome.public_reward import validate_pack

        validate_pack(pack)
        if len(context.get("products", [])) != 1:
            raise ValueError("Independent reward requires exactly one candidate")
        validate_binding(context, evidence)
        install(registry)
    context, evidence, input_sha256 = _bound_measurement_inputs(pack, context, evidence)
    mandatory = []
    if pack.get("public_delivery_contract"):
        from reward_engine.core.outcome.public_delivery import install, mandatory_tests

        install(registry)
        mandatory = mandatory_tests(pack)
    tests = pack["tests"]
    for test in tests:
        if test.get("execution_recipe") and test["checker"] != "evidence.review":
            registry.bind_recipe(test["checker"], test["execution_recipe"])
    if test_ids is not None:
        test_ids = list(dict.fromkeys(list(test_ids) + mandatory))
        if not set(test_ids) <= {t["test_id"] for t in tests}:
            raise ValueError("Unknown test selection")
        tests = [t for t in tests if t["test_id"] in test_ids]
    from reward_engine.core.outcome.resume_inputs import measurement_results

    reused = measurement_results(
        completed,
        input_sha256,
        pack,
        tests,
        evidence,
        policy=getattr(registry.tools, "role_contract_policy", None),
        tools=registry.tools,
    )
    if previous:
        old_context, old_evidence, old_result = previous
        old = {r["test_id"]: r for r in old_result["results"]}
        for test in tests:
            row = old.get(test["test_id"], {})
            if (
                registry.dependencies.get(test["checker"]) is not None
                and not context.get("measurement_inquiries", {}).get(test["test_id"])
                and row.get("status") == "pass"
                and not row.get("error")
                and registry.inputs(test, context, evidence)
                == registry.inputs(test, old_context, old_evidence)
            ):
                reused[test["test_id"]] = {
                    **deepcopy(row),
                    "reused_unchanged_input": True,
                }
    batch = await collect_batch(
        {
            t["test_id"]: lambda t=t: registry.run(t, context, evidence)
            for t in tests
            if t["test_id"] not in reused
        },
        policy or BatchPolicy(),
    )
    results = [
        r.value
        if r.status == "ok"
        else {
            "test_id": r.job_id,
            "criterion_id": next(
                t["criterion_id"] for t in tests if t["test_id"] == r.job_id
            ),
            **unknown("Check timed out or failed", error=True),
        }
        for r in batch.items
    ]
    results.extend(reused.values())
    for row in results:
        inquiry = context.get("measurement_inquiries", {}).get(row["test_id"])
        if inquiry:
            row["measurement_inquiry"] = deepcopy(
                inquiry
            )  # Keep the question even on batch timeout.
    summary = {
        **summarize_results(pack, results),
        "measurement_input_schema": MEASUREMENT_INPUT_SCHEMA,
        "measurement_input_sha256": input_sha256,
        "batch": batch.summary(),
        "execution_summary": {
            "selected_checks": len(tests),
            "submitted_checks": len(tests) - len(reused),
            "reused_checks": len(reused),
            "reported_model_requests": sum(
                r.get("execution_usage", {}).get("model_requests", 0)
                for r in results
                if not r.get("reused_unchanged_input")
            ),
            "reported_tool_calls": sum(
                r.get("execution_usage", {}).get("tool_calls", 0)
                for r in results
                if not r.get("reused_unchanged_input")
            ),
            "boundary": "Returned check receipts only; interrupted calls may be missing. See model request logs for usage.",
        },
    }
    if pack.get("public_delivery_contract"):
        from reward_engine.core.outcome.public_delivery import TEST

        rows = [r for r in results if r["test_id"] == TEST]
        summary["public_delivery_gate"] = (
            deepcopy(rows[0].get("public_delivery_receipt")) if len(rows) == 1 else None
        )
    if pack.get("reward_protocol"):
        from reward_engine.core.outcome.public_reward import score

        summary["independent_reward"] = score(pack, results, context, evidence)
    return summary


def summarize_results(pack, results):
    """Summarize actual receipts, including explicitly bound checkpoint reuse."""
    criteria = []
    for criterion in pack["criteria"]:
        rows = [r for r in results if r["criterion_id"] == criterion["id"]]
        expected = [t for t in pack["tests"] if t["criterion_id"] == criterion["id"]]
        known = (
            bool(rows)
            and len(rows) == len(expected)
            and all(r["status"] != "unknown" for r in rows)
        )
        criteria.append(
            {"id": criterion["id"], "known": known, "kind": criterion["kind"]}
        )
    outcome_coverage = report(pack, results)
    return {
        "pack_id": pack["pack_id"],
        "results": results,
        "criteria": criteria,
        "coverage": sum(c["known"] for c in criteria) / len(criteria)
        if criteria
        else 0,
        "outcome_coverage": outcome_coverage,
        "ready_under_estimator": bool(results)
        and len(results) == len(pack["tests"])
        and all(r["status"] == "pass" for r in results)
        and (not expectations(pack) or complete(outcome_coverage)),
        "interpretation": "Check evidence and gaps only. Coverage is NOT quality. Ranking compares quality; official grader is separate.",
    }
