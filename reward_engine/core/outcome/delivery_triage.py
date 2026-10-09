"""One short Outcome review, then only requested measurements. No extra judge hierarchy."""

from copy import deepcopy

from reward_engine.core.batch import BatchPolicy, collect_batch
from reward_engine.core.outcome.construction import fingerprint
from reward_engine.core.outcome.context import (
    EvidenceReadLimit,
    ResponseContractError,
    ask,
    measurement_test_view,
    require_refs,
)
from reward_engine.core.outcome.harness import measure, measurement_context
from reward_engine.core.outcome.hypotheses import question_context
from reward_engine.core.outcome.improvement import INSTRUCTION, schema
from reward_engine.core.outcome.improvement import validate as validate_improvements


class ReviewValidationError(ResponseContractError):
    """Safe schema diagnostics; never include model text or transport secrets."""


def review_refs(refs, evidence, field):
    try:
        return require_refs(refs, evidence)
    except ValueError as error:
        raise ReviewValidationError(
            field, "Evidence references must exist in this context"
        ) from error


def route_claim_checks(row, evidence, tests):
    """Route admitted uncertainty, not decide semantic truth or demand tool activity."""
    claims = row.get("claim_checks")
    if not isinstance(claims, list):
        raise ReviewValidationError(
            "claim_checks", "Declare a claim-check list (empty if not applicable)"
        )
    # Three is prompt guidance, not a measurement budget. Validate every claim;
    # the actual tool budget is enforced on requested tests below.
    for index, item in enumerate(claims):
        field = f"claim_checks[{index}]"
        if not isinstance(item, dict):
            raise ReviewValidationError(field, "Claim check must be an object")
        ref, quote = item.get("claim_ref"), item.get("claim")
        if (
            not isinstance(ref, str)
            or not ref.startswith("product_1:")
            or ref not in evidence
        ):
            raise ReviewValidationError(
                field + ".claim_ref", "Claim must cite the current product"
            )
        if not isinstance(quote, str) or not quote.strip():
            raise ReviewValidationError(
                field + ".claim", "Claim needs a concrete assertion"
            )
        # Paraphrases/translations are normal model output. Source identity is a
        # transport invariant; lexical identity is not a test of semantic truth.
        if item.get("assessment") not in (
            "supported",
            "needs_verification",
            "not_required",
        ):
            raise ReviewValidationError(
                field + ".assessment",
                "Choose supported, needs_verification or not_required",
            )
        if not isinstance(item.get("reason"), str) or not item["reason"].strip():
            raise ReviewValidationError(
                field + ".reason", "Claim needs an evidence assessment reason"
            )
        if not isinstance(item.get("support_refs", []), list):
            raise ReviewValidationError(
                field + ".support_refs", "Supporting references must be a list"
            )
        for support in item.get("support_refs", []):
            review_refs([support], evidence, field + ".support_refs")
        test_id = item.get("test_id")
        if test_id is not None and (
            not isinstance(test_id, str) or test_id not in tests
        ):
            raise ReviewValidationError(
                field + ".test_id", "Claim verification must select an existing test"
            )
        if item["assessment"] != "needs_verification":
            continue
        if test_id:
            if not any(t["test_id"] == test_id for t in row["verify"]):
                row["verify"].append(
                    {"test_id": test_id, "reason": quote + ": " + item["reason"]}
                )
        else:
            if row["status"] == "target_satisfied":
                row["status"] = "unknown"
            row["reason"] += (
                " Unverified claim without a bound test: "
                + quote
                + ": "
                + item["reason"]
            )


def validate_review(row, evidence, tests):
    """Validate evidence and derive the primary verdict from all reported findings."""
    if row.get("status") not in (
        "target_satisfied",
        "needs_improvement",
        "needs_verification",
        "unknown",
    ):
        raise ReviewValidationError(
            "status",
            "Choose target_satisfied, needs_improvement, needs_verification or unknown",
        )
    if not isinstance(row.get("reason"), str) or not row["reason"].strip():
        raise ReviewValidationError("reason", "Need a short Outcome conclusion")
    for field in ("gaps", "verify"):
        if not isinstance(row.get(field), list):
            raise ReviewValidationError(field, "Declare a list, possibly empty")
    for index, gap in enumerate(row["gaps"]):
        field = f"gaps[{index}]"
        if (
            not isinstance(gap, dict)
            or not isinstance(gap.get("gap"), str)
            or not gap["gap"].strip()
        ):
            raise ReviewValidationError(
                field + ".gap", "A defect needs a concrete description"
            )
        refs = review_refs(gap.get("evidence_refs"), evidence, field + ".evidence_refs")
        if not any(r.startswith("product_1:") for r in refs):
            raise ReviewValidationError(
                field + ".evidence_refs", "A defect needs current-product evidence"
            )
    for index, test in enumerate(row["verify"]):
        field = f"verify[{index}]"
        if not isinstance(test, dict):
            raise ReviewValidationError(field, "Verification request must be an object")
        if not isinstance(test.get("test_id"), str) or test["test_id"] not in tests:
            raise ReviewValidationError(field + ".test_id", "Choose an existing test")
        if not isinstance(test.get("reason"), str) or not test["reason"].strip():
            raise ReviewValidationError(
                field + ".reason", "Verification needs a specific doubt"
            )
    route_claim_checks(row, evidence, tests)
    row["factual_assessment"] = (
        "defects_identified" if row["gaps"] else "no_confirmed_defect"
    )
    if row["gaps"] or row.get("full_score_gaps") or row.get("improvements"):
        row["status"] = "needs_improvement"
    elif row["status"] == "needs_improvement":
        raise ReviewValidationError(
            "status", "Needs improvement requires concrete remaining findings"
        )
    elif any(
        c["assessment"] == "needs_verification" and not c.get("test_id")
        for c in row["claim_checks"]
    ):
        row["status"] = "unknown"
    elif row["verify"]:
        row["status"] = "needs_verification"
    elif row["status"] == "needs_verification":
        raise ReviewValidationError(
            "status", "Needs verification requires a concrete verification request"
        )
    return row


REVIEW_INPUT_SCHEMA = "targeted-review-input/v1"


def prepare_review_inputs(pack, context, evidence):
    context, evidence = measurement_context(pack, context, evidence)
    return question_context(
        context,
        evidence,
        {
            "coverage": pack.get("expectation_coverage", []),
            "independent_expectations": pack.get("independent_expectations", []),
            "proposal_expectations": pack.get("proposal_expectations", []),
        },
    )


def review_input_fingerprint(
    pack, context, evidence, *, previous=None, max_tests=3, draft_checks=()
):
    """Bind the overview's actual public inputs and prior state, without inference."""
    context, evidence, pending = prepare_review_inputs(pack, context, evidence)
    return fingerprint(
        {
            "schema": REVIEW_INPUT_SCHEMA,
            "pack": pack,
            "context": context,
            "evidence": evidence,
            "open_questions": pending["open_questions"],
            "previous": previous,
            "max_tests": max_tests,
            "draft_checks": list(draft_checks),
        }
    )


def review_prior_rows(previous):
    return (
        {
            r["test_id"]: r
            for r in (previous[2].get("unresolved_checks", []) + previous[2]["results"])
        }
        if previous
        else {}
    )


def unresolved(row):
    return (
        row["status"] != "pass"
        or row.get("inquiry_result", {}).get("status", "pass") != "pass"
    )


def execution_error(row):
    return row.get("error") or row.get("inquiry_result", {}).get("error")


def measurement_plan(pack, overview_result, *, previous=None, max_tests=3):
    """Reconstruct requested checks/questions from observed review and prior state."""
    from reward_engine.core.outcome import admission

    hardened = pack.get("feedback_policy") == "evidence_admitted"
    prior_rows = review_prior_rows(previous)
    # Retest unresolved checks against the NEW product, never inherit an earlier pass.
    # Confirmed defects remain queued. Execution errors need an explicit new
    # decision, not an automatic retry or an instruction to rewrite the product.
    retry = [
        r["test_id"]
        for r in prior_rows.values()
        if unresolved(r)
        and (
            not execution_error(r)
            or r["status"] == "fail"
            or r.get("inquiry_result", {}).get("status") == "fail"
        )
    ]
    requested = list(
        dict.fromkeys(retry + [t["test_id"] for t in overview_result["verify"]])
    )
    if hardened:
        requested = list(
            dict.fromkeys(
                sorted(admission.critical_tests(pack))
                + requested
                + [
                    t
                    for a in overview_result.get("improvements", [])
                    for t in a.get("verification_test_ids", [])
                    if t in {test["test_id"] for test in pack["tests"]}
                ]
            )
        )
    inquiries = {
        r["test_id"]: deepcopy(r["measurement_inquiry"])
        for r in prior_rows.values()
        if unresolved(r) and r.get("measurement_inquiry")
    }
    for item in overview_result["verify"]:
        inquiry = inquiries.setdefault(
            item["test_id"], {"questions": [], "claim_checks": []}
        )
        if item["reason"] not in inquiry["questions"]:
            inquiry["questions"].append(item["reason"])
        for claim in overview_result.get("claim_checks", []):
            if (
                claim.get("test_id") == item["test_id"]
                and claim not in inquiry["claim_checks"]
            ):
                inquiry["claim_checks"].append(deepcopy(claim))
    return {
        "requested_tests": requested,
        "selected_tests": requested[:max_tests],
        "deferred_tests": requested[max_tests:],
        "measurement_inquiries": inquiries,
    }


async def review(
    pack,
    registry,
    context,
    evidence,
    *,
    previous=None,
    max_tests=3,
    policy=None,
    draft_checks=(),
):
    hardened = pack.get("feedback_policy") == "evidence_admitted"
    from reward_engine.core.outcome import admission

    # Snapshot before any asynchronous role/checker call can mutate its inputs.
    draft_checks = deepcopy(list(draft_checks))
    input_receipt = {
        "schema": REVIEW_INPUT_SCHEMA,
        "sha256": review_input_fingerprint(
            pack,
            context,
            evidence,
            previous=previous,
            max_tests=max_tests,
            draft_checks=draft_checks,
        ),
        "max_tests": max_tests,
        "draft_checks": deepcopy(draft_checks),
        "previous_measurement_sha256": fingerprint(previous[2]) if previous else None,
    }
    context, evidence, pending = prepare_review_inputs(pack, context, evidence)
    prior_rows = review_prior_rows(previous)

    async def overview():
        if registry.review_call is None:
            raise ValueError("Registry has no pre-delivery review transport")
        action_schema = schema()
        if hardened:
            action_schema["improvements"][0].update(admission.fields())
        row = await ask(
            registry.review_call,
            "outcome_delivery_review",
            context,
            evidence,
            (admission.INSTRUCTION if hardened else "")
            + INSTRUCTION
            + "This is the current deliverable before public delivery. Review correctness AND "
            "completeness and correctness together; select the most vulnerable consequential claims. "
            "Do not silently repair a wrong assertion into a correct paraphrase. "
            "A verification plan is not an executed check; a closing disclaimer cannot repair contradictory facts. "
            "Use sufficient existing evidence and never impose a search/tool-call quota. "
            "the path to full quality. Explain how it could earn full Outcome reward. Check vulnerable "
            "consequential claims with supplied evidence/derivations; do not approve a repaired paraphrase "
            "of a wrong assertion. Cite current-product evidence for defects. A source preview is not "
            "its full record. Full-score opportunities may improve depth/completeness/usability even "
            "when no factual defect exists. Preserve already-correct work. Recheck previous findings "
            "and improvements against the new deliverable, including possible regressions. "
            "prior_draft_checks are historical observations, not current passes. A plan to verify is "
            "not executed verification. Select zero to "
            + str(max_tests)
            + " supplied test_ids for "
            "concrete doubts. Unknown/checker errors do not prove a defect. Do not demand empirical "
            "proof when a sufficient derivation answers the Task. Return gaps for supported defects, "
            "verify for measurable uncertainty, and full_score_gaps/improvements for the full-quality path.",
            {
                **action_schema,
                "claim_checks": [
                    {
                        "claim": "Short faithful summary of a current product assertion; zero to three important claims, not a quota",
                        "claim_ref": "product_1 evidence ref containing that assertion",
                        "support_refs": ["available supporting refs, if any"],
                        "assessment": "supported/needs_verification/not_required",
                        "reason": "Actual support/derivation or what is missing; why material or not required by Task",
                        "test_id": "available test id for needed verification, otherwise null",
                    }
                ],
                "reason": "Briefly identify the decisive claim, its actual support or missing evidence, then conclude",
                "status": "needs_improvement if any full-quality gap remains; needs_verification for pending checks; target_satisfied only when the full target is met; unknown if evidence cannot support a conclusion",
                "gaps": [
                    {
                        "gap": "consequential defect",
                        "evidence_refs": ["actual product refs"],
                    }
                ],
                "verify": [{"test_id": "available id", "reason": "specific doubt"}],
            },
            extra={
                "outcome_criteria": [
                    measurement_test_view({"rubric": c})["rubric"]
                    for c in pack["criteria"]
                ],
                **(
                    {"validation_records": admission.evidence_index(pack)}
                    if hardened
                    else {}
                ),
                "open_questions": pending["open_questions"],
                "prior_draft_checks": deepcopy(list(draft_checks)),
                "available_tests": [
                    {
                        k: t[k]
                        for k in (
                            "test_id",
                            "criterion_id",
                            "question",
                            "binding_status",
                        )
                        if k in t
                    }
                    for t in pack["tests"]
                ],
                "previous_findings": {
                    "overview": previous[2].get("delivery_review", {}),
                    "checks": [
                        {
                            k: r[k]
                            for k in (
                                "test_id",
                                "status",
                                "gap",
                                "reason",
                                "error",
                                "diagnostic",
                                "measurement_inquiry",
                                "inquiry_result",
                            )
                            if k in r
                        }
                        for r in prior_rows.values()
                        if r["status"] != "pass"
                        or r.get("inquiry_result", {}).get("status", "pass") != "pass"
                    ],
                }
                if previous
                else {},
            },
            verify_used_previews=True,
            max_evidence_rounds=2,
            close_evidence=True,
            max_contract_retries=1,
            validate=lambda value: validate_improvements(
                validate_review(value, evidence, {t["test_id"] for t in pack["tests"]}),
                evidence,
            ),
        )
        return row

    batch = await collect_batch({"overview": overview}, policy or BatchPolicy())
    overview_item = batch.items[0]
    overview_result = (
        overview_item.value
        if overview_item.status == "ok"
        else {
            "status": "unknown",
            "reason": "Outcome review unavailable; verify evidence, not arbitrary rewrites",
            "gaps": [],
            "verify": [],
            "error": True,
            "error_type": type(overview_item.error).__name__,
            "error_kind": "review_unavailable",
        }
    )
    if isinstance(overview_item.error, ResponseContractError):
        overview_result.update(
            error_kind="invalid_review_output",
            diagnostic=deepcopy(overview_item.error.details),
        )
    if isinstance(overview_item.error, EvidenceReadLimit):
        overview_result.update(
            error_kind="source_read_budget_exhausted",
            unread_evidence_refs=overview_item.error.refs,
            reason="Review still relies on unread sources after the bounded follow-up; not a product defect or API failure",
        )

    overview_before_measurement = deepcopy(overview_result)
    plan = measurement_plan(
        pack, overview_result, previous=previous, max_tests=max_tests
    )
    context = {**context, "measurement_inquiries": plan["measurement_inquiries"]}
    result = await measure(
        pack,
        registry,
        context,
        evidence,
        policy=policy,
        test_ids=plan["selected_tests"],
        previous=previous,
    )
    measured = {r["test_id"] for r in result["results"]}
    pending_checks = [
        deepcopy(r)
        for r in prior_rows.values()
        if unresolved(r) and r["test_id"] not in measured
    ]
    result.update(
        review_mode="targeted",
        deferred_tests=plan["deferred_tests"],
        unresolved_checks=pending_checks,
    )
    result["delivery_clear"] = (
        overview_result["status"] in {"target_satisfied", "needs_verification"}
        and not result["deferred_tests"]
        and not pending_checks
        and not overview_result["gaps"]
        and not overview_result.get("full_score_gaps")
        and not overview_result.get("improvements")
        and not any(unresolved(r) for r in result["results"])
    )
    outstanding = [r for r in result["results"] + pending_checks if unresolved(r)]
    result["review_status_before_measurement"] = overview_result["status"]
    if result["delivery_clear"]:
        overview_result["status"] = "target_satisfied"
    elif any(
        (r["status"] == "fail" and not r.get("error"))
        or (
            r.get("inquiry_result", {}).get("status") == "fail"
            and not r["inquiry_result"].get("error")
        )
        for r in outstanding
    ):
        overview_result["status"] = "needs_improvement"
    elif overview_result["status"] == "target_satisfied":
        overview_result["status"] = "needs_verification"
    result["delivery_review"] = deepcopy(overview_result)
    if hardened:
        overview_result = admission.admit(
            overview_result, pack, context, evidence, result
        )
        result["delivery_review"] = overview_result
        result["delivery_clear"] = (
            not overview_result.get("error")
            and not overview_result["improvements"]
            and not overview_result["gaps"]
            and not result["deferred_tests"]
        )
    result["checks_unavailable_only"] = (
        bool(outstanding)
        and all(
            execution_error(r)
            and r["status"] != "fail"
            and r.get("inquiry_result", {}).get("status") != "fail"
            for r in outstanding
        )
        and not overview_result["gaps"]
        and not result["deferred_tests"]
        and not overview_result.get("improvements")
    )
    result["review_input"] = {
        **input_receipt,
        "overview_before_measurement": overview_before_measurement,
        "measurement_plan": deepcopy(plan),
    }
    result["review_batch"] = batch.summary()
    if isinstance(overview_item.error, ResponseContractError):
        result["review_batch"]["items"][0]["error_details"] = deepcopy(
            overview_item.error.details
        )
    claim_count = len(overview_result.get("claim_checks", []))
    result["review_diagnostics"] = (
        [
            {
                "reason": "claim_count_above_guidance",
                "suggested": 3,
                "received": claim_count,
                "all_claims_preserved": True,
            }
        ]
        if claim_count > 3
        else []
    )
    return result
