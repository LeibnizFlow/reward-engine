"""Controller-owned admission of fallible advice; assertions cannot self-authorize."""

from copy import deepcopy

from reward_engine.core.outcome.clauses import bound_premise
from reward_engine.core.outcome.clauses import targets as clause_targets
from reward_engine.core.outcome.construction import fingerprint


def validation_payload(pack):
    """Content identity excludes only controller-added validation annotations."""
    value = deepcopy(pack)
    value.pop("validation", None)
    value.pop("feedback_policy", None)
    for criterion in value["criteria"] + [t.get("rubric", {}) for t in value["tests"]]:
        for key in ("validation_status", "revision_authority", "validation_ids"):
            criterion.pop(key, None)
    for test in value["tests"]:
        test.pop("critical_preservation", None)
        test.pop("validation_status", None)
    return value


def publication_digest(pack):
    value = deepcopy(pack)
    value.get("validation", {}).pop("publication_sha256", None)
    return fingerprint(value)


def require_publication(pack):
    if pack.get("feedback_policy") == "evidence_admitted" and pack.get(
        "validation", {}
    ).get("publication_sha256") != publication_digest(pack):
        raise ValueError(
            "Published Pack content or validation records changed after freezing"
        )


def evidence_index(pack):
    """Small navigation records; case material stays in immutable audit artifacts."""
    return [
        {
            **{
                k: deepcopy(r[k])
                for k in (
                    "validation_id",
                    "target_id",
                    "status",
                    "task_sha256",
                    "target_sha256",
                    "reason",
                    "boundary",
                    "claim_binding",
                )
                if k in r
            },
            "cases_ref": "reference:validation:" + r["validation_id"]
            if r.get("cases")
            else None,
        }
        for r in pack.get("validation", {}).get("records", [])
    ]


INSTRUCTION = (
    "Classify each action as confirmed_defect, outcome_gain, verify or advisory. "
    "For outcome_gain cite an EXACT quote from the original Task, locate the current shortfall "
    "using current_evidence_refs, and explain substantive improvement and how to falsify it. "
    "Pure style preferences or topic/word-count additions are advisory. List EVERY consequential "
    "factual premise your prescribed correction depends on in factual_premises, including numeric "
    "values, hardware, formulas and causal diagnostic rules. Cite existing validation_ids when "
    "available, binding the EXACT target_id, claim text and applicability conditions in claim_binding; "
    "an ID alone cannot certify another assertion. An empty list means unverified. "
    "A concrete outcome_gain may still be tried as a fallible candidate: explicitly state "
    "assumptions, expected_gain and verification, preserve the original and check the result. "
    "Use verification_test_ids only for real published tests. Controller determines admission "
    "after measurements; your category is a proposal, not authority. Aim for full Outcome quality "
    "without declaring a private score or requiring optional preferences."
)


def fields():
    return {
        "kind": "confirmed_defect/outcome_gain/verify/advisory",
        "task_quote": "exact relevant text from the original Task",
        "current_evidence_refs": ["current product refs locating the shortfall"],
        "verification_test_ids": ["published tests, or empty"],
        "factual_premises": [
            {
                "claim": "specific factual dependency, including implied prescriptions",
                "target_id": "exact assessed clause target ID, or empty if unverified",
                "conditions": [
                    "exact assessed applicability conditions, if binding validation"
                ],
                "validation_ids": ["actual supported validation record IDs, or empty"],
            }
        ],
    }


def admit(row, pack, context, evidence, measurement=None):
    """Scope/receipt gate, not a claim that reference matching proves semantics."""
    row = deepcopy(row)
    require_publication(pack)
    bound = pack.get("validation", {}).get("payload_sha256") == fingerprint(
        validation_payload(pack)
    )
    records = {
        r["validation_id"]: r
        for r in pack.get("validation", {}).get("records", [])
        if bound
        and r["status"] == "supported_on_cases"
        and r.get("task_sha256") == fingerprint(context["task"])
    }
    tests = {t["test_id"] for t in pack["tests"]}
    targets = clause_targets(pack["criteria"])
    trial_policy = (
        pack.get("validation", {}).get("decision_policy") == "bounded_candidate"
    )
    trusted_tests = critical_tests(pack)
    results = {r["test_id"]: r for r in (measurement or {}).get("results", [])}
    task = context["task"]["prompt"]
    admitted, deferred = [], []
    for action in row.get("improvements", []):
        kind = action.get("kind", "advisory")
        refs = action.get("current_evidence_refs", [])
        located = (
            bool(refs)
            and isinstance(refs, list)
            and all(
                isinstance(ref, str)
                and ref.startswith("product_1:")
                and ref in evidence
                for ref in refs
            )
        )
        quote = action.get("task_quote", "")
        grounded = isinstance(quote, str) and bool(quote.strip()) and quote in task
        chosen = action.get("verification_test_ids", [])
        valid_tests = isinstance(chosen, list) and all(
            isinstance(t, str) and t in tests for t in chosen
        )
        premises = action.get("factual_premises")
        supported = isinstance(premises, list) and all(
            isinstance(p, dict)
            and isinstance(p.get("claim"), str)
            and p["claim"].strip()
            and isinstance(p.get("validation_ids"), list)
            and p["validation_ids"]
            and all(
                isinstance(v, str)
                and v in records
                and bound_premise(p, records[v], targets)
                for v in p["validation_ids"]
            )
            for p in premises
        )
        failed = valid_tests and any(
            t in trusted_tests
            and t in results
            and results[t]["status"] == "fail"
            and not results[t].get("error")
            for t in chosen
        )
        eligible = (
            located
            and grounded
            and supported
            and valid_tests
            and (kind == "outcome_gain" or kind == "confirmed_defect" and failed)
        )
        # A concrete quality proposal can be TRIED without pretending that all
        # its premises were proved. Missing/forged validation never grants authority.
        trial = bool(
            trial_policy
            and located
            and grounded
            and valid_tests
            and kind == "outcome_gain"
            and all(
                isinstance(action.get(k), str) and action[k].strip()
                for k in ("action", "expected_gain", "verification")
            )
        )
        action["admission"] = (
            "eligible_for_trial"
            if trial_policy and (eligible or trial)
            else "eligible_for_revision"
            if eligible
            else "verification_only"
            if kind != "advisory"
            else "advisory"
        )
        action["unverified_premises"] = deepcopy(premises) if not supported else []
        action["admission_evidence"] = {
            "current_work_located": located,
            "task_quote_matches": grounded,
            "declared_premises_supported_on_cases": supported,
            "known_test_ids": valid_tests,
            "current_defect_measured": failed,
            "boundary": "Necessary evidence bindings; semantic correctness remains model-assisted and rechecked",
        }
        (admitted if eligible or trial else deferred).append(action)
    row["proposed_improvements"] = deepcopy(row.get("improvements", []))
    row["improvements"], row["advisory_improvements"] = admitted, deferred
    row["reported_gaps"] = row.get("gaps", [])
    row["gaps"] = [
        {"gap": r["gap"], "evidence_refs": r.get("evidence_refs", [])}
        for r in results.values()
        if r["test_id"] in trusted_tests
        and r["status"] == "fail"
        and not r.get("error")
        and r.get("gap")
    ]
    if not row.get("error"):
        row["status"] = (
            "needs_improvement"
            if admitted or row["gaps"]
            else "unknown"
            if any(a["admission"] == "verification_only" for a in deferred)
            else "target_satisfied"
        )
    row["admission_policy"] = "evidence_admitted"
    return row


def critical_tests(pack):
    require_publication(pack)
    return {t["test_id"] for t in pack["tests"] if t.get("critical_preservation")}


def preservation(before, after, pack):
    """Compare the SAME definitions; missing/unknown never preserves a critical pass."""
    if before.get("pack_id") != after.get("pack_id"):
        return False, ["pack_changed"]
    protected = {
        r["test_id"]
        for r in before.get("results", [])
        if r["status"] == "pass"
        and not r.get("error")
        and r["test_id"] in critical_tests(pack)
    }
    current = {r["test_id"]: r for r in after.get("results", [])}
    lost = [
        key
        for key in sorted(protected)
        if key not in current
        or current[key]["status"] != "pass"
        or current[key].get("error")
    ]
    prior_failures = {
        r["test_id"]
        for r in before.get("results", [])
        if r["status"] == "fail" and not r.get("error")
    }
    lost += [
        r["test_id"]
        for r in after.get("results", [])
        if r["status"] == "fail"
        and not r.get("error")
        and r["test_id"] in critical_tests(pack)
        and r["test_id"] not in prior_failures
    ]
    return not lost, sorted(set(lost))
