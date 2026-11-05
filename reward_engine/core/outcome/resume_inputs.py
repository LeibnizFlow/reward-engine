"""Reuse historical substage results only with exact inputs and complete validation."""

from copy import deepcopy

from reward_engine.core.outcome.construction import fingerprint


def reduction_checkpoint(saved, context, evidence):
    if saved is None:
        return None
    if fingerprint(saved.get("context")) != fingerprint(context) or fingerprint(saved.get("evidence")) != fingerprint(evidence):
        raise ValueError("Historical Reduce checkpoint public inputs changed")
    from reward_engine.core.outcome import jury
    from reward_engine.core.outcome.jury_revision import criterion_body, materialize

    inputs = saved.get("inputs", {})
    members = inputs.get("proposals", [])
    if len(members) != 3:
        return None
    initial = {r["expectation_id"]: r for r in inputs.get("independent_expectations", [])}
    observed = {r["expectation_id"]: r for r in inputs.get("proposal_expectations", [])}
    seen_before, seen_after = set(), set()
    try:
        for member in members:
            before_ids = member["initial_draft"]["criterion_refs"]
            after_ids = member["criterion_refs"]
            if set(before_ids) & seen_before or set(after_ids) & seen_after:
                return None
            seen_before.update(before_ids)
            seen_after.update(after_ids)
            before = [initial[key] for key in before_ids]
            after = [observed[key] for key in after_ids]
            if not before or not after:
                return None
            for rows in (before, after):
                jury.validate({"criteria": [criterion_body(r) for r in rows]}, evidence, context, clause_contract=True)
            proposal = materialize({**member, "criteria": [criterion_body(r) for r in after]}, before, evidence)
            if proposal.get("unclaimed_additions") or any(d.get("code_deferred") for d in proposal["revision_decisions"]):
                return None
    except (ValueError, KeyError, TypeError):
        return None  # An unproven draft must be re-authored, never promoted.
    if seen_before != set(initial) or seen_after != set(observed):
        return None
    return {"context_sha256": fingerprint(context), "evidence_sha256": fingerprint(evidence),
            "jury_reduce": deepcopy(saved)}


def validate_measurement(row, test, evidence, *, policy=None, tools=None):
    """Current validation shared by batch resumes and completed-operation caches."""
    from reward_engine.core.outcome.context import measurement_test_view
    from reward_engine.core.outcome.harness import (
        has_product_evidence,
        unresolved_measurements,
        validate_claims,
        validate_result,
    )
    from reward_engine.core.outcome.measurement_requests import (
        POLICY,
        MeasurementRequestRejected,
        preflight,
    )

    if row.get("error") or row.get("unresolved_evidence_refs") or row.get("unresolved_measurement_refs"):
        raise ValueError("Historical measurement is incomplete")
    if any(row.get(k) != test[k] for k in ("test_id", "checker", "criterion_id")):
        raise ValueError("Historical measurement checker binding changed")
    test = measurement_test_view(test) if policy == POLICY else test
    supplied = {**deepcopy(evidence), test["test_id"]: test}
    if test.get("rubric", {}).get("id"):
        supplied[test["rubric"]["id"]] = test["rubric"]
    receipts = deepcopy(row.get("tool_observations", {}))
    observed = {**supplied, **receipts}
    validate_result(deepcopy(row), observed)
    if row.get("measurement_kind") == "model_assisted":
        validate_claims(deepcopy(row), observed)
    if policy == POLICY:
        if row["status"] != "unknown" and not has_product_evidence(row["evidence_refs"], evidence, receipts):
            raise ValueError("Historical known result lacks observed product evidence")
        if row.get("measurement_inquiry"):
            inquiry = row.get("inquiry_result")
            validate_result(deepcopy(inquiry), observed)
            if inquiry["status"] != "unknown" and not has_product_evidence(inquiry["evidence_refs"], evidence, receipts):
                raise ValueError("Historical inquiry lacks observed product evidence")
        # Reconstruct proof from checkpoint-bound inputs, including the exact
        # standard view installed by Registry; never rewrite historical bytes.
        for ref, receipt in receipts.items():
            if tools is not None:
                tools.validate_historical(receipt, supplied)
            # A stronger admission check cannot rewrite a recorded execution
            # failure into a preflight rejection or erase its later recovery.
            # Still reconstruct exact selected inputs and require successful
            # same-input replacement receipts below. Observed outputs always
            # face the full current program check.
            failed_execution = (receipt.get("status") == "unknown"
                                and receipt.get("request_rejected") is False
                                and receipt.get("execution_phase") == "execution"
                                and receipt.get("contract_policy") == POLICY
                                and "input_bindings" in receipt)
            try:
                bindings = (tools.validate_request(receipt, supplied,
                            require_available=bool(receipt.get("request_rejected")),
                            check_program=not failed_execution)
                            if tools is not None else preflight(receipt, supplied,
                                                               check_program=not failed_execution))
            except MeasurementRequestRejected:
                if not receipt.get("request_rejected"):
                    raise ValueError("Historical executed result fails current preflight") from None
            else:
                if receipt.get("request_rejected"):
                    capability = tools.catalog_for(supplied).get(receipt.get("tool"), {}) if tools else {}
                    if capability.get("available") is not False:
                        raise ValueError("Historical rejection is not proven by current preflight")
                if "input_bindings" in receipt and receipt["input_bindings"] != bindings:
                    raise ValueError("Historical selected inputs changed")
                receipt.update(input_bindings=bindings, contract_policy=POLICY)
            supplied[ref] = row["tool_observations"][ref]
    if unresolved_measurements(row, receipts):
        raise ValueError("Historical execution recovery is not proven")
    return deepcopy(row)


def measurement_results(saved, input_sha256, pack, tests, evidence, *, policy=None, tools=None):
    """A failed batch can still contain independently completed measurements."""
    if saved is None or "measurement_input_sha256" not in saved:
        return {}
    if saved.get("measurement_input_sha256") != input_sha256 or saved.get("pack_id") != pack["pack_id"]:
        raise ValueError("Historical measurement checkpoint inputs changed")
    catalog = {t["test_id"]: t for t in tests}
    seen, reusable = set(), {}
    for row in saved.get("results", []):
        key = row.get("test_id")
        if key not in catalog or key in seen:
            raise ValueError("Historical measurement has unknown or repeated test")
        seen.add(key)
        if row.get("error") or row.get("unresolved_evidence_refs") or row.get("unresolved_measurement_refs"):
            continue
        if row.get("checker") != catalog[key]["checker"] or row.get("criterion_id") != catalog[key]["criterion_id"]:
            raise ValueError("Historical measurement checker binding changed")
        try:
            validate_measurement(row, catalog[key], evidence, policy=policy, tools=tools)
        except (ValueError, KeyError, TypeError, AttributeError):
            continue
        reusable[key] = {**deepcopy(row), "reused_completed_measurement": True,
                         "source_measurement_sha256": fingerprint(saved)}
    return reusable
