"""Harness claims for the existing public entry point."""

from copy import deepcopy

from reward_engine.core.outcome.context import (
    ResponseContractError,
    require_refs,
)


def validate_result(row, evidence):
    if row.get("status") not in {"pass", "fail", "unknown"} or not row.get("reason"):
        field = (
            "status"
            if row.get("status") not in {"pass", "fail", "unknown"}
            else "reason"
        )
        error = ResponseContractError(
            field, "Check needs a typed status and explanation"
        )
        error.details.update(
            actual=deepcopy(row.get(field)),
            missing=field not in row,
            repair_instruction="Explicitly author the missing or invalid summary from the preserved claims "
            "and supplied evidence. A status correction must include consistent status, reason and gap. "
            "Preserve every claim and its unknown extension fields, citations, judgment and relevance; "
            "do not regenerate claims or request new evidence/tools. No host-default verdict is supplied.",
        )
        if field == "status":
            error.details["allowed_values"] = ["pass", "fail", "unknown"]
        raise error
    if "score" in row:
        raise ValueError(
            "Outcome checks return facts/defects, not absolute quality grades"
        )
    if row.get("error") and row["status"] != "unknown":
        raise ValueError("An execution error is not a product pass or fail")
    if row.get("gap") is None and row["status"] != "fail":
        row = {**row, "gap": ""}
    if not isinstance(row.get("gap", ""), str):
        raise ValueError("Check gap must be text; failure requires a concrete gap")
    if row["status"] != "unknown":
        try:
            require_refs(row.get("evidence_refs"), evidence)
        except ValueError as ref_error:
            error = ResponseContractError(
                "evidence_refs",
                "Top-level evidence_refs of this verdict: " + str(ref_error),
            )
            error.details.update(
                actual=deepcopy(row.get("evidence_refs")),
                missing="evidence_refs" not in row,
                repair_instruction=(
                    "Repair only the invalid top-level evidence_refs field of this verdict "
                    "with a nonempty list of exact supplied evidence handles supporting the "
                    "existing assessment. Claim-level evidence_refs do not replace this root field. "
                    "Preserve the claims, judgments, relevance, valid claim citations and summary; "
                    "do not regenerate claims, invent handles/proof or change the verdict. "
                    "The model must author the citations using the same evidence; the host does not fill them."
                ),
            )
            raise error from ref_error
        if row["status"] == "fail" and not row.get("gap"):
            raise ValueError("Failure needs an actionable gap")
    return row


def has_product_evidence(refs, evidence, receipts):
    """Follow only trusted, executed read receipts to their product sources.

    This establishes observation, not correctness or complete-file coverage.
    Caller-supplied model JSON must never populate ``receipts``.
    """
    for ref in refs:
        if ref in evidence and ref.startswith("product_"):
            return True
        receipt = receipts.get(ref, {})
        if (
            receipt.get("tool") != "workspace.read"
            or receipt.get("status") != "observed"
        ):
            continue
        result = receipt.get("result", {})
        if result.get("text") and any(
            source in evidence and source.startswith("product_")
            for source in result.get("observed_product_refs", [])
        ):
            return True
    return False


def claim_contract_issues(row, evidence):
    """Collect independent claim contract errors without changing the assessment."""
    issues = []

    def reject(field, reason, **details):
        issues.append({"field": field, "reason": reason, **details})

    claims = row.get("claims")
    if not isinstance(claims, list) or (row.get("status") != "unknown" and not claims):
        return [
            {
                "field": "claims",
                "reason": "Semantic verdict needs a claim/evidence audit",
            }
        ]
    material_claims = 0
    for index, claim in enumerate(claims):
        if not isinstance(claim, dict):
            reject(
                f"claims[{index}]", "Each claim must be an object", claim_index=index
            )
            continue

        def invalid(field, reason, *, index=index, claim=claim):
            reject(
                f"claims[{index}].{field}",
                reason,
                claim_index=index,
                claim_field=field,
                actual=deepcopy(claim.get(field)),
                missing=field not in claim,
            )

        def refs_valid(field, *, claim=claim, invalid=invalid):
            try:
                require_refs(claim.get(field), evidence)
            except ValueError as ref_error:
                invalid(field, str(ref_error))

        relevance = claim.get("verdict_relevance", "material")
        if relevance not in ("material", "advisory"):
            invalid(
                "verdict_relevance",
                "Claim verdict_relevance must be material or advisory",
            )
        if relevance == "advisory" and not (
            isinstance(claim.get("relevance_reason"), str)
            and claim["relevance_reason"].strip()
        ):
            invalid(
                "relevance_reason",
                "Advisory findings need a reason tied to the supplied standard",
            )
        material_claims += relevance == "material"
        for field in ("claim", "observation"):
            if not claim.get(field):
                invalid(field, "Each claim needs its observed support or gap")
        refs_valid("claim_refs")
        refs = claim.get("evidence_refs")
        if not isinstance(refs, list):
            invalid(
                "evidence_refs",
                "Declare evidence refs, even when unavailable: supply a list of exact observed handles. "
                "Supported/contradicted needs nonempty observed evidence; only unverified may use []. "
                "Do not invent refs or change the existing judgment to satisfy the schema.",
            )
        elif refs:
            refs_valid("evidence_refs")
        if claim.get("judgment") not in ("supported", "contradicted", "unverified"):
            invalid("judgment", "Invalid claim judgment")
        if claim.get("judgment") in ("supported", "contradicted") and refs == []:
            field = f"claims[{index}].evidence_refs"
            error = ResponseContractError(
                field,
                f"{field}: Supported/contradicted needs observed evidence; "
                f"judgment={claim['judgment']!r}, actual=[]. Cite nonempty exact "
                "supplied evidence handles that support this claim's existing judgment. "
                "Do not invent refs, copy claim_refs automatically, or flip the "
                "judgment merely to satisfy validation. Empty refs are allowed only for unverified claims.",
            )
            error.details.update(
                claim_index=index,
                judgment=claim["judgment"],
                actual=deepcopy(refs),
                claim_field="evidence_refs",
            )
            issues.append(error.details)
        if (
            row.get("status") == "pass"
            and relevance == "material"
            and claim.get("judgment") in ("unverified", "contradicted")
        ):
            error = ResponseContractError(
                "status",
                f"status='pass' conflicts with claims[{index}].judgment="
                f"{claim['judgment']!r} and claims[{index}].verdict_relevance='material'. "
                "Pass requires ALL material claims to be supported. "
                "Unresolved material claims cannot silently become pass.",
            )
            error.details.update(
                claim_index=index,
                actual=row["status"],
                depends_on={
                    f"claims[{index}].judgment": claim["judgment"],
                    f"claims[{index}].verdict_relevance": relevance,
                },
                repair_instruction=(
                    "For this status/claim contradiction only, minimally correct "
                    "status/reason/gap to be consistent with the preserved claim assessment "
                    "using the same evidence. This contract consistency correction is permitted; "
                    "the model must author the verdict, not the host. Preserve claims, "
                    "judgments, relevance and evidence refs except citation fields separately "
                    "reported invalid. Do not upgrade an unsupported "
                    "claim to supported, downgrade material to advisory, delete a claim, "
                    "invent proof or request new evidence to make pass."
                ),
            )
            issues.append(error.details)
    if row.get("status") == "pass" and not material_claims:
        reject(
            "status",
            "A pass needs supported material claims, not only advisory findings",
            actual=row["status"],
            material_claim_count=0,
            depends_on={
                f"claims[{index}].verdict_relevance": claim.get(
                    "verdict_relevance", "material"
                )
                for index, claim in enumerate(claims)
                if isinstance(claim, dict)
            },
            repair_instruction=(
                "For this all-advisory status contradiction only, minimally correct "
                "status/reason/gap to be consistent with the preserved claim assessment "
                "using the same evidence. The model must author that summary. Preserve claims, "
                "judgments, relevance and valid evidence refs; do not promote advisory claims "
                "to material, add/delete claims or invent support to force pass. "
                "Only other fields separately reported invalid may also be repaired."
            ),
        )
    return issues


def validate_claims(row, evidence):
    """Validate an inspectable audit, not the truth of the model's reasoning."""
    issues = claim_contract_issues(row, evidence)
    if issues:
        error = ResponseContractError(
            issues[0]["field"],
            "; ".join(f"{issue['field']}: {issue['reason']}" for issue in issues),
        )
        if len(issues) == 1:
            error.details = deepcopy(issues[0])
        else:
            error.details["issues"] = issues
        raise error
    return row


def preserve_claim_assessment(original, issues, *, current):
    """A same-evidence repair may edit invalid fields, not reroll valid claims."""
    claims = original.get("claims")
    if not isinstance(claims, list) or (
        not claims and any(issue["field"] == "claims" for issue in issues)
    ):
        return  # There was no usable claim structure to preserve.
    revised = current.get("claims")
    if not isinstance(revised, list) or len(revised) != len(claims):
        raise ResponseContractError(
            "claims", "Structure repair must preserve claim count and order"
        )
    for index, claim in enumerate(claims):
        if not isinstance(claim, dict):
            continue  # Only this malformed row may be replaced.
        allowed = {
            issue["claim_field"]
            for issue in issues
            if issue.get("claim_index") == index and "claim_field" in issue
        }
        fixed = revised[index]
        if not isinstance(fixed, dict) or {
            k: v for k, v in claim.items() if k not in allowed
        } != {k: v for k, v in fixed.items() if k not in allowed}:
            raise ResponseContractError(
                f"claims[{index}]",
                f"Structure repair must preserve already valid fields of claims[{index}]; "
                f"only these invalid fields may change: {sorted(allowed)}",
            )
    if (
        original.get("status") in ("pass", "fail", "unknown")
        and not any(issue["field"] == "status" for issue in issues)
        and current.get("status") != original["status"]
    ):
        raise ResponseContractError(
            "status", "Structure repair must preserve the already consistent verdict"
        )


def field_correction_targets(row, issues):
    """Finite host-identified targets; model strings never drive path traversal."""
    targets = {}
    for issue in issues:
        field = issue["field"]
        if field == "status":
            for name in ("status", "reason", "gap"):
                targets[name] = (name,)
        elif field in {"evidence_refs", "reason"}:
            targets[field] = (field,)
        else:
            index, name = issue.get("claim_index"), issue.get("claim_field")
            claims = row.get("claims")
            if not (
                type(index) is int
                and isinstance(claims, list)
                and 0 <= index < len(claims)
                and isinstance(claims[index], dict)
                and name in ("claim_refs", "evidence_refs", "judgment")
                and field == f"claims[{index}].{name}"
            ):
                return {}  # Other repairs continue to require the full reply.
            targets[field] = ("claims", index, name)
    return targets


def compose_field_correction(raw, state, evidence, attempt):
    """Compose only explicit model replacements, retaining their source attempts."""
    if state is None or set(raw) != {"field_corrections"}:
        raise ResponseContractError(
            "field_corrections",
            "Field corrections require an advertised active repair and no sibling fields",
        )
    values = raw["field_corrections"]
    targets = state["targets"]
    if not isinstance(values, dict) or set(values) != set(targets):
        raise ResponseContractError(
            "field_corrections",
            f"Provide exactly the advertised required fields: {sorted(targets)}; no extra fields or paths",
        )
    # A correction cannot exploit unknown's optional root citations or gap
    # normalization. Every replacement must be explicitly authored and valid.
    for field, target in targets.items():
        value = values[field]
        if target[-1] in ("evidence_refs", "claim_refs"):
            if (
                target != ("evidence_refs",)
                and target[-1] == "evidence_refs"
                and value == []
            ):
                continue  # The complete claim validator checks the judgment.
            require_refs(value, evidence)
        elif target[-1] == "judgment":
            if value not in ("supported", "contradicted", "unverified"):
                raise ResponseContractError(
                    field,
                    "Judgment correction requires an explicit supported/contradicted/unverified value",
                )
        elif not isinstance(value, str) or (field != "gap" and not value.strip()):
            raise ResponseContractError(
                field, "Summary corrections require explicit text values"
            )
    composed = deepcopy(state["base"])
    fields = {
        key: {
            "source_attempt": state["attempt"],
            "source_field": key,
            "kind": "preserved",
        }
        for key in composed
        if key != "claims"
    }
    for index, claim in enumerate(composed.get("claims", [])):
        if isinstance(claim, dict):
            for name in claim:
                path = f"claims[{index}].{name}"
                fields[path] = {
                    "source_attempt": state["attempt"],
                    "source_field": path,
                    "kind": "preserved",
                }
    for field, target in targets.items():
        if len(target) == 1:
            composed[target[0]] = deepcopy(values[field])
        else:
            composed["claims"][target[1]][target[2]] = deepcopy(values[field])
        fields[field] = {
            "source_attempt": attempt,
            "source_field": f"field_corrections.{field}",
            "kind": "replacement",
        }
    return composed, {
        "base_attempt": state["attempt"],
        "correction_attempt": attempt,
        "claim_order_source_attempt": state["attempt"],
        "fields": fields,
    }


def compose_response_unwrap(raw, state, attempt):
    """Relocate an explicitly selected, unchanged model assessment; infer nothing."""
    if (
        state is None
        or state.get("expected") is None
        or not state.get("offer")
        or raw != {"response_unwrap": "response"}
    ):
        raise ResponseContractError(
            "response_unwrap",
            "Use only the advertised response_unwrap operation, without replacement values or siblings",
        )
    composed = deepcopy(state["expected"])
    fields = {}
    for key, value in composed.items():
        if key == "claims":
            paths = [
                f"claims[{i}].{field}"
                for i, claim in enumerate(value)
                for field in claim
            ]
        else:
            paths = [key]
        for path in paths:
            nested = key != "inquiry_result"
            fields[path] = {
                "source_attempt": state["attempt"],
                "source_field": ("response." if nested else "") + path,
                "kind": "relocated" if nested else "preserved",
            }
    return composed, {
        "base_attempt": state["attempt"],
        "claim_order_source_attempt": state["attempt"],
        "operation": {
            "source_attempt": attempt,
            "source_field": "response_unwrap",
            "value": "response",
        },
        "fields": fields,
    }


def claim_locations(row, evidence):
    """Separate a known source handle from an unverified human-readable locator.

    This resolves document identity, never the truth/existence of the quoted
    section. Unknown documents and ambiguous parent handles still fail validation.
    """
    row = deepcopy(row)
    for claim in row.get("claims", []):
        refs = claim.get("claim_refs")
        if not isinstance(refs, list):
            continue
        resolved, locations = [], []
        for ref in refs:
            matches = [
                key
                for key in evidence
                if isinstance(ref, str)
                and ref not in evidence
                and any(ref.startswith(key + sep) for sep in (": ", " (", "#"))
            ]
            if len(matches) == 1:
                parent = matches[0]
                locations.append(
                    {
                        "original_ref": ref,
                        "source_ref": parent,
                        "annotation": ref[len(parent) :],
                        "location_verified": False,
                    }
                )
                ref = parent
            resolved.append(ref)
        claim["claim_refs"] = resolved
        if locations:
            claim["location_resolution"] = locations
    return row


def unresolved_measurements(result, receipts):
    """A failed attempt can be superseded only by later executed, cited evidence.

    This validates provenance/coverage, not the model's semantic interpretation.
    All failed receipts remain in the audit; no arbitrary waiver is accepted.
    A request rejected before execution (invalid refs/paths or oversized input)
    is a correctable request error, not an unavailable measurement, so it does
    not by itself force the check to unknown.
    """
    failed = {
        ref
        for ref, value in receipts.items()
        if value["status"] != "observed" and not value.get("request_rejected")
    }
    cited = set(result.get("evidence_refs", []))
    cited.update(
        ref
        for claim in result.get("claims", [])
        for ref in claim.get("evidence_refs", [])
    )
    order = {ref: index for index, ref in enumerate(receipts)}
    recoveries = result.get("measurement_recovery", [])
    if not isinstance(recoveries, list):
        return failed
    for recovery in recoveries:
        if not isinstance(recovery, dict):
            continue
        ref, replacements = recovery.get("failed_ref"), recovery.get("replacement_refs")
        if (
            not isinstance(ref, str)
            or ref not in failed
            or not recovery.get("reason")
            or not isinstance(replacements, list)
            or not replacements
            or any(not isinstance(key, str) for key in replacements)
        ):
            continue
        old = receipts[ref]
        if not isinstance(old.get("arguments"), dict):
            continue
        if any(
            key not in receipts
            or key not in cited
            or receipts[key]["status"] != "observed"
            or receipts[key]["tool"] != old["tool"]
            for key in replacements
        ):
            continue
        # A cited earlier success may accompany the actual recovery. It cannot
        # supply recovery coverage, but must not invalidate later successful
        # probes that independently cover the failed operation's inputs.
        replacements = [key for key in replacements if order[key] > order[ref]]
        if not replacements:
            continue
        from reward_engine.core.outcome.measurement_requests import POLICY, comparable

        if old.get("contract_policy") == POLICY or any(
            receipts[k].get("contract_policy") == POLICY for k in replacements
        ):
            if comparable(old, [receipts[k] for k in replacements]):
                failed.remove(ref)
            continue
        # Exact retry is always comparable. Revised numerical programs may
        # split a large input into several probes over the same cited sources.
        exact = any(
            receipts[key]["arguments"] == old["arguments"] for key in replacements
        )
        raw_refs = old["arguments"].get("input_refs", [])
        if not isinstance(raw_refs, list) or any(
            not isinstance(source, str) for source in raw_refs
        ):
            continue
        old_refs = set(raw_refs)
        new_refs = {
            source
            for key in replacements
            for source in receipts[key]["arguments"].get("input_refs", [])
        }
        if exact or old["tool"] == "code.run" and old_refs and old_refs <= new_refs:
            failed.remove(ref)
    return failed
