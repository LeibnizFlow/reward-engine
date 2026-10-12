"""Open checker registry. Keys are extensible; execution is real registered code."""

import json
from copy import deepcopy
from time import monotonic

from reward_engine.core.batch import BatchPolicy, collect_batch
from reward_engine.core.outcome.construction import fingerprint
from reward_engine.core.outcome.context import (
    EvidenceReadLimit,
    ResponseContractError,
    ask,
    measurement_prior_view,
    measurement_test_view,
    require_refs,
    response_payload,
    used_previews,
    validate_current_product_refs,
)
from reward_engine.core.outcome.coverage import complete, expectations, report

MEASUREMENT_INPUT_SCHEMA = "outcome-measurement-input/v1"


def unknown(reason, *, error=False):
    return {
        "status": "unknown",
        "reason": reason,
        "evidence_refs": [],
        "error": error,
        "gap": "",
    }


# Provider/SDK transport, timeout and budget failures are physical: the check
# never produced a semantic verdict. They must be retried or retained as
# unknown, never treated as evidence of a defective standard or checker.
INFRASTRUCTURE_FAILURES = frozenset(
    {
        "ReadTimeout",
        "ConnectTimeout",
        "WriteTimeout",
        "PoolTimeout",
        "TimeoutException",
        "ReadError",
        "ConnectError",
        "RemoteProtocolError",
        "TransportError",
        "ProviderStreamError",
        "APITimeoutError",
        "APIConnectionError",
        "InternalServerError",
        "RateLimitError",
        "HTTPStatusError",
        "TimeoutError",
        "FrameworkError",
        "BudgetExhausted",
        "RequestBudgetExceeded",
    }
)


def failure_layer(error):
    """Separate a physical/infrastructure failure from an evaluator defect."""
    return (
        "infrastructure"
        if type(error).__name__ in INFRASTRUCTURE_FAILURES
        else "evaluation_execution"
    )


async def smoke(pack, registry):
    """Zero-model infrastructure probe; deliberately NOT a product-quality verdict."""
    from reward_engine.core.outcome.context import build_context

    bindings = {t["test_id"]: t.get("binding_status", "unknown") for t in pack["tests"]}
    receipt = None
    if "json.parse" in registry.checkers:
        context, evidence = build_context(
            {"task_id": "infra-smoke", "prompt": "Parse fixture"},
            [{"artifact": {"final_response": "{}"}}],
        )
        receipt = await registry.run(
            {
                "test_id": "infra-parser",
                "criterion_id": "infra",
                "checker": "json.parse",
                "arguments": {"ref": "product_1:artifact:final_response"},
            },
            context,
            evidence,
        )
    return {
        "bindings": bindings,
        "receipt": receipt,
        "status": "receipt_path_checked"
        if receipt and receipt["status"] == "pass"
        else "binding_only",
        "boundary": "Synthetic parser receipt only; per-check semantic quality and GPU execution NOT validated.",
    }


def validate_result(row, evidence):
    if row.get("status") not in {"pass", "fail", "unknown"} or not row.get("reason"):
        raise ValueError("Check needs a typed status and explanation")
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
        elif field == "evidence_refs":
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


class Registry:
    def __init__(self):
        self.checkers = {}
        self.kinds = {}
        self.dependencies = {}
        self.recipes = {}
        self.tools = None
        self.review_call = None

    def bind_recipe(self, key, recipe):
        validate_current_product_refs(recipe)
        if key in self.checkers and self.recipes.get(key) != recipe:
            raise ValueError("Recipe cannot replace an existing checker")
        if key not in self.checkers:
            self.register(
                key,
                self.semantic,
                "Tool-assisted generated measurement recipe",
                kind="model_assisted",
            )
        self.recipes[key] = deepcopy(recipe)

    def register(
        self, key, run, description, *, kind="deterministic", dependencies=None
    ):
        if (
            not isinstance(key, str)
            or not key.strip()
            or not callable(run)
            or not description
        ):
            raise ValueError(
                "Checker requires key, executable async function and description"
            )
        if key in self.checkers:
            raise ValueError("Do not silently replace an existing checker")
        if kind not in {"deterministic", "model_assisted"}:
            raise ValueError("Declare whether the check depends on model judgment")
        if dependencies is not None and (
            kind != "deterministic" or not callable(dependencies)
        ):
            raise ValueError(
                "Reuse dependencies must be trusted deterministic checker code"
            )
        self.checkers[key] = (run, description)
        self.kinds[key] = kind
        self.dependencies[key] = dependencies

    def inputs(self, test, context, evidence):
        dependency = self.dependencies.get(test["checker"])
        if dependency is None:
            return {"context": context, "evidence": evidence}
        return {
            "task": context["task"],
            "users": [p["user_updates"] for p in context["products"]],
            "input": dependency(test, context, evidence),
        }

    def catalog(self):
        return {key: description for key, (_, description) in self.checkers.items()}

    async def run(self, test, context, evidence):
        started = monotonic()
        test = measurement_test_view(test)
        key = test["checker"]
        inquiry = deepcopy(
            context.get("measurement_inquiries", {}).get(test["test_id"])
        )
        if key not in self.checkers:
            return {
                "test_id": test["test_id"],
                "criterion_id": test["criterion_id"],
                **unknown(f"Checker {key} needs implementation", error=True),
                **({"measurement_inquiry": inquiry} if inquiry else {}),
            }
        try:
            validate_current_product_refs(
                {k: test.get(k) for k in ("arguments", "probes", "execution_recipe")}
            )
            observed = evidence
            evidence, context = deepcopy(evidence), deepcopy(context)
            context.pop("measurement_inquiries", None)
            if inquiry:
                context["measurement_inquiry"] = inquiry
            # Supplied standards may be cited, but never count as observed product facts.
            rubric = test.get("rubric", {})
            for ref, value in ((test.get("test_id"), test), (rubric.get("id"), rubric)):
                if ref:
                    if ref in evidence:
                        raise ValueError(
                            "Check/criterion ID collides with an evidence handle"
                        )
                    evidence[ref] = deepcopy(value)
            context["available_evidence_refs"] = list(evidence)
            result = await self.checkers[key][0](
                deepcopy(test), deepcopy(context), deepcopy(evidence)
            )
            # Only trusted registered checker code may introduce measurement receipts.
            receipts = result.pop("_tool_evidence", {})
            evidence.update(receipts)
            result = validate_result(result, evidence)
            if result["status"] != "unknown" and not has_product_evidence(
                result["evidence_refs"], observed, receipts
            ):
                raise ValueError(
                    "Known check results need observed product evidence, not just the Task"
                )
        except Exception as error:
            result = unknown(f"Checker failed: {type(error).__name__}", error=True)
            result["diagnostic"] = str(error)[:400]
            result["failure_layer"] = failure_layer(error)
        return {
            **result,
            "test_id": test["test_id"],
            "criterion_id": test["criterion_id"],
            "checker": key,
            "measurement_kind": self.kinds[key],
            "elapsed_s": monotonic() - started,
            **({"measurement_inquiry": inquiry} if inquiry else {}),
        }


def defaults(semantic_call, *, tools=None):
    from reward_engine.core.outcome.measurement_tools import MeasurementTools

    registry = Registry()
    registry.review_call = semantic_call
    registry.tools = tools or MeasurementTools()

    async def semantic(test, context, evidence):
        from reward_engine.core.outcome.clauses import INSTRUCTION as CLAUSE_INSTRUCTION

        context = measurement_prior_view(context, evidence, test)
        receipts = {}
        model_requests = 0
        validation_attempts, validated_parts, selected_attempts = [], {}, {}
        requested_inquiry = bool(context.get("measurement_inquiry"))
        structure_repair = False
        claim_repair = None
        correction_state = None
        unwrap_state = None
        request_context, attached_refs = context, set()
        pending_evidence, read_failure = {}, {}

        def charged_tools():
            # A probe rejected before execution (bad refs/paths, oversized input)
            # is correctable feedback, not an executed measurement, so it does
            # not consume the four-execution budget.
            return sum(
                1 for row in receipts.values() if not row.get("request_rejected")
            )

        async def execute(probes):
            if (
                not isinstance(probes, list)
                or not 1 <= len(probes) <= 3
                or charged_tools() + len(probes) > 4
                or len(receipts) + len(probes) > 8
            ):
                raise ValueError(
                    "Use at most four executed bounded measurements per check"
                )
            for probe in probes:
                if not isinstance(probe, dict):
                    raise ValueError("Malformed tool request")
                ref = f"tool:{test['test_id']}:{len(receipts)}"
                if ref in evidence:
                    raise ValueError("Measurement receipt identity collision")
                receipt = await registry.tools.run(probe, evidence)
                receipts[ref] = receipt
                evidence[ref] = receipt

        recipe = test.get("execution_recipe", {})
        if recipe.get("probes"):
            await execute(recipe["probes"])

        async def tool_call(request):
            nonlocal \
                model_requests, \
                structure_repair, \
                request_context, \
                attached_refs, \
                claim_repair, \
                correction_state, \
                unwrap_state
            # Syntax recovery replaces response_recovery details. It cannot
            # reopen the tools/reads of an already active same-evidence repair.
            structure_repair = structure_repair or bool(
                request.get("response_recovery", {})
                .get("details", {})
                .get("invalid_fields")
            )
            if request.get("evidence_followup") and not request.get(
                "response_recovery"
            ):
                newly_attached = (
                    set(request.get("requested_evidence", {})) - attached_refs
                )
                if used_previews(
                    {"evidence_refs": list(newly_attached)}, request_context
                ):
                    # Only an actual expansion beyond the preceding view reopens
                    # assessment. Duplicate/inline reads and budget closure do not.
                    validated_parts.clear()
                    selected_attempts.clear()
                    structure_repair = False
                    claim_repair = None
                    correction_state = None
                    unwrap_state = None
                    request["evidence_followup"] += (
                        " After assessing the newly supplied evidence, return a full response whose "
                        "status/reason/gap agree with its structured claims. Omitted verdict_relevance "
                        "means material; calling a claim nonmaterial only in prose does not change "
                        "that field. Any advisory assessment must explicitly declare "
                        "verdict_relevance=advisory and a model-authored relevance_reason tied to "
                        "THIS criterion. Do not change relevance just to obtain pass. New evidence "
                        "may legitimately change judgment or relevance when supported; the prior "
                        "provisional verdict is not fixed. Pass still requires all material claims "
                        "to be supported. Keep the global verdict and any inquiry_result independent, "
                        "each consistent with its own assessment."
                    )
            if structure_repair and request.get("response_recovery"):
                # ask retains the previous preview expansion in its outer request.
                # This emitted retry repairs the latest rejected assessment, not
                # that older provisional one. Keep the actual attached evidence;
                # a subsequent genuine expansion still uses ask's original state.
                request.pop("evidence_followup", None)
                request.pop("provisional_check", None)
            request.pop("field_correction_reply", None)
            request.pop("response_unwrap_reply", None)
            if (
                unwrap_state is not None
                and unwrap_state.get("offer")
                and structure_repair
            ):
                request["response_unwrap_reply"] = deepcopy(unwrap_state["offer"])
                if request.get("response_recovery"):
                    request["response_recovery"]["instruction"] = (
                        'Preferred envelope repair: return ONLY {"response_unwrap":"response"} '
                        "to explicitly move the preserved assessment to the root without changing any value. "
                        + request["response_recovery"]["instruction"]
                    )
            if correction_state is not None and structure_repair:
                # Also survives syntax recovery replacing response_recovery.
                request["field_correction_reply"] = deepcopy(correction_state["offer"])
                if request.get("response_recovery"):
                    request["response_recovery"]["instruction"] = (
                        "Preferred repair reply: return ONLY field_corrections with every "
                        "field_correction_reply.required_fields entry and model-authored values. "
                        "Do not regenerate unchanged claims. The original full-response schema "
                        "is a compatibility alternative under the same strict preservation checks. "
                        + request["response_recovery"]["instruction"]
                    )
            request_context = request["context"]
            attached_refs = set(request.get("requested_evidence", {}))
            # Two measurement waves at most, then a final evidence-grounded verdict.
            # A failed or rejected probe needs one correction wave before the
            # final verdict; capping at one wave would drop that recovery.
            for wave in range(3):
                request["context"]["available_evidence_refs"] = list(evidence)
                request["context"]["tool_observations"] = deepcopy(receipts)
                request["measurement_budget"] = {
                    "remaining_tool_calls": max(0, 4 - charged_tools()),
                    "max_calls_per_wave": min(3, max(0, 4 - charged_tools())),
                    "remaining_tool_waves": 2 - wave,
                    "final_verdict_required": wave == 2 or charged_tools() == 4,
                    "instruction": "Reuse recorded observations. Do not repeat completed reads/calculations. "
                    "A request rejected before execution (invalid refs/paths, oversized input) is "
                    "correctable and does not consume the executed budget; fix the cited refs/paths and "
                    "retry within the cap. When no tool budget remains, give the required final verdict "
                    "from existing evidence; return unknown for any material fact that still needs an "
                    "unperformed measurement.",
                }
                if structure_repair:
                    request["measurement_budget"].update(
                        remaining_tool_calls=0,
                        max_calls_per_wave=0,
                        remaining_tool_waves=0,
                        final_verdict_required=True,
                    )
                model_requests += 1
                try:
                    result = await semantic_call(deepcopy(request))
                except Exception as error:
                    validation_attempts.append(
                        {
                            "error_type": type(error).__name__,
                            "diagnostic": str(error)[:400],
                            **(
                                {"raw_response": error.doc}
                                if isinstance(error, json.JSONDecodeError)
                                else {}
                            ),
                        }
                    )
                    raise
                envelope = response_payload(result, request["response"])
                if (
                    structure_repair
                    and isinstance(envelope, dict)
                    and ("need_tools" in envelope or "need_evidence" in envelope)
                ):
                    reason = "Structure repair must use existing evidence, not new measurements/reads"
                    validation_attempts.append(
                        {"response": deepcopy(result), "errors": {"repair": reason}}
                    )
                    raise ResponseContractError("response", reason)
                if not isinstance(result, dict) or "need_tools" not in result:
                    return result
                if wave == 2:
                    return {
                        **unknown(
                            "Measurement wave budget exhausted; requested tools were not executed",
                            error=True,
                        ),
                        "claims": [],
                    }
                probes = result["need_tools"]
                if (
                    not isinstance(probes, list)
                    or not 1 <= len(probes) <= 3
                    or charged_tools() + len(probes) > 4
                    or len(receipts) + len(probes) > 8
                    or any(not isinstance(probe, dict) for probe in probes)
                ):
                    # Do not throw away actual receipts when the model asks for
                    # too much. Use the next already-bounded model wave to close
                    # with evidence or correct the request within the same cap.
                    request["measurement_feedback"] = {
                        "status": "request_not_executed",
                        "reason": "Request exceeds remaining tool budget or has malformed need_tools",
                        "instruction": "No tools in this rejected request ran. Use measurement_budget and "
                        "recorded tool_observations; produce an evidence-grounded verdict or unknown.",
                    }
                    continue
                request.pop("measurement_feedback", None)
                await execute(probes)
            raise ValueError("Missing final measurement verdict")

        class MeasurementCall:
            # Wrapping the role to execute probes must preserve its capacity,
            # compression, policy overlay and bounded reply-repair protocol.
            # Losing these silently restores a 160 KB / one-read / no-repair
            # default even when the configured role supports more.
            def __getattr__(self, name):
                return getattr(semantic_call, name)

            async def __call__(self, request):
                return await tool_call(request)

        async def ask_with_receipts(*args, **kwargs):
            try:
                return await ask(*args, **kwargs)
            except EvidenceReadLimit as error:
                read_failure.update(
                    {
                        **unknown(
                            "Evidence read allowance exhausted; final assessment unavailable",
                            error=True,
                        ),
                        "diagnostic": str(error),
                        "unresolved_evidence_refs": list(error.refs),
                        "claims": [],
                    }
                )
                # This operation did not finish its evidence protocol. A
                # provisional global verdict must never replace that failure.
                validated_parts.pop("global", None)
                selected_attempts.pop("global", None)
                validation_attempts.append(
                    {
                        "error_type": type(error).__name__,
                        "diagnostic": str(error),
                        "unresolved_evidence_refs": list(error.refs),
                    }
                )
                return deepcopy(read_failure)
            except (ResponseContractError, json.JSONDecodeError) as error:
                # The bounded repair already ran. A rejected interpretation
                # must not erase computations/reads that actually completed.
                return {
                    **unknown(
                        f"Measurement reply rejected: {type(error).__name__}",
                        error=True,
                    ),
                    "diagnostic": str(error)[:400],
                    "claims": [],
                }
            except Exception as error:
                if not requested_inquiry or not validated_parts:
                    raise
                # A failed repair call cannot erase an independently validated
                # counterpart. Preserve the failure; do not retry transport here.
                return {
                    **unknown("Measurement structure repair unavailable", error=True),
                    "diagnostic": str(error)[:400],
                    "failure_layer": failure_layer(error),
                    "claims": [],
                }

        def validate_global(row):
            # Validate the whole final verdict inside the bounded retry, not
            # just its claim audit. Otherwise a well-explained `partial` or a
            # missing reason escapes repair and is discarded by Registry.run.
            row = validate_claims(
                claim_locations(validate_result(row, evidence), evidence), evidence
            )
            if row["status"] != "unknown" and not has_product_evidence(
                row["evidence_refs"], evidence, receipts
            ):
                raise ValueError(
                    "Known check results need observed product evidence, not just the Task"
                )
            return row

        def validate_inquiry(row):
            if not isinstance(row, dict):
                raise ValueError("Requested inquiry_result must be a verdict object")
            row = validate_result(deepcopy(row), evidence)
            if row["status"] != "unknown" and not has_product_evidence(
                row["evidence_refs"], evidence, receipts
            ):
                raise ValueError("An inquiry conclusion needs actual product evidence")
            return row

        def validate_reply(row):
            nonlocal structure_repair, claim_repair, correction_state, unwrap_state
            raw = deepcopy(row)
            composition = {}
            if "response_unwrap" in row:
                try:
                    row, provenance = compose_response_unwrap(
                        raw, unwrap_state, len(validation_attempts) + 1
                    )
                    composition = {
                        "response_kind": "response_unwrap",
                        "composed_response": deepcopy(row),
                        "field_provenance": provenance,
                    }
                except (ValueError, TypeError, KeyError) as error:
                    validation_attempts.append(
                        {"response": raw, "errors": {"envelope": str(error)}}
                    )
                    raise ResponseContractError(
                        "response_unwrap", str(error)
                    ) from error
            envelope_error = None
            if isinstance(row.get("response"), dict):
                if unwrap_state is None:
                    expected = None
                    nested = row["response"]
                    if set(row) <= {"response", "inquiry_result"} and not (
                        set(nested)
                        & {
                            "response",
                            "inquiry_result",
                            "field_corrections",
                            "response_unwrap",
                            "need_tools",
                            "need_evidence",
                        }
                    ):
                        try:
                            validate_global(deepcopy(nested))
                            if requested_inquiry:
                                validate_inquiry(deepcopy(row.get("inquiry_result")))
                        except (ValueError, TypeError, KeyError, AttributeError):
                            pass  # No unwrapping offer for an invalid/ambiguous assessment.
                        else:
                            expected = deepcopy(nested)
                            if "inquiry_result" in row:
                                expected["inquiry_result"] = deepcopy(
                                    row["inquiry_result"]
                                )
                    unwrap_state = {
                        "expected": expected,
                        "attempt": len(validation_attempts) + 1,
                    }
                    # Deliberate narrow limit: ambiguous/invalid nested verdicts
                    # get no relocation offer, and cannot select a different
                    # assessment during this same-evidence repair. Supporting
                    # nested field repair would need its own preservation proof.
                envelope_error = ResponseContractError(
                    "response",
                    "Misplaced verdict envelope: claims/status/reason/gap/evidence_refs belong at the TOP LEVEL, "
                    "not inside response. A full reply must move the unchanged nested assessment to the root, "
                    "preserve any inquiry_result sibling, and omit the response wrapper. "
                    "Competing root fields or an invalid nested assessment cannot be implicitly selected/unwrapped.",
                )
                envelope_error.details.update(
                    actual_root_fields=sorted(row),
                    missing_root_fields=[
                        k
                        for k in ("claims", "status", "reason", "gap", "evidence_refs")
                        if k not in row
                    ],
                    nested_assessment_valid=unwrap_state["expected"] is not None,
                )
            elif unwrap_state is not None and row != unwrap_state["expected"]:
                envelope_error = ResponseContractError(
                    "response",
                    "Envelope repair must preserve the entire original nested assessment and inquiry sibling; "
                    "no changed judgment/relevance/claims/count/order/status/citations or ambiguous source selection",
                )
            if "field_corrections" in row:
                try:
                    row, provenance = compose_field_correction(
                        raw, correction_state, evidence, len(validation_attempts) + 1
                    )
                    composition = {
                        "response_kind": "field_correction",
                        "composed_response": deepcopy(row),
                        "field_provenance": provenance,
                    }
                except (ValueError, TypeError, KeyError) as error:
                    validation_attempts.append(
                        {
                            "response": raw,
                            "response_kind": "field_correction",
                            "errors": {"correction": str(error)},
                        }
                    )
                    raise ResponseContractError(
                        "field_corrections", str(error)
                    ) from error
            errors = {}
            issues = []
            candidates = {}
            global_row = (
                {k: v for k, v in row.items() if k != "inquiry_result"}
                if requested_inquiry
                else row
            )
            parts = [("global", global_row, validate_global)]
            if requested_inquiry:
                local = row.get("inquiry_result")
                if "inquiry_result" not in row and not getattr(
                    semantic_call, "bounded_stage", False
                ):
                    # Legacy roles never promised the separate reply contract.
                    # An omitted answer remains unresolved and eligible for
                    # review feedback/retesting, not an execution failure.
                    local = unknown(
                        "Specific inquiry was not explicitly answered; the rubric verdict is separate"
                    )
                parts.append(("inquiry", local, validate_inquiry))
            for scope, value, validator in parts:
                try:
                    if scope == "global" and envelope_error is not None:
                        raise envelope_error
                    if scope == "global" and claim_repair is not None:
                        preserve_claim_assessment(*claim_repair, current=value)
                    candidates[scope] = validator(deepcopy(value))
                except (ValueError, TypeError, KeyError, AttributeError) as error:
                    errors[scope] = str(error)[:400]
                    if isinstance(error, ResponseContractError):
                        issues.extend(
                            {"scope": scope, **deepcopy(issue)}
                            for issue in error.details.get("issues", [error.details])
                        )
            pending_evidence.clear()
            for scope, value in candidates.items():
                if scope not in validated_parts:
                    unread = sorted(
                        (
                            set(used_previews(value, context))
                            | set(used_previews(value, request_context))
                        )
                        - attached_refs
                    )
                    if unread:
                        pending_evidence[scope] = unread
                    else:
                        validated_parts[scope] = deepcopy(value)
                        selected_attempts[scope] = len(validation_attempts) + 1
            validation_attempts.append(
                {
                    "response": raw,
                    **composition,
                    "errors": errors,
                    **({"issues": deepcopy(issues)} if issues else {}),
                    **(
                        {"pending_evidence": deepcopy(pending_evidence)}
                        if pending_evidence
                        else {}
                    ),
                }
            )
            missing = {
                scope: reason
                for scope, reason in errors.items()
                if scope not in validated_parts
            }
            if missing:
                structure_repair = True
                if claim_repair is None:
                    # Keep the raw assessment; normalization only determines
                    # which citation fields were actually invalid.
                    try:
                        assessed = claim_locations(deepcopy(global_row), evidence)
                    except (ValueError, TypeError, AttributeError):
                        assessed = global_row
                    claim_repair = (
                        deepcopy(global_row),
                        claim_contract_issues(assessed, evidence),
                    )
                error = ResponseContractError("response", str(missing))
                error.details["invalid_fields"] = missing
                error.details["preserved_fields"] = list(validated_parts)
                if issues:
                    error.details["issues"] = [
                        issue for issue in issues if issue["scope"] in missing
                    ]
                if (
                    unwrap_state is not None
                    and unwrap_state["expected"] is not None
                    and set(missing) == {"global"}
                    and getattr(semantic_call, "bounded_stage", False)
                ):
                    offer = {
                        "schema": "measurement-response-unwrap/v1",
                        "base_attempt": unwrap_state["attempt"],
                        "reply": {"response_unwrap": "response"},
                        "instruction": 'Return ONLY {"response_unwrap":"response"} to explicitly relocate '
                        "the unchanged model-authored response object to the root. Preserve every field, "
                        "claim count/order, judgment, relevance, citations, summary and inquiry_result. "
                        "No replacements/defaults/tools/reads. Faithful full JSON is also accepted. "
                        "All validators and mandatory preview reads still apply; this is not a verdict acceptance.",
                    }
                    unwrap_state["offer"] = offer
                    error.details["response_unwrap_reply"] = deepcopy(offer)
                if (
                    correction_state is None
                    and getattr(semantic_call, "bounded_stage", False)
                    and set(missing) == {"global"}
                    and error.details.get("issues")
                ):
                    correction_issues = deepcopy(claim_repair[1])
                    for issue in error.details["issues"]:
                        detail = {
                            key: value for key, value in issue.items() if key != "scope"
                        }
                        if detail not in correction_issues:
                            correction_issues.append(detail)
                    targets = field_correction_targets(raw, correction_issues)
                    if targets:
                        offer = {
                            "schema": "measurement-field-correction/v1",
                            "base_attempt": len(validation_attempts),
                            "allowed_fields": sorted(targets),
                            "required_fields": sorted(targets),
                            "issues": deepcopy(correction_issues),
                            "instruction": (
                                "Preferred repair reply when this offer is present: return ONLY "
                                '{"field_corrections": {"advertised field": "model-authored replacement value"}}. '
                                "Supply every required field and no others. For a status conflict, author "
                                "status, reason and gap together. Obey allowed_values when supplied; "
                                "the model chooses the values, with no defaults. All unlisted values are copied exactly "
                                "from the original rejected model response, never inferred. The composed "
                                "response must pass all validators. Same evidence only; no tools/reads. "
                                "After new evidence expansion return a fresh full assessment."
                            ),
                        }
                        judgment_values = {
                            field: ["supported", "contradicted", "unverified"]
                            for field, target in targets.items()
                            if target[-1] == "judgment"
                        }
                        if judgment_values:
                            offer["allowed_values"] = judgment_values
                            offer["instruction"] += (
                                " For each missing/invalid judgment, author only its explicit enum value "
                                "from the preserved claim, observation and actual evidence. Do not rewrite "
                                "those fields or change claim count/order. A judgment correction does not "
                                "authorize changing a valid judgment, citations or summary. Existing "
                                "evidence and status/claim consistency constraints still apply."
                            )
                        if "status" in targets and raw.get("status") == "pass":
                            # These targets arise from existing pass/claim conflicts.
                            # Expose that constraint; the validators still decide acceptance.
                            offer.setdefault("allowed_values", {})["status"] = [
                                "fail",
                                "unknown",
                            ]
                            offer["instruction"] += (
                                " For this locked assessment, pass is invalid. Author fail or unknown "
                                "with consistent reason/gap; all existing verdict/evidence checks still apply."
                            )
                        correction_state = {
                            "base": raw,
                            "attempt": len(validation_attempts),
                            "targets": targets,
                            "offer": offer,
                        }
                        error.details["field_correction_reply"] = deepcopy(offer)
                error.recovery_instruction = (
                    "Repair only the invalid response structure using the same evidence. "
                    "Preserve already valid global/inquiry verdicts; do not reassess them, "
                    "change the standard, or seek a different semantic outcome. "
                    "Copy already valid claim fields exactly from rejected_response, including "
                    "claim text, observation, claim_location, relevance_reason, judgment, "
                    "verdict_relevance and valid citation lists; preserve claim count/order "
                    "wherever the claim structure is valid. Correct only fields identified as "
                    "invalid by the reported errors. A status error does not authorize editing "
                    "claims; depends_on fields are constraints to preserve, not repair targets. "
                    "Do not paraphrase or clarify valid claim text or observations."
                )
                for issue in error.details.get("issues", []):
                    if "repair_instruction" in issue:
                        error.recovery_instruction += " " + issue["repair_instruction"]
                raise error
            # Structurally valid but unread parts remain provisional. Return
            # them to ask's preview-read protocol, never to the preservation cache.
            current = {**candidates, **validated_parts}
            result = deepcopy(current["global"])
            if requested_inquiry:
                result["inquiry_result"] = deepcopy(current["inquiry"])
            return result

        result = await ask_with_receipts(
            MeasurementCall(),
            "outcome_measure",
            context,
            evidence,
            CLAUSE_INSTRUCTION
            + "Evaluate ONLY the supplied Outcome standard against current evidence and anchors. "
            "Return claims, status, reason, gap and evidence_refs at the TOP LEVEL; do not wrap them "
            "inside response. Any inquiry_result is a separate sibling of these fields. "
            "Find factual contradictions, missing required outcomes and unsupported claims. "
            "If measurement_inquiry is supplied, address that specific doubt using current evidence. "
            "It is an unverified question, not an established defect or a replacement standard. "
            "Answer it separately in inquiry_result: pass=no material issue found, fail=evidenced Task "
            "defect, unknown=unresolved/out of scope. A rubric pass does not erase an inquiry defect. "
            "Artifact existence/read success is not correctness. Compare technical claims against "
            "available sources; request exact evidence if needed. Without enough evidence say unknown. "
            "Historical user-specific standards apply only if THIS product's user actually required them. "
            "A missing applicability fact is unknown, not failure; never impose another user's answers. "
            "If test.knowledge is present: the Jury criterion is the standard; method_refs may define how to "
            "measure; reference_refs illustrate and are not requirements. applicability_refs are related notes, "
            "not mandatory preconditions merely because they share a source. Establish relevance to the chosen "
            "method first; unrelated notes add no requirement. Essential unverified applicability prevents "
            "that method from establishing a verdict, not other independent measurements. conditions_unstated "
            "and unclassified_refs are uncertainties, not automatic product failures or verified methods. "
            "Before the verdict, audit the material claims relevant to THIS standard across ALL "
            "delivery channels, including saved artifacts and environment-held drafts, not just the "
            "Actor's summary. For each claim list where it appears, what the evidence actually establishes, "
            "and supported/contradicted/unverified. An artifact proves what was written, not that its "
            "external assertions are true. Separate a proposed step, a confirmed arrangement, an executed "
            "action and an outcome commitment. Non-execution does not establish the truth of a draft. "
            "Do not infer confirmation from a person's role or a tool's availability. Missing support "
            "is unverified, not proof the assertion is false. Explain whether that uncertainty violates "
            "the standard (e.g. asserting certainty without support) or needs verification. "
            "Every supported or contradicted claim REQUIRES nonempty evidence_refs citing exact "
            "supplied observed evidence handles for that judgment. Empty evidence_refs are allowed "
            "only for unverified claims. Every claim MUST include evidence_refs as a list, "
            "including unverified claims ([] when unavailable). claim_refs locate a claim; they are not automatically "
            "evidence for its truth or contradiction. Never invent citations to complete the schema. "
            "Use the evidence-access option when a preview is insufficient. Write claims first, then "
            "claim_refs must contain exact supplied handles only; put chapter names or quotes in "
            "claim_location, not inside a handle. The supplied rubric/test is a standard, not proof "
            "that the product satisfies it. Audit only claims material to this test's standard. "
            "summarize status/reason/gap; pass requires ALL material claims to be supported "
            "by actual observed evidence. A material unverified or contradicted claim is "
            "incompatible with pass. If status contradicts the preserved claim assessment, "
            "minimally correcting status/reason/gap for consistency is permitted; the model "
            "must author that verdict. Do not upgrade an unsupported claim to supported, "
            "downgrade material to advisory, delete a claim or invent proof to make pass. "
            "Keep ancillary improvement observations as advisory claims with verdict_relevance=advisory "
            "and relevance_reason explaining why they do not affect THIS standard's verdict. "
            "Material means needed to judge THIS criterion, including a soft criterion. "
            "A soft criterion does not make its claims advisory; advisory is reserved for "
            "ancillary observations outside what is needed to judge this criterion. "
            "They may remain unverified without blocking supported required outcomes; do not delete "
            "the uncertainty or relabel it supported. Missing required outcomes, contradictions of "
            "required facts and unsupported claims that affect this standard remain material. "
            "A pass must still have supported material claims; calling every claim advisory is invalid. "
            "Give a localized gap for fail. Do not return an absolute score or official success claim.",
            {
                "claims": [
                    {
                        "claim": "material assertion or required outcome under this standard",
                        "claim_refs": [
                            "exact refs locating the assertion or requirement"
                        ],
                        "claim_location": "optional chapter/quote; this locator is not independently verified",
                        "evidence_refs": [
                            "exact supplied observed evidence refs; nonempty REQUIRED for supported/contradicted; empty allowed only for unverified"
                        ],
                        "observation": "what the evidence establishes, does not establish, or contradicts",
                        "judgment": "supported/contradicted/unverified",
                        "verdict_relevance": "material/advisory; omitted means material",
                        "relevance_reason": "for advisory: why the uncertainty does not affect the supplied standard",
                    }
                ],
                "status": "pass/fail/unknown",
                "reason": "why",
                "gap": "specific defect if fail",
                "evidence_refs": ["exact refs"],
                "measurement_recovery": [
                    {
                        "failed_ref": "optional failed tool receipt ref",
                        "replacement_refs": [
                            "later observed same-tool receipt refs covering the same inputs"
                        ],
                        "reason": "why the cited successful measurements resolve the original required check",
                    }
                ],
                **(
                    {
                        "inquiry_result": {
                            "status": "pass/fail/unknown",
                            "reason": "answer to the specific question",
                            "gap": "concrete Task defect if fail",
                            "evidence_refs": ["exact observed product/source refs"],
                        }
                    }
                    if context.get("measurement_inquiry")
                    else {}
                ),
            },
            extra={
                "test": test,
                "measurement_tools": registry.tools.catalog(),
                "tool_protocol": 'Before a verdict you may return ONLY {"need_tools":[{"tool":"catalog key","arguments":{}}]}. '
                "Use actual tools for numeric/code/source questions, not mental simulation. "
                "Receipts establish computations/reads, not automatically the claim's correctness or a Task pass. "
                "Cite actual artifact refs or executed read receipts carrying observed_product_refs. "
                "A receipt proves only its observed excerpt, not unseen pages. Missing tools are unknown, not product failure. "
                "If a failed probe was corrected, explicitly declare measurement_recovery with failed_ref, "
                "later successful replacement_refs over the same inputs, and reason. Cite those observed "
                "receipts in the verdict/claims; a failed attempt alone cannot be waived. "
                "At most three probes per request, four probes total and two probe waves; "
                "reuse supplied tool_observations instead of executing the same probe again. "
                "No obligation to execute GPU workloads for a report-only task.",
            },
            verify_used_previews=True,
            **(
                {"validate": validate_reply}
                if getattr(semantic_call, "bounded_stage", False) or requested_inquiry
                else {}
            ),
        )
        # The specific question is a separate verdict. A malformed global claim
        # audit must not erase a well-formed, evidenced local correction (or vice versa).
        inquiry_result = None
        inquiry_diagnostic = None
        if requested_inquiry:
            if "global" in validated_parts and not read_failure:
                result = deepcopy(validated_parts["global"])
            try:
                if "inquiry" in validated_parts:
                    inquiry_result = deepcopy(validated_parts["inquiry"])
                else:
                    rejected = next(
                        (
                            attempt["errors"]["inquiry"]
                            for attempt in reversed(validation_attempts)
                            if "inquiry" in attempt.get("errors", {})
                        ),
                        None,
                    )
                    if rejected:
                        raise ValueError(rejected)
                    inquiry_result = validate_inquiry(result.get("inquiry_result"))
            except (ValueError, TypeError, AttributeError) as error:
                inquiry_result = {
                    **unknown("Specific inquiry validation failed", error=True),
                    "diagnostic": str(error)[:400],
                }
        elif "inquiry_result" in result:
            # Unrequested metadata is not a local verdict. Preserve its exact
            # value outside authoritative fields so malformed replies remain
            # visible without breaking consumers of requested inquiry results.
            inquiry_diagnostic = {
                "reason": "no_measurement_inquiry_requested",
                "authoritative": False,
                "original_value": deepcopy(result["inquiry_result"]),
            }
            result = {
                key: value for key, value in result.items() if key != "inquiry_result"
            }
        try:
            result = validate_claims(claim_locations(result, evidence), evidence)
        except ValueError as error:
            result = {
                **unknown("Claim audit validation failed", error=True),
                "diagnostic": str(error)[:400],
            }
        if inquiry_result is not None:
            result["inquiry_result"] = inquiry_result
        if inquiry_diagnostic is not None:
            result["inquiry_result_diagnostic"] = inquiry_diagnostic
        unread = sorted(
            set(result.get("unresolved_evidence_refs", []))
            | {ref for refs in pending_evidence.values() for ref in refs}
            | set(read_failure.get("unresolved_evidence_refs", []))
        )
        if unread:
            result["unresolved_evidence_refs"] = unread
        if any(
            attempt.get("errors")
            or attempt.get("error_type")
            or attempt.get("pending_evidence")
            for attempt in validation_attempts
        ):
            result["response_validation"] = {
                "attempts": deepcopy(validation_attempts),
                "selected_attempts": deepcopy(selected_attempts),
            }
        # Never trust model-authored receipts; overwrite with executed observations.
        result["tool_observations"] = deepcopy(receipts)
        result["execution_usage"] = {
            "model_requests": model_requests,
            "tool_calls": charged_tools(),
        }
        result["_tool_evidence"] = deepcopy(receipts)
        unresolved = unresolved_measurements(result, receipts)
        result["unresolved_measurement_refs"] = sorted(unresolved)
        if unresolved:
            result.update(
                status="unknown",
                gap="",
                reason="A requested measurement was unavailable; " + result["reason"],
            )
            if "inquiry_result" in result:
                result["inquiry_result"] = unknown(
                    "A requested measurement was unavailable"
                )
        return result

    async def json_valid(test, context, evidence):
        ref = test.get("arguments", {}).get("ref")
        if ref not in evidence or not isinstance(evidence[ref], str):
            return unknown("No exact textual evidence at supplied ref")
        try:
            json.loads(evidence[ref])
            return {
                "status": "pass",
                "reason": "Parses as JSON; no semantic claim",
                "gap": "",
                "evidence_refs": [ref],
            }
        except json.JSONDecodeError as error:
            return {
                "status": "fail",
                "reason": "JSON parser rejected text",
                "gap": f"Invalid JSON at line {error.lineno}",
                "evidence_refs": [ref],
            }

    registry.register(
        "evidence.review",
        semantic,
        "Grounded semantic Outcome check; pass/fail/unknown and evidence.",
        kind="model_assisted",
    )
    registry.register(
        "json.parse",
        json_valid,
        "Parse exact text at arguments.ref. Syntax only, not data accuracy.",
        dependencies=lambda t, c, e: e.get(t.get("arguments", {}).get("ref")),
    )
    registry.semantic = semantic
    return registry


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
    pack, registry, context, evidence, *, policy=None, test_ids=None, previous=None
):
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
    reused = {}
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
