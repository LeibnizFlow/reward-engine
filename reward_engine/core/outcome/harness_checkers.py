"""Harness checkers for the existing public entry point."""

import json
from copy import deepcopy

from reward_engine.core.outcome.context import (
    EvidenceReadLimit,
    ResponseContractError,
    ask,
    measurement_prior_view,
    response_payload,
    used_previews,
)


def defaults(semantic_call, *, tools=None):
    from reward_engine.core.outcome.harness import Registry, failure_layer, unknown
    from reward_engine.core.outcome.harness_claims import (
        claim_contract_issues,
        claim_locations,
        compose_field_correction,
        compose_response_unwrap,
        field_correction_targets,
        has_product_evidence,
        preserve_claim_assessment,
        unresolved_measurements,
        validate_claims,
        validate_result,
    )
    from reward_engine.core.outcome.measurement_tools import MeasurementTools

    registry = Registry()
    registry.review_call = semantic_call
    registry.tools = tools or MeasurementTools()
    registry.tools.role_contract_policy = getattr(
        semantic_call, "role_contract_policy", None
    )

    async def semantic(test, context, evidence):
        from reward_engine.core.outcome.clauses import INSTRUCTION as CLAUSE_INSTRUCTION
        from reward_engine.core.outcome.response_receipts import ACTIVE_RESPONSE

        context = measurement_prior_view(context, evidence, test)
        from reward_engine.core.outcome.authored_json import enabled

        v2 = enabled(semantic_call)
        tool_waves = 0
        receipts = {}
        model_requests = 0
        validation_attempts, validated_parts, selected_attempts = [], {}, {}
        requested_inquiry = bool(context.get("measurement_inquiry"))
        structure_repair = False
        closed_evidence_repair = False
        claim_repair = None
        correction_state = None
        unwrap_state = None
        request_context, attached_refs = context, set()
        pending_evidence, read_failure = {}, {}
        final_receipts = []

        def charged_tools():
            # A probe rejected before execution (bad refs/paths, oversized input)
            # is correctable feedback, not an executed measurement, so it does
            # not consume the four-execution budget.
            return sum(
                1 for row in receipts.values() if not row.get("request_rejected")
            )

        def executable_count(probes):
            from reward_engine.core.outcome.measurement_requests import (
                MeasurementRequestRejected,
            )
            from reward_engine.core.outcome.measurement_requests import (
                enabled as recovery_enabled,
            )

            if not recovery_enabled(registry.tools):
                return len(probes)
            count = 0
            for probe in probes:
                try:
                    registry.tools.validate_request(probe, evidence)
                except MeasurementRequestRejected:
                    continue
                count += 1
            return count

        async def execute(probes):
            if (
                not isinstance(probes, list)
                or not 1 <= len(probes) <= 3
                or charged_tools() + executable_count(probes) > 4
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
                run = (
                    registry.tools.run_checked
                    if v2 and hasattr(registry.tools, "run_checked")
                    else registry.tools.run
                )
                receipt = await run(probe, evidence)
                receipts[ref] = receipt
                evidence[ref] = receipt

        recipe = test.get("execution_recipe", {})
        if recipe.get("probes"):
            await execute(recipe["probes"])

        async def tool_call(request):
            nonlocal \
                model_requests, \
                structure_repair, \
                closed_evidence_repair, \
                request_context, \
                attached_refs, \
                claim_repair, \
                correction_state, \
                unwrap_state, \
                tool_waves
            # Syntax recovery replaces response_recovery details. It cannot
            # reopen the tools/reads of an already active same-evidence repair.
            structure_repair = not v2 and (
                structure_repair
                or bool(
                    request.get("response_recovery", {})
                    .get("details", {})
                    .get("invalid_fields")
                )
            )
            closed_evidence_repair = not v2 and (
                closed_evidence_repair
                or bool(
                    request.get("response_recovery", {})
                    .get("details", {})
                    .get("evidence_access_closed")
                )
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
            # Preflight-only rejections do not consume an execution wave. The
            # operation still shares its eight-request and eight-receipt caps.
            for wave in range(8 if v2 else 3):
                request["context"]["available_evidence_refs"] = list(evidence)
                request["context"]["tool_observations"] = deepcopy(receipts)
                if getattr(semantic_call, "bounded_stage", False):
                    failed = [
                        {
                            "failed_ref": ref,
                            "tool": row["tool"],
                            "reason": row.get("reason", "Measurement unavailable"),
                            "input_refs": deepcopy(
                                row.get("arguments", {}).get("input_refs", [])
                            ),
                        }
                        for ref, row in receipts.items()
                        if row["status"] != "observed"
                        and not row.get("request_rejected")
                    ]
                    if failed:
                        request["measurement_recovery_required"] = {
                            "failed_measurements": failed,
                            "instruction": "These executed probes failed; printed stdout is partial output, "
                            "not a successful result. Within the remaining tool waves/calls, correct the "
                            "actual program or arguments and execute a replacement using the same inputs. "
                            "code.run requires an explicit result assignment; printing alone does not "
                            "complete it. If a later successful replacement already exists, reuse it. "
                            "The final verdict must explicitly declare measurement_recovery with failed_ref, "
                            "later observed same-tool replacement_refs covering the failed inputs, and reason; "
                            "cite the replacement receipts in the assessment. Do not waive the failure, "
                            "simulate success, or repeat the unchanged failing probe. If recovery is "
                            "unavailable within the existing budget, leave the measurement unresolved.",
                        }
                request["measurement_budget"] = {
                    "remaining_tool_calls": max(0, 4 - charged_tools()),
                    "max_calls_per_wave": min(3, max(0, 4 - charged_tools()))
                    if not v2 or tool_waves < 2
                    else 0,
                    "remaining_tool_waves": max(0, 2 - (tool_waves if v2 else wave)),
                    "final_verdict_required": (tool_waves >= 2 if v2 else wave == 2)
                    or charged_tools() == 4,
                    "instruction": "Reuse recorded observations. Do not repeat completed reads/calculations. "
                    "A request rejected before execution (invalid refs/paths, oversized input) is "
                    "correctable and does not consume the executed budget; fix the cited refs/paths and "
                    "retry within the cap. When no tool budget remains, give the required final verdict "
                    "from existing evidence; return unknown for any material fact that still needs an "
                    "unperformed measurement.",
                }
                if structure_repair or closed_evidence_repair:
                    request["measurement_budget"].update(
                        remaining_tool_calls=0,
                        max_calls_per_wave=0,
                        remaining_tool_waves=0,
                        final_verdict_required=True,
                    )
                if request.get("measurement_recovery_required"):
                    budget = request["measurement_budget"]
                    recovery = request["measurement_recovery_required"]
                    recovery.update(
                        remaining_tool_calls=budget["remaining_tool_calls"],
                        remaining_tool_waves=budget["remaining_tool_waves"],
                        replacement_execution_allowed=(
                            budget["remaining_tool_calls"] > 0
                            and budget["remaining_tool_waves"] > 0
                        ),
                    )
                    recovery["instruction"] += (
                        " The explicit replacement_execution_allowed field and remaining_tool_calls/waves "
                        "above govern this request. Ordinary evidence_access_closed closes source expansion, "
                        "not these remaining tool slots: when replacement_execution_allowed is true, "
                        "return need_tools to execute the corrected program over already supplied inputs. "
                        "Do not claim the tool budget is exhausted while these counters allow execution."
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
                envelope = (
                    result if v2 else response_payload(result, request["response"])
                )
                if (
                    (structure_repair or closed_evidence_repair)
                    and isinstance(envelope, dict)
                    and ("need_tools" in envelope or "need_evidence" in envelope)
                ):
                    reason = "Structure repair must use existing evidence, not new measurements/reads"
                    validation_attempts.append(
                        {"response": deepcopy(result), "errors": {"repair": reason}}
                    )
                    raise ResponseContractError("response", reason)
                if not isinstance(result, dict) or "need_tools" not in result:
                    receipt = ACTIVE_RESPONSE.get()
                    if receipt is not None:
                        final_receipts.append(receipt)
                    return result
                if tool_waves >= 2 if v2 else wave == 2:
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
                    or charged_tools() + executable_count(probes) > 4
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
                before = charged_tools()
                await execute(probes)
                tool_waves += int(not v2 or charged_tools() > before)
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
                    "diagnostic_details": deepcopy(getattr(error, "details", {})),
                    "claims": [],
                }
            except Exception as error:
                from reward_engine.core.outcome.measurement_requests import (
                    enabled as recovery_enabled,
                )

                if recovery_enabled(registry.tools):
                    return {
                        **unknown(
                            "Measurement operation failed before an accepted assessment",
                            error=True,
                        ),
                        "error_type": type(error).__name__,
                        "diagnostic": str(error)[:400],
                        "operation_budget": deepcopy(
                            getattr(error, "operation_budget", {})
                        ),
                        "failure_layer": failure_layer(error),
                        "claims": [],
                    }
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
                error = ResponseContractError(
                    "evidence_refs",
                    "Known check results need observed product evidence, not just the Task",
                )
                error.details.update(
                    actual=deepcopy(row["evidence_refs"]),
                    missing=False,
                    repair_instruction="Correct only the top-level evidence_refs list for this verdict. "
                    "Explicitly cite existing observed product evidence supporting the unchanged assessment: "
                    "a supplied product reference, or a trusted observed workspace.read receipt with text "
                    "and observed_product_refs bound to supplied product evidence. Task/rubric refs and "
                    "workspace.inspect receipts alone do not meet this requirement, even when an inspection "
                    "lists nested source refs. The model must verify and author the root references; do not "
                    "invent or automatically copy refs. Preserve every claim, its citations, observation, "
                    "judgment and relevance, and the existing status/reason/gap. Do not request new tools "
                    "or evidence during this same-evidence correction.",
                )
                raise error
            return row

        def validate_inquiry(row):
            if not isinstance(row, dict):
                raise ValueError("Requested inquiry_result must be a verdict object")
            row = validate_result(deepcopy(row), evidence)
            if row["status"] != "unknown" and not has_product_evidence(
                row["evidence_refs"], evidence, receipts
            ):
                error = ResponseContractError(
                    "inquiry_result.evidence_refs",
                    "An inquiry conclusion needs actual product evidence",
                )
                error.details.update(
                    actual=deepcopy(row["evidence_refs"]),
                    missing=False,
                    repair_instruction="Correct only inquiry_result.evidence_refs using existing observed "
                    "product evidence supporting the unchanged inquiry assessment. Explicitly author a "
                    "supplied product reference, or a trusted observed workspace.read receipt with text "
                    "and observed_product_refs bound to supplied product evidence. Task/rubric refs and "
                    "workspace.inspect receipts alone do not qualify; do not invent or automatically copy "
                    "their nested refs. Preserve inquiry_result status/reason/gap and all global claims, "
                    "citations, observations, judgments, relevance and summary. No new tools or evidence "
                    "during this same-evidence correction.",
                )
                raise error
            return row

        def validate_reply(row):
            nonlocal structure_repair, claim_repair, correction_state, unwrap_state
            raw = deepcopy(row)
            if v2:
                from reward_engine.core.outcome.measurement_contract import (
                    issues as field_issues,
                )

                issues = field_issues(row, requested_inquiry)
                for scope, value, validator in [
                    (
                        "global",
                        {k: v for k, v in row.items() if k != "inquiry_result"},
                        validate_global,
                    ),
                    *(
                        [("inquiry", row.get("inquiry_result"), validate_inquiry)]
                        if requested_inquiry
                        else []
                    ),
                ]:
                    try:
                        validator(deepcopy(value))
                    except (ValueError, TypeError, KeyError, AttributeError) as error:
                        issues.append(
                            {
                                "scope": scope,
                                "reason": str(error),
                                "details": getattr(error, "details", {}),
                            }
                        )
                try:
                    unresolved = unresolved_measurements(row, receipts)
                except (TypeError, AttributeError):
                    unresolved = set(receipts)
                if unresolved:
                    issues.append(
                        {
                            "field": "measurement_recovery",
                            "reason": "Executed failures need later successful same-tool receipts covering failed inputs",
                            "failed_refs": sorted(unresolved),
                        }
                    )
                validation_attempts.append(
                    {
                        "response": raw,
                        "issues": deepcopy(issues),
                        "errors": {"response": "invalid"} if issues else {},
                    }
                )
                if issues:
                    error = ResponseContractError(
                        "response", "Measurement requires complete correction"
                    )
                    error.details["issues"] = issues
                    from reward_engine.core.outcome.measurement_requests import (
                        comparable,
                    )
                    from reward_engine.core.outcome.measurement_requests import (
                        enabled as recovery_enabled,
                    )

                    if (
                        unresolved
                        and recovery_enabled(registry.tools)
                        and (
                            tool_waves >= 2
                            or charged_tools() >= 4
                            or len(receipts) >= 8
                        )
                    ):
                        # A full reply can still declare an existing valid recovery.
                        # Stop only when further execution is required but forbidden.
                        order = list(receipts)
                        impossible = [
                            ref
                            for ref in unresolved
                            if not comparable(
                                receipts[ref],
                                [
                                    receipts[key]
                                    for key in order[order.index(ref) + 1 :]
                                    if receipts[key]["status"] == "observed"
                                    and receipts[key]["tool"] == receipts[ref]["tool"]
                                ],
                            )
                        ]
                        if impossible:
                            error.details.update(
                                recovery_terminal=True,
                                code="MEASUREMENT_RECOVERY_EXHAUSTED",
                                failed_refs=impossible,
                                remaining_tool_calls=max(0, 4 - charged_tools()),
                                remaining_tool_waves=max(0, 2 - tool_waves),
                            )
                    raise error
                # Reassess the full draft after correction. No earlier malformed
                # verdict can pin claim values or close unused tool slots.
                validated_parts.clear()
                selected_attempts.clear()
                pending_evidence.clear()
                return row
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
                        if "status" in targets:
                            offer.setdefault("allowed_values", {})["status"] = [
                                "pass",
                                "fail",
                                "unknown",
                            ]
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

        from reward_engine.core.outcome.measurement_contract import (
            template as measurement_template,
        )

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
            "Actor's summary. In the initial assessment, failure anchors and investigative hypotheses "
            "are alternatives to evaluate, not assertions attributable to the Task or product. "
            "For outcome attainment, assess the criterion-aligned requirement and explain rejected "
            "defect hypotheses in observations. Preserve the meaning, including negation, of assertions "
            "the product actually makes. For each claim list where it appears, what the evidence actually establishes, "
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
            measurement_template(requested_inquiry)
            if v2
            else {
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
                "measurement_tools": registry.tool_catalog(evidence, authored=v2),
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
        if unresolved and getattr(semantic_call, "bounded_stage", False):
            failure = ResponseContractError(
                "measurement_recovery",
                "Final assessment leaves executed measurement failures unresolved",
            )
            failure.details["unresolved_measurement_refs"] = sorted(unresolved)
            # An earlier premature final can otherwise pin the next resume to
            # format-only repair, where replacement tools are correctly closed.
            # Reject only final-assessment replies, never prior tool requests.
            for receipt in final_receipts:
                receipt.reject(failure)
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

    def json_input(test, context, evidence):
        resolver = getattr(
            registry.tools, "text_evidence", MeasurementTools().text_evidence
        )
        return resolver(test.get("arguments", {}), evidence)

    def json_dependencies(test, context, evidence):
        try:
            return json_input(test, context, evidence)
        except (ValueError, TypeError, KeyError) as error:
            return {
                "unavailable": str(error),
                "arguments": test.get("arguments", {}),
                "evidence": evidence,
            }

    async def json_valid(test, context, evidence):
        try:
            resolved = json_input(test, context, evidence)
        except (ValueError, TypeError, KeyError) as error:
            return unknown(str(error))
        refs = resolved["evidence_refs"]
        try:
            json.loads(resolved["text"])
            return {
                "status": "pass",
                "reason": "Parses as JSON; no semantic claim",
                "gap": "",
                "evidence_refs": refs,
            }
        except json.JSONDecodeError as error:
            return {
                "status": "fail",
                "reason": "JSON parser rejected text",
                "gap": f"Invalid JSON at line {error.lineno}",
                "evidence_refs": refs,
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
        dependencies=json_dependencies,
    )
    registry.semantic = semantic
    return registry
