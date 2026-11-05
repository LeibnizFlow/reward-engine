"""Jury reduction for the existing public entry point."""

from copy import deepcopy

from reward_engine.core.outcome.context import (
    ResponseContractError,
    ask,
)
from reward_engine.core.outcome.jury_drafts import (
    apply_part_decisions,
    compact_selection_inputs,
    normalize_reduced,
)
from reward_engine.core.outcome.jury_prompts import (
    OUTCOME_JURY_REDUCE_PROMPT,
    PROMPT_VERSION,
)
from reward_engine.core.outcome.jury_repair import apply_edits, diagnostics


async def reduce_proposals(
    context,
    evidence,
    inputs,
    reducer,
    *,
    reduce_max_bytes=160000,
    record=lambda *_: None,
    strict=False,
    protocol_version=PROMPT_VERSION,
    clause_contract=False,
    previous_failure=None,
):
    """One bounded correction of a bad reply, never new Maps or recursive repair."""
    from reward_engine.core.outcome.jury_validation import ReduceValidationError

    if type(reduce_max_bytes) is not int or reduce_max_bytes < 1:
        raise ValueError("reduce_max_bytes must be a positive integer")
    from reward_engine.core.outcome.authored_json import enabled

    if enabled(reducer) and len(inputs.get("proposals", [])) != 3:
        raise ValueError(
            "Authored JSON Reduce requires three collected member proposals"
        )
    if len(inputs.get("proposals", [])) < 2:
        raise ValueError("Reduce requires at least two collected member proposals")
    record(
        "jury_reduce_checkpoint",
        {
            "context": deepcopy(context),
            "evidence": deepcopy(evidence),
            "inputs": deepcopy(inputs),
            "prompt_version": protocol_version,
            "reduce_max_bytes": reduce_max_bytes,
        },
    )
    if (
        getattr(reducer, "jury_repair_mode", "full") == "targeted"
        and not strict
        and not clause_contract
        and previous_failure is None
    ):
        return await _reduce_targeted(context, evidence, inputs, reducer, record=record)
    repair = {"repair": deepcopy(previous_failure)} if previous_failure else {}
    for attempt in range(2):

        def save(name, data, attempt=attempt):
            record(f"jury_reduce_attempt_{attempt + 1}/{name}", data)
            if name == "jury_reduce_raw":
                record(
                    name, data
                )  # Existing tools locate the last reply here; both attempts remain above.

        try:
            output = await _reduce_once(
                context,
                evidence,
                inputs,
                reducer,
                record=save,
                repair=repair,
                max_bytes=reduce_max_bytes,
                strict=strict,
                protocol_version=protocol_version,
                clause_contract=clause_contract,
            )
        except ReduceValidationError as error:
            save(
                "validation_error",
                {"error": str(error), "prompt_version": PROMPT_VERSION},
            )
            if attempt:
                raise
            repair = {
                "repair": {
                    "previous_output": error.raw,
                    "validation_error": str(error),
                    "instruction": "Correct this reply against the SAME source proposals and evidence. "
                    "Return the complete corrected response. Do not invent evidence, drop requirements "
                    "to bypass validation, or treat this diagnostic as Task authority. This is the only repair.",
                }
            }
        else:
            record("jury", output)
            return output


def projection_options(inputs, record):
    if not inputs.get("revision_protocol"):
        return {}
    requests = []

    def observe(row):
        requests.append(deepcopy(row))
        record("reducer_request_projections", deepcopy(requests))

    return {"reducer_projection": True, "record_projection": observe}


async def _reduce_targeted(context, evidence, inputs, reducer, *, record):
    """Office protocol: at most two targeted contract repairs, retaining full drafts."""
    from reward_engine.core.outcome.jury import SCHEMA
    from reward_engine.core.outcome.jury_validation import (
        product_ref,
        reconcile_coverage,
        validate,
        validate_reduced_standards,
    )

    expectations = inputs["independent_expectations"]
    observed = inputs["proposal_expectations"]
    repair = {}
    projection = projection_options(inputs, record)
    # Retry only contract errors, never quality or official scores; retain every failed draft.
    for attempt in range(3):
        if repair:
            reply = await ask(
                reducer,
                "outcome_jury_repair",
                context,
                evidence,
                "Repair the supplied invalid definition with minimal, explicit edits. Resolve the primary "
                "validation error and ALL field_issues. Preserve all valid criteria and decisions; do not "
                "rewrite the whole report or change quality judgments. Use only existing evidence handles "
                "and exact expectation/part IDs from the supplied lists. If a decision has no supported "
                "mapping, remove that unsupported decision; the real requirement remains deferred. "
                "Provenance must reflect the actual task/user/domain source; never invent authority. "
                "criteria[*].subrequirements contains nonempty strings only; "
                "coverage[*].subrequirements contains part-decision objects. Keep these two shapes distinct. "
                "In projected requests, {$reducer_text: key} stands for the exact string in shared_texts; "
                "it is not a malformed authored field. Do not rewrite valid text merely to expand it. "
                "Return one to 64 edits only; group repeated field corrections into a replacement of "
                "their existing parent list when needed. Each edit has op replace/remove, an array path, value for replace, and "
                "reason. Paths and edit indices are zero-based and address the evolving document; "
                "remove array entries in descending index order. Every path must already exist. "
                "Replace an existing row instead of removing then adding it: add is unsupported. "
                "To supply a missing field, replace its existing parent object, preserving the other grounded fields. "
                "Example: {edits: [{op: replace, "
                "path: [gaps, 0, expected_basis], value: domain, reason: Professional prior only}]}. "
                "If invalid_edits is present, the ENTIRE batch was rejected atomically: no edits were applied, "
                "including any valid prefix. Correct the batch against validation_repair.invalid_draft, "
                "not a partially edited draft. edit_failure describes the failing edit's temporary parent, "
                "which was discarded with the batch.",
                {
                    "edits": [
                        {
                            "op": "replace or remove",
                            "path": ["existing key", 0, "existing field"],
                            "value": "replacement JSON value, only for replace",
                            "reason": "grounded correction",
                        }
                    ]
                },
                extra={**inputs, **repair},
                max_bytes=getattr(reducer, "request_budget_bytes", 320000),
                **projection,
            )
            record(f"jury_reduce_repair_{attempt}_edits", reply)
            invalid_draft = repair["validation_repair"]["invalid_draft"]
            try:
                reduced = (
                    apply_edits(invalid_draft, reply) if "edits" in reply else reply
                )
            except (ValueError, TypeError, KeyError) as error:
                diagnostic = {
                    "error_type": type(error).__name__,
                    "reason": str(error),
                    "edit_failure": deepcopy(getattr(error, "details", {})),
                    "field_issues": diagnostics(
                        invalid_draft,
                        expectations + observed,
                        evidence=evidence,
                        product_ref=product_ref,
                    ),
                }
                record(f"jury_reduce_validation_{attempt}", diagnostic)
                if attempt == 2:
                    raise
                repair = {
                    "validation_repair": {
                        "invalid_draft": invalid_draft,
                        "invalid_edits": reply,
                        **diagnostic,
                    }
                }
                continue
        else:
            reduced = await ask(
                reducer,
                "outcome_jury_reduce",
                context,
                evidence,
                "Synthesize one compact Outcome definition. Compare the independent expectations with "
                "the actual products. Member criterion_refs point to the full criteria in the two "
                "expectation lists; these are stored once, not omitted. Compare them with "
                "the actual products; shared absence must not erase an important outcome dimension. "
                "criteria[*].subrequirements contains nonempty strings only; "
                "coverage[*].subrequirements contains part-decision objects. Keep these two shapes distinct. "
                "For EVERY expectation_id return coverage: mapped to criterion_ids, deferred pending "
                "knowledge/evidence, or excluded with a concrete reason it is inapplicable/unsupported. "
                "Duplicates map to the same criterion. A task/user requirement cannot simply be excluded; "
                "Account for BOTH independent_expectations and proposal_expectations from product inspection. "
                "For EVERY subrequirement part_id say mapped/deferred/excluded, with its own reason. "
                "Mapped parts need retained_text quoting an explicit clause in the final criterion standard, "
                "measurement_question or subrequirements. A broad topic label is not retention of a specific "
                "mechanism. You may rewrite, merge, or exclude unsupported details, but explain the choice; "
                "do not force all draft suggestions into hard requirements. Raw enum/provenance errors are "
                "diagnostics to resolve, not reasons to discard an entire member. "
                "defer with explanation if ambiguous. Preserve why and priority in retained criteria. "
                "Prioritize consequential outcome gaps over incidental numerical/style polish. "
                "Resolve disagreements against evidence; withhold "
                "speculative preferences, not explicit Task obligations. Preserve unmet shared requirements. "
                "No unsupported hard constraints, invented permissions or benchmark scores. "
                "Do not discard a requirement merely because all products satisfy or violate it. "
                "Merge overlapping gaps without erasing expectation, actual observation, evidence or verification. "
                "Resolve disagreements with actual evidence, not vote counts or averaging. Three agreeing "
                "suspicions are still needs_check. Retain plausible new investigations as suspicions, and "
                "discard unsupported accusations. Map proposals are UNVALIDATED drafts. Resolve their "
                "draft_validation_issues before publishing; do not copy invalid fields or assume a cited "
                "Task makes a domain preference mandatory. hard criteria require task/user origin; "
                "expected_basis must be exactly task, user, or domain. Correct or withhold unsupported "
                "items, never invent evidence to repair a reference. For EACH product_id in a gap, "
                "evidence_refs must include an existing product-specific ref or a recorded check for that product. "
                "Task refs and another product's evidence cannot establish its observation. "
                "If validation_repair is supplied, correct that invalid draft against the same evidence and "
                "return the full definition including coverage; preserve supported requirements and explain "
                "any unsupported claims you withhold. No absolute Outcome grade or satisfaction label.",
                {
                    **SCHEMA,
                    "coverage": [
                        {
                            "expectation_id": "supplied id",
                            "criterion_ids": ["retained IDs, empty when not mapped"],
                            "status": "mapped/deferred/excluded",
                            "reason": "basis for this disposition",
                            "subrequirements": [
                                {
                                    "part_id": "exact supplied id",
                                    "criterion_ids": ["final id"],
                                    "status": "mapped/deferred/excluded",
                                    "reason": "explicit retention or exclusion basis",
                                    "retained_text": "exact final clause when mapped",
                                }
                            ],
                        }
                    ],
                },
                extra={**inputs, **repair},
                # A repair carries both the original evidence and a complete invalid draft.
                # Give that bounded correction its own allowance; never truncate requirements.
                max_bytes=getattr(
                    reducer, "request_budget_bytes", 320000 if repair else 160000
                ),
                **projection,
            )
        record(
            "jury_reduce_raw" if attempt == 0 else f"jury_reduce_repair_{attempt}_raw",
            reduced,
        )
        try:
            normalized = normalize_reduced(reduced, context, evidence)
            try:
                output = validate(normalized, evidence, context)
            except ValueError as error:
                if str(error) != "User criterion must cite an actual user statement":
                    raise
                output = validate_reduced_standards(
                    normalized, evidence, context, record=record
                )
            output["independent_expectations"] = expectations
            output["proposal_expectations"] = observed
            output["coverage"] = reconcile_coverage(expectations + observed, output)
        except (ValueError, TypeError) as error:
            diagnostic = {
                "error_type": type(error).__name__,
                "reason": str(error),
                "field_issues": diagnostics(
                    reduced,
                    expectations + observed,
                    evidence=evidence,
                    product_ref=product_ref,
                ),
            }
            if str(error) == "Unknown expectation in coverage":
                allowed = {r["expectation_id"] for r in expectations + observed}
                diagnostic["unknown_expectation_ids"] = [
                    r.get("expectation_id")
                    for r in output["coverage"]
                    if r.get("expectation_id") not in allowed
                ]
            if any(
                isinstance(gap, dict)
                and gap.get("expected_basis") not in {"task", "user", "domain"}
                for gap in reduced.get("gaps", [])
            ):
                diagnostic["invalid_fields"] = [
                    {
                        "path": f"gaps[{i}].expected_basis",
                        "gap_id": gap.get("gap_id"),
                        "value": gap.get("expected_basis"),
                        "allowed_values": ["task", "user", "domain"],
                        "instruction": "Choose the actual source of the expectation and cite it, or withhold "
                        "an unsupported accusation. An observed failed check is evidence of a "
                        "problem, not itself a source of task/user requirements.",
                    }
                    for i, gap in enumerate(reduced.get("gaps", []))
                    if isinstance(gap, dict)
                    and gap.get("expected_basis") not in {"task", "user", "domain"}
                ]
            record(f"jury_reduce_validation_{attempt}", diagnostic)
            if attempt == 2:
                raise
            repair = {
                "validation_repair": {
                    "invalid_draft": reduced,
                    **diagnostic,
                }
            }
        else:
            break
    record("jury", output)
    return output


async def _reduce_once(
    context,
    evidence,
    inputs,
    reducer,
    *,
    record,
    repair,
    max_bytes=160000,
    strict=False,
    protocol_version=PROMPT_VERSION,
    clause_contract=False,
):
    from reward_engine.core.outcome.jury_validation import (
        ReduceValidationError,
        clause_schema,
        reconcile_coverage,
        validate_reduced_standards,
    )
    from reward_engine.core.outcome.task_inputs import (
        instruction as task_input_instruction,
    )

    expectations = inputs["independent_expectations"]
    observed = inputs["proposal_expectations"]
    from reward_engine.core.outcome.authored_json import enabled

    schema, clause_instruction = clause_schema(clause_contract)

    def finalize(raw):
        record("jury_reduce_raw", raw)
        if clause_contract and not enabled(reducer):
            from reward_engine.core.outcome.jury_drafts import normalize_part_bindings

            raw = normalize_part_bindings(raw, inputs, evidence)
        if strict and "part_decisions" not in raw:
            raise ValueError(
                "Construction Reduce must return part_decisions; omitted clauses remain deferred"
            )
        if (
            "part_decisions" in raw
            and compact_selection_inputs(inputs).get("selection_aliases")
            and raw.get("selection_protocol") != "exact-alias-v1"
        ):
            raise ValueError(
                "Compact clause decisions require selection_protocol=exact-alias-v1"
            )
        decisions = _apply_reduced_decisions(raw, inputs, evidence, strict=strict)
        output = validate_reduced_standards(
            decisions
            if enabled(reducer)
            else normalize_reduced(
                decisions,
                context,
                evidence,
            ),
            evidence,
            context,
            clause_contract=clause_contract,
            record=record,
        )
        output["independent_expectations"] = expectations
        output["proposal_expectations"] = observed
        output["coverage"] = reconcile_coverage(expectations + observed, output)
        output["prompt_version"] = protocol_version
        return output

    reduced = await ask(
        reducer,
        "outcome_jury_reduce",
        context,
        evidence,
        OUTCOME_JURY_REDUCE_PROMPT
        + clause_instruction
        + task_input_instruction(context)
        + (
            " Group semantically equivalent clauses by an explicit rewrite into ONE final clause; "
            "preserve distinct conditions and alternatives. Each retain/rewrite decision names clause_id "
            "in the final criterion. Its text must match that clause. Never accumulate the three "
            "members' illustrative methods into cumulative obligations."
            if clause_contract
            else ""
        ),
        {
            **schema,
            "selection_protocol": "exact-alias-v1",
            "part_decisions": [
                {
                    "parts": ["selection_id, e.g. p0"],
                    "action": "retain/rewrite/defer/exclude",
                    "criterion_id": "final criterion ID for retain/rewrite; omit otherwise",
                    **(
                        {
                            "clause_id": "exact ID in final criterion.clauses for retain/rewrite"
                        }
                        if clause_contract
                        else {}
                    ),
                    "reason": "required except unchanged retain",
                    "text": "only for rewrite: full replacement clause",
                    "evidence_refs": ["required for rewrite: supplied supporting refs"],
                }
            ],
        },
        extra={
            **compact_selection_inputs(inputs),
            **repair,
            "prompt_version": protocol_version,
        },
        max_bytes=max_bytes,
        **projection_options(inputs, record),
        **(
            {
                "validate": finalize,
                "max_contract_retries": 1,
                "max_evidence_rounds": 2,
                "close_evidence": True,
            }
            if strict or enabled(reducer)
            else {}
        ),
    )
    if strict or enabled(reducer):
        return reduced
    record("jury_reduce_raw", reduced)
    try:
        output = finalize(reduced)
    except (ValueError, TypeError, KeyError, AttributeError) as error:
        raise ReduceValidationError(error, reduced) from error
    return output


def _apply_reduced_decisions(raw, inputs, evidence, *, strict):
    """Report independent retirement and binding failures in the same repair.

    Neither validator's result can be adopted if either rejects the draft.
    Keep their diagnostics tied to the same normalized input and array indices.
    """
    failures = []
    if strict:
        from reward_engine.core.outcome.jury_refinement import check_retired_clauses

        try:
            check_retired_clauses(raw, inputs)
        except ValueError as failure:
            failures.append(failure)
    try:
        result = apply_part_decisions(raw, inputs, evidence, mark_omissions=strict)
    except ValueError as failure:
        failures.append(failure)
    if failures:
        first = failures[0]
        if len(failures) > 1:
            if not isinstance(first, ResponseContractError):
                first = ResponseContractError("part_decisions", str(first))
            first.details["validation_errors"] = [
                deepcopy(f.details)
                if isinstance(f, ResponseContractError)
                else {"field": "part_decisions", "reason": str(f)}
                for f in failures
            ]
            first.details["instruction"] = (
                "Correct all validation_errors in the same draft. "
                + first.details.get("instruction", "")
            )
        raise first
    return result
