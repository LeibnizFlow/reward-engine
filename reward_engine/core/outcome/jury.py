"""Three concurrent outcome proposals, one evidence-aware Reduce. No ranking gate."""

from copy import deepcopy

from reward_engine.core.batch import BatchPolicy, collect_batch
from reward_engine.core.outcome.construction import fingerprint
from reward_engine.core.outcome.context import (
    ResponseContractError,
    ask,
    requirement_context,
)

# Preserve the established module import surface.
from reward_engine.core.outcome.context import require_refs as require_refs
from reward_engine.core.outcome.context import used_previews as used_previews
from reward_engine.core.outcome.jury_drafts import (
    apply_part_decisions as apply_part_decisions,
)
from reward_engine.core.outcome.jury_drafts import (
    compact_selection_inputs as compact_selection_inputs,
)
from reward_engine.core.outcome.jury_drafts import expectations as draft_expectations
from reward_engine.core.outcome.jury_drafts import (
    keep_draft,
    reduction_inputs,
)
from reward_engine.core.outcome.jury_drafts import (
    normalize_reduced as normalize_reduced,
)
from reward_engine.core.outcome.jury_drafts import resolve_parts as resolve_parts
from reward_engine.core.outcome.jury_prompts import (
    OUTCOME_EXPECTATIONS_PROMPT,
    OUTCOME_JURY_MAP_PROMPT,
    OUTCOME_JURY_REDUCE_PROMPT,
    PROMPT_VERSION,
)
from reward_engine.core.outcome.jury_ranking import _RankingRepair as _RankingRepair
from reward_engine.core.outcome.jury_ranking import (
    continuation_decision as continuation_decision,
)
from reward_engine.core.outcome.jury_ranking import rank as rank
from reward_engine.core.outcome.jury_ranking import validate_ranking as validate_ranking
from reward_engine.core.outcome.jury_reduction import (
    _apply_reduced_decisions as _apply_reduced_decisions,
)
from reward_engine.core.outcome.jury_reduction import _reduce_once as _reduce_once
from reward_engine.core.outcome.jury_reduction import (
    _reduce_targeted as _reduce_targeted,
)
from reward_engine.core.outcome.jury_reduction import (
    projection_options as projection_options,
)
from reward_engine.core.outcome.jury_reduction import (
    reduce_proposals as reduce_proposals,
)
from reward_engine.core.outcome.jury_repair import apply_edits as apply_edits
from reward_engine.core.outcome.jury_repair import diagnostics as diagnostics
from reward_engine.core.outcome.jury_validation import (
    ReduceValidationError as ReduceValidationError,
)
from reward_engine.core.outcome.jury_validation import clause_schema as clause_schema
from reward_engine.core.outcome.jury_validation import product_ref as product_ref
from reward_engine.core.outcome.jury_validation import (
    reconcile_coverage as reconcile_coverage,
)
from reward_engine.core.outcome.jury_validation import validate as validate
from reward_engine.core.outcome.jury_validation import (
    validate_proposal as validate_proposal,
)
from reward_engine.core.outcome.jury_validation import (
    validate_reduced_standards as validate_reduced_standards,
)
from reward_engine.core.outcome.response_receipts import defer_rejections


async def ask_draft(*args, **kwargs):
    """Keep a completed substantive proposal after its one bounded correction.

    Drafts have no modification authority. Reduce must validate the final
    standard; transport, syntax and missing-evidence failures still propagate.
    """
    call, *rest = args
    from reward_engine.core.outcome.authored_json import enabled

    if enabled(call):
        return await ask(call, *rest, **kwargs)
    earlier = None

    async def remember(request):
        nonlocal earlier
        value = await call(request)
        if (
            isinstance(value, dict)
            and "need_evidence" not in value
            and isinstance(value.get("criteria"), list)
            and value["criteria"]
        ):
            earlier = deepcopy(value)
        return value

    for name in (
        "protocol_edits",
        "bounded_stage",
        "contract_retries",
        "stage_timeout_s",
        "max_request_bytes",
        "compress_request",
        "prepare_request",
        "request_budget_bytes",
        "evidence_expansion_limit",
    ):
        if hasattr(call, name):
            setattr(remember, name, getattr(call, name))
    try:
        return await ask(remember, *rest, **kwargs)
    except ResponseContractError as error:
        raw = getattr(error, "rejected_response", None)
        if not isinstance(raw, dict) or "need_evidence" in raw:
            raise
        if not isinstance(raw.get("criteria"), list) or not raw["criteria"]:
            if earlier is None:
                raise
            return {
                **deepcopy(earlier),
                "draft_response_failure": deepcopy(error.details),
                "draft_correction_response": deepcopy(raw),
                "draft_recovery": "Earlier substantive draft retained after correction changed schema. "
                "Neither response is authoritative; inspect the correction for factual objections "
                "and resolve them against Task/evidence before adopting any standard.",
            }
        return {**deepcopy(raw), "draft_response_failure": deepcopy(error.details)}


SCHEMA = {
    "criteria": [
        {
            "id": "open string",
            "standard": "Task outcome requirement",
            "kind": "hard or soft",
            "priority": "critical/important/minor: contribution to the requested outcome, not veto authority",
            "priority_reason": "why this dimension matters relative to the other requested outcomes",
            "origin": "task or user or domain or observed_gap",
            "evidence_refs": ["task or product refs"],
            "measurement_question": "how to check",
            "subrequirements": [
                "nonempty string only, never a {part_id,text} object; short explicit important subrequirement, not a quota"
            ],
        }
    ],
    "gaps": [
        {
            "gap_id": "unique key",
            "product_ids": ["supplied product IDs"],
            "expected": "desired result",
            "expected_basis": "exactly one enum string: task OR user OR domain; no explanation or combined values",
            "expected_refs": [
                "Task, actual user update or source refs; domain prior may have none"
            ],
            "observed": "what the product/environment actually shows",
            "gap": "specific discrepancy",
            "evidence_refs": ["actual product or check refs"],
            "verification": "what observation/test could support OR reject this suspected gap",
            "status": "needs_check or check_supported",
        }
    ],
    "uncertainties": ["unresolved questions"],
}


def field_repair_call(call):
    """Let the originating Jury repair metadata without rewriting its substantive draft."""

    from reward_engine.core.outcome.authored_json import enabled

    if enabled(call):
        return call

    async def wrapped(request):
        recovery = request.get("response_recovery", {})
        fields = recovery.get("details", {}).get("criterion_field_repairs")
        original = recovery.get("rejected_response")
        if not fields or not isinstance(original, dict):
            return await call(request)
        current = deepcopy(request)
        current["mode"] += "_field_repair"
        current["response"] = {
            "updates": [
                {
                    "id": r["id"],
                    "kind": "hard OR soft",
                    "origin": "task OR user OR domain OR observed_gap",
                }
                for r in fields
            ]
        }
        current["instruction"] = (
            "Repair ONLY the listed kind/origin fields in your rejected criteria. Return updates for "
            "exactly the listed IDs; never rewrite standard, subrequirements, rationale or evidence. "
            "kind means hard/soft obligation strength; important/critical/minor are priorities, not kind. "
            "origin means task/user/domain/observed_gap; shared_instruction is not a Task requirement. "
            "Domain or observed_gap origins MUST have kind=soft. If the original Task does not actually "
            "require a rule, retain its domain provenance and make it soft; never invent Task support. "
            "For actual Task requirements use task provenance with the appropriate strength. "
            "The host applies this exact patch and revalidates the entire unchanged draft."
        )
        current["response_recovery"]["instruction"] = current["instruction"]
        patch = await call(current)
        updates = patch.get("updates") if isinstance(patch, dict) else None
        if (
            not isinstance(updates, list)
            or len(updates) != len(fields)
            or any(
                not isinstance(r, dict) or set(r) != {"id", "kind", "origin"}
                for r in updates
            )
            or {r["id"] for r in updates} != {r["id"] for r in fields}
            or any(
                r["kind"] not in {"hard", "soft"}
                or r["origin"] not in {"task", "user", "domain", "observed_gap"}
                for r in updates
            )
        ):
            raise ResponseContractError(
                "updates",
                "Return exactly the listed criterion IDs and allowed kind/origin fields",
            )
        result = deepcopy(original)
        by_id = {r["id"]: r for r in updates}
        for row in result["criteria"]:
            # Revision drafts can retain an earlier criterion by inherit_from.
            # Those references have no local id and must remain untouched while
            # repairing metadata on the explicitly supplied criteria.
            if row.get("id") in by_id:
                row.update(by_id[row["id"]])
        return result

    for name in (
        "protocol_edits",
        "bounded_stage",
        "contract_retries",
        "stage_timeout_s",
        "max_request_bytes",
        "compress_request",
        "prepare_request",
        "request_budget_bytes",
        "evidence_expansion_limit",
    ):
        if hasattr(call, name):
            setattr(wrapped, name, getattr(call, name))
    return wrapped


async def propose(
    context,
    evidence,
    members,
    reducer,
    *,
    policy=None,
    expectation_cache=None,
    reduce_max_bytes=160000,
    record=lambda *_: None,
    construction_rounds=1,
    revision_protocol=False,
):
    if type(construction_rounds) is not int or construction_rounds not in (1, 2, 3):
        raise ValueError("Use one to three Jury construction rounds")
    if construction_rounds != 1:
        from reward_engine.core.outcome.jury_refinement import construct

        return await construct(
            context,
            evidence,
            members,
            reducer,
            rounds=construction_rounds,
            policy=policy,
            expectation_cache=expectation_cache,
            reduce_max_bytes=reduce_max_bytes,
            record=record,
            revision_protocol=revision_protocol,
        )
    return await _propose_once(
        context,
        evidence,
        members,
        reducer,
        policy=policy,
        expectation_cache=expectation_cache,
        reduce_max_bytes=reduce_max_bytes,
        record=record,
        revision_protocol=revision_protocol,
    )


async def _propose_once(
    context,
    evidence,
    members,
    reducer,
    *,
    policy=None,
    expectation_cache=None,
    reduce_max_bytes=160000,
    record=lambda *_: None,
    strict=False,
    task_only=False,
    protocol_version=PROMPT_VERSION,
    clause_contract=False,
    minimum_members=None,
    revision_protocol=False,
):
    from reward_engine.core.outcome.context import requirement_scope

    if type(reduce_max_bytes) is not int or reduce_max_bytes < 1:
        raise ValueError("reduce_max_bytes must be a positive integer")
    minimum_members = (
        (3 if strict else 2) if minimum_members is None else minimum_members
    )
    if type(minimum_members) is not int or minimum_members not in (2, 3):
        raise ValueError("Jury requires two or three independent members")
    from reward_engine.core.outcome.authored_json import enabled

    if enabled(reducer):
        minimum_members = 3
        if len(members) != 3:
            raise ValueError("Authored JSON requires exactly three independent members")
    if strict:
        members = {key: field_repair_call(call) for key, call in members.items()}
    blind_context, blind_evidence = requirement_context(context, evidence)
    if task_only:
        blind_context["skills"] = []
        blind_context.pop("reference_documents", None)
        allowed = {"task"} | {
            u["ref"] for p in blind_context["products"] for u in p["user_updates"]
        }
        blind_evidence = {r: v for r, v in blind_evidence.items() if r in allowed}
        blind_context["available_evidence_refs"] = list(blind_evidence)
    scope = (
        requirement_scope(blind_context, blind_evidence)
        if task_only
        else requirement_scope(context, evidence)
    )
    schema, clause_instruction = clause_schema(clause_contract)
    from reward_engine.core.outcome.authored_json import VERSION as AUTHORED_VERSION
    from reward_engine.core.outcome.authored_json import enabled

    if enabled(reducer):
        scope = {**scope, "role_output_protocol": AUTHORED_VERSION}
    if clause_contract:
        scope = {**scope, "clause_protocol": "outcome-clauses/2026-09-11"}
    record(
        "jury_protocol",
        {
            "version": protocol_version,
            "expectations": OUTCOME_EXPECTATIONS_PROMPT,
            "map": OUTCOME_JURY_MAP_PROMPT,
            "reduce": OUTCOME_JURY_REDUCE_PROMPT,
        },
    )
    if expectation_cache is None:
        expectation_cache = {}
    if (
        expectation_cache.get("scope") != scope
        or expectation_cache.get("prompt_version") != protocol_version
    ):
        expectation_cache.clear()
        expectation_cache.update(scope=scope, seats={}, prompt_version=protocol_version)

    @defer_rejections
    async def initial_proposal(key, call):
        cached = expectation_cache["seats"].get(key)
        initial = (
            deepcopy(cached)
            if cached is not None
            else await (ask_draft if clause_contract else ask)(
                call,
                "outcome_expectations",
                blind_context,
                blind_evidence,
                OUTCOME_EXPECTATIONS_PROMPT + clause_instruction,
                {
                    "criteria": [
                        {
                            **schema["criteria"][0],
                            "why": "importance to the Task outcome",
                            "priority": "critical/important/minor",
                        }
                    ]
                },
                extra={"prompt_version": protocol_version},
                validate=lambda row: (
                    validate(
                        row,
                        blind_evidence,
                        blind_context,
                        clause_contract=clause_contract,
                    )
                    if strict
                    else keep_draft(row, blind_evidence, blind_context, validate)
                ),
                max_contract_retries=1,
                max_syntax_retries=1,
                max_evidence_rounds=2,
                close_evidence=True,
            )
        )
        initial = keep_draft(
            initial,
            blind_evidence,
            blind_context,
            lambda r, e, c: validate(r, e, c, clause_contract=clause_contract),
        )
        # A substantive draft already accepted for Reduce remains reusable after
        # another seat fails. Preserve its validation issues; caching does not
        # promote it to standards or bypass final provenance validation.
        expectation_cache["seats"][key] = deepcopy(initial)
        expectations = draft_expectations(initial["criteria"], key, "before_products")
        record(
            f"expectations/{key}",
            {
                **initial,
                "criteria": expectations,
                "cache_hit": cached is not None,
                "prompt_version": protocol_version,
            },
        )
        return initial

    from reward_engine.core.outcome import jury_revision

    full_input_identity = fingerprint(
        {
            "context": context,
            "evidence": evidence,
            "map_prompt": OUTCOME_JURY_MAP_PROMPT,
            "clause_contract": clause_contract,
            "protocol_version": protocol_version,
            **(
                {"revision_protocol": jury_revision.VERSION}
                if revision_protocol
                else {}
            ),
        }
    )
    pending_proposals = expectation_cache.setdefault("pending_proposals", {})

    @defer_rejections
    async def one(key, call, initial):
        from reward_engine.core.outcome.task_inputs import (
            instruction as task_input_instruction,
        )

        identity = fingerprint({"input": full_input_identity, "initial": initial})
        cached = pending_proposals.get(key)
        if cached and cached.get("identity") == identity:
            proposal = deepcopy(cached["proposal"])
            record(
                f"proposal_recovery/{key}",
                {
                    "cache_hit": True,
                    "identity": identity,
                    "boundary": "Same-input partial batch draft, not an adopted standard or verdict.",
                },
            )
            record(f"proposals/{key}", proposal)
            return proposal
        expectations = draft_expectations(initial["criteria"], key, "before_products")

        def prepare_draft(row):
            from reward_engine.core.outcome.source_projection import (
                enabled as canonical_enabled,
            )

            if revision_protocol:
                row = jury_revision.materialize(
                    row, expectations, evidence, canonical=canonical_enabled(call)
                )
            return (
                validate(
                    validate_proposal(
                        row, evidence, context, canonical=canonical_enabled(call)
                    ),
                    evidence,
                    context,
                    clause_contract=clause_contract,
                )
                if strict
                else keep_draft(row, evidence, context, validate)
            )

        draft = await (ask_draft if clause_contract or revision_protocol else ask)(
            call,
            "outcome_jury_map",
            context,
            evidence,
            OUTCOME_JURY_MAP_PROMPT
            + clause_instruction
            + task_input_instruction(context)
            + (jury_revision.INSTRUCTION if revision_protocol else ""),
            {
                **schema,
                "criteria": [
                    {**schema["criteria"][0], "why": (
                        "Preserve the original value and presence for retain; omit if originally absent. "
                        "For add/rewrite/merge, explain importance to the Task outcome."
                        if revision_protocol else "importance to the Task outcome"
                    )}
                ],
                **(
                    {"revision_decisions": [jury_revision.SCHEMA]}
                    if revision_protocol
                    else {}
                ),
            },
            extra={
                "independent_expectations": expectations,
                "prompt_version": protocol_version,
                **(
                    {"revision_protocol": jury_revision.VERSION}
                    if revision_protocol
                    else {}
                ),
                "initial_draft_validation_issues": initial.get(
                    "draft_validation_issues", []
                ),
            },
            validate=prepare_draft,
            max_contract_retries=1,
            max_syntax_retries=1,
            max_evidence_rounds=2,
            close_evidence=True,
        )
        if revision_protocol:
            try:
                draft = jury_revision.materialize(draft, expectations, evidence)
            except ResponseContractError as error:
                if strict or not draft.get("draft_response_failure"):
                    raise
                # A failed draft ledger has no authority to retain, rewrite or
                # delete requirements. Preserve its substantive text for Reduce,
                # with ALL originals unresolved and ALL additions unclaimed.
                # Do not repair its assertions or relax final-standard validation.
                rejected = deepcopy(draft.get("revision_decisions"))
                rejected_criteria = deepcopy(draft.get("criteria"))
                try:
                    draft = jury_revision.materialize(
                        {**draft, "revision_decisions": []}, expectations, evidence
                    )
                except ResponseContractError as criteria_error:
                    # The bounded correction can also corrupt the criteria
                    # structure itself (for example inherit_from overrides or
                    # empty clauses). Fall back to this seat's already validated
                    # independent expectations. They remain drafts and every
                    # source clause is deferred for Reduce; the rejected revision
                    # stays available as audit evidence, never as authority.
                    draft = jury_revision.materialize(
                        {
                            **draft,
                            "criteria": [
                                jury_revision.criterion_body(row)
                                for row in expectations
                            ],
                            "revision_decisions": [],
                        },
                        expectations,
                        evidence,
                    )
                    draft["revision_criteria_failure"] = deepcopy(
                        criteria_error.details
                    )
                draft["rejected_revision_decisions"] = rejected
                draft["rejected_revision_criteria"] = rejected_criteria
                draft["revision_ledger_failure"] = deepcopy(error.details)
            record(f"revisions/{key}", draft)
        # Map creates drafts, not publishable authority. Keep a substantive draft
        # and its contract error for the existing Reduce instead of losing all
        # other criteria because one enum/provenance field was wrong.
        draft = keep_draft(
            draft,
            evidence,
            context,
            lambda r, e, c: validate(r, e, c, clause_contract=clause_contract),
        )
        proposal = {
            **draft,
            "independent_expectations": expectations,
            "proposal_expectations": draft_expectations(
                draft["criteria"], f"{key}:observed", "after_products"
            ),
            "initial_draft": initial,
        }
        pending_proposals[key] = {"identity": identity, "proposal": deepcopy(proposal)}
        record(f"proposals/{key}", proposal)
        return proposal

    # Each phase is one model call per seat. Do not let time spent proposing
    # expectations consume the later product-review call's entire deadline.
    policy = policy or BatchPolicy()
    initial_batch = await collect_batch(
        {
            key: lambda k=key, c=call: initial_proposal(k, c)
            for key, call in members.items()
        },
        policy,
    )
    record("jury_expectations_map", initial_batch.summary())
    initial = {
        row.job_id: row.value for row in initial_batch.items if row.status == "ok"
    }
    if len(initial) < minimum_members:
        failure = RuntimeError(
            "Fewer than three usable independent expectations; keep raw failures"
            if minimum_members == 3
            else "Fewer than two usable independent expectations; keep raw failures"
        )
        failure.jury_failures = seat_failures(initial_batch)
        record("jury_failure_details", failure.jury_failures)
        raise failure
    batch = await collect_batch(
        {key: lambda k=key: one(k, members[k], initial[k]) for key in initial}, policy
    )
    record("jury_map", batch.summary())
    good = [row.value for row in batch.items if row.status == "ok"]
    if len(good) < minimum_members:
        failure = RuntimeError(
            "Fewer than three usable outcome proposals; keep raw failures"
            if minimum_members == 3
            else "Fewer than two usable outcome proposals; keep raw failures"
        )
        failure.jury_failures = seat_failures(batch)
        record("jury_failure_details", failure.jury_failures)
        raise failure
    usable = [row.job_id for row in batch.items if row.status == "ok"]
    record(
        "jury_participation",
        {
            "requested_seats": list(members),
            "usable_seats": usable,
            "missing_seats": [key for key in members if key not in usable],
            "minimum_members": minimum_members,
            "degraded": len(usable) < len(members),
            "boundary": "Missing output is unavailable, not agreement; Reduce still validates the full standard contract.",
        },
    )
    # Full batches use the existing Reduce checkpoint; this cache only bridges
    # partial member failure, not later requests for new semantic deliberation.
    pending_proposals.clear()
    return await reduce_proposals(
        context,
        evidence,
        reduction_inputs(good),
        reducer,
        reduce_max_bytes=reduce_max_bytes,
        record=record,
        strict=strict,
        protocol_version=protocol_version,
        clause_contract=clause_contract,
    )


def seat_failures(batch):
    """Keep the quorum failure and each rejected seat's actual underlying cause."""
    return [
        {
            "seat": row.job_id,
            "status": row.status,
            "error_type": type(row.error).__name__ if row.error else None,
            "reason": str(row.error) if row.error else None,
            "details": deepcopy(getattr(row.error, "details", None)),
            "operation_budget": deepcopy(getattr(row.error, "operation_budget", None)),
        }
        for row in batch.items
    ]
