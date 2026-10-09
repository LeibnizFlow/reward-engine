"""Three concurrent outcome proposals, one evidence-aware Reduce. No ranking gate."""

from copy import deepcopy

from reward_engine.core.batch import BatchPolicy, collect_batch
from reward_engine.core.outcome.construction import fingerprint
from reward_engine.core.outcome.context import (
    ResponseContractError,
    ask,
    require_refs,
    requirement_context,
)
from reward_engine.core.outcome.jury_drafts import (
    apply_part_decisions,
    compact_selection_inputs,
    keep_draft,
    normalize_reduced,
    reduction_inputs,
    resolve_parts,
)
from reward_engine.core.outcome.jury_drafts import expectations as draft_expectations
from reward_engine.core.outcome.jury_prompts import (
    OUTCOME_EXPECTATIONS_PROMPT,
    OUTCOME_JURY_MAP_PROMPT,
    OUTCOME_JURY_REDUCE_PROMPT,
    PROMPT_VERSION,
)
from reward_engine.core.outcome.jury_repair import apply_edits, diagnostics


async def ask_draft(*args, **kwargs):
    """Keep a completed substantive proposal after its one bounded correction.

    Drafts have no modification authority. Reduce must validate the final
    standard; transport, syntax and missing-evidence failures still propagate.
    """
    call, *rest = args
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
        "bounded_stage",
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


def clause_schema(enabled):
    schema = deepcopy(SCHEMA)
    if not enabled:
        return schema, ""
    from reward_engine.core.outcome.clauses import INSTRUCTION
    from reward_engine.core.outcome.clauses import SCHEMA as CLAUSE_SCHEMA

    schema["criteria"][0].pop("subrequirements", None)
    schema["criteria"][0]["clauses"] = [deepcopy(CLAUSE_SCHEMA)]
    return schema, INSTRUCTION


class ReduceValidationError(ValueError):
    """A model reply failed publication; successful Maps remain reusable."""

    def __init__(self, error, raw):
        super().__init__(str(error))
        self.raw = deepcopy(raw)


def product_ref(ref, product, evidence):
    value = evidence.get(ref)
    return ref.startswith(product + ":") or (
        ref.startswith("check:")
        and isinstance(value, dict)
        and value.get("product_id") == product
    )


def validate(result, evidence, context=None, *, clause_contract=False):
    """Without context only non-user criteria can be checked; scoped claims need it."""
    if not isinstance(result, dict):
        raise ValueError("Outcome definition must be an object")
    rows = result.get("criteria")
    if (
        not isinstance(rows, list)
        or not rows
        or len(rows) > 20
        or any(not isinstance(r, dict) for r in rows)
    ):
        raise ResponseContractError(
            "criteria",
            "Need a small nonempty Outcome definition in the exact top-level criteria field: "
            "a list of 1..20 criterion objects; preserve the supplied draft content when fixing its wrapper",
        )
    from reward_engine.core.outcome.clauses import normalize_criteria

    result = {
        **deepcopy(result),
        "criteria": normalize_criteria(
            rows, context, evidence, required=clause_contract
        ),
    }
    rows = result["criteria"]
    # Surface wire-shape and reference errors in the same bounded redo. Draft
    # input clauses have IDs, but this output contract requires plain strings.
    # Never infer provenance or silently discard fields to repair a response.
    structure_issues = []
    field_repairs = []
    metadata_issue_count = 0
    for index, row in enumerate(rows):
        invalid_metadata = False
        for field, allowed in (
            ("kind", ("hard", "soft")),
            ("origin", ("task", "user", "domain", "observed_gap")),
        ):
            if row.get(field) not in allowed:
                invalid_metadata = True
                metadata_issue_count += 1
                structure_issues.append(
                    f"criteria[{index}].{field}: actual={row.get(field)!r}; "
                    f"allowed: {', '.join(allowed)}. "
                    + (
                        "kind is obligation strength, NOT priority; important belongs in priority."
                        if field == "kind"
                        else "origin identifies actual Task/user support or a soft domain proposal; "
                        "shared_instruction is not Task evidence. Do not invent provenance."
                    )
                )
        if row.get("kind") == "hard" and row.get("origin") in {
            "domain",
            "observed_gap",
        }:
            invalid_metadata = True
            metadata_issue_count += 1
            structure_issues.append(
                f"criteria[{index}]: Domain preferences cannot become hard Task requirements; "
                "domain/observed_gap origin requires kind=soft. Keep actual provenance; do not relabel it task."
            )
        if invalid_metadata:
            field_repairs.append({"id": row.get("id"), "fields": ["kind", "origin"]})
        if "subrequirements" not in row:
            continue
        parts = row["subrequirements"]
        field = f"criteria[{index}].subrequirements"
        if not isinstance(parts, list):
            structure_issues.append(
                f"{field}: actual={parts!r}; required: list of nonempty strings"
            )
            continue
        for part_index, part in enumerate(parts):
            if not isinstance(part, str) or not part.strip():
                structure_issues.append(
                    f"{field}[{part_index}]: actual={part!r}; required: nonempty string, "
                    "not an object (do not copy input part_id/text records)"
                )
    gaps = result.get("gaps", [])
    for index, row in enumerate(gaps if isinstance(gaps, list) else []):
        if isinstance(row, dict) and row.get("expected_basis") not in (
            "task",
            "user",
            "domain",
        ):
            structure_issues.append(
                f"gaps[{index}].expected_basis: actual={row.get('expected_basis')!r}; "
                "allowed: exactly one of 'task', 'user', 'domain'; "
                "put explanations in expected/verification, not this enum"
            )

    def check_reference_fields(value, path=""):
        if isinstance(value, dict):
            for key, item in value.items():
                field = f"{path}.{key}" if path else key
                if key not in {"evidence_refs", "expected_refs"}:
                    check_reference_fields(item, field)
                elif isinstance(item, list):
                    for index, ref in enumerate(item):
                        if not isinstance(ref, str) or ref not in evidence:
                            structure_issues.append(
                                f"{field}[{index}]: actual={ref!r}; required: an exact "
                                "supplied evidence handle; do not invent or change its namespace"
                            )
                elif item or key == "evidence_refs":
                    structure_issues.append(
                        f"{field}: actual={item!r}; required: a list of exact supplied evidence handles"
                    )
        elif isinstance(value, list):
            for index, item in enumerate(value):
                check_reference_fields(item, f"{path}[{index}]")

    check_reference_fields(result)
    if structure_issues:
        error = ResponseContractError(
            "response", "Response structure errors:\n" + "\n".join(structure_issues)
        )
        if (
            field_repairs
            and len(structure_issues) == metadata_issue_count
            and all(isinstance(r.get("id"), str) and r["id"] for r in rows)
            and len({r["id"] for r in rows}) == len(rows)
        ):
            error.details["criterion_field_repairs"] = field_repairs
        raise error
    ids = set()
    scoped_users = {
        p["product_id"]: {r["ref"] for r in p["user_updates"]}
        for p in (context or {}).get("products", [])
    }
    user_refs = set().union(*scoped_users.values()) if scoped_users else set()
    for row in rows:
        if "priority" in row and (
            not isinstance(row["priority"], str)
            or row["priority"] not in {"critical", "important", "minor"}
        ):
            raise ValueError(
                "Criterion priority must be critical, important or minor; it is not a gate"
            )
        if "priority_reason" in row and (
            not isinstance(row["priority_reason"], str)
            or not row["priority_reason"].strip()
        ):
            raise ValueError(
                "Explain the criterion priority in terms of the requested outcome"
            )
        for key in ("id", "standard", "measurement_question"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(
                    f"Incomplete Outcome criterion: {key} must be nonempty text"
                )
        if row["id"] in ids:
            raise ValueError(f"Duplicate criterion id: {row['id']!r}")
        for field, allowed in (
            ("kind", ("hard", "soft")),
            ("origin", ("task", "user", "domain", "observed_gap")),
        ):
            if row.get(field) not in allowed:
                raise ValueError(
                    f"Criterion {row['id']!r}.{field}={row.get(field)!r}; allowed: {', '.join(allowed)}"
                )
        ids.add(row["id"])
        if "subrequirements" in row and (
            not isinstance(row["subrequirements"], list)
            or any(
                not isinstance(p, str) or not p.strip() for p in row["subrequirements"]
            )
        ):
            raise ValueError("Published subrequirements must be explicit text")
        require_refs(row.get("evidence_refs"), evidence)
        if row["kind"] == "hard" and row["origin"] not in {"task", "user"}:
            raise ValueError("Domain preferences cannot become hard Task requirements")
        if (
            row["kind"] == "hard"
            and row["origin"] == "task"
            and "task" not in row["evidence_refs"]
        ):
            raise ValueError("A hard Task requirement must cite the Task")
        if row["origin"] == "user" and context is None:
            raise ValueError("User criterion validation requires scoped context")
        if row["origin"] == "user" and not set(row["evidence_refs"]) & user_refs:
            raise ValueError("User criterion must cite an actual user statement")
    gap_ids = set()
    products = {p["product_id"] for p in (context or {}).get("products", [])}
    gaps = result.get("gaps", [])
    if not isinstance(gaps, list) or any(not isinstance(r, dict) for r in gaps):
        raise ValueError("Gaps must be a list of objects")
    if not products:
        # Pre-product questions are useful plans, not observed defects. Preserve
        # their exact content outside the gap list; never invent a product ID.
        pending = [
            row
            for row in gaps
            if row.get("product_ids") == [] and row.get("status") == "needs_check"
        ]
        if pending:
            result.setdefault("unobserved_gap_hypotheses", []).extend(deepcopy(pending))
            gaps = [row for row in gaps if row not in pending]
            result["gaps"] = gaps
    for row in gaps:
        if any(
            not isinstance(row.get(k), str) or not row[k].strip()
            for k in ("gap_id", "expected", "observed", "gap", "verification")
        ):
            raise ValueError(
                "Gap needs expectation, observation, discrepancy and verification"
            )
        if row["gap_id"] in gap_ids:
            raise ValueError("Duplicate gap id")
        gap_ids.add(row["gap_id"])
        targets = row.get("product_ids")
        if not isinstance(targets, list) or not targets or not set(targets) <= products:
            raise ValueError("Gap must identify observed products")
        refs = require_refs(row.get("evidence_refs"), evidence)
        if row.get("expected_basis") not in {"task", "user", "domain"}:
            raise ValueError("Expectation needs provenance")
        expected_refs = row.get("expected_refs", [])
        if expected_refs:
            require_refs(expected_refs, evidence)
        if row["expected_basis"] == "task" and "task" not in expected_refs:
            raise ValueError("Task expectation must refer to the Task")
        if row["expected_basis"] == "user" and not set(expected_refs) & user_refs:
            raise ValueError(
                "User expectation requires an actual user statement, not Actor narration"
            )
        for product in targets:
            if row["expected_basis"] == "user" and not set(
                expected_refs
            ) & scoped_users.get(product, set()):
                raise ValueError(
                    "Do not transfer one product's user requirements to another product"
                )
            if not any(product_ref(r, product, evidence) for r in refs):
                raise ValueError("Gap must cite evidence for every affected product")
        if row.get("status") not in {"needs_check", "check_supported"}:
            raise ValueError("Gap is a suspicion unless a real check supports it")
        if row["status"] == "check_supported":
            for product in targets:
                if not any(
                    r.startswith("check:")
                    and evidence[r].get("product_id") == product
                    and evidence[r].get("status") == "fail"
                    and not evidence[r].get("error")
                    for r in refs
                ):
                    raise ValueError("Votes cannot replace a recorded failed check")
    return result


def field_repair_call(call):
    """Let the originating Jury repair metadata without rewriting its substantive draft."""

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
            if row["id"] in by_id:
                row.update(by_id[row["id"]])
        return result

    for name in (
        "bounded_stage",
        "stage_timeout_s",
        "max_request_bytes",
        "compress_request",
        "prepare_request",
    ):
        if hasattr(call, name):
            setattr(wrapped, name, getattr(call, name))
    return wrapped


def validate_proposal(result, evidence, context):
    """A draft cannot manufacture an executed check; keep unsupported gaps provisional."""
    result = deepcopy(result)
    changes = []
    for gap in result.get("gaps", []):
        if gap.get("status") != "check_supported":
            continue
        products = gap.get("product_ids", [])
        refs = gap.get("evidence_refs", [])
        supported = bool(products) and all(
            any(
                isinstance(ref, str)
                and ref.startswith("check:")
                and isinstance(evidence.get(ref), dict)
                and evidence[ref].get("product_id") == product
                and evidence[ref].get("status") == "fail"
                and not evidence[ref].get("error")
                for ref in refs
            )
            for product in products
        )
        if not supported:
            changes.append(
                {
                    "gap_id": gap.get("gap_id"),
                    "claimed_status": "check_supported",
                    "admitted_status": "needs_check",
                    "reason": "No supplied failed-check receipt for every target",
                }
            )
            gap["status"] = "needs_check"
    if changes:
        result["gap_authority_corrections"] = changes
    return validate(result, evidence, context)


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
                **(
                    {
                        "validate": lambda row: validate(
                            row,
                            blind_evidence,
                            blind_context,
                            clause_contract=clause_contract,
                        ),
                        "max_contract_retries": 1,
                        "max_evidence_rounds": 2,
                        "close_evidence": True,
                    }
                    if strict
                    else {}
                ),
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
            if revision_protocol:
                row = jury_revision.materialize(row, expectations, evidence)
            return (
                validate(
                    validate_proposal(row, evidence, context),
                    evidence,
                    context,
                    clause_contract=clause_contract,
                )
                if strict
                else row
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
            **({"validate": prepare_draft} if strict or revision_protocol else {}),
            **({"max_contract_retries": 1} if revision_protocol and not strict else {}),
            **(
                {
                    "max_contract_retries": 1,
                    "max_evidence_rounds": 2,
                    "close_evidence": True,
                }
                if strict
                else {}
            ),
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
        raise RuntimeError(
            "Fewer than three usable independent expectations; keep raw failures"
            if minimum_members == 3
            else "Fewer than two usable independent expectations; keep raw failures"
        )
    batch = await collect_batch(
        {key: lambda k=key: one(k, members[k], initial[k]) for key in initial}, policy
    )
    record("jury_map", batch.summary())
    good = [row.value for row in batch.items if row.status == "ok"]
    if len(good) < minimum_members:
        raise RuntimeError(
            "Fewer than three usable outcome proposals; keep raw failures"
            if minimum_members == 3
            else "Fewer than two usable outcome proposals; keep raw failures"
        )
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
    if type(reduce_max_bytes) is not int or reduce_max_bytes < 1:
        raise ValueError("reduce_max_bytes must be a positive integer")
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
                "Return one to 64 edits only; group repeated field corrections into a replacement of "
                "their existing parent list when needed. Each edit has op replace/remove, an array path, value for replace, and "
                "reason. Paths address the evolving document; remove array entries in descending index "
                "order. Every path must already exist. Example: {edits: [{op: replace, "
                "path: [gaps, 0, expected_basis], value: domain, reason: Professional prior only}]}. "
                "If invalid_edits is present, correct those edits against invalid_draft.",
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
                diagnostic = {"error_type": type(error).__name__, "reason": str(error)}
                record(f"jury_reduce_validation_{attempt}", diagnostic)
                if attempt == 2:
                    raise
                repair = {
                    "validation_repair": {
                        "invalid_draft": invalid_draft,
                        "invalid_edits": reply,
                        "field_issues": diagnostics(
                            invalid_draft, expectations + observed
                        ),
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
            diagnostic = {"error_type": type(error).__name__, "reason": str(error)}
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
                    "field_issues": diagnostics(reduced, expectations + observed),
                }
            }
        else:
            break
    record("jury", output)
    return output


def validate_reduced_standards(
    result, evidence, context, *, clause_contract=False, record=lambda *_: None
):
    """Keep invalid proposed criteria and observations outside adopted standards.

    Never repair provenance by guessing a reference. A rejected criterion makes
    its covered requirement deferred; a rejected observation stays diagnostic.
    """
    raw_criteria = result.get("criteria", [])
    if not isinstance(raw_criteria, list):
        return validate(result, evidence, context, clause_contract=clause_contract)
    output = None
    rejected_criteria = []
    for raw in raw_criteria:
        try:
            if not isinstance(raw, dict):
                raise ValueError("Criterion must be an object")
            candidate = validate(
                {
                    **(result if output is None else output),
                    "criteria": [raw] if output is None else [*output["criteria"], raw],
                    "gaps": [],
                },
                evidence,
                context,
                clause_contract=clause_contract,
            )
        except ValueError as error:
            rejected_criteria.append(
                {
                    "id": raw.get("id") if isinstance(raw, dict) else None,
                    "reason": str(error),
                    "raw": deepcopy(raw),
                }
            )
        else:
            output = candidate
    if output is None:
        if not rejected_criteria or any(
            row["reason"] != "User criterion must cite an actual user statement"
            for row in rejected_criteria
        ):
            return validate(result, evidence, context, clause_contract=clause_contract)
        # No provenance-valid draft survived. Adopt only this explicit Task
        # baseline, while retaining every rejected draft as diagnostic evidence.
        output = validate(
            {
                **result,
                "criteria": [
                    {
                        "id": "task_outcome_requirement",
                        "standard": "The delivery satisfies the requested task outcome.",
                        "kind": "hard",
                        "origin": "task",
                        "evidence_refs": ["task"],
                        "measurement_question": "Compare the delivery against the task request.",
                    }
                ],
                "gaps": [],
            },
            evidence,
            context,
            clause_contract=clause_contract,
        )
    if rejected_criteria:
        rejected_ids = {
            row["id"] for row in rejected_criteria if isinstance(row["id"], str)
        }

        def references_rejected(value):
            return isinstance(value, str) and any(
                value == rejected or rejected.startswith(value + "_")
                for rejected in rejected_ids
            )

        for coverage in output.get("coverage", []):
            if not isinstance(coverage, dict) or not any(
                references_rejected(value)
                for value in coverage.get("criterion_ids", [])
            ):
                continue
            coverage.update(
                status="deferred",
                criterion_ids=[],
                reason="Deferred because a mapped criterion failed provenance validation.",
            )
            for part in coverage.get("subrequirements", []):
                if isinstance(part, dict):
                    part.update(
                        status="deferred",
                        criterion_ids=[],
                        reason="Deferred because a mapped criterion failed provenance validation.",
                    )
                    part.pop("retained_text", None)
        record(
            "jury_criterion_validation",
            {
                "rejected": rejected_criteria,
                "boundary": "Unvalidated criteria are retained for diagnosis only; not adopted requirements.",
            },
        )
        output["criterion_validation_issues"] = [
            {key: row[key] for key in ("id", "reason")} for row in rejected_criteria
        ]
    raw_gaps = result.get("gaps", [])
    rejected = []
    if not isinstance(raw_gaps, list):
        rejected.append(
            {
                "gap_id": None,
                "reason": "Gaps must be a list of objects",
                "raw": deepcopy(raw_gaps),
            }
        )
        raw_gaps = []
    for raw in raw_gaps:
        try:
            if not isinstance(raw, dict):
                raise ValueError("Gap must be an object")
            candidate = validate(
                {**output, "gaps": [*output["gaps"], raw]},
                evidence,
                context,
                clause_contract=clause_contract,
            )
        except ValueError as error:
            rejected.append(
                {
                    "gap_id": raw.get("gap_id") if isinstance(raw, dict) else None,
                    "reason": str(error),
                    "raw": deepcopy(raw),
                }
            )
        else:
            output = candidate
    if rejected:
        record(
            "jury_gap_validation",
            {
                "rejected": rejected,
                "boundary": "Unvalidated observations retained for diagnosis only; not adopted gaps or requirements.",
            },
        )
        output["gap_validation_issues"] = [
            {key: row[key] for key in ("gap_id", "reason")} for row in rejected
        ]
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
    from reward_engine.core.outcome.task_inputs import (
        instruction as task_input_instruction,
    )

    expectations = inputs["independent_expectations"]
    observed = inputs["proposal_expectations"]
    schema, clause_instruction = clause_schema(clause_contract)

    def finalize(raw):
        record("jury_reduce_raw", raw)
        if clause_contract:
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
        if strict:
            from reward_engine.core.outcome.jury_refinement import check_retired_clauses

            check_retired_clauses(raw, inputs)
        output = validate_reduced_standards(
            normalize_reduced(
                apply_part_decisions(raw, inputs, evidence, mark_omissions=strict),
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
            if strict
            else {}
        ),
    )
    if strict:
        return reduced
    record("jury_reduce_raw", reduced)
    try:
        output = finalize(reduced)
    except (ValueError, TypeError, KeyError, AttributeError) as error:
        raise ReduceValidationError(error, reduced) from error
    return output


def reconcile_coverage(expectations, output):
    """Missing Reduce rows become visible unknowns, never silent deletion or pass."""
    valid = {c["id"] for c in output["criteria"]}
    criteria = {c["id"]: c for c in output["criteria"]}
    supplied = output.get("coverage", [])
    keys = {e["expectation_id"] for e in expectations}
    if not isinstance(supplied, list) or any(not isinstance(r, dict) for r in supplied):
        raise ValueError("Coverage must be a list")

    def canonical_id(value):
        if not isinstance(value, str) or value in valid:
            return value
        matches = [
            criterion_id
            for criterion_id in valid
            if criterion_id.startswith(value + "_")
        ]
        return matches[0] if len(matches) == 1 else value

    normalized = []
    for raw in supplied:
        row = deepcopy(raw)
        if isinstance(row.get("criterion_ids"), list):
            row["criterion_ids"] = [
                canonical_id(value) for value in row["criterion_ids"]
            ]
        if isinstance(row.get("subrequirements"), list):
            for part in row["subrequirements"]:
                if isinstance(part, dict) and isinstance(
                    part.get("criterion_ids"), list
                ):
                    part["criterion_ids"] = [
                        canonical_id(value) for value in part["criterion_ids"]
                    ]
        normalized.append(row)
    supplied = normalized
    if len({r.get("expectation_id") for r in supplied}) != len(supplied):
        raise ValueError("Duplicate expectation coverage")
    if any(r.get("expectation_id") not in keys for r in supplied):
        raise ValueError("Unknown expectation in coverage")
    lookup = {r["expectation_id"]: r for r in supplied}
    rows = []
    for expected in expectations:
        key = expected["expectation_id"]
        row = deepcopy(
            lookup.get(
                key,
                {
                    "expectation_id": key,
                    "criterion_ids": [],
                    "status": "deferred",
                    "reason": "Reduce omitted this independent expectation.",
                },
            )
        )
        ids = row.get("criterion_ids")
        if (
            not isinstance(ids, list)
            or not set(ids) <= valid
            or not row.get("reason")
            or row.get("status") not in {"mapped", "deferred", "excluded"}
            or (row["status"] == "mapped") != bool(ids)
        ):
            raise ValueError("Invalid outcome coverage mapping")
        if (
            row["status"] == "excluded"
            and expected.get("kind") == "hard"
            and not expected.get("draft_validation_issues")
        ):
            row.update(
                status="deferred",
                reason="Hard expectation requires resolution: " + row["reason"],
            )
        if expected.get("subrequirements"):
            row["subrequirements"] = resolve_parts(expected, row, criteria)
            mapped_ids = {i for p in row["subrequirements"] for i in p["criterion_ids"]}
            if (
                any(p["status"] == "deferred" for p in row["subrequirements"])
                or (row["status"] == "mapped" and mapped_ids != set(ids))
                or (row["status"] == "excluded" and mapped_ids)
            ):
                row.update(
                    status="deferred",
                    criterion_ids=[],
                    reason="Important subrequirements unresolved or inconsistent with parent mapping; see per-part decisions.",
                )
        rows.append(row)
    return rows


def validate_ranking(result, context, evidence):
    groups, unknown = result.get("groups"), result.get("unknown")
    if (
        not isinstance(groups, list)
        or not isinstance(unknown, list)
        or any(not isinstance(g, list) or not g for g in groups)
    ):
        raise ValueError("Ranking needs nonempty tie groups and explicit unknown list")
    ordered = [p for group in groups for p in group] + unknown
    products = {p["product_id"] for p in context["products"]}
    if len(ordered) != len(products) or set(ordered) != products:
        raise ValueError("Ranking must account for each supplied product once")
    observations = result.get("observations", [])
    if (
        len(observations) != len(products)
        or {r["product_id"] for r in observations} != products
    ):
        raise ValueError("Ranking needs product-specific observations")
    for row in observations:
        refs = require_refs(row.get("evidence_refs"), evidence)
        if not row.get("reason") or not any(
            product_ref(r, row["product_id"], evidence) for r in refs
        ):
            raise ValueError("Ranking must cite the actual product")
    if not result.get("reason"):
        raise ValueError("Ranking needs an evidence-based explanation")
    return result


def continuation_decision(raw, context, evidence, known_gaps):
    """Validate spending authority separately; bad/missing advice cannot erase ranking."""
    stop = {
        "decision": "stop",
        "reason": "No supported important new gap from terminal Jury",
    }
    if not isinstance(raw, dict):
        return stop
    if raw.get("decision") == "stop":
        return {**stop, "reason": raw.get("reason") or stop["reason"]}
    try:
        fields = ("reason", "gap_id", "gap", "why_important", "why_new", "next_step")
        if raw.get("decision") != "continue" or any(
            not isinstance(raw.get(k), str) or not raw[k].strip() for k in fields
        ):
            raise ValueError(
                "Continuation needs a concrete new gap, importance, novelty and next step"
            )
        targets = raw.get("product_ids")
        products = {p["product_id"] for p in context["products"]}
        if (
            not isinstance(targets, list)
            or not targets
            or not all(isinstance(p, str) and p in products for p in targets)
        ):
            raise ValueError("Continuation must identify actual products")
        refs = require_refs(raw.get("evidence_refs"), evidence)
        for product in targets:
            if not any(product_ref(r, product, evidence) for r in refs):
                raise ValueError(
                    "Continuation must cite evidence for each affected product"
                )

        def normalize(text):
            return " ".join(str(text).casefold().split())

        if any(
            raw["gap_id"] == g.get("gap_id")
            or normalize(raw["gap"]) == normalize(g.get("gap", ""))
            for g in known_gaps
        ):
            raise ValueError(
                "Previously known gap is not a new reason to repeat the loop"
            )
        return {
            k: deepcopy(raw[k])
            for k in ("decision", *fields, "product_ids", "evidence_refs")
        }
    except (ValueError, TypeError, KeyError) as error:
        return {**stop, "reason": "Invalid continuation advice: " + str(error)}


async def rank(
    context,
    evidence,
    members,
    reducer,
    *,
    policy=None,
    record=lambda *_: None,
    loop_review=None,
    reduce_max_bytes=160000,
):
    """Same three Map/one Reduce mechanism, after delivery only; no absolute grade."""
    if type(reduce_max_bytes) is not int or reduce_max_bytes < 1:
        raise ValueError("reduce_max_bytes must be a positive integer")
    schema = {
        "groups": [["product ids, best first; ties allowed"]],
        "unknown": ["unjudgeable product ids"],
        "observations": [
            {
                "product_id": "id",
                "reason": "specific outcome difference",
                "evidence_refs": ["refs"],
            }
        ],
        "reason": "why these relative differences matter for the Task",
    }
    instruction = (
        "Compare final Task outcomes, not effort, narrative polish or following our guidance. "
        "Examine specific quality differences before overall ordering. Consult the actual user "
        "requirements attached to EACH product; do not transfer undisclosed requirements between them. "
        "Ranking checks improvement/regression, never proves absolute completion. Shared defects "
        "remain defects even for the winner. If no meaningful difference is supported use tie; "
        "if evidence is insufficient use unknown. No absolute scores."
    )

    async def one(call):
        return validate_ranking(
            await ask(
                call,
                "outcome_ranking_map",
                context,
                evidence,
                instruction,
                schema,
                **(
                    {
                        "validate": lambda value: validate_ranking(
                            value, context, evidence
                        )
                    }
                    if getattr(call, "bounded_stage", False)
                    else {}
                ),
            ),
            context,
            evidence,
        )

    batch = await collect_batch(
        {k: lambda c=c: one(c) for k, c in members.items()}, policy or BatchPolicy()
    )
    record("ranking_map", batch.summary())
    good = [r.value for r in batch.items if r.status == "ok"]
    if len(good) < 2 or (
        getattr(reducer, "bounded_stage", False) and not batch.complete
    ):
        result = {
            "status": "unavailable",
            "reason": "Not enough valid comparisons; do not claim improvement",
        }
    else:
        extra, reduce_schema = {"comparisons": good}, deepcopy(schema)
        if loop_review is not None:
            extra["loop_review"] = deepcopy(loop_review)
            reduce_schema["continuation"] = {
                "decision": "stop (default) or continue",
                "reason": "why another fresh round is or is not worth doing",
                "gap_id": "stable semantic ID; reuse an old ID for the same known problem",
                "gap": "only for continue: concrete remaining Outcome discrepancy",
                "why_important": "why this materially affects Task completion, not incidental polish",
                "why_new": "what new evidence reveals beyond known_gaps/current_criteria/earlier decisions",
                "next_step": "a specific feasible direction for the next round, not just try again",
                "product_ids": ["affected current product IDs"],
                "evidence_refs": ["actual supporting product/check refs"],
            }
        result = validate_ranking(
            await ask(
                reducer,
                "outcome_ranking_reduce",
                context,
                evidence,
                instruction
                + " Resolve member disagreements against evidence, not vote counts; preserve uncertainty."
                + (
                    " Default to ending after this round. Recommend continue ONLY for a materially "
                    "important NEW gap supported by supplied evidence that still affects the best available "
                    "current deliverable. Compare known gaps, current criteria and prior decisions; renamed "
                    "old issues, inferior historical works, minor polish, uncertainty or failed checkers alone "
                    "do not justify another round. Respect Task scope. This is a fallible semantic judgment, "
                    "not an official grade. Do not invent a gap to use the remaining round budget."
                    if loop_review is not None
                    else ""
                ),
                reduce_schema,
                extra=extra,
                max_bytes=reduce_max_bytes,
                **(
                    {
                        "validate": lambda value: validate_ranking(
                            value, context, evidence
                        )
                    }
                    if getattr(reducer, "bounded_stage", False)
                    else {}
                ),
            ),
            context,
            evidence,
        )
        result["status"] = "compared"
        if loop_review is not None:
            record("continuation_raw", result.get("continuation"))
            result["continuation"] = continuation_decision(
                result.get("continuation"), context, evidence, loop_review["known_gaps"]
            )
    record("ranking", result)
    return result
