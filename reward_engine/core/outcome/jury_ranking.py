"""Jury ranking for the existing public entry point."""

from copy import deepcopy

from reward_engine.core.batch import BatchPolicy, collect_batch
from reward_engine.core.outcome.construction import fingerprint
from reward_engine.core.outcome.context import (
    ResponseContractError,
    ask,
    require_refs,
    used_previews,
)


def validate_ranking(result, context, evidence):
    """Report all binding defects together without supplying a model verdict."""
    from reward_engine.core.outcome.jury_validation import product_ref

    if not isinstance(result, dict):
        raise ResponseContractError("ranking", "Ranking must be a JSON object")
    issues = []

    def issue(field, reason):
        issues.append({"field": field, "reason": reason})

    groups, unknown = result.get("groups"), result.get("unknown")
    products = {p["product_id"] for p in context["products"]}
    if (
        not isinstance(groups, list)
        or not isinstance(unknown, list)
        or any(not isinstance(g, list) or not g for g in groups)
    ):
        issue(
            "groups/unknown",
            "Ranking needs nonempty tie groups and explicit unknown list",
        )
    else:
        ordered = [p for group in groups for p in group] + unknown
        if (
            any(not isinstance(p, str) for p in ordered)
            or len(ordered) != len(products)
            or set(ordered) != products
        ):
            issue(
                "groups/unknown", "Ranking must account for each supplied product once"
            )
    observations = result.get("observations", [])
    if not isinstance(observations, list) or any(
        not isinstance(r, dict) for r in observations
    ):
        issue(
            "observations",
            "Ranking needs product-specific observations as a list of objects",
        )
    else:
        ids = [r.get("product_id") for r in observations]
        if (
            any(not isinstance(p, str) for p in ids)
            or len(ids) != len(products)
            or set(ids) != products
        ):
            issue(
                "observations",
                "Ranking needs product-specific observations, one per supplied product",
            )
        for index, row in enumerate(observations):
            field = f"observations[{index}]"
            try:
                refs = require_refs(row.get("evidence_refs"), evidence)
            except (ValueError, TypeError) as error:
                issue(field + ".evidence_refs", str(error))
                continue
            product = row.get("product_id")
            if (
                not isinstance(product, str)
                or product not in products
                or not isinstance(row.get("reason"), str)
                or not row["reason"].strip()
                or not any(product_ref(r, product, evidence) for r in refs)
            ):
                issue(
                    field,
                    "Ranking must cite the actual product and explain the observation",
                )
    if not isinstance(result.get("reason"), str) or not result["reason"].strip():
        issue("reason", "Ranking needs an evidence-based explanation")
    if issues:
        error = ResponseContractError(
            "ranking", "\n".join(f"{r['field']}: {r['reason']}" for r in issues)
        )
        error.details.update(issues=issues, product_ids=sorted(products))
        raise error
    return result


class _RankingRepair:
    """Preserve validated fields during repair, never bypass evidence closure."""

    def __init__(self, call, context, evidence, record=lambda _: None):
        self.call = call
        self.context = context
        self.evidence = evidence
        self.record = record
        self.attached = {}
        self.unread = None
        self.preserved = None
        self.fields = {}
        self.correction = None
        self.repair_active = False
        self.attempts = []

    def __getattr__(self, name):
        return getattr(self.call, name)

    def prepare_request(self, request):
        from reward_engine.core.outcome.authored_json import enabled

        if enabled(self.call):
            return (
                self.call.prepare_request(request)
                if hasattr(self.call, "prepare_request")
                else request
            )
        if hasattr(self.call, "prepare_request"):
            request = self.call.prepare_request(request)
        if "response_recovery" in request:
            self.repair_active = True
            request["response_recovery"]["instruction"] += (
                " Evidence and tools are closed during same-evidence ranking repair, "
                "including syntax retries. Correct only invalid fields; preserve "
                "valid assessments. Automatic verification of cited unread previews "
                "runs after a valid full or composed response."
            )
        if self.correction and "response_recovery" in request:
            # Syntax recovery replaces generic recovery details. Keep this
            # operation's offer and closed tools independently of that envelope.
            recovery = request["response_recovery"]
            recovery["field_correction_reply"] = deepcopy(self.correction["offer"])
            recovery["instruction"] += (
                " Preferred repair reply: return ONLY field_corrections with "
                "explicit model-authored groups and unknown. Preserve all other "
                "fields; do not regenerate observations. No evidence/tool requests "
                "during this same-evidence structural repair. A complete JSON "
                "reply remains supported only with unchanged valid fields."
            )
        return request

    async def __call__(self, request):
        if self.unread is None:
            # Use the actual projected view: inline/previously read records
            # cannot unlock a same-evidence assessment.
            self.unread = set(
                used_previews(
                    {"evidence_refs": list(self.evidence)}, request["context"]
                )
            )
        attached = request.get("requested_evidence", {})
        newly_read = (set(attached) - set(self.attached)) & self.unread
        if newly_read and "response_recovery" not in request:
            self.preserved = None
            self.fields = {}
            self.correction = None
            self.repair_active = False
            self.unread.difference_update(newly_read)
        self.attached = deepcopy(attached)
        attempt = {
            "schema": "ranking-response-attempt/v1",
            "attempt": len(self.attempts) + 1,
            "request_sha256": fingerprint(request),
        }
        self.attempts.append(attempt)
        try:
            value = await self.call(request)
            attempt["response"] = deepcopy(value)
            attempt["response_kind"] = (
                "field_correction"
                if isinstance(value, dict) and "field_corrections" in value
                else "full_response"
            )
            # ask also accepts a single response envelope. Check its control
            # payload before it can bypass the persistent repair-time tool lock.
            control = (
                value["response"]
                if isinstance(value, dict)
                and set(value) == {"response"}
                and isinstance(value["response"], dict)
                else value
            )
            if (
                self.repair_active
                and isinstance(control, dict)
                and any(
                    key in control
                    for key in (
                        "need_evidence",
                        "need_tools",
                        "tool_calls",
                        "tool_call",
                    )
                )
            ):
                raise ResponseContractError(
                    "groups/unknown",
                    "Evidence and tools are closed during same-evidence ranking repair",
                )
            return value
        except Exception as error:
            attempt.update(error_type=type(error).__name__, error=str(error))
            if hasattr(error, "doc"):
                attempt["raw_response"] = error.doc
            raise
        finally:
            self.record(deepcopy(self.attempts))

    def check_preserved(self, value):
        if self.preserved is not None:
            try:
                # Order within a tie/unknown set carries no preference.
                same = [set(g) for g in value["groups"]] == [
                    set(g) for g in self.preserved["groups"]
                ] and set(value["unknown"]) == set(self.preserved["unknown"])
            except (KeyError, TypeError):
                same = False
            if not same:
                error = ResponseContractError(
                    "groups/unknown",
                    "Ranking repair changed an already-valid ordering or unknown "
                    "assignment on the same evidence",
                )
                error.details["preserved_ranking"] = deepcopy(self.preserved)
                raise error
        for field, original in self.fields.items():
            if field not in value or value[field] != original:
                raise ResponseContractError(
                    field,
                    "Ranking repair must preserve the already-valid "
                    + field
                    + " on the same evidence",
                )

    def compose(self, raw):
        if (
            self.correction is None
            or set(raw) != {"field_corrections"}
            or not isinstance(raw["field_corrections"], dict)
            or set(raw["field_corrections"]) != {"groups", "unknown"}
        ):
            raise ResponseContractError(
                "field_corrections",
                "An active offer requires exactly groups and unknown; "
                "no sibling fields, extra fields, or paths",
            )
        state = self.correction
        value = deepcopy(state["base"])
        value.update(deepcopy(raw["field_corrections"]))
        fields = {
            key: {
                "kind": "replacement" if key in ("groups", "unknown") else "preserved",
                "source_attempt": len(self.attempts)
                if key in ("groups", "unknown")
                else state["attempt"],
                "source_field": "field_corrections." + key
                if key in ("groups", "unknown")
                else state["source_prefix"] + key,
            }
            for key in value
        }
        self.attempts[-1].update(
            response_kind="field_correction",
            composed_response=deepcopy(value),
            field_provenance={
                "base_attempt": state["attempt"],
                "correction_attempt": len(self.attempts),
                "fields": fields,
            },
        )
        return value

    def validate(self, value):
        from reward_engine.core.outcome.authored_json import enabled

        if enabled(self.call):
            return validate_ranking(value, self.context, self.evidence)
        try:
            if "field_corrections" in value:
                value = self.compose(value)
            self.check_preserved(value)
            try:
                result = validate_ranking(value, self.context, self.evidence)
            except ResponseContractError as error:
                issues = error.details.get("issues", [])
                invalid = {r["field"] for r in issues}
                if issues:
                    if "groups/unknown" not in invalid:
                        if self.preserved is None:
                            self.preserved = deepcopy(
                                {k: value[k] for k in ("groups", "unknown")}
                            )
                        error.details["preserved_ranking"] = deepcopy(self.preserved)
                    for field in ("observations", "reason"):
                        if not any(
                            k == field or k.startswith(field + "[") for k in invalid
                        ):
                            self.fields.setdefault(field, deepcopy(value[field]))
                    error.recovery_instruction = (
                        "Correct only reported contract defects against the same "
                        "evidence. Preserve already-valid groups/unknown, observations "
                        "and reason. Do not reverse valid ordering, split/merge ties, "
                        "change unknown assignments, or invent evidence."
                    )
                    if (
                        invalid == {"groups/unknown"}
                        and getattr(self.call, "bounded_stage", False)
                        and self.correction is None
                    ):
                        self.correction = {
                            "base": deepcopy(value),
                            "source_prefix": "response."
                            if self.attempts[-1]["response"].get("response") == value
                            else "",
                            "attempt": len(self.attempts),
                            "offer": {
                                "schema": "ranking-field-correction/v1",
                                "base_attempt": len(self.attempts),
                                "allowed_fields": ["groups", "unknown"],
                                "required_fields": ["groups", "unknown"],
                                "product_ids": sorted(
                                    p["product_id"] for p in self.context["products"]
                                ),
                                "instruction": 'Return {"field_corrections": {"groups": <model-authored tie groups>, '
                                '"unknown": <model-authored unjudgeable IDs>}}. Each supplied product '
                                "must occur exactly once across these two fields. No default ordering "
                                "or unknown assignment is supplied. All other fields are preserved.",
                            },
                        }
                raise
            self.attempts[-1]["contract_valid"] = True
            # This is not a final accepted ranking: ask still verifies previews
            # and may fail at the evidence read limit. Never restore an old value.
            return result
        except ResponseContractError as error:
            # This lock belongs to every failed ranking contract, not just the
            # optional partition-correction offer. Syntax retries cannot erase it.
            self.repair_active = True
            self.attempts[-1].update(
                contract_valid=False,
                error_type=type(error).__name__,
                details=deepcopy(error.details),
            )
            raise
        finally:
            self.record(deepcopy(self.attempts))


def continuation_decision(raw, context, evidence, known_gaps):
    """Validate spending authority separately; bad/missing advice cannot erase ranking."""
    from reward_engine.core.outcome.jury_validation import product_ref

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
        "if evidence is insufficient use unknown. No absolute scores. Return top-level groups "
        "as arrays of product IDs (one array per tie group), and unknown as an array. Account "
        "for every supplied product exactly once across groups and unknown; include one "
        "observation for EVERY product, including ties and unknowns. Cite exact available "
        "evidence handles for that product. Do not infer check indices or use filesystem paths "
        "as handles. A repair corrects the rejected contract against the same evidence; "
        "preserve any already-valid groups/unknown partition while correcting unrelated "
        "observation or explanation defects. Do not manufacture certainty."
    )

    async def one(call, key):
        repair = _RankingRepair(
            call,
            context,
            evidence,
            record=lambda rows: record(f"ranking_member_{key}_responses", rows),
        )
        return validate_ranking(
            await ask(
                repair,
                "outcome_ranking_map",
                context,
                evidence,
                instruction,
                schema,
                validate=repair.validate,
                verify_used_previews=True,
                max_contract_retries=1,
                max_syntax_retries=1,
                max_evidence_rounds=2,
                close_evidence=True,
            ),
            context,
            evidence,
        )

    batch = await collect_batch(
        {k: lambda c=c, k=k: one(c, k) for k, c in members.items()},
        policy or BatchPolicy(),
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
        repair = _RankingRepair(
            reducer,
            context,
            evidence,
            record=lambda rows: record("ranking_reduce_responses", rows),
        )
        result = validate_ranking(
            await ask(
                repair,
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
                validate=repair.validate,
                verify_used_previews=True,
                max_contract_retries=1,
                max_syntax_retries=1,
                max_evidence_rounds=2,
                close_evidence=True,
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
