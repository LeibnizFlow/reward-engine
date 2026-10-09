"""Same Outcome context, different question: what practical method could improve it?"""

import json
from copy import deepcopy

from reward_engine.core.batch import BatchPolicy, collect_batch
from reward_engine.core.outcome.construction import (
    audit_round,
    fingerprint,
    validate_rounds,
)
from reward_engine.core.outcome.context import (
    ask,
    compact_learning_context,
    require_refs,
)

PROMPT_VERSION = "outcome-hypothesis/2026-09-10-task-first-construction"

TASK_FIRST = (
    "Task is the primary source of the desired outcome; the final Jury clarifies its quality goals. "
    "Knowledge packs are supplemental, fallible evidence and possible methods, not a replacement for "
    "reading the Task and not a checklist of mandatory techniques. First identify the largest "
    "consequential opportunity to improve complete, accurate, useful and sufficiently deep delivery "
    "of the actual Task. A missing or irrelevant knowledge pack does not erase the Task's goals. "
    "Do not prefer a tiny well-documented defect over a major missing Task outcome. "
)

AUDIT = (
    TASK_FIRST
    + "Independently audit the WHOLE current hypothesis, including a semantic keep. Check factual "
    "correctness against the actual Task and available evidence, unstated applicability conditions, "
    "and whether the method targets the largest Task outcome improvement. Check that actions are "
    "concrete and executable and that verification specifies observable benefit and possible "
    "refutation; an expected benefit is not a measured result. The final Jury is frozen: suggest "
    "methods, not altered scoring criteria or new user obligations. Record a suspected Jury defect "
    "as upstream uncertainty and propose an investigation without assuming the accusation true. "
    "A keep decision is valid only after checking the full Task; do not invent changes for novelty. "
    "Read the current artifact, not old drafts. Preserve earlier accepted corrections, constraints "
    "and unresolved limits. In round 3 additionally check whether the previous revision introduced "
    "a regression or omitted an important Task goal."
)

SCHEMA = {
    "method": "specific method, or empty for keep",
    "target": "Outcome gap",
    "expected_improvement": "unverified expectation",
    "verification": "how to observe improvement",
    "evidence_refs": ["supplied refs"],
    "target_gap_ids": ["Jury gap IDs, or empty for new exploration"],
    "status": "propose or keep",
    "clarifications": [{
        "question": "concise neutral question to the current user, not a presumed answer",
        "why": "which Outcome decision this answer changes",
        "when": "ask only if still unresolved in this fresh conversation/environment",
        "if_unanswered": "safe progress or explicit unresolved boundary without inventing an answer",
        "evidence_refs": ["Task or observed gap refs"],
    }],
}

CLARIFY = (
    "Proactively consider whether asking the current user would improve the outcome. Propose a few "
    "high-value clarifications when missing requirements, preferences or authority materially change "
    "the solution. First use facts already available in the current conversation and tools; do not "
    "ask the user to supply records the Actor can inspect. Questions must be neutral, not smuggle "
    "another rollout's answers into a fresh session. Include why, when and a safe unanswered path. "
    "Do not ask for the sake of asking, ask again after an answer, or invent consent. Return "
    "clarifications=[] when none are useful. Use propose for a clarification-first method; keep has "
    "no questions. This is guidance for the Actor's user dialogue, not permission to contact third parties. "
)


def validate(result, evidence, jury=None):
    if result.get("status") not in {"propose", "keep"}:
        raise ValueError("Invalid hypothesis status")
    if result.get("availability") == "unavailable":
        raise ValueError("Execution availability is not a model's semantic hypothesis verdict")
    questions = result.get("clarifications", [])
    if not isinstance(questions, list) or len(questions) > 3:
        raise ValueError("Use zero to three high-value clarifications")
    if result["status"] == "keep" and questions:
        raise ValueError("Clarification-first is a proposal, not keep")
    for question in questions:
        if not isinstance(question, dict) or any(not isinstance(question.get(k), str) or not question[k].strip()
               for k in ("question", "why", "when", "if_unanswered")):
            raise ValueError("Clarification needs question, decision value, condition and fallback")
        require_refs(question.get("evidence_refs"), evidence)
    if result["status"] == "propose":
        if any(
            not isinstance(result.get(k), str) or not result[k].strip()
            for k in ("method", "target", "expected_improvement", "verification")
        ):
            raise ValueError("Method must target and test an Outcome improvement")
        require_refs(result.get("evidence_refs"), evidence)
        targets = result.get("target_gap_ids", [])
        if not isinstance(targets, list) or not set(targets) <= {
            r["gap_id"] for r in (jury or {}).get("gaps", [])
        }:
            raise ValueError("Hypothesis must reference existing gap IDs")
    return result


def require_validated_projection(result, *, allow_unverified=False):
    """The new frozen entry cannot accidentally pair an old unvalidated method with a new Pack."""
    if result.get('status') != 'propose':
        return
    audit = result.get('validation', {})
    body = {k: v for k, v in result.items() if k not in {'validation', 'actor_projection'}}
    projection = result.get('actor_projection', {})
    if allow_unverified and audit.get('status') == 'skipped':
        if (audit.get('reason') != 'adversarial_review_disabled' or audit.get('records') != []
                or audit.get('artifact_sha256') != fingerprint(body)
                or projection.get('validated') is not False
                or projection.get('method') != result['method']
                or projection.get('source_sha256') != fingerprint(result['method'])):
            raise ValueError('Unverified Hypothesis projection or skipped-review binding changed')
        return
    if (audit.get('artifact_sha256') != fingerprint(body) or not audit.get('records')
            or any(r.get('status') != 'supported_on_cases' or r.get('contrast_qualified') is not True
                   for r in audit['records'])
            or projection.get('method') != result['method'] or projection.get('validated') is not True
            or projection.get('source_sha256') != fingerprint(result['method'])):
        raise ValueError('Fresh Actor requires the exact independently validated Hypothesis projection')


def actor_prompt(result, *, allow_unverified=False):
    """Pass questions, not historical answers, through the existing fresh Actor prompt."""
    if result.get("status") != "propose":
        return ""
    if 'actor_projection' in result:
        from reward_engine.core.outcome.construction import fingerprint
        projection = result['actor_projection']
        if projection.get('validated') is False:
            require_validated_projection(result, allow_unverified=allow_unverified)
            return ('Optional Task method, NOT verified by adversarial review. Treat it as a fallible '
                    'suggestion, not a new requirement or established fact. Respect the original Task; '
                    'verify applicability and consequential claims using available evidence.\n'
                    + projection['method'])
        if (not projection.get('validated') or projection.get('method') != result['method']
                or projection.get('source_sha256') != fingerprint(result['method'])):
            raise ValueError('Actor projection no longer matches the validated method')
        return ('Optional Task method, independently checked on limited cases. Respect the original '
                'Task; verify applicability and consequential claims with available evidence.\n'
                + projection['method'])
    text = (
        "Source: an exploration proposal that may draw on other sampled work, NOT this fresh "
        "Actor's conversation history. Candidate IDs, earlier versions, disagreements and reported "
        "checks in the proposal are research leads, not your prior actions, observed results or "
        "confirmed user statements. Apply useful ideas without presenting that history as your own.\n"
        + result["method"]
    )
    text += (f"\nOutcome gap: {result.get('target', '')}"
             f"\nExpected improvement (unverified): {result.get('expected_improvement', '')}"
             f"\nVerify by: {result.get('verification', '')}")
    text += ("\nDuring execution, proactively resolve knowledge gaps using available search/read tools: "
             "choose relevant sources, read the supporting passages, refine the search when evidence "
             "is missing, and stop when the important question is answered. A plan to search is not "
             "retrieved evidence. For missing user requirements or authority, ask the CURRENT user "
             "when their answer changes the solution; never invent their reply. Reuse available facts, "
             "avoid redundant searches/questions, and state unresolved limits if the channel is unavailable.")
    questions = result.get("clarifications", [])
    if questions:
        text += (
            "\nSuggested clarifications, NOT confirmed requirements. Check the CURRENT dialogue and "
            "available records first. Ask the user through the normal conversation only when the answer "
            "will change the outcome; do not repeat answered questions or send unsolicited external "
            "messages. Await the real answer, never simulate it. If interaction is unavailable, state "
            "the uncertainty and continue safe independent work; do not guess approval.\n"
        )
        text += "\n".join(f"- Ask: {q['question']} Why: {q['why']} When: {q['when']} "
                          f"If unanswered: {q['if_unanswered']}" for q in questions)
    return text


async def propose(context, evidence, jury, members, reducer, *, policy=None, record=lambda *_: None,
                  construction_rounds=1):
    rounds = validate_rounds(construction_rounds)
    if rounds > 1 and len(members) != 3:
        raise ValueError("Multi-round hypothesis construction requires three independent seats")
    context = compact_learning_context(context)
    context, evidence, jury = question_context(context, evidence, jury)
    prepared = {"context": context, "evidence": evidence, "jury": jury,
                "construction_rounds": rounds, "prompt_version": PROMPT_VERSION}
    if rounds > 1:
        record("construction/hypothesis/round_1/input", {
            **deepcopy(prepared), "fingerprint": fingerprint(prepared),
        })

    async def one(call):
        return await ask(
                call,
                "outcome_hypothesis_map",
                context,
                evidence,
                TASK_FIRST + "Propose one concrete way to improve Task outcome, informed by Jury gaps. "
                "Use optional Task Review knowledge in skills: retain its concrete mechanisms, source "
                "references and applicability limits when proposing a method. Pending findings are "
                "investigation leads, not verified facts; professional advice is not a user obligation. "
                "Read open_questions too: an unverified outcome question may matter "
                "more than a tiny error in an already-covered dimension. Target the largest consequential "
                "gap, explain why it matters and how to verify the improvement. If expertise is missing, "
                "propose a specific source search or real check, not an invented answer or tool result. "
                "For check_supported gaps propose a repair; for needs_check first investigate rather than "
                "assuming the accusation true. Read expected/observed/gap/verification together. "
                "open_questions are draft concerns, NOT product failures: a missing text mapping is not "
                "evidence of an unmet requirement. Check applicability against Task and actual observations. "
                "You may explore a new method; do not merely imitate the highest-ranked old product. "
                "You propose a method, not a different scoring rule. Mark expected effects as unverified. "
                "Do not reveal another rollout's private-to-that-dialogue answers as facts to a fresh Actor; "
                "instead suggest what it should clarify or verify. Do not preselect fixed test IDs. " + CLARIFY,
                SCHEMA,
                extra={"jury": jury},
                validate=lambda result: validate(result, evidence, jury),
                max_contract_retries=int(rounds > 1),
                max_evidence_rounds=2 if rounds > 1 else 1,
                close_evidence=rounds > 1,
            )

    batch = await collect_batch({k: lambda c=c: one(c) for k, c in members.items()}, policy or BatchPolicy())
    record("hypothesis_map", batch.summary())
    good = [r.value for r in batch.items if r.status == "ok"]
    if rounds > 1:
        record("construction/hypothesis/round_1/map", batch.summary())
        record("construction/hypothesis/round_1/proposals", deepcopy(good))
    completed = 0
    if not good or (rounds > 1 and not batch.complete):
        output = {
            "status": "keep",
            "method": "",
            "availability": "unavailable",
            "reason": ("All three construction seats must be valid; not a semantic keep verdict"
                       if rounds > 1 else "No usable proposals; not a semantic keep verdict"),
        }
    else:
        try:
            output = await _reduce_prepared(context, evidence, jury, reducer, good,
                                            contract_retry=rounds > 1)
            completed = 1
            if rounds > 1:
                record("construction/hypothesis/round_1/output", deepcopy(output))
            for number in range(2, rounds + 1):
                output = await _refine(context, evidence, jury, members, reducer, output,
                                       number=number, policy=policy, record=record)
                completed = number
        except Exception as error:
            output = {
                "status": "keep",
                "method": "",
                "availability": "unavailable",
                "reason": ("Hypothesis construction failed; not a semantic keep verdict"
                           if rounds > 1 else "Reduce failed; proceed without an exploration prompt"),
                "error_type": type(error).__name__,
            }
    if rounds > 1:
        output["construction"] = {"requested_rounds": rounds, "completed_rounds": completed,
            "complete": completed == rounds, "input_fingerprint": fingerprint(prepared),
            "prompt_version": PROMPT_VERSION}
    record("hypothesis", output)
    return output


async def reduce(context, evidence, jury, reducer, proposals):
    """One synthesis, no recursively added judge or quality retry."""
    context, evidence, jury = question_context(compact_learning_context(context), evidence, jury)
    return await _reduce_prepared(context, evidence, jury, reducer, proposals)


async def _reduce_prepared(context, evidence, jury, reducer, proposals, *, contract_retry=False):
    """Reuse the same lossless question projection as Map; standalone reduce still prepares it."""
    return await ask(
            reducer,
            "outcome_hypothesis_reduce",
            context,
            evidence,
            TASK_FIRST + "Decide ONE actionable method, or keep. You may refine the strongest proposal or combine "
            "compatible ideas; do not force incompatible methods into a patchwork. Remove redundant "
            "steps and keep the method concise and directly usable by an Actor. Judge whether the method "
            "addresses the most consequential missing outcome rather than repeatedly polishing small "
            "details. A source search or clarification can precede a repair; never assume it was done. "
            "Preserve the source and applicability limits of optional Task Review knowledge in skills; "
            "do not promote an unresolved finding or a domain suggestion into a mandatory Task requirement. "
            "Judge whether the method "
            "addresses an Outcome gap, not whether it sounds novel. No copied candidate-specific facts, "
            "invented user answers or extra authority. Preserve investigation before repair for unverified "
            "gaps. Explain how to test its expected improvement and what evidence would refute it. "
            "Synthesize and deduplicate useful questions from members; retain only those that change "
            "an Outcome decision, at most three. " + CLARIFY,
            SCHEMA,
            extra={"jury": jury, "proposals": proposals},
            validate=lambda result: validate(result, evidence, jury),
            max_contract_retries=int(contract_retry),
            max_evidence_rounds=2 if contract_retry else 1,
            close_evidence=contract_retry,
        )


async def _refine(context, evidence, jury, members, reducer, current, *, number, policy, record):
    """Revisit the accepted current method; discarded first-round drafts never return."""
    reviews = await audit_round("hypothesis", number, context, evidence, current, members,
        instruction=AUDIT, policy=policy, extra={"jury": jury, "prompt_version": PROMPT_VERSION},
        record=record)
    instruction = (
        TASK_FIRST + "Revise ONE current actionable method, or keep, using the three independent "
        "whole-method audits. Evaluate their evidence rather than counting votes. Preserve accepted "
        "corrections and valid applicability limits; never restore a discarded earlier draft. Resolve "
        "supported problems, explain rejected suggestions, and explicitly retain unresolved issues. "
        "The final Jury and Task are frozen: do not alter scoring criteria, invent user requirements "
        "or assume an upstream suspicion is established. If investigation is needed, specify what "
        "to read or measure before repair. The method must advance the largest consequential Task "
        "outcome with executable actions and falsifiable verification. Do not assume a source search, "
        "clarification or experiment has already happened. A semantic keep must still be reconsidered "
        "against the full Task. Return revision_summary explaining changes, rejected suggestions and "
        "remaining uncertainty; do not manufacture changes when none are supported. " + CLARIFY
    )
    if number == 3:
        instruction += " This is round 3: explicitly check regression and omitted Task goals after revision."

    def final_contract(result):
        validate(result, evidence, jury)
        if not isinstance(result.get("revision_summary"), str) or not result["revision_summary"].strip():
            raise ValueError("Hypothesis refinement requires a revision summary and unresolved limits")
        return result

    extra = {"jury": jury, "current_hypothesis": deepcopy(current), "reviews": deepcopy(reviews),
             "construction_round": number, "prompt_version": PROMPT_VERSION}
    record(f"construction/hypothesis/round_{number}/reduce_input", {
        **deepcopy(extra), "fingerprint": fingerprint({"context": context, "evidence": evidence, **extra}),
    })
    output = await ask(reducer, "outcome_hypothesis_refine", context, evidence, instruction,
        {**SCHEMA, "revision_summary": "changes, rejected suggestions and unresolved limits"},
        extra=extra, validate=final_contract, max_contract_retries=1, max_evidence_rounds=2,
        close_evidence=True)
    record(f"construction/hypothesis/round_{number}/output", deepcopy(output))
    return output


def compact_jury(jury):
    """Front-facing questions, not the internal coverage ledger. Safe to apply twice."""
    if "open_questions" in jury:
        return {k: jury.get(k, []) for k in ("criteria", "gaps", "open_questions")}
    expectations = {r.get("expectation_id"): r for r in jury.get("independent_expectations", [])
                    + jury.get("proposal_expectations", [])}
    questions = {}
    for row in jury.get("coverage", []):
        pending = [p for p in row.get("subrequirements", []) if p.get("status") not in {"mapped", "excluded"}]
        if row.get("status") == "excluded" or (not pending and (
                row.get("subrequirements") or row.get("status") == "mapped")):
            continue
        original = expectations.get(row.get("expectation_id"), {})
        parts = {p.get("part_id"): p.get("text", "") for p in original.get("subrequirements", [])}
        texts = [p.get("current_text") or p.get("retained_text") or parts.get(p.get("part_id"))
                 or p.get("proposed_text", p.get("text", "")) for p in pending]
        texts = list(dict.fromkeys(t for t in texts if t)) or [
            row.get("current_standard") or original.get("standard", "")]
        texts = [t for t in texts if t]
        if not texts:
            continue
        origin = original.get("origin", "unresolved")
        current = "current_standard" in row or any("current_text" in p for p in pending)
        verification = row.get("current_verification") or (
            "Verify the current pending clauses against Task and actual product evidence" if current
            else original.get("measurement_question", "Verify against Task and actual product evidence"))
        why = row.get("current_why") or ("Current unresolved outcome concern; applicability still needs checking"
            if current else original.get("why", "Draft outcome concern; applicability still needs checking"))
        key = (origin, verification, tuple(texts))
        if key not in questions:
            questions[key] = {"questions": texts, "origin": origin,
                "why": why,
                "verification": verification, "status": "unverified_proposal",
                "evidence_refs": [], "source_expectation_ids": [], "pending_sources": []}
        item = questions[key]
        # Preserve disposition, not a guessed diagnosis derived from reason-string prefixes.
        # A reducer omission is NOT evidence that the product failed this requirement.
        item['pending_sources'].append({
            'expectation_id': row.get('expectation_id'), 'kind': original.get('kind'),
            'priority': original.get('priority'), 'status': row.get('status'),
            'reason': row.get('reason', ''),
            'parts': [{k: p[k] for k in ('part_id', 'status', 'reason') if k in p} for p in pending],
        })
        for ref in row.get("current_evidence_refs", original.get("evidence_refs", [])):
            if ref not in item["evidence_refs"]:
                item["evidence_refs"].append(ref)
        source_id = row.get("expectation_id")
        if source_id and source_id not in item["source_expectation_ids"]:
            item["source_expectation_ids"].append(source_id)
    return {"criteria": jury.get("criteria", []), "gaps": jury.get("gaps", []),
            "open_questions": list(questions.values())}


def question_context(context, evidence, jury, *, inline_bytes=8000):
    """All questions stay visible; long rationales/verification stay readable by handle.

    No model summary, semantic deletion, or extra read budget. These are draft
    concerns, never observed product facts or additional user instructions.
    """
    jury = compact_jury(jury)
    questions = jury["open_questions"]
    if len(json.dumps(questions, ensure_ascii=False).encode()) <= inline_bytes:
        return context, evidence, jury
    context, evidence, jury = deepcopy(context), dict(evidence), deepcopy(jury)
    for index, row in enumerate(jury["open_questions"]):
        if not any(key in row for key in ("why", "verification")):
            continue  # Already indexed; Reduce uses the same handles and records.
        ref = f"reference:outcome_question:{index}"
        record = deepcopy(row)
        if ref in evidence and evidence[ref] != record:
            raise ValueError("Outcome question detail conflicts with existing evidence")
        evidence[ref] = record
        row.pop("why", None)
        row.pop("verification", None)
        if "pending_sources" in row:
            row["pending_sources"] = [{k: deepcopy(source[k]) for k in
                ("expectation_id", "kind", "priority", "status") if k in source}
                for source in row["pending_sources"]]
        row["details_ref"] = ref
        docs = context.setdefault("reference_documents", [])
        if not any(d["ref"] == ref for d in docs):
            docs.append({"ref": ref, "title": "Draft question rationale and verification; NOT a product fact",
                         "preview_only": True})
    context["available_evidence_refs"] = list(evidence)
    return context, evidence, jury
