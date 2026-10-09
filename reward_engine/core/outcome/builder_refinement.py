"""Task-centred semantic review of Builder drafts, before checker installation."""

from copy import deepcopy

from reward_engine.core.outcome.construction import audit_round, fingerprint
from reward_engine.core.outcome.context import ask, require_refs

CONSTRUCTION_VERSION = "outcome-builder-construction/2026-09-12-partial-outcomes"

AUDIT_INSTRUCTION = (
    "Review the WHOLE Reward/Harness draft against the original Task and frozen final Jury. "
    "Task completion is the goal. Knowledge only supplements facts and methods; missing knowledge "
    "does not remove Task requirements, and irrelevant knowledge must not add obligations. "
    "Check factual correctness, complete and sufficiently deep full-quality anchors, and whether "
    "each executable test measures its actual criterion rather than keywords, source coverage, "
    "or an easier proxy. Check fail/partial/full boundaries, valid alternative solutions, "
    "applicability, source uncertainty, tool capability and future-current-product binding. "
    "Do not lower the standard merely to make seed works pass. Do not require a particular "
    "reference implementation unless the Task requires it. A callable check is not a correct check. "
    "Jury criterion identities, meaning and kind are frozen: report suspected Jury defects as "
    "scope=upstream, never silently replace them or restart Jury. For scope=current provide "
    "a specific anchor/check correction, grounded in Task or supplied evidence. "
    "Audit all dimensions independently; agreement with another reviewer is not evidence. "
    "Check that partial fulfillment is distinguished from unusable work, that an individual checker "
    "does not assert whole-outcome rejection, and that anchors do not add approval or presentation "
    "requirements absent from the frozen criterion. Report upstream overreach for QC to address. "
)

REDUCE_INSTRUCTION = (
    "Revise the entire Reward/Harness draft using the independent reviews, original Task and "
    "frozen final Jury. Evaluate reasons and evidence, not vote counts. Return a complete draft "
    "even if every review recommends keep. Keep EVERY Jury criterion identity, meaning and kind "
    "unchanged; only improve fail/partial/full anchors and executable measurement plans. "
    "Task decides the outcome and scope. Knowledge is supplementary, not the standard or a "
    "mandatory reference solution. Full anchors must require the appropriate completeness, "
    "correctness and useful depth. Preserve valid alternative solutions and applicability. "
    "Do not soften standards to fit seed works or equate checkability with quality. "
    "Preserve partial credit and distinguish missing core outcomes from localized quality gaps. "
    "Record upstream doubts as unresolved rather than rewriting Jury. Never add a criterion. "
    "Use product_1 as the future-current-product handle; no historical answer values or "
    "product_1:public:N bindings. Inline probes belong to evidence.review, use at most three "
    "actual measurement tools, and never claim unavailable execution. Declare missing "
    "implementations; do not emit execution_recipe or binding_status. Code installs only "
    "the final approved draft. Return a concise revision_summary explaining changes or why "
    "no change was warranted, plus unresolved_issues that remain after this revision."
)

DRAFT_SCHEMA = {
    "rubrics": [
        {
            "criterion_id": "frozen Jury id",
            "anchors": {
                "fail": "failure evidence",
                "partial": "partial evidence",
                "full": "full-quality evidence",
            },
        }
    ],
    "tests": [
        {
            "test_id": "unique key",
            "criterion_id": "frozen Jury id",
            "checker": "catalog key or missing implementation key",
            "question": "actual measurement question",
            "arguments": {},
            "probes": [{"tool": "optional measurement tool", "arguments": {}}],
            "implementation_needed": "when capability is missing",
            "knowledge_refs": ["optional supplied skill:task_review refs"],
        }
    ],
    "revision_summary": "specific revisions and reasons, or evidence-based reason for no change",
    "unresolved_issues": [
        {
            "scope": "current or upstream",
            "target": "draft location or criterion id",
            "problem": "remaining uncertainty",
            "suggested_change": "verification or correction",
            "evidence_refs": [
                "exact supplied refs, including Task for a Task uncertainty"
            ],
        }
    ],
}


def completed_construction(pack, rounds):
    """A published one-round bundle or incomplete review history cannot skip reviews."""
    row = pack.get("construction", {})
    history = row.get("history", [])
    return (
        row.get("version") == CONSTRUCTION_VERSION
        and row.get("rounds") == rounds
        and isinstance(history, list)
        and len(history) == rounds
        and [r.get("round") for r in history] == list(range(1, rounds + 1))
        and all(
            isinstance(r.get("reviews"), list) and len(r["reviews"]) == 3
            for r in history[1:]
        )
        and all(
            r.get("output_fingerprint") == fingerprint(r.get("output")) for r in history
        )
        and all(
            history[index].get("input_fingerprint")
            == history[index - 1].get("output_fingerprint")
            for index in range(1, len(history))
        )
        and row.get("published_fingerprint")
        == fingerprint({"criteria": pack.get("criteria"), "tests": pack.get("tests")})
        and pack.get("status") == "callable"
    )


def scope_generated_checkers(tests, checker_catalog, scope):
    """An authored implementation belongs to its final contract, not a model-chosen name."""
    for test in tests:
        key = test["checker"]
        if key in checker_catalog:
            continue
        logical_key = key.split("::builder:", 1)[0]
        contract = {**deepcopy(test), "checker": logical_key}
        test["checker_logical_key"] = logical_key
        test["checker"] = (
            logical_key
            + "::builder:"
            + fingerprint({"scope": scope, "test": contract})[:24]
        )


def _validate_revision(value, validate, evidence):
    validate(value)
    if (
        not isinstance(value.get("revision_summary"), str)
        or not value["revision_summary"].strip()
    ):
        raise ValueError("Builder revision needs a nonempty revision_summary")
    issues = value.get("unresolved_issues")
    if not isinstance(issues, list):
        raise ValueError(
            "Builder revision needs unresolved_issues, including an empty list when none"
        )
    for issue in issues:
        if (
            not isinstance(issue, dict)
            or issue.get("scope") not in {"current", "upstream"}
            or any(
                not isinstance(issue.get(k), str) or not issue[k].strip()
                for k in ("target", "problem", "suggested_change")
            )
        ):
            raise ValueError("Malformed unresolved Builder issue")
        require_refs(issue.get("evidence_refs"), evidence)
    return value


async def refine_draft(
    draft,
    rounds,
    context,
    evidence,
    jury,
    call,
    *,
    reviewers,
    policy,
    validate,
    record,
    checker_catalog,
    measurement_tools,
    knowledge_roles,
):
    members = reviewers or {str(index + 1): call for index in range(3)}
    history = [
        {
            "round": 1,
            "output": deepcopy(draft),
            "output_fingerprint": fingerprint(draft),
        }
    ]
    record("construction/builder/round_1/output", history[0])
    upstream_concerns = []
    for number in range(2, rounds + 1):
        current = deepcopy(draft)
        extra = {
            "jury": {"criteria": deepcopy(jury["criteria"])},
            "checker_catalog": deepcopy(checker_catalog),
            "measurement_tools": deepcopy(measurement_tools),
            "knowledge_roles": deepcopy(knowledge_roles),
            "prior_revision_summary": current.get("revision_summary"),
            "unresolved_issues": deepcopy(current.get("unresolved_issues", [])),
            "upstream_concerns": deepcopy(upstream_concerns),
        }
        instruction = AUDIT_INSTRUCTION
        if number == 3:
            instruction += (
                " Also check the prior round changes for regressions, newly introduced "
                "errors and lost Task coverage. Previously corrected errors must not return."
            )
        reviews = await audit_round(
            "builder",
            number,
            context,
            evidence,
            current,
            members,
            instruction=instruction,
            policy=policy,
            extra=extra,
            record=record,
        )
        for review in reviews:
            for issue in review["issues"]:
                if issue["scope"] == "upstream" and issue not in upstream_concerns:
                    upstream_concerns.append(deepcopy(issue))
        draft = await ask(
            call,
            "outcome_builder_reduce",
            deepcopy(context),
            deepcopy(evidence),
            REDUCE_INSTRUCTION,
            deepcopy(DRAFT_SCHEMA),
            extra={
                **extra,
                "draft": current,
                "reviews": deepcopy(reviews),
                "construction_round": number,
                "construction_version": CONSTRUCTION_VERSION,
            },
            validate=lambda value: _validate_revision(value, validate, evidence),
            max_contract_retries=1,
            max_evidence_rounds=2,
            close_evidence=True,
        )
        row = {
            "round": number,
            "input_fingerprint": fingerprint(current),
            "reviews": deepcopy(reviews),
            "output": deepcopy(draft),
            "output_fingerprint": fingerprint(draft),
        }
        history.append(row)
        record(f"construction/builder/round_{number}/output", row)
    return draft, {
        "version": CONSTRUCTION_VERSION,
        "rounds": rounds,
        "history": history,
        "upstream_concerns": upstream_concerns,
        "unresolved_issues": deepcopy(draft.get("unresolved_issues", [])),
    }
