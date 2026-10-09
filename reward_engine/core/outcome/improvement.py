"""One full-score Outcome objective shared by delivery and runtime feedback."""

from reward_engine.core.outcome.context import ResponseContractError, require_refs

INSTRUCTION = (
    "Your objective is to help the Actor deliver a FULL-SCORE OUTCOME. Work backwards from "
    "the best complete result for the actual Task, using the supplied Outcome criteria/full "
    "anchors and relevant knowledge. Identify what separates the current work from that result, "
    "then recommend concretely HOW to close that distance. Correct core claims are only part "
    "of full quality: examine completeness, explanatory depth, quantitative reasoning where "
    "useful, applicability, and whether the proposed solution can actually be used. Assess all "
    "relevant Outcome dimensions; tool measurements remain selective, not a checklist quota. "
    "The supplied criteria and memories are aids, not a ceiling on quality: they can omit "
    "important aspects or contain wrong premises. Independently derive what the user needs "
    "to understand, decide or implement from this Task. For explanations and technical reviews, "
    "naming a mechanism or a remedy is not the same as explaining or making it usable: locate "
    "missing causal steps, useful quantitative reasoning, concrete solution/design details, "
    "tradeoffs and applicability constraints. Recommend the actual substantive content to add "
    "and how it answers the Task; do not merely request longer text or more topic names. "
    "Prioritize the largest remaining quality gains. Do not let instructions to write the "
    "deliverable, correctness caveats and evidence-boundary wording consume the entire improvement "
    "plan while central analysis or solution depth remains missing. Verify specific factual "
    "premises from memories before prescribing them as corrections. "
    "Do not stop just because there is no factual error or the minimum requirements are met. "
    "Use full_score_gaps for the complete remaining quality-gap ledger, including improvement "
    "opportunities that are not factual defects. Use gaps only for evidenced defects. "
    "Select at most THREE prioritized improvements to execute next. Each must locate the "
    "current shortfall, specify a concrete change, explain its expected contribution to the "
    "Task outcome, and state how to verify the change. More words, more tools or a keyword "
    "list alone are not quality gains. Ground opportunities in this Task and current evidence; "
    "do not invent user requirements, private grading rules, exact scores or new facts. "
    "Retain stable gap/improvement IDs across follow-ups. Reassess earlier suggestions using "
    "the actual new work: adoption is not proof of improvement, and a change can introduce a "
    "regression. Keep unresolved gaps in the ledger. When no gaps remain, explain how the "
    "current work satisfies the full-quality target; lack of an idea is not that explanation. "
    "The primary verdict assesses the FULL outcome, not just factual correctness. Any remaining "
    "quality gap requires improvement or verification; absence of factual defects is only a "
    "subfinding, never a release verdict. Unfinished work can have next-step opportunities "
    "without being defective. Private independent grading happens separately. "
    "Structured clauses distinguish Task constraints, fallible quality judgments and illustrative "
    "methods. A hard parent does not make every method mandatory. Preserve applicability and "
    "alternatives. Disputed/unverified guidance may motivate a bounded candidate revision, not "
    "a declaration that the old work is wrong. Compare real outcomes and preserve the original. "
)


def schema():
    return {
        "outcome_goal": "Concrete full-score result for this Task, not merely minimum compliance",
        "full_score_gaps": [
            {
                "id": "stable gap ID",
                "gap": "Current shortfall from full quality",
                "evidence_refs": [
                    "Task/current-work/knowledge refs supporting this opportunity"
                ],
            }
        ],
        "improvements": [
            {
                "id": "stable action ID",
                "gap_ids": ["full_score_gaps IDs addressed"],
                "action": "Specific next change or verification; locate what to improve",
                "expected_gain": "Why this helps achieve the full-score Task outcome",
                "verification": "Observable evidence that would show the change helped",
                "evidence_refs": ["actual available refs"],
            }
        ],
    }


def validate(row, evidence):
    """Require actionable grounded opportunities, not a new numeric self-judge."""

    def fail(field, reason):
        raise ResponseContractError(field, reason)

    def string(value, field):
        if not isinstance(value, str) or not value.strip():
            fail(field, "Expected nonempty text")

    def refs(value, field):
        try:
            require_refs(value, evidence)
        except ValueError as error:
            raise ResponseContractError(
                field, "Use supplied evidence references"
            ) from error

    string(row.get("outcome_goal"), "outcome_goal")
    gaps, actions = row.get("full_score_gaps"), row.get("improvements")
    if not isinstance(gaps, list) or not isinstance(actions, list):
        fail("full_score_gaps/improvements", "Both lists are required, even when empty")
    gap_ids, action_ids = set(), set()
    for i, gap in enumerate(gaps):
        field = f"full_score_gaps[{i}]"
        if not isinstance(gap, dict):
            fail(field, "Expected a quality-gap object")
        for key in ("id", "gap"):
            string(gap.get(key), field + "." + key)
        if gap["id"] in gap_ids:
            fail(field + ".id", "Duplicate gap ID")
        gap_ids.add(gap["id"])
        refs(gap.get("evidence_refs"), field + ".evidence_refs")
    if len(actions) > 3:
        fail(
            "improvements",
            "Prioritize at most three next actions; keep other gaps in the ledger",
        )
    for i, action in enumerate(actions):
        field = f"improvements[{i}]"
        if not isinstance(action, dict):
            fail(field, "Expected an actionable improvement object")
        for key in ("id", "action", "expected_gain", "verification"):
            string(action.get(key), field + "." + key)
        if action["id"] in action_ids:
            fail(field + ".id", "Duplicate action ID")
        action_ids.add(action["id"])
        ids = action.get("gap_ids")
        if (
            not isinstance(ids, list)
            or not ids
            or any(not isinstance(g, str) or g not in gap_ids for g in ids)
        ):
            fail(field + ".gap_ids", "Reference existing full-score gap IDs")
        refs(action.get("evidence_refs"), field + ".evidence_refs")
    if gaps and not actions:
        fail(
            "improvements",
            "Remaining full-score gaps need a next improvement or verification action",
        )
    if row.get("status") == "observe" and (
        gaps or actions or row.get("gaps") or row.get("verify")
    ):
        fail(
            "status",
            "Remaining gaps or verification require suggest/clarify, not observe",
        )
    return row
