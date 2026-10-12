"""Jury proposal prompts; version travels with requests and replay checkpoints."""

PROMPT_VERSION = "outcome-jury/2026-10-12-bounded-contract"
CONSTRUCTION_PROMPT_VERSION = "outcome-jury/2026-10-12-three-round-bounded-contract"

OUTCOME_VALUE_INSTRUCTION = (
    " Organize standards around the main requested results and their relative importance. Preserve "
    "priority and priority_reason in every final criterion. kind=hard means the Task requires the "
    "work; it does not mean any imperfection makes the entire outcome unacceptable. Separate core "
    "completion, partial fulfillment, and polish. Distinguish an explicit Task prohibition from "
    "a quality recommendation; do not invent approval workflows, permissions, or mandatory evidence "
    "presentation formats. Finding or verifying information in actual tool results is different from "
    "repeating every detail in final prose. Require both only when the Task or outcome needs both. "
    "Distinguish proposed next steps and conditional commitments from claiming an action was approved "
    "or already executed. Ground the scope of each restriction in actual Task/user text; a generic "
    "source quote does not justify an added prohibition. Use public task facts and source records to "
    "make outcome anchors concrete without importing any hidden rubric."
)

OUTCOME_JURY_AUDIT_PROMPT = (
    "Independently audit the ENTIRE previous synthesized Outcome definition against the original Task "
    "and each product's scoped actual user amendments. Those define scope; optional knowledge and old "
    "products are fallible supplemental evidence, not a ceiling on domain correctness or full quality. "
    "Read the Task again before the previous synthesis. Review every current criterion, including its "
    "subrequirements, factual assumptions, conditions, meaning, source support and missing major outcomes. "
    "Do not limit your audit to previous disagreements. The previous synthesis is the object being "
    "examined, not new authority. Distinguish a supported correction from uncertainty requiring source "
    "reading; no voting, invented facts, user replies or hidden grader criteria. An unknown launch, "
    "runtime setting or applicability condition is not an established fact. "
    "Return an audit row for EVERY supplied criterion ID, then one complete proposed definition using "
    "the original Task's scope. Retain useful precise conditions and equivalent valid approaches. "
    "Explain changes with actual evidence, and keep unresolved expectations visible. current_text in "
    "the pending ledger is the latest clause to resolve, not its superseded historical wording. "
    "Criteria may be kept, refined or reorganized; do not create a new scoring task or require the "
    "particular solution demonstrated by an old product. No absolute score or claim that agreement "
    "proves correctness."
)

OUTCOME_EXPECTATIONS_PROMPT = (
    "Before seeing ANY old deliverables, outline the major results a good answer must achieve. Start "
    "from Task and scoped user amendments; optional skills are fallible domain knowledge. Do not "
    "invent user requests. Explore distinct dimensions of correctness, completeness and applicable "
    "domain mechanisms, not a quota or one prescribed solution. Explain WHY each matters to the "
    "requested outcome, and HOW one could verify it. Use a few broad outcome questions, not a long "
    "checklist of incidental details. Mark priority critical/important/minor. Use task/user hard "
    "criteria only for actual requirements; professional suggestions are soft. If expertise is "
    "missing, name the source/question to investigate in measurement_question. List important "
    "distinct subrequirements explicitly, so later synthesis cannot silently replace them with one "
    "broad label. No exhaustive quota or new hard user obligations. No product observations, gaps, "
    "ranking or absolute score in this phase. Return the exact top-level criteria list in the "
    "response contract; do not substitute an assessment report or rename its fields. Use exact "
    "supplied evidence handles, not paths or invented citations."
)

OUTCOME_JURY_MAP_PROMPT = (
    "Now inspect the products AGAINST the independently proposed expectations. These expectations are "
    "UNVALIDATED drafts, not authority: correct field/provenance errors against evidence. Keep "
    "original suggestions visible for Reduce even when correcting them. List important "
    "subrequirements explicitly, including NEW ones discovered here. Keep important shared omissions "
    "visible even when no product demonstrates the missing skill. Define a usable Outcome estimator: "
    "necessary Task checks and graded quality rubrics. Look for shared omissions as well as "
    "differences. Do not require ranking evidence to retain a Task requirement. Domain suggestions "
    "are soft priors, not new user constraints. For each meaningful suspected gap supply expected "
    "result, observed result, discrepancy, provenance and a check that could support OR reject it. "
    "Inspect actual user amendments and environment state, not just prose. Domain experience may "
    "propose a testable suspicion but cannot create a hard obligation. check_supported requires an "
    "existing failed check; otherwise needs_check. Never invent a gap to fill a quota. No absolute "
    "total score. Return the exact top-level criteria list in the response contract, not an "
    "assessment of pass/fail or scores. Cite exact supplied evidence handles; check indices and "
    "source paths must not be guessed into references. Avoid reprinting product content or whole "
    "prior drafts; preserve distinct requirements and cite their existing evidence handles."
)

OUTCOME_JURY_REDUCE_PROMPT = (
    "Synthesize a compact Outcome definition by selecting and grouping the supplied clauses. Member "
    "criterion_refs locate the two complete expectation lists. They are UNVALIDATED drafts, not "
    "truth: decide using Task, scoped user requirements and actual evidence. Write a SHORT standard "
    "and measurement question for each final criterion, preserving why/priority. Do not repeat all "
    "source clauses in standard or subrequirements. Use each clause's short selection_id (p0, p1, "
    "...) in part_decisions, not parent IDs. selection_aliases groups EXACT duplicates with identical "
    "scope; decide once on the canonical ID and code applies it to all listed aliases. Do not "
    "separately decide aliases. Return selection_protocol='exact-alias-v1'. All original sources "
    "remain recorded. retain means code copies the EXACT original text into that criterion's "
    "subrequirements. Group many retained clauses in one decision; no retyping or separate parent "
    "coverage. Prefer retain for supported detail; do not silently strengthen qualifiers, change "
    "assumptions, or replace equivalent alternatives by one mandatory answer. If a clause is wrong or "
    "conflicting, rewrite with explicit replacement text, reason and supplied evidence_refs; several "
    "duplicate clauses may share a rewrite. Defer uncertain clauses or exclude irrelevant suggestions "
    "with a reason. Each selection_id gets at most one decision; omitted clauses remain unresolved, "
    "never implicit approval. Do not discard a Task obligation or important shared omission merely "
    "because all products satisfy or violate it. Domain advice stays soft/conditional, never silently "
    "mandatory inside a hard criterion. New subrequirements may be added only when needed and "
    "justified by the supplied Task/evidence. Resolve draft_validation_issues, not discard the whole "
    "member. No invented permissions, sources or absolute scores. Combine consequential gaps with "
    "expected/observed/evidence/verification intact; votes cannot turn a suspicion into "
    "check_supported. expected_basis is task/user/domain, hard origin is task/user. Inspect factual "
    "consistency, not only topic coverage."
)

OUTCOME_EXPECTATIONS_PROMPT += OUTCOME_VALUE_INSTRUCTION
OUTCOME_JURY_MAP_PROMPT += OUTCOME_VALUE_INSTRUCTION
OUTCOME_JURY_REDUCE_PROMPT += OUTCOME_VALUE_INSTRUCTION
OUTCOME_JURY_AUDIT_PROMPT += OUTCOME_VALUE_INSTRUCTION
