# Outcome core

Reward Engine constructs task-grounded criteria and executable checks, freezes them for actor feedback and selection, and records experience for later construction rounds. Start with the [runnable core example](../../../docs/quickstart.md) or [native benchmark entry points](../../../experiments/README.md).

## Construction and execution

1. Jury members form independent expectations from the public task, then review candidate evidence. Reduce reconciles the proposals while retaining unresolved requirements and evidence links.
2. Builder turns criteria into executable checks. Testcases express required correctness; rubrics express graded quality. Missing implementations and uncertain evidence remain explicit.
3. Hypothesis proposes optional methods for a new actor. Suggestions do not become additional user requirements.
4. A frozen RewardPack supports targeted measurement, same-conversation revisions, and candidate comparison. Native grader scores are excluded from this process.
5. Memory preserves observed changes, methods, evidence, and limitations for later rounds.

Integrations using `TaskSequence` establish a new scope when public requirements change. Internal Jury, Attack, or Critic messages do not create user instructions. See [multi-turn behavior](MULTITURN.md).

## Default and optional construction

All current construction entry points default to `adversarial_review_enabled=false`. Ordinary construction runs Jury, Builder, and basic integrity checks without Attack, independent counterexample references, or related repair/retest steps. Unvalidated criteria remain `unverified`; they gain neither case support nor critical-protection authority. Hypothesis still runs, but its actor projection explicitly labels guidance as optional and unvalidated.

Enable adversarial review only when explicitly requested for the run. Set the flag at experiment configuration top level, or under `runtime` in a QC manifest. `construction_strategy="counterexample_validated"` alone does not enable it. Dedicated counterexample diagnostics reject disabled configurations before model calls.

When enabled, counterexample construction tests selected high-risk criteria/checks on correct controls, incorrect variants, and boundary cases. Independent public references are compared with actual check receipts. A candidate repair is adopted only when it improves the common controls without losing existing support and survives fresh boundary checks. Failure, uncertainty, or a tie retains the original pack. `supported_on_cases` describes evidence from those cases, not universal correctness; reference/check disagreements remain `disputed`.

Structured clauses preserve their use, source text, applicability, and alternatives. A hard parent criterion does not turn illustrative methods into hard requirements. New clauses are not silently synthesized into old packs. Model consensus and source-model counts do not establish truth.

`construction_rounds` controls repeated construction review and is distinct from outer actor rounds. Compatibility configurations can request three construction rounds; the counterexample strategy does not stack that legacy loop automatically. Model seats, transport parameters, and budgets come from the actual run configuration.

## Feedback and selection

Feedback targets the complete desired outcome, including quality, depth, completeness, and usability. `outcome_goal`, `full_score_gaps`, and `improvements` describe the target, unresolved gaps, and prioritized actions. Actions have stable IDs, linked gaps, evidence, expected contribution, and a verification method.

Delivery review distinguishes `needs_improvement`, `needs_verification`, `target_satisfied`, and `unknown`. Target satisfaction is the current review's conclusion, not a native perfect score. Revision budgets are bounded and actual versions are retained. Contract errors receive bounded repair with the same evidence; valid low scores do not trigger resampling.

Targeted review is the default, with configurable `max_tests=3`. Unselected checks remain unknown. Explicit diagnostic full audit uses `DeliveryReview(audit_mode="full")` or `run_task(full_audit=True)`. A narrow criterion pass cannot erase a valid failed local inquiry. Execution errors and insufficient evidence are not, by themselves, product defects.

Selections use actual completed candidates and reject known checked regressions. A fresh task-only candidate is the fallback when strict preference is not established; this rule does not guarantee native-score improvement. Within-candidate review can retain an earlier delivery and its measurement as immutable evidence. This does not restore native workspace or service state: retention records explicitly report `native_state_restored=false`. Native grading observes the actual terminal state, and its score cannot be reassigned to retained evidence.

Outer `rounds` bounds the run (one to three); it is not generally a promise to execute all rounds. Continuation requires a valid Jury request identifying an important evidence-grounded gap and a concrete next direction, plus eligible completed candidates. Native grader scores never drive continuation. The deterministic example deliberately exercises all three rounds.

## Interfaces

| Responsibility | Implementation | Interface guide |
|---|---|---|
| Task and evidence context | [context.py](context.py) | [Public task scopes](MULTITURN.md) |
| Jury and Reduce | [jury.py](jury.py) | [Revision and reducer context](REDUCER_CONTEXT.md) |
| Builder and execution | [builder.py](builder.py), [harness.py](harness.py) | [Measurement tools](MEASUREMENT_TOOLS.md) |
| Optional methods | [hypotheses.py](hypotheses.py) | [Arm protocols](ARMS.md) |
| Memory | [memory.py](memory.py) | Evidence, methods, observed changes, and limitations |
| Delivery and task loop | [runner.py](runner.py), [loop.py](loop.py) | [Runtime critic](POLICY_CRITIC.md) |
| Independent public reward | [public_reward.py](public_reward.py) | [Public-task reward protocol](PUBLIC_REWARD.md) |

Optional task review and agentic search retrieve task-relevant public evidence. Knowledge supplies fallible methods, references, or conditions; it cannot replace the original task. Missing knowledge and unanswered questions remain explicit. Generated check recipes and their tool receipts persist with the pack, enabling restoration without relying on process-local memory.

Requests and checkpoints preserve raw role responses, repairs, measurements, selection decisions, and memory updates. Oversized inputs fail with size diagnostics rather than silently truncating unique requirements. Review execution, actual feedback injection, revision count, selection, and independent task score are separate outcomes.
