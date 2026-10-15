# Arm protocols

`arms.py` defines the shared routing used by the core loop and native entry points. Choose `arm_protocol="hypothesis_feedback_v1"` explicitly to use the following independent arms:

| Arm | Initial input | Additional feedback |
|---|---|---|
| A | Task | None; original actor baseline |
| B | Task and Hypothesis | None |
| C | Task | Pre-delivery reward review, same-session revision, and remeasurement |
| D | Task | Periodic runtime Critic, then final review by the same Critic and at most one continuation |

Only B receives the generated Hypothesis. C/D share the frozen RewardPack and HarnessPack without Hypothesis or historical Battle Memory; they retain their own current-session feedback and revisions. They receive no additional static reward prompt. D does not also run C's delivery revision loop.

Each arm starts independently. Zero revisions in C means C retained its own first draft, not that it equals B. Differences between independent arms include generation and grading variability. To measure revision gains, compare a single arm's preserved initial and final versions; do not reinterpret independent samples as paired revisions.

## Configuration

Use a valid native task/model configuration with explicit protocol fields, for example:

```json
{
  "arm_protocol": "hypothesis_feedback_v1",
  "arms": ["A", "B", "C", "D"],
  "rounds": 1,
  "construction_rounds": 3,
  "max_revisions": 2,
  "max_tests": 3,
  "actor_harness": {"enabled": false},
  "policy_critic": {"enabled": true, "every_k_turns": 1}
}
```

`construction_rounds` expands construction review only; it does not multiply actor runs, feedback frequency, or independent grading. The protocol defaults to four arms, one outer round, and Critic frequency one. An explicitly supplied frequency is retained. Including D while explicitly disabling Critic is a configuration error.

C resumes the same actor rather than resampling a fresh run; its normal revision budget is two. D intervenes after an eligible root actor tool turn and before the next root model request. At delivery it reviews the actual final product even if periodic reviews occurred, then permits at most one same-session continuation and recheck. Counterexample-validated terminal control may impose a stricter revision budget; inspect the frozen configuration.

A semantic Hypothesis `keep` skips only B. C/D continue and record `hypothesis_applied=false`; the four-arm result is partial, with no invented B score. Unavailable construction is different from semantic keep and stops full initialization before sampling. Explicit manual A/C/D controls without Hypothesis do not constitute a complete four-arm run.

Feedback costs are additional to the common actor baseline budget. Preserve requests, measurements, revisions, elapsed time, and native traces; this is not an equal-cost comparison. Native grading occurs after sampling and does not enter Hypothesis, RewardPack, or Critic inputs.

## Compatibility and execution status

The default for configurations without an explicit new protocol remains `legacy_abcd`: A is original, B uses static reward and delivery review, C uses Hypothesis, and D combines both. Optional legacy Critic settings remain separate. Do not relabel historical arms using current routing; frozen source and actual requests define historical behavior.

`feedback_execution` distinguishes delivery review and policy critic. States include `disabled`, `completed`, `degraded`, `unavailable`, and `unknown`; periodic D can also be `not_triggered`. Receipts retain effective reviews, unavailable reviews, revision requests, checkpoints, failed checkpoints, and actual injections. A completed review is not proof of a correct product. Valid `observe` can produce no injection. Missing logs yield unknown, not assumed coverage.
