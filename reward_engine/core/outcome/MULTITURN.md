# Public task scopes and internal revisions

Claw-Eval simulator follow-ups, `ask_user` answers, and WorkBuddy native `[[steps]]` are public task updates. Jury, Attack, and Critic messages are internal feedback and do not create user requirements.

`TaskSequence` preserves initial instructions, actual public messages in arrival order, and original input-file versions. Later conflicting instructions supersede earlier ones. Actor claims, future simulator instructions, and grader information do not become requirements. Scope hashes bind snapshots, version comparisons, and independent content grading.

When public requirements change, the old RewardPack is rebound as unverified reference material: validation support and requirement coverage are cleared, and criteria become soft. Review follows the complete current task, so withdrawn requirements are not reported as current defects. This does not run a fresh three-round construction pipeline or claim all new criteria are validated. Existing Hypothesis guidance explicitly belongs to the initial scope.

## Claw-Eval

The native `JiuwenContainerRuntime.prompt` manages simulator messages, DONE, follow-ups, sessions, request counts, and wall-clock budgets. The task scope updates before public messages are sent to `/prompt` or `/answer`. Each delivery boundary runs the applicable C/D review.

Internal revisions continue the native session without entering the public user-message list. At most one terminal revision is permitted per public scope. New requirements clear old review state, close prior Critic requests, and enable a Critic for the current scope.

Initial and revised versions are compared only within the current scope; selection cannot revert to an earlier user turn. Independent native conversations may receive different follow-ups, so candidates from different scopes are not forced into a common winner. Content grading carries the actual public requirements and does not reuse scores across changed scopes. Native trajectory grading remains separate; independent content scores are not official leaderboard-equivalent scores.

The bridge supports text messages. Unsupported attachment messages fail explicitly rather than being silently dropped.

## WorkBuddy

The [Office adapter](../../adapters/workbuddy/README.md) preserves the native workspace and actual public history across steps. Each step has independent review and snapshot state. Legitimate native low-score stops are retained as results rather than retried as execution failures.
