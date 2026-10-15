# Public task scopes and internal revisions

`TaskSequence` in [task_sequence.py](task_sequence.py) provides a core interface for public task updates. The current Claw-Eval OpenCode runtime and WorkBuddy OpenCode agent do not connect simulator follow-ups or native `[[steps]]` to this interface. Same-session delivery revisions are supported; they do not imply support for changing public requirements.

## Scope contract

An integration creates a sequence from the initial task, then calls `advance(text, event_id=..., documents=...)` only for actual public messages. `task` and `scope` expose the current requirements and scope identity. `observe(candidate)` attaches current scope information to evidence; `bind(pack)` and `rebase(pack)` bind reward criteria to the applicable task scope.

The sequence preserves initial instructions, actual public messages in arrival order, and input-file evidence. Later conflicting instructions supersede earlier ones. Actor claims, future simulator instructions, hidden answers, and grader information do not become requirements. Jury, Attack, Critic, and revision messages are internal feedback and must not be added as public updates.

When public requirements change, rebasing marks the old RewardPack as unverified reference material: validation support and requirement coverage are cleared, and criteria become soft. This does not run a new construction pipeline or establish validation for new requirements. Existing Hypothesis guidance remains tied to its original scope.

## Integration responsibilities

Custom adapters must advance the sequence before acting on a public update and attach the actual current task to returned candidates. They must preserve message and file evidence, reset scope-specific feedback state, and prevent comparisons or score reuse across incompatible scopes. Unsupported public message types must be handled explicitly.

The core's scope checks cannot replace native conversation handling. Adapters remain responsible for native sessions, user interaction, task completion, and grading. Retaining an earlier delivery as review evidence does not restore native workspace or service state and does not transfer its official grade.
