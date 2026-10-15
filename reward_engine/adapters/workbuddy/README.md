# WorkBuddy Office adapter

The adapter connects WorkBuddy's native Office execution to the shared Outcome core. `OutcomeOfficeAgent` inherits native behavior; the worker hook verifies source anchors and fails explicitly if the expected upstream interface differs. It does not maintain a second reward algorithm.

## Public task steps

Native `[[steps]]` preserve the workspace and actual public conversation. Only instructions already disclosed to the actor enter the task scope; future steps and private verifier data are excluded. Each step has its own review state and restorable snapshots.

A new public step rebinds the existing pack as unverified reference material, clearing earlier validation support and requirement coverage. Review and version selection use the complete current task. Existing Hypothesis guidance is optional and cannot override new instructions. The adapter does not rebuild the entire Jury/Attack pipeline for every public step.

## Artifact snapshots and selection

At a delivery boundary the controller pauses native requests, checks file inventories before and after transfer, and preserves file bytes, sizes, modes, and SHA-256. Office XML inspection retains text, structure, cell addresses, formulas, and cached values. Unexecuted formulas, absent rendering, and incomplete parsing remain explicit limitations. These observations alone do not establish content or visual correctness.

Full file contents have evidence handles for subsequent reading; binary originals remain preserved. After version comparison, the adapter restores the selected snapshot's actual files and verifies their hashes before releasing the worker. With no clear improvement, or with selection or revision failure, it retains the first complete delivery. Versions may only be compared and restored within the same public requirement scope.

Runtime Critic reads actual tool observations, not actor plans presented as completed artifacts. The native verifier runs after the actor process stops. Valid low-score stops remain low scores rather than triggering resampling.

See [native execution](../../../experiments/workbuddy/README.md) and [multi-turn scopes](../../core/outcome/MULTITURN.md).
