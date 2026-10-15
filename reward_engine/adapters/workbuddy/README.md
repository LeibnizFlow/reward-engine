# WorkBuddy Office adapter

The adapter connects Harbor's WorkBuddy Office execution to the shared Outcome core. `OutcomeOpenCodeAgent` extends `OpenCodeTransport`, runs OpenCode in the native workspace, and collects delivery evidence for the core review loop. See [native execution](../../../experiments/workbuddy/README.md) for setup and configuration.

## Execution and review

The agent starts from the supplied instruction and preserves the actual OpenCode session for internal revisions. It checks terminal session receipts and bounds execution time and output-limit continuations. When a frozen pack is supplied, `DeliveryReview` performs targeted checks and can request up to two same-session revisions. Feedback is internal advice, not a new public requirement.

The current adapter consumes one supplied instruction per run. It does not implement a bridge for native `[[steps]]` updates or periodic Policy Critic callbacks. The core's [task scope interface](../../core/outcome/MULTITURN.md) is available to custom integrations.

## Artifact snapshots and grading

The agent archives the initial workspace and each completed delivery. Evidence collection records file hashes and extracts supported Office content while retaining the original files. Missing rendering, unexecuted formulas, and incomplete parsing remain limitations; extracted text alone does not establish visual or content correctness.

Review may retain an earlier candidate and measurement as immutable evidence. It does not restore the native workspace or services: `retained_delivery.native_state_restored` remains false. The native verifier observes the actual terminal workspace, so its grade must not be reassigned to an earlier retained snapshot.

The worker runs the native verifier after the actor trial. Native grades remain separate from reward construction and candidate selection. A valid low score is a result, not a reason to resample.
