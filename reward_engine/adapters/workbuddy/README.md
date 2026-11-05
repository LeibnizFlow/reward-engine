# WorkBuddy Office adapter

The adapter connects Harbor's WorkBuddy Office execution to the shared Outcome core. `OfficeDelivery` shares snapshot and review behavior between the default `OutcomeJiuwenSwarmAgent` and the optional `OutcomeOpenCodeAgent`. Each transport runs in the native workspace and collects delivery evidence for the core review loop. See [native execution](../../../experiments/workbuddy/README.md) for setup and configuration.

## Execution and review

The agent starts from the supplied instruction and preserves the actual actor session for internal revisions. It checks terminal session receipts and bounds execution time. The OpenCode transport also bounds output-limit continuations. When a frozen pack is supplied, `DeliveryReview` performs full checks for the three B protocol, or the existing targeted checks for legacy runs, and can request up to two same-session revisions. Feedback is internal advice, not a new public requirement.

The current adapter consumes one supplied instruction per run. It does not implement a bridge for native `[[steps]]` updates or periodic Policy Critic callbacks. The core's [task scope interface](../../core/outcome/MULTITURN.md) is available to custom integrations.

## Artifact snapshots and grading

The agent archives the initial workspace and each completed delivery. Evidence collection records file hashes and extracts supported Office content while retaining the original files. Missing rendering, unexecuted formulas, and incomplete parsing remain limitations; extracted text alone does not establish visual or content correctness.

Review may retain an earlier candidate and measurement as immutable evidence. It does not restore the native workspace or services: `retained_delivery.native_state_restored` remains false. The native verifier observes the actual terminal workspace, so its grade must not be reassigned to an earlier retained snapshot.

For three B runs, a separate worker restores the hash-bound terminal delivery into a fresh native environment and runs the verifier without invoking an actor. Completed native scores can be reused by delivery identity; failed grading attempts retain their records. Legacy runs keep their existing post-actor verifier lifecycle. Native grades remain separate from reward construction and candidate selection. A valid low score is a result, not a reason to resample.

## Gateway admission

The authenticated `POST /control/concurrency` endpoint accepts a `global_limit` and optional `model_limits` mapping. Each model limit must name a configured model and be an integer from one through the configured `per_model_concurrency`. The global limit must remain within `max_model_concurrency`. For example, `{"global_limit": 4, "model_limits": {"slow-model": 1}}` limits future requests for that model while allowing other configured models to use the remaining global capacity.

Queued requests share capacity in turns between roles, Actors and native graders. Requests keep arrival order within each class and model. A model at its limit leaves other eligible models free to execute. Cancelled queued requests are removed immediately; lowering a limit preserves requests already running. The health endpoint reports each model's queued request count and oldest waiting time in `model_queues`.

Admission checks the global and model limits together. A queued request occupies neither capacity slot. Lowering a limit preserves requests already executing and makes queued requests obey the new limit when execution begins. Partial model updates preserve the other limits; updates are recorded in transport limit events and restored on gateway restart. `/health` reports effective limits in `model_limits`; its existing `per_model_limit` field reports the configured per-model ceiling.
