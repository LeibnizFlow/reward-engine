# Independent public-task reward

`public-task-reward-v1` is an explicit opt-in protocol for independent single-trajectory evaluation. It adds public fact checks, reviewable requirement bindings, and tiered aggregation without automatically migrating legacy packs or runtime selection.

## Binding requirements and observations

1. Build single-candidate public context with `build_context(task, [candidate])`.
2. A trusted adapter calls `public_facts.bind_observations(context, evidence, descriptors)`. Each descriptor identifies source handle, JSON path, object, property, event sequence, capture time, phase, completeness, and execution state. Values come from the actual source, not a supplied expected answer.
3. Call `public_reward.with_reward_protocol(pack, requirements, context, evidence)`. Every original criterion is classified as `quality`, `gate`, or `advisory`. The function returns a new content-identified pack with its parent ID and does not mutate inputs.
4. Call `harness.measure()`. The protocol registers `public.fact` and returns `independent_reward` alongside ordinary measurements.

Requirements are explicitly reviewed bindings, not inferred oracle labels. Scored requirements include exact public-source quotations and obligations. Each obligation binds a test on the same criterion or a fact compiled into a test. Executed material failures cannot be overwritten by another passing obligation.

Allowed sources are the original task, task attachments, and actual user supplements for the current candidate. Exact quote validation establishes source presence, not that the interpretation follows from the quote. Additional advice is non-scoring. Gates require a prohibition/delivery-prerequisite classification and justification; importance alone does not create a gate.

`scope.delivery_channel` records the delivery boundary, which must also be implemented by an obligation on the actual object. Conditions are sourced public fact predicates: known false makes an item inapplicable; unknown does not force a score. Date-dependent rules use explicit `scope.reference_date={value,sources}`. Missing, conflicting, or inconsistent dates remain unknown, without substituting the machine's current date. Changed task attachments or user supplements invalidate old bindings until reviewed again.

## Facts, time, and snapshots

`public.fact` supports structured equals, contains, and forbids. Membership is exact: basenames, substrings, and aliases are not equivalent delivery paths. Trusted adapters map object/property pairs to observation sources; this is not a general natural-language parser.

- Terminal facts use the latest sequence for the same object/property. Intermediate-only records or inconsistent terminal ordering remain unknown.
- History retains an observed prohibited action even after deletion or honest disclosure. Proving absence requires complete history that is not older than subsequent observations.
- Partial inventories can establish presence, but not absence. Execution failure and missing evidence retain distinct unknown reasons.
- Candidate identity and complete public inputs bind observations. Hashes enforce identity, not the truth of an adapter's completeness or capture-phase assertions.

`adapters.public_reward_snapshot.inspect_snapshot()` checks exact file locations and SQLite table columns in a specified public snapshot. It verifies archived hashes, rejects traversal, symlinks, and uncheckpointed databases, uses read-only immutable SQLite, and never runs candidate scripts. Missing paths establish absence only with an explicitly complete snapshot. A private grader snapshot or rerun cannot be substituted for a missing original public terminal snapshot.

## Aggregation

Checks return pass/fail/unknown. Explicit obligations produce criterion tiers: all satisfied is full (1), all violated is none (0), and a mixture of satisfied and violated is partial (0.5). Partial records actual satisfied/missing evidence. Unresolved obligations produce a possible tier interval; known satisfied and violated obligations already fix partial while still retaining uncertainty.

Quality criteria have equal weight in Q. Adding tests or splitting obligations does not increase a criterion's weight. Duplicate IDs, exact duplicate obligation descriptions, and repeated scoring of the same test are rejected; semantic duplication still needs review.

- Any established gate failure gives R=0, even if other checks are unknown.
- With all gates passed, R=0.5+0.5Q.
- Outcome-relevant uncertainty gives `reward=null`, preserves `r_min/r_max`, and sets `training_mask=false`.
- With no quality criteria and valid gates all passed, Q=1. When every requirement is inapplicable, the result remains undetermined rather than full credit.

This is an ordering encoding, not a success probability. Scope, sources, satisfied/missing/unresolved obligations, tests, and evidence remain available. Real ties are retained. The [minimal core example](../../../docs/quickstart.md) does not demonstrate the effectiveness of this optional protocol.
