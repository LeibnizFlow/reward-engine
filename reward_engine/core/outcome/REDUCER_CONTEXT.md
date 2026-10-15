# Jury revision and Reducer context

Normal Outcome loop, validated construction, and Engine Jury entry points enable `revision_protocol=True`. Low-level `jury.propose` and `_propose_once` retain the legacy default for compatibility; new callers should choose the option explicitly.

## Member revisions

The first stage forms expectations from the task, actual user supplements, and permitted independent sources, without products or historical methods. The second stage receives each member's `independent_expectations` and product evidence, returning a complete revision plus `revision_decisions`:

- `retain`: preserve the original; `{"inherit_from":"expectation_id"}` may inherit a complete criterion.
- `rewrite` / `merge`: provide complete target clauses, reasons, and existing evidence references.
- `delete` / `defer`: retain the original for Reducer review; deletion proposals require reasons and evidence.
- `add`: explain the new clause's basis. Undeclared additions appear as `unclaimed_additions`.

Each decision maps original `source_parts` to target `criterion_id` / `part_index`. Validation checks text, parent meaning, conditions, priority, sources, and verification method. Changed fields cannot be labeled retain. Undisposed original requirements remain unresolved; omission by every candidate does not justify deleting a requirement. Member edits become criteria only after validated Reduce output.

## Lossless request projection

The request preserves task, actual user supplements, criteria, revision grounds, disputes, unresolved original requirements, check results, and uncertainty. Verified identical retained items use selector aliases that expand back into original part IDs. Complete originals remain in checkpoints; changed, deleted, and unresolved originals remain visible for review.

Text sharing is byte-for-byte: `{"$reducer_text":"t0"}` refers to `shared_texts.t0`. Parent scopes, identity, conditions, and decision objects remain separate. No normalization, synonym matching, or embedding similarity establishes equivalence. Reversible sharing is used only when the whole encoded request is smaller. Adapter revision proposals remain literal.

Historical memory method bodies become navigable indexes with byte counts and SHA-256. Scope, observed change, status, limitations, and unknown fields remain inline. `need_evidence` retrieves complete records. These navigation handles are not final criterion evidence; criteria must cite the original task, product, or check. Retrieved content is subject to the complete request's byte limit and is never silently truncated or falsely labeled complete.

Projection adds neither summarization calls nor evidence-read budget. Requests that cannot fit fail explicitly and preserve checkpoints.

## Identity and recovery

Map cache identity includes `outcome-member-revision/2026-09-15-v1`; completed Engine criteria also include `outcome-reducer-context/2026-09-15-v1`. Unchanged independent drafts may be reused, but legacy completed criteria are not treated as new-protocol results. Checkpoints without the revision marker recover through the legacy Reduce protocol, including explicit `qc_recover_jury` support.

Complete member revisions persist under `revisions/<seat>` and in proposals/checkpoints. Downstream criteria, tests, and coverage schemas remain compatible. `reducer_request_projections` records request and field sizes, hashes, evidence stage, repair attempt, and budget eligibility. Unsent oversized entries have `within_budget=False`; ordinary retries retain separate attempt files.
