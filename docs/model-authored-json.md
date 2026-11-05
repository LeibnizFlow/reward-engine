# Model-authored role JSON

`model_authored_json_v2` is an explicit implementation revision of the internal role interface. Configurations without `role_output_protocol` retain their existing parser, editing protocol, budgets and checkpoint interpretation. OpenCode and JiuwenSwarm share this interface.

Enable it with `python -m experiments.revision --output RUN --parent-revision REVISION --model-authored-json`. The revision freezes source code and sets `role_output_protocol="model_authored_json_v2"`, `role_recovery_policy="strict_bounded_v2"` and `failure_policy="pause_repeated_v1"`. Actor models, tasks, runtime and native grading settings cannot change. Existing actor deliveries and frozen Packs remain bound to their original checkpoints.

Independent expectations contain criteria with semantic metadata and `clauses`. Only `clauses[].text` defines detailed requirement text. The engine generates criterion/clause IDs, parent text and `subrequirements`. Do not submit these derived fields.

A member can retain a whole requirement using only `{"inherit_from":"supplied expectation_id"}`. Otherwise it returns a complete criterion. Each clause includes `source_parts`, `action`, `reason` and its explicit `evidence_refs`. Actions are `retain`, `rewrite`, `merge` or `add`; only additions have empty sources. Retain requires exact text and unchanged meaning, provenance, conditions and priority. Withheld sources go in `dispositions` with `delete` or `defer` and a reason; deletion needs evidence. Every source needs exactly one explicit disposition.

Inherited requirements from different reviewers can share a seat-local internal ID. The compiler assigns a distinct target ID when necessary while preserving every semantic field and the original source handles. Repeating the same source is still rejected. Retain diagnostics identify the exact source and present two choices: inherit the entire unchanged requirement, or author a complete revision with an explicit change action, reason and evidence. The model makes that choice.

Reduce uses the same representation, with canonical `selection_id` handles in `source_parts` and `exclude`/`defer` in `dispositions`. Exact aliases expand deterministically. The engine generates legacy bindings from the final clause and its declared sources. It never matches similar text, changes retain to rewrite, supplies evidence or accepts a claimed execution. Original clause, provenance, retirement, coverage and counterexample validators remain authoritative for admission.

Reduce also compares optional model-authored parent fields when checking `retain`. Adding, removing or changing such a field requires an explicit revision; only internal identifiers, derived text projections and validation annotations are excluded from this comparison.

Only a single strict JSON object or a complete unambiguous JSON code fence is parsed. Duplicate keys, prose, competing objects, non-JSON constants and truncated completions require model correction. Recovery supplies the whole rejected draft, its identity, original public input and diagnostics to the originating role. The model resubmits a complete result; no patch paths, hashes or restoration IDs are required in its output.

Canonical requests also specify JSON string escaping, including embedded quotes, backslashes and line breaks. Syntax correction includes `response_recovery.details.syntax` with the parser-reported `line`, `column`, `offset`, `context_start` and an exact `context` excerpt of at most 160 characters. Coordinates refer to `rejected_raw_response`: offsets count Unicode characters from zero, while lines and columns start at one. These are parser locations, not inferred semantic field paths. The model must resubmit the complete JSON object; diagnostics do not repair or accept invalid text, and the single syntax-correction budget is unchanged.

Each logical operation allows three content corrections, one syntax correction, two evidence expansions, eight physical model requests including transport recovery, and 1800 seconds. Identical consecutive drafts with identical errors stop early. Role cache identities and completed-operation receipts include protocol, budgets and implementation identity. Completed-operation records bind all public inputs and payload hashes; corrupted checkpoints fail closed. Unknown historical substage provenance does not qualify for reuse.

`response` describes the output schema; its fields belong directly at the output root. Requests expose remaining evidence rounds and reference capacity in `evidence_budget`. After access closes, a new read request or a final answer citing unread previews receives complete-draft diagnostics within the existing content-correction budget. No additional evidence expansion is granted, and the corrected judgment must remain grounded in supplied observations.

Measurement fields and request templates share `measurement_contract.py`. Numeric and file probes execute real tools. At most two waves and four executions are allowed; a first wave can use at most three executions. A premature final response leaves unused correction slots available. A failed execution is resolved only by explicitly cited, later successful receipts from the same tool covering the failed inputs. Failed receipts remain recorded. Independently completed measurements can resume without executing their model or tools again.

Tool availability can depend on the supplied evidence. An unavailable capability is rejected before invocation and recorded with `request_rejected=true`; it consumes no execution and supplies no observed result. For example, workspace inspection requires a collected file inventory. A supplied but corrupt inventory still reaches validation and remains an execution failure. Legacy tool invocation retains its original behavior.

The failure policy persists in the implementation revision. In the latest ten terminal tasks, three failures with the same stage/error code or five failures in total stop new admissions. Already running tasks finish. Restarting the same revision retains the pause. Terminal scheduler state reports zero active tasks, pending count and the pause reason. Adaptive admission starts at four tasks and doubles after each ten consecutive completions without a failure, capped at 128 and the existing resource/service limits. `campaign run --only-task ID` (repeatable) restricts admission to a subset of frozen tasks for a small batch without changing experiment identity.

Native scores and private grading inputs remain outside role context, Pack construction, measurement and six-candidate selection.

Requests include `source_catalog` for Map/Reduce: its keys are requirement clause IDs. These are distinct from evidence handles such as `task` or product/source references. Citation diagnostics identify every invalid parent/clause `evidence_refs` field without filling it. Completed authored drafts from an explicit parent revision may be reused only when role settings and exact public inputs match and the full current contract revalidates the original compact draft; a reuse receipt binds both checkpoint identities. Unrelated revisions are not searched.

Other completed review outputs, including successful selection seats, require the same identity checks and full current validation of their completed JSON before reuse. A failed seat does not invalidate another seat's proven completion.

## Literal source input and single-attempt gateway

A new implementation revision may opt into `role_input_projection="canonical_sources_v1"`
while retaining `model_authored_json_v2` and `strict_bounded_v2`. The default keeps
historical input and transport behavior. Revision manifests, request cache keys,
protocol receipts and reports record this setting. Completed operations bind the
original semantic inputs before projection; ancestor reuse requires the same
inputs, role/model settings, lineage proof and full current validation.

Map and Reduce expose `source_requirements`, keyed by `expectation_id`, with exact
clause selectors in `source_catalog`. Phase lists contain directory keys. A parent
`standard` is omitted only if it exactly equals the newline join of clause text.
A subrequirement text is omitted only if its explicit part index points to exactly
the same clause text; `clause_index` records that position. All other historical
fields remain. `_local/` runtime role receipts retain the original directory and
omission mappings. Text remains literal; this path does not emit `$reducer_text`.

The response template describes an actual `criteria` array of 1–20 items. Each Map
or Reduce item is either an object containing only `inherit_from`, or a complete
requirement with explicit clause actions, source selectors, reasons and evidence.
Independent expectations permit only complete requirements. Preflight collects
structure, source coverage, exact retain consistency, evidence and gap provenance
errors. Diagnostics include codes, fields, actual values and rules; sections that
cannot be inspected due to malformed structure are identified. Publication still
requires all existing validators. Unsupported `check_supported` claims are
rejected; code does not replace them with `needs_check` or add missing evidence.

Every `source_quotes[].ref` must also appear in the same clause's `evidence_refs`;
parent requirement citations do not satisfy this relationship. Canonical response
templates state this rule explicitly. A `QUOTE_REFERENCE` diagnostic identifies
both the destination `field` and the offending `quote_field` and `quoted_ref`,
alongside the actual citation list. The model must explicitly revise the complete
draft; diagnostics never insert references or change clause actions.

The role owns one transport recovery per logical operation. It shares the eight
request / 1800 second bound with content and syntax correction and evidence
expansion. Empty final content is a provider failure, separate from JSON syntax;
reasoning content is never accepted as an answer. Complete JSON resubmission is
required; duplicate keys, multiple documents, prose and truncated responses remain
invalid. Model and reasoning parameters stay fixed during recovery.

With canonical sources, a nonempty reply ending in `finish_reason="length"` is
`TruncatedFinalAnswer`, not a JSON syntax error. It shares the single transport
recovery with empty answers and service failures. Recovery resends the complete
unchanged request; it never joins partial replies or accepts even parseable text
from an incomplete completion. Failure receipts retain the provider usage,
requested output limit and partial answer. A second transport failure is terminal.
An omitted finish marker is not treated as a successful completion.

Set `role_max_tokens` explicitly for internal roles that produce large drafts.
Omitting a role leaves its output limit to the provider, whose default may be too
small for a complete response. These limits are frozen in the implementation
revision and request cache identity. They do not apply to actor or native scoring
requests and never increase automatically during recovery. Changing a role's
limit prevents reuse of that role's incompatible unfinished-operation cache;
completed delivery and measurement stages retain their existing frozen results.

For this setting only, `/roles` requires `X-Role-Operation-ID`,
`X-Role-Request-ID` and `X-Role-Operation-Deadline`. They are internal HTTP headers,
never model inputs. The gateway performs one upstream attempt per admitted request,
reserves at most eight admissions durably across restarts, rejects repeated physical
IDs, and returns `X-Role-Gateway-Receipt`. Receipts distinguish gateway calls,
upstream attempts and pre-execution rejection. Actor, grading and historical role
paths retain their own transport behavior. Reports retain the quorum failure and
underlying seat receipts. Historical requests without operation IDs are explicitly
unproven against the logical budget; observed gateway attempts remain reportable.

## Canonical recovery contract

An explicit implementation revision can additionally set
`role_contract_policy="canonical_recovery_v1"`. It requires canonical source
projection and authored JSON. The policy is part of revision, request-cache and
report identities; configurations without it retain historical tool behavior.

Only `source_catalog` keys are actionable source selectors. Exact historical alias
bindings remain in the local projection audit. A known alias in a rejected draft
is diagnosed with its canonical target and any existing coverage; the model must
resolve conflicting decisions and resubmit a complete draft.

Recipe authoring and measurement execution share parameter and static syntax
checks. `calculate.expression` is one expression, without assignment or multiple
statements. `code.run.program` is numerical Python assigning `result`. Invalid
parameters, syntax, evidence references and selection paths are rejected before
execution, with field diagnostics. Such rejections consume no executed-tool slot;
the existing model-request, two-wave and eight-receipt-attempt limits still apply.
Runtime errors, missing results and corrupt supplied inventories remain failures.

Executed numerical receipts bind each input reference, exact selected JSON path
and selected-content SHA-256. Recovery must explicitly cite later successful
same-tool receipts covering those bindings. Changing a source's selected path or
content cannot establish recovery. A changed expression alone cannot waive a
real failed calculation. Original receipts remain intact. When no legal execution
slot or existing comparable successful receipt remains, recovery terminates with
`MEASUREMENT_RECOVERY_EXHAUSTED` instead of soliciting impossible corrections.
Resuming measurements reconstructs provenance from exact checkpoint-bound inputs
and applies the current validator before accepting historical success.

Workspace path patterns operate only on collected evidence. `*` matches one path
segment; `**` matches zero or more directory segments. `**/*.md` therefore includes
root-level Markdown files. The default `**` inspects all collected files. A valid
empty match is distinct from missing or corrupt inventory evidence.

Internal requests also carry `X-Role-Operation-Started`, never sent to the model.
The gateway enforces the original deadline while waiting for capacity as well as
during execution. Budget audits reconcile the terminal role budget, admission
ledger and gateway receipts one-to-one, checking physical identities, attempts,
recovery allowances and timestamps. Attempt counts remain separately reportable
when historical records cannot prove the complete contract; missing proof is not
reported as compliance.
