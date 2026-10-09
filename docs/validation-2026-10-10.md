# Reward Engine validation, 2026-10-10

**Result: both strict pipeline audits FAIL. WorkBuddy completed three rounds; Claw stopped at third-round Jury construction after completing two.**

The earlier actor/native-grader smoke tests did not test the Reward Engine loop.
This validation runs `T111zh_expense_email_check` (Claw) and
`api-usage-explain-cli-l3-001` (WorkBuddy) through the production runners, with
three fresh seeds each, three fixed outer rounds, legacy A/B/C, generated-pack
measurements on each delivered artifact, and separate native benchmark grades.
See [the reproduction and evidence rules](pipeline-validation.md).

## Source findings and repairs

| Finding | Repair and verification |
|---|---|
| Claw's task runner hard-coded one round and the native runners did not forward full-audit mode. | Forward configured rounds/full audit; production-runner argument tests. |
| A targeted actor review could be reused as though it were a full delivered-pack audit. | Full audit remeasures the actual delivery using every generated check; tests verify changed deliveries and partial actor receipts. |
| An intentional absent B treatment could be counted as a failed arm. | Use the controller's expected arms; tests retain failure for missing outputs and unsuccessful samplers. |
| Context compression could rewrite canonical artifact fields named `summary`, `preview`, or `rationale`. | Restrict compression to explicit navigation material; preserve exact canonical evidence and source access. |
| Regression handling did not export immutable delivery versions to selection. | Export version history and retention decisions; reject missing/tampered histories. Native terminal candidates and grades are not relabelled or rolled back. |
| PPTX table and grouped-shape text was omitted. | Extract both; test real XLSX, DOCX, PPTX and PDF bytes, and declare the parser dependencies. |
| Structured reply handling had stream-choice, terminal-boundary and duplicate-key ambiguities. | Reject ambiguous streams and duplicate JSON keys; normalize only request-advertised native measurement tool calls. Controlled malformed-stream tests and retained-stream replay cover the changes. |
| Provider SSE error events were indistinguishable from generic checker exceptions. | Preserve typed provider errors and exact captured events; apply the existing bounded retry budget to narrowly identified transient events. Exhausted failures remain unknown/infrastructure errors. |
| Measurements lacked a binding to their exact input; the new audit initially trusted candidate metadata without checking retained bytes. | Record an input fingerprint at execution and verify snapshot inventory, hashes, sizes, containment and candidate identity. Tests reject substituted measurements and changed/deleted native files. |
| An unsolicited scalar `inquiry_result` could crash review bookkeeping. | Preserve it explicitly as a nonauthoritative diagnostic; validate requested inquiry verdicts strictly. A real-controller regression confirms history is recorded and processing completes. This repair postdates the frozen v3 runs. |
| Embedded server signal handlers could prevent the validation driver from stopping. | Leave SIGINT ownership to the outer runner; subprocess tests use real servers and one interrupt. |
| The new auditor incorrectly required Builder anchors on raw Jury criteria and a positive revision in every round. | Validate stage-specific contracts. Require three valid reviews and separately track at least one actual changed-artifact revision; a valid no-change review does not count as revision coverage. An independent read-only scope review preceded actor outcomes. |

These are implementation repairs, not retroactive changes to historical model
answers, native grades, or the code frozen into running experiments.

A fresh read-only reviewer reproduced additional audit and native-score validation
defects. Targeted B reviews now bind their overview inputs, prior delivery state,
inquiries and executed-check plan. Measurement validation checks batch/result/count
consistency and verifies reuse against actual dependencies and prior passing results.
Native WorkBuddy validation checks the authored judge/item roster, verdict schemas,
raw payload scorability, derived counts, statuses and component/aggregate arithmetic.
Genuine zero and fractional product scores remain valid; unavailable or contradictory
grading fails validation. Independent bounded rechecks closed the reproduced cases.
This does not prove rejection of every possible malformed input. New provenance
fields cannot be backfilled into historical live reviews.

## Controlled validation

The final source revision bound to `payload_tests.json` passed **2,364 tests**, with **zero skips or xfails**.
The enabled Docker/OpenCode integration uses a scripted local provider; it is not
a live-model quality experiment. Source hashes were stable before and after the
suite. Runtime package versions, native fixture paths/image, source hashes and the
exact command are retained. Ruff and diff checks passed. An earlier Node log records 29 passing tests, but that invocation lacks independent
before/after source binding; it is separate from the current 2,364-test evidence.

| Scope | Statements executed | Branch exits executed |
|---|---:|---:|
| Core | 6,859/7,723 (88.81%) | 3,014/3,690 (81.68%) |
| Adapters | 1,504/1,766 (85.16%) | 484/620 (78.06%) |
| Experimental tools/runners | 2,517/4,866 (51.73%) | 689/1,744 (39.51%) |

All 62 implementation modules with function bodies have some body execution in
these controlled tests. This does not mean every function, branch or optional
configuration was tested. Experimental tools include historical, analysis and
scheduling utilities beyond the two default task runners.

An offline parser replay matched all **194 saved successful Role replies**, with
zero mismatches. It separately records six saved failed requests, two requests
still in progress at capture, rejected failure streams and unavailable consumer
stream reconstruction. No new provider call, regrading or original-run mutation
was performed. This validates parser compatibility, not new model performance.

## Final live results

Current roots:

- `runs/full_claw_20261010_v3/`
- `runs/full_workbuddy_20261010_v3/`

Claw completed two rounds and stopped during round-three Jury construction.
All three expectation maps succeeded, but only one of three outcome proposals was
usable: GLM exhausted its 32,768-token output limit and Qwen failed the structured
stream completion contract. The runner recorded `jury_unavailable`, retained the
recovery state, and exited with failure. There are six Claw A/B/C outputs and six
native grades; no third-round Claw outputs or grades exist.

WorkBuddy completed all three rounds in 7,792 seconds (about 130 minutes), with
nine A/B/C outputs and nine native grades. Claw's attempt took 4,892 seconds (about
82 minutes). Each task also produced three fresh seed outputs, separately from the
15 graded A/B/C deliveries. All 15 native-grade receipts pass the current strict
native-grade validator without altering any score.

WorkBuddy B's first-round review was unavailable: the provider spent 32,768 tokens
on reasoning and returned no answer. First-round ranking was also unavailable,
so selection retained A. In rounds 2 and 3 B made two same-session revisions and
exhausted its revision budget with unresolved suggestions. Its third-round final
review also retained a checker JSON error. Rounds 2 and 3 produced valid rankings;
selection chose B in round 2 and retained A in round 3. Final memory/checkpoint and
three-round completion receipts pass their structural checks.

**Both terminal audits fail.** Degraded Jury/hypothesis panels, unavailable checks,
malformed or unsupported inquiry verdicts, and missing targeted-review provenance
remain in the original receipts. All completed rounds produced full delivered-pack
measurements (seven checks per Claw output, ten per WorkBuddy output), but none of
those 15 complete measurement receipts passes every strict evidence requirement.
For example, third-round WorkBuddy checks contain known inquiry verdicts without
the supporting evidence required by their contract, even though the enclosing
checks report no execution error. The audit's validated-measurement count of zero
means zero certified receipts, not zero executions.

The frozen runs lack targeted-review provenance added later. Raw native sessions
show the revisions, but the new audit cannot certify those historical reviews.
A permissible two-member Jury quorum also does not establish a clean three-member
panel. Successful actor completion or a high native grade cannot override these
failures. Original failures, scores and source snapshots remain unchanged; there
was no quality-based rerun or post-hoc provenance backfill.

Live coverage is separately bound to each frozen tree, with no path remapping onto
newer source. Each task has function-body evidence in **36 of 62 implementation
modules**. Claw recorded 3,009 of 9,452 statement lines; WorkBuddy recorded 3,168
of 9,452. The remaining modules are not established as live-tested by these runs.
Import execution, a helper-function hit and controlled-test coverage are separate
from activation or correctness of an optional feature.


## Raw native grades

These are the benchmark graders' recorded outputs, separate from Reward Engine
Jury scores. No rescaling, cross-benchmark averaging or causal improvement claim
is made. A = baseline, B = reward-guided review/revision, C = hypothesis treatment.

| Task | Round | A | B | C | Reward Engine selection |
|---|---:|---:|---:|---:|---|
| Claw | 1 | 0.916 | 0.918 | 0.868 | B |
| Claw | 2 | 0.892 | 0.944 | 0.918 | B |
| Claw | 3 | — | — | — | Not reached: Jury unavailable |
| WorkBuddy | 1 | 0.9213 | 0.9213 | 0.9344 | A fallback; ranking unavailable |
| WorkBuddy | 2 | 0.9344 | 0.9475 | 0.9344 | B |
| WorkBuddy | 3 | 0.9082 | 0.9344 | 0.9213 | A retained after valid ranking |

## Module-level interpretation

| Module | Observed live behavior | Validation limit |
|---|---|---|
| Native actors and evidence capture | Three fresh seeds per task; 6 Claw and 9 WorkBuddy A/B/C outputs | Two tasks do not cover every native format or environment. |
| Jury construction | Executed repeatedly; Claw round three stopped at 1/3 usable proposals | Failed seats remain failures even when the 2-member quorum permits continuation. |
| Reward Builder | Generated 7 Claw checks and 10 WorkBuddy checks per completed construction | Pack generation is separate from successful execution of every generated check. |
| Generated measurements | Construction and actual delivered-artifact measurements ran | Original receipts include checker errors and malformed inquiry metadata; strict audits fail. |
| Hypothesis generation/application | Proposals and C treatment executed in all five completed rounds | Failed panel seats remain visible; no hypothesis-quality or causal benefit claim. |
| Reward-guided critique/revision | Claw B rounds 1/2 and WorkBuddy B rounds 2/3 each made two same-session revisions | WorkBuddy B round 1 unavailable; round 3 retains a checker error. Unresolved improvements and missing later-added provenance prevent a clean review claim. |
| Candidate selection | Claw rounds 1/2 and WorkBuddy round 2 selected B; WorkBuddy rounds 1/3 retained A | The first WorkBuddy fallback is not a successful Jury comparison. |
| Memory/checkpoint continuity | Claw retained two rounds of state; WorkBuddy completed three checkpoints and the fixed-round stopping decision | Claw's failed third construction does not establish a completed three-round loop. |
| Native benchmark grading | Six Claw and nine WorkBuddy native grades retained and validated | Native grading does not validate Jury, learning or overall pipeline success. |
| Optional modes | Selected contracts exercised by controlled tests | No live claim for every optional mode, function, branch or provider. |

## Provenance and scope

`v1` and `v2` are retained diagnostic attempts, not successful runs. They were
stopped before any A/B/C actor result or native quality score became available;
source/protocol corrections and interruption receipts are retained. No artifact
or run was replaced because of its native quality score.

The live v3 processes ran their original frozen source. Several fixes—including
provider-error classification, unsolicited inquiry handling, targeted-review input
provenance, and stricter native-grade validation—were made afterward. The final
2,364-test suite covers current source; it is not a full live rerun of that newer
source. Historical receipts and grades were neither rewritten nor backfilled.

Local evidence is under `runs/pipeline_validation_20261010/`, including
`payload_test_summary.json`, the test-command/source-hash and environment manifests,
`payload_test_coverage.json`, `payload_module_test_inventory.json`, `parser_replay.json`,
`live_module_inventory.json`, `live_module_summary.json`, `pipeline_facts.json`,
`claw_v3_terminal_audit.json`, `workbuddy_v3_terminal_audit.json` and
`terminal_audit_manifest.json`. Large run data stays in the local workspace.
Read-only reviewer traces live under `.aris/traces/experiment-audit/`; reviews are
same-family and provisional, not independent cross-family certification.

The final read-only review also found an unresolved historical memory discrepancy:
Claw's saved round-one B receipt contains four memory entries, while that round's
entry in the final aggregate contains ten (the original four remain an exact prefix).
Use the round-local receipt for review-time state. Scores and delivery counts agree,
but the aggregate must not be treated as an immutable snapshot of that earlier round.
No repair or clean validation of this aggregate-history behavior is claimed here.

The live protocol does not activate every optional mode: counterexample-driven
construction, runtime critic D, public reward compilation, task research/updates,
independent pool selection, isolated CUDA measurements and regression-safe version
selection need separate evidence. Controlled tests exercise selected contracts;
they do not establish live integration of every optional combination. Two tasks
also cannot establish general benchmark quality or a causal improvement claim.
