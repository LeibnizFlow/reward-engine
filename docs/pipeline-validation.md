# Pipeline validation

A native actor smoke test does not validate Reward Engine construction or learning.
`experiments.validate_pipeline` runs the production task runners with three fresh
seeds, three fixed rounds, A/B/C treatments, full generated-check measurements,
and native benchmark grading. It freezes the source, configuration and benchmark
inputs before model calls. Historical benchmark outputs are not seed inputs.

For a small example without benchmark assets, start with the
[quickstart](quickstart.md). For these native runs, complete the
[native environment setup](native-smoke.md) first and run from the source
checkout. The `configs/public_delivery/` files are included in the source
export; paths below are relative to that checkout. Credentials are never copied
into the public report.

The same `--credentials`, `--key-env`, `--backend-url` and `--workbuddy-root`
options as the native smoke launcher are supported. See its
[provider configuration](native-smoke.md#run-fresh-tasks) for precedence.

```bash
.venv/bin/python -m experiments.validate_pipeline \
  --benchmark claw_eval --task-id T111zh_expense_email_check \
  --role-thinking disabled \
  --output runs/full_claw_validation

.venv/bin/python -m experiments.validate_pipeline \
  --benchmark workbuddy --task-id api-usage-explain-cli-l3-001 \
  --role-thinking disabled \
  --public-delivery-contract configs/public_delivery/api-usage-explain-cli-l3-001.v1.json \
  --output runs/full_workbuddy_validation

.venv/bin/python -m experiments.pipeline_audit runs/full_claw_validation
.venv/bin/python -m experiments.pipeline_audit runs/full_workbuddy_validation
```

The WorkBuddy example explicitly supplies a public delivery contract. Its required
root `final_answer.md` path is an executor translation of the quoted public task
instructions, bound to the exact prompt hash. This is not automatic extraction of
all natural-language requirements. The runtime supports other explicitly supplied
safe relative file paths; matching filenames in other directories are not aliases.
Claw has no workspace-path contract in this validation.

This opt-in prerequisite runs in every B review and final measurement, outside the
semantic targeted-check quota. Missing files fail when collection establishes
absence; incomplete collection remains unknown. A selected delivery must satisfy
the prerequisite. File presence alone makes no content-quality claim. Raw model
Jury/Pack records remain in `jury_semantic.json` and `pack_semantic.json`; effective
records explicitly identify the added executor prerequisite.

After WorkBuddy terminates, run its additional public-contract audit using the
frozen implementation. It reconstructs semantic Jury and Pack contents from
retained provider responses and declared repairs or reuse, then checks the public
projection, actual snapshot bytes, reviews and selection. Reconstruction makes no
provider or measurement-tool calls. Both this audit and the original pipeline
audit must pass:

```bash
(cd runs/full_workbuddy_validation/code && \
  ../../../.venv/bin/python -m reward_engine.adapters.workbuddy.public_delivery_audit ..)
```

The examples explicitly request disabled thinking for structured roles only. Providers
may still report reasoning tokens; inspect retained usage rather than assuming the
request was honored. Native actor and grader parameters are unchanged by this flag.

Use fresh output directories. For an explicitly inspected recoverable execution
failure, the frozen driver accepts `--execute --output ABSOLUTE_RUN_DIRECTORY`
when invoked from that run's `code` directory. Preserve the failed attempt and
record the operational correction before resuming; a failed artifact must not be
replaced because of its quality score.

## What constitutes evidence

The execution driver deliberately records `stage_audit: pending`, even after a
zero exit. The separate deterministic auditor checks the native snapshots and
session logs, Jury maps and criteria, generated packs, actual check results,
hypothesis proposal/application, same-session feedback and revision, persistent
memory/checkpoints, selected candidate, and all nine native grades. It preserves
raw scores; it does not normalize them against the run's own outputs.

The version-two auditor separates receipt validity from observed branch coverage.
It requires valid B reviews in all three rounds and at least one bound, changed-artifact
revision for the revision-coverage claim. A valid decision that no revision is needed
is review evidence, but does not exercise the revision branch. An unavailable role,
checker execution error, missing requested continuation, or incomplete round cannot
silently become a passing stage. The engine itself permits a two-member Jury quorum; the
strict validation audit additionally records failed panel seats rather than
calling such a construction a clean three-member result.

A checker may correctly find a product defect. That does not mean the checker
failed to execute, and a valid native score need not be perfect. Conversely, a
high native score does not prove that any Reward Engine stage ran. Official
benchmark grading remains separate from Jury construction and learning inputs.

`experiments.audit_deliveries --output RUN_DIRECTORY` can create separately
labelled full measurements of exact delivered snapshots after an execution
stops. This preserves original measurements, learning, selection, artifacts and
grades. Supplements bind their source files, native candidate, effective pack,
code and complete measurements. They cannot establish that the original loop
used a later supplemental measurement.

## Coverage boundaries

The three-round runs exercise the default legacy A/B/C pipeline. Other paths
require separate evidence: adversarial counterexample construction (which allows
one outer round), runtime policy critic D, public fact/reward compilation, task
research, task updates, independent pool selection, and regression-safe version
selection. Controlled tests cover these modules; that is not a claim of live
native integration for every optional combination. Native state rollback is not
provided by retaining an earlier immutable artifact.

Likewise, these two tasks cannot establish all benchmark tasks, document types,
model providers, or GPU execution. Real generated XLSX, DOCX, PPTX and PDF files
are tested separately through the document parsers. Mocked HTTP/model fixtures
are identified as controlled tests, including the offline Docker/OpenCode test.

## Current validation artifacts

The paired 2026-10-12 native validation is still incomplete. Claw v11 passed its
strict three-round audit on frozen commit `33d39e0`; WorkBuddy has no clean full
strict pass. Later attempts on `bf85cdd` stopped with provider failures, including
a separately confirmed quota blocker. That source passed 3,035 controlled tests;
neither these tests nor the earlier Claw success certify later source changes.
No native run is currently active. See the
[current evidence and failure history](validation-2026-10-12.md).

## Historical validation artifacts

The 2026-10-10 validation ended with both strict pipeline audits failing.
Claw completed two rounds before a third-round Jury failure; WorkBuddy completed
three rounds. There are 15 validated native-grade receipts, but no clean full-pipeline
pass. See [the final results and module limits](validation-2026-10-10.md).
Local evidence lives under `runs/pipeline_validation_20261010/` and the fresh
`runs/full_{claw,workbuddy}_20261010_v3/` directories. Earlier `v1` and `v2` directories are
retained diagnostic attempts, stopped after code/protocol defects were confirmed
and before any A/B/C actor result or native quality score was available.

The final source revision bound to `payload_tests.json` passed 2,364 tests with no skips or xfails,
including the real Docker/OpenCode integration against a scripted local provider.
Source hashes remained unchanged during the suite. The latest native-payload validation fixes
are included in this suite. An earlier Node log records 29 passes, but lacks independent source binding for
that invocation. Controlled-test statement coverage
is 88.81% for core code and 85.16% for adapters (branch coverage: 81.68% and 78.06%).
All 62 Python implementation modules with function bodies have some body execution;
this is not complete function/branch coverage or a correctness guarantee.

An offline replay of the saved provider streams matched 194 successful structured
Role replies against the current parser, with zero mismatches. The report separately
retains failures, unavailable consumer-stream reconstruction and in-progress files.
This is parser replay, not another live model experiment.

For the final result, inspect the stage audit, raw receipts and module inventory
rather than inferring success from a process exit or this document's test counts.
