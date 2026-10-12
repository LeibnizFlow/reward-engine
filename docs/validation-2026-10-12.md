# Full-pipeline validation — 2026-10-12

**Incomplete. Neither benchmark has yet passed a new complete three-round run.**

The objective remains one Claw-eval task and one WorkBuddy task through the
Reward Engine Jury, Reward Builder, hypothesis generation, reward-guided
critique/revision, selection, and memory updates across three learning rounds.
Native benchmark grading is separate and cannot certify those modules.

The tasks remain `T111zh_expense_email_check` and
`api-usage-explain-cli-l3-001`. Each fresh run uses three new initial candidates
and A/B/C deliveries in each of three fixed rounds. No historical candidates or
scores are reused. The existing strict receipt auditor is unchanged.

## Repairs and validation so far

The old checkout was read as a reference, particularly
`experiments/w1_finish/provider.py`. Its explicit internal-role prompt, JSON
response format, and bounded operation protocol informed the new adapter.
Unbounded outer retries were not adopted.

Commit `2e2aaf0` fixes historical memory aliasing, supports the captured Qwen
duplicate empty termination/usage trailer, and enables explicitly configured
bounded role recovery. The driver freezes JSON response format, a 1,800-second
stage timeout, one syntax and one contract repair per stage, and two evidence
expansion rounds. Output limits are 65,536 tokens, or 131,072 for the reducer.
The limits are bound into the frozen configuration and checked before dispatch.
Providers may still produce reasoning tokens despite the disabled-thinking
request. The six-call role budget counts adapter-to-gateway requests; gateway
upstream retries have their own existing bounded policy and retained receipts.

The first full regression run had 2 failures and 2,465 passes. A compatibility
fix restored unanswered-inquiry feedback for legacy, unbounded callables.
The subsequent run passed **2,469 tests**, with no skips and nine dependency
warnings. It included real Docker/OpenCode integration with a scripted local
provider; that is controlled testing, not a live benchmark result. Source hashes
were unchanged throughout that successful run.

Five real provider sentinel probes returned the requested JSON. These establish
interface compatibility only. Their recorded source hashes identify an earlier
adapter snapshot; they do not certify the final source or semantic judgments.

## Diagnostic attempt v4

`runs/full_claw_20261012_v4` and `runs/full_workbuddy_20261012_v4` each generated
three initial candidates and entered Jury construction. Both were interrupted
with SIGINT after an independent source review reproduced three defects:

1. Preserved measurement verdicts could bypass full-source reads and replace an
   evidence-read-limit error.
2. A syntax retry could reopen tools during a structural repair.
3. A ranking repair could reverse an already-valid preference while repairing
   an unrelated observation field.

At interruption, both runs had **zero A/B/C final results and zero native grades**.
Their frozen code, model attempts, cancellation receipts and diagnostic-stop
records are retained. They are failed diagnostic attempts, not successful
validation. No quality outcome triggered the interruption.

Fixes now restrict preservation to evidence-complete verdicts, retain read
failures, keep structural repair restrictions across syntax retries, and bind
valid ranking partitions on unchanged evidence. The full regression suite then
passed **2,526 tests**, with no skips and nine dependency warnings; source hashes
were unchanged during execution. Independent follow-up closed the two
measurement findings but reproduced another ranking path: reattaching an
already-inline Task or artifact incorrectly unlocked its preference. That fix
now distinguishes previously unread previews from data already supplied in the
actual model request. Redundant Task, artifact, environment and inlined-source
reads retain the lock; genuine unread-source expansion still allows reassessment.
The final regression run passed **2,534 tests**, with no skips and nine dependency
warnings. Independent follow-up 2 closed all four recorded findings for the
tested hashes, provisionally and within that review scope. Both genuine
unread-source positive controls continue to permit reassessment. New native
attempts follow these checks; neither these fixes nor passing tests alone
establish pipeline success.

## Local evidence

- `runs/pipeline_validation_20261012/full_tests.json` and `.log`: first failed suite.
- `runs/pipeline_validation_20261012/full_tests_after_inquiry_fix.json` and `.log`:
  2,469-test passing suite and exact source hashes.
- `runs/pipeline_validation_20261012/source_integrity_review.md` and `.json`:
  independent findings and adversarial reproductions; same-family, provisional.
- `runs/pipeline_validation_20261012/auditor_probes.py` and
  `auditor_probe_results_frozen.json`: original scripted counterexamples.
- `runs/pipeline_validation_20261012/full_tests_after_integrity_fix.json` and
  `.log`: 2,526-test passing suite and exact source hashes.
- `runs/pipeline_validation_20261012/source_integrity_followup_1.md` and `.json`:
  measurement fixes accepted provisionally; redundant-read ranking bypass retained.
- `runs/pipeline_validation_20261012/full_tests_after_ranking_read_fix.json` and
  `.log`: 2,534-test passing suite and exact source hashes.
- `runs/pipeline_validation_20261012/source_integrity_followup_2.md` and `.json`:
  independent closure of the four recorded source findings; same-family/provisional.
- Each v4 run's `diagnostic_stop.json` and `validation_status.json`: reason,
  observed output counts, and actual termination status.

The earlier [2026-10-10 report](validation-2026-10-10.md) remains historical.
Successful native process exit, grader scores, and source-line execution will
not substitute for complete stage receipts or the final strict audit.
