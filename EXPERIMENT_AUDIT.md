**Final actual-evidence review — 2026-10-10 (Asia/Shanghai)**

**Overall: WARN. Qualified failure/count/score reporting: PASS. Clean pipeline validation: FAIL.**
The final documents honestly report the failed validation; they do not establish that every Reward Engine module works. Reviewer: inherited Codex model, no explicit override; **same-family, provisional**. This is a direct artifact review, reusing the closed 004-C source findings.

| Check | Verdict | Finding |
|---|---|---|
| A — Ground truth | PASS | Authored benchmark fixtures/gold/rubrics; Jury-generated criteria remain proxy signals. No model-output-derived gold found in checked paths. |
| B — Normalization | PASS | Fixed native formulas and raw values; no prediction-derived denominator. All 15 scores reconcile. |
| C — Existence/counts | WARN | Final counts/scores are real and correct. Round-one Claw aggregate memory and historical Node source binding need the qualifications below. |
| D — Executed code | WARN | Actual stage receipts exist; only 36/62 implementation modules have live function-body evidence per task. Execution does not certify correctness. |
| E — Scope/binding | WARN | Two tasks, older frozen live source, newer controlled tests; optional paths and general quality remain unestablished. |
| F — Evaluation types | PASS | Native: real_gt in simulated environments with model judgments. Jury: synthetic_proxy. Controlled integration/replay: simulation_only/compatibility. No human evaluation. |

A/B evidence: [Claw authored rubric](/data0/jiqi/reward-engine/runs/full_claw_20261010_v3/benchmark/tasks/T111zh_expense_email_check/grader.py:23), [Claw fixed formula](/data0/jiqi/reward-engine/runs/full_claw_20261010_v3/benchmark/src/claw_eval/models/scoring.py:11), [WorkBuddy authored gold loading](/data0/jiqi/reward-engine/runs/full_workbuddy_20261010_v3/dataset/wb-bench-office-v1.0/tasks/api-usage-explain-cli-l3-001/tests/grading/eval_core.py:85), [WorkBuddy fixed merge](/data0/jiqi/reward-engine/runs/full_workbuddy_20261010_v3/dataset/wb-bench-office-v1.0/tasks/api-usage-explain-cli-l3-001/tests/judge.yaml:242). Claw uses a custom native judge route, not certified leaderboard-equivalent grading. WorkBuddy raw rule checks number 61 per output; fractional rule failures remain valid product grades.

C evidence: [final claims](/data0/jiqi/reward-engine/docs/validation-2026-10-10.md:76), both [Claw](/data0/jiqi/reward-engine/runs/full_claw_20261010_v3/validation_status.json:2) / [WorkBuddy](/data0/jiqi/reward-engine/runs/full_workbuddy_20261010_v3/validation_status.json:2) terminal statuses, and the hashed native receipts listed in the companion JSON.

| Task | Round | A | B | C | Selection |
|---|---:|---:|---:|---:|---|
| claw | 1 | 0.916 | 0.918 | 0.868 | B |
| claw | 2 | 0.892 | 0.944 | 0.918 | B |
| workbuddy | 1 | 0.9213 | 0.9213 | 0.9344 | A (fallback) |
| workbuddy | 2 | 0.9344 | 0.9475 | 0.9344 | B |
| workbuddy | 3 | 0.9082 | 0.9344 | 0.9213 | A |

Claw round three produced **no A/B/C outputs or grades**: only one usable Jury proposal, then `jury_unavailable`. WorkBuddy reached `round_limit`, completed three rounds and retained A after a valid final ranking; its final pool and memory each contain 12 entries. There are **15 graded deliveries, not 18**, plus six separately counted fresh seeds. See [Claw failure](/data0/jiqi/reward-engine/runs/full_claw_20261010_v3/tasks/T111zh_expense_email_check/round_3/construction_failure.json:1) and [WorkBuddy stop](/data0/jiqi/reward-engine/runs/full_workbuddy_20261010_v3/tasks/api-usage-explain-cli-l3-001/round_3/loop_decision.json:1).

Both [Claw terminal audit](/data0/jiqi/reward-engine/runs/pipeline_validation_20261010/claw_v3_terminal_audit.json:1) and [WorkBuddy terminal audit](/data0/jiqi/reward-engine/runs/pipeline_validation_20261010/workbuddy_v3_terminal_audit.json:1) fail. All 15 native-grade checks pass in those audits; independent raw-record/arithmetic checks agree. This does not validate the surrounding pipeline.

All 15 full delivered-pack measurements exist: seven checks per Claw delivery and ten per WorkBuddy delivery (132 result entries). **Zero complete receipts meet every strict evidence contract.** WorkBuddy round-three measurements can report no execution error while `inquiry_result` asserts a verdict without required evidence. See [actual round-three inquiry](/data0/jiqi/reward-engine/runs/full_workbuddy_20261010_v3/tasks/api-usage-explain-cli-l3-001/round_3/A/outcome_estimate.json:63) and [accurate execution/validation distinction](/data0/jiqi/reward-engine/docs/validation-2026-10-10.md:97).

Native sessions support two B continuations in Claw rounds 1/2 and WorkBuddy rounds 2/3. WorkBuddy round-one review/ranking were unavailable; round-three B retained a checker JSON error. Missing later-added review-input/prior-state/inquiry provenance prevents certification of historical targeted reviews. A valid stop or unchanged selection is not missing coverage; unavailable evidence still cannot pass.

D/E evidence: [live inventory totals](/data0/jiqi/reward-engine/runs/pipeline_validation_20261010/live_module_summary.json:346) independently recount to **36/62 modules with function bodies** per frozen run (70 Python files including eight without named bodies): Claw **3,009/9,452** statement lines, WorkBuddy **3,168/9,452**. Imports/helper hits do not activate every optional feature. Coverage export paths and frozen hashes match; shard collection remains [explicitly operator-attested](/data0/jiqi/reward-engine/runs/pipeline_validation_20261010/live_module_summary.json:20).

The current-source suite terminal log supports **2,364 passed, zero skips/xfails**, with 266 matching source hashes. Controlled coverage recount matches core **6,859/7,723 statements; 3,014/3,690 branches**, adapters **1,504/1,766; 484/620**, experiments **2,517/4,866; 689/1,744**. Some controlled execution in all 62 modules is not every-function correctness. The Docker/OpenCode provider is scripted. See [bound test summary](/data0/jiqi/reward-engine/runs/pipeline_validation_20261010/payload_test_summary.json:1) and [controlled-test scope](/data0/jiqi/reward-engine/docs/validation-2026-10-10.md:45).

F and compatibility evidence: [parser replay limits](/data0/jiqi/reward-engine/docs/validation-2026-10-10.md:63) matches 194 saved successful replies, with six failed and two still in progress at capture. Later completion is not retroactively replay evidence. No new provider calls or grading occurred in this review.

Evidence qualifications retained:

- Claw final aggregate `rounds[0].arms.B.memory` contains ten entries; the [saved round-one receipt](/data0/jiqi/reward-engine/runs/full_claw_20261010_v3/tasks/T111zh_expense_email_check/round_1/summary.private.json:342) contains four. Those four are an exact prefix of the [final aggregate](/data0/jiqi/reward-engine/runs/full_claw_20261010_v3/tasks/T111zh_expense_email_check/result.private.json:347). Use round-local receipts for review-time state; scores/counts are unaffected. This remains an evidence limitation, not a reopened broad source audit.
- The 29 earlier Node tests have a passing log, but no independent before/after source witness for that historical invocation. Environment metadata is an earlier capture with versions rechecked. Neither is full process attestation.
- Frozen manifests checked: 748 Claw entries and 792 WorkBuddy entries, all matched; 678 original WorkBuddy dataset hashes matched. Current source differs from live snapshots. The 2,364-test result cannot retroactively validate repaired code in those live executions.
- Earlier v1/v2 stopped attempts remain with zero A/B/C results/grades; Claw v3 failure and all prior source-review FAIL history remain. No quality-culling evidence was found in retained files; local hashes cannot prove historical non-deletion.
- No general benchmark, causal improvement, every-module, or optional-mode success claim is supported. Known 004-C source findings remain closed at unchanged hashes.

Documentation-only rebind verified: the [results document](/data0/jiqi/reward-engine/docs/validation-2026-10-10.md:49) and [pipeline guide](/data0/jiqi/reward-engine/docs/pipeline-validation.md:97) now qualify the historical Node source binding. The [results document explicitly retains the unresolved four-versus-ten Claw memory discrepancy](/data0/jiqi/reward-engine/docs/validation-2026-10-10.md:174) and directs readers to round-local receipts. WARN and the open memory qualification remain unchanged; no source or execution investigation was repeated.

Bounded verification used direct JSON/stream reads, SHA-256, arithmetic and AST/coverage recounts; no tests, experiments, graders, auditor calls or network calls were rerun. `/tmp/reward_final_terminal_check.py` completed with zero unexpected mismatches and the aggregate-memory difference explicitly retained. No source/test/live files were changed.

Exact source/report/receipt SHA-256 values and claim limits are in [the compact JSON review](/data0/jiqi/reward-engine/runs/pipeline_validation_20261010/final_actual_review.json:1). The prior source-review hash remains `8660282afbdff4824509658df97261f21ce7005f4e9be8fceaccf042aae52dce`. This verdict supports the qualified failed-validation account; it does not certify a clean pipeline pass.

The three durable test-log copies match their original `/tmp` files byte for byte; their manifest and hashes are included in the JSON review.
