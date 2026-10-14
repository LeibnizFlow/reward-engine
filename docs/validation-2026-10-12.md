# Full-pipeline validation — 2026-10-12

**Paired native validation remains incomplete.** Claw v11 passed its strict
three-round audit on frozen commit `33d39e0`; WorkBuddy has no clean complete
strict pass. Subsequent attempts on `bf85cdd` stopped with provider failures,
including a separately confirmed quota-reservation rejection. No native driver
remains active. The chronology below preserves earlier failures and repairs;
the successful Claw run does not validate later source changes or establish
quality improvement.

An earlier diagnostic pair (Claw v7 / WorkBuddy v6) stopped after confirmed
execution failures. Adapter and bounded field-correction repairs have passed
2,697 unit/integration tests and bounded independent review. Fresh Claw v8 and
WorkBuddy v7 runs started on commit `1d6e90c`. Claw v8 subsequently failed its
first B review and was stopped. Claw v9 on `b382448` and WorkBuddy v7 on
its original `1d6e90c` also ended after one completed round: Claw failed ranking,
and WorkBuddy failed second-round Jury construction. No native run is currently
active at that point and no attempt had yet completed successfully. Later
attempts and the Claw v11 success are recorded below.

Claw v7 subsequently recorded a first-round Jury adapter failure: GLM's ordinary
JSON reply included empty `tool_calls: []` placeholders, which were mistaken
for a native tool request despite `finish_reason=stop`. Only two of three
expectation members completed, so this attempt cannot pass the strict audit.
Its original stream and failure are retained. Both drivers are now stopped.

Both new attempts later recorded correctly rejected claim-changing repairs in
construction measurements. Claw v7 tried promoting all advisory findings to
material while retaining pass, and rewrote claims while adding an omitted
root-level citation list. WorkBuddy v6 corrected a conflicting pass to unknown
but also rewrote an existing claim and observation. These remain execution
failures; neither attempt is a clean success. The runtime preservation guard is
not relaxed. A bounded field-correction reply is implemented so models can
submit explicit repairs without regenerating immutable analysis. Raw replies,
composed results and field provenance must remain distinguishable; all existing
validation and retry budgets still apply.

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

## Native attempt v5

Both fresh runs use code matching the final 2,534-test snapshot and the closed
source-review findings. WorkBuddy is running and has completed its three seeds.
Claw exited before producing a seed because `/data0` fell just below its existing
20 GiB free-space admission floor. This is an infrastructure failure, not a
benchmark verdict. Its failure receipt is retained.

The passive `data/claw_eval/transport` archive was relocated to
`/var/tmp/reward-engine-storage/jiqi/claw-transport-20261012` on another volume.
All 111,392 regular files (4,684,211,782 bytes) were compared by SHA-256 before
replacing the original path with a symlink and removing the verified duplicate.
Historical evidence is
retained; neither old run results nor the storage admission floor are changed.
The reserve was restored to over 24 GiB. `runs/full_claw_20261012_v6` is now
running alongside the unchanged WorkBuddy v5 run. Neither is yet certified.

## Claw v6 construction failure and repair

Both tasks completed first-round Jury expectations, Jury proposals and hypothesis
panels (three of three each), and generated their Reward Builder packs. WorkBuddy
has nine checks per seed and no execution errors in its 27 construction checks.
Claw has eight checks per seed; one of its 24 construction checks failed to
execute correctly. The original model twice returned a `contradicted` claim with
empty `claims[7].evidence_refs`. The generic recovery message did not identify
that claim. This retained error prevents Claw v6 from passing the strict audit;
its remaining execution is diagnostic evidence, not a clean successful run.

The harness now identifies the offending claim index, field, judgment and empty
value in its recovery diagnostics. Its prompt also states the existing nonempty
evidence requirement explicitly. Acceptance rules, repair budgets and verdicts
are unchanged; the harness does not invent citations or fill them from claim
locations. Portable tests retain the captured eight-claim response, exercise the
actual Role transport with scripted replies, and reject repeated or invented
citations. The full suite passed **2,541 tests**, with no skips, nine dependency
warnings and no source changes during execution.

A separate prospective live measurement on the original seed/check completed
successfully with two model requests and full-source evidence expansion. The
input fingerprint matches the historical candidate. This fresh response did not
repeat the missing-reference error, so it demonstrates successful execution on
that input, not live coverage of the indexed structural-repair branch. Its
receipt is stored separately and was never used by the original learning loop.
Neither that measurement nor controlled tests erase Claw v6's failure.

At the retained progress snapshot, both first-round B actors had completed their
initial delivery plus two same-session revisions. Both final delivery reviews
still requested improvement within the exhausted revision budget; that outcome
is distinct from a measurement execution error. C applied the generated
hypothesis in both tasks. WorkBuddy's raw first-round native scores were
**A: 0.9475, B: 0.9344, C: 0.9475**. These scores do not certify Reward Engine
correctness or establish an improvement from revision. Candidate selection and
the complete three-round loop were still pending.

WorkBuddy subsequently recorded an execution error in A's full delivered-pack
measurement, `caveats_unsupported_ambiguous_check`: after reading the full
artifact, the model returned `status=pass` while retaining an
`unverified`/`material` claim. Its repair repeated that contradiction. The
unchanged strict audit rejects this original measurement, so WorkBuddy v5 also
cannot count as a clean full-pipeline success. The native scores above remain
raw benchmark results, not a substitute for that failed Reward Engine check.
The original response ledger is retained while a further diagnostic repair is
developed; the 2,541-test result applies only to its recorded earlier hashes.

Both diagnostic attempts subsequently completed round one and entered round two.
Reviewing all intermediate versions, rather than only the final B measurement,
also exposed a Claw B version-one check with all 12 claim-level `evidence_refs`
fields omitted. Its generic repair repeated the omission. A further WorkBuddy
B full-pack check repeated the same material-claim/pass contradiction as A.
The after-round-one strict audits retain these failures. Claw's three full
delivered-pack measurements did execute without errors, and its native scores
were **A: 0.868, B: 0.904, C: 0.882**; neither fact cancels its earlier errors.
Claw's selector chose B. These are diagnostic execution observations, not
successful benchmark-pipeline validation or evidence of general quality gains.

The next indexed status-dependency diagnostic snapshot passed **2,549 tests**
(no skips, nine dependency warnings, unchanged source hashes). That result is
bound to `full_tests_after_claim_status_diagnostic.json`; a narrow copy of its
harness and new tests is retained in `claim_status_tested_snapshot/`. It is an
intermediate snapshot: the omitted-reference case requires reporting multiple
simultaneous claim errors, and same-evidence repairs are being checked for
semantic drift in otherwise valid claim assessments before new native runs.

Independent review then reproduced **CS-1** through the actual Role/harness:
same-evidence structural repair accepted a pass after upgrading an unverified
material claim, relabeling it advisory, or deleting it. The issue also exists in
the preceding diagnostic snapshot; prompt wording alone does not enforce
preservation. The failing review is retained in
`source_integrity_claim_status_review.md/.json`. Runtime preservation of valid
claim assessments and aggregated claim-error reporting are now being repaired.

Claw v6 and WorkBuddy v5 were interrupted with SIGINT after that confirmation.
Each had completed one round, three A/B/C deliveries and three native grades,
and had entered round two. Each `diagnostic_stop.json` records the reason and
hashes of completed measurements and review histories. These are failed
diagnostic attempts, never clean pipeline successes. The interruption and need
for fresh runs follow code integrity and execution errors, not native quality
scores. Earlier receipts remain unchanged.

The latest patch collects all claim-field errors and summary dependencies in
the single existing repair request. It compares each repaired claim with the
original, permitting changes only to invalid fields and an inconsistent
summary. It preserves valid assessments through syntax recovery; only newly
attached, previously unread preview evidence reopens assessment. The model
still authors every semantic verdict. The focused suite passed **425 tests**,
including 29 new cases for the 12 omitted fields, the CS-1 counterexamples,
conservative summaries and genuine evidence expansion. The full suite then
passed **2,578 tests**, with no skips, nine dependency warnings and unchanged
source hashes. Independent follow-up closed CS-1 for the reviewed hashes:
79 focused tests, 90 acceptance comparisons, the original attacks, aggregate
diagnostics and the earlier SI-1/2 controls passed within that review's scope.
This is same-family, provisional source review, not a native benchmark verdict.

## Fresh pair on commit 3c77aa5

`runs/full_claw_20261012_v7` and `runs/full_workbuddy_20261012_v6` ran the
same tasks from fresh seeds. Their frozen runtime files match the 2,578-test
snapshot (72 package/driver files verified in each). Three rounds, full delivered
pack measurements, the role limits and `quality_resampling=false` are unchanged.
No stopped run's candidates, grades or measurements were reused. Both drivers
were interrupted after the execution failures above. Claw completed one round;
WorkBuddy completed no full round, despite producing three A/B/C outputs and
three native grades. Both have terminal cancellation receipts and
`diagnostic_stop.json` records. Retained receipt hashes remain unchanged.

## Adapter and bounded field-correction validation

The adapter distinguishes empty streaming tool placeholders from actual native
tool records, retaining raw chunks and rejecting malformed or unsupported tool
payloads. Its author suite passed 310 focused tests. An independent bounded
review passed 287 focused tests and 55 actual-Role controls, including replay of
the captured GLM failure. This is source/protocol evidence, not a native pass.

Field corrections allow only explicitly advertised invalid fields. Models must
author all correction values, including the complete summary when required.
The host composes those values with the original valid fields, records per-field
provenance, and applies the existing validators, claim preservation and evidence
read rules. Retry and tool budgets are unchanged. Full-response repairs that
rewrite valid claims remain rejected.

The first field-correction snapshot passed 2,694 tests. Independent review of
that exact snapshot passed 193 focused tests, 44 actual-Role controls and 80
baseline validator comparisons. The subsequent guidance-only change exposes
the existing permitted summary statuses when locked claims invalidate pass.
The current snapshot passed **2,697 tests**, with nine dependency warnings and
no source changes during execution. These tests include real Docker/OpenCode
with a scripted provider; they are not real benchmark pipeline runs.

Separate hybrid fault-injection probes supplied one retained invalid response
and called the real configured model for its repair. Missing root citations and
the material-unverified contradiction repaired successfully. The first
all-advisory probe retained pass and was correctly rejected; after the explicit
allowed-values guidance change, a new probe received model-authored unknown
without an execution error. The initial failure is retained. These probes are
supplementary repair evidence only; none replaces a historical measurement or
counts as a fresh native pipeline run.

The guidance follow-up passed 38 focused tests and 24 independent offline Role
replays, with outcomes matching the prior snapshot. The original invalid pass
remained rejected. Review remains same-family and provisional, restricted to
source/protocol integrity.

## Fresh pair on commit 1d6e90c

`runs/full_claw_20261012_v8` and `runs/full_workbuddy_20261012_v7` started from
new seeds. Their frozen source matches the latest test snapshot (147 common
files for Claw, 85 for WorkBuddy). Three rounds, full audit, the existing role
limits and `quality_resampling=false` are unchanged. Complete native receipts
and strict audit results remain pending.

Claw v8's first construction completed 21 checks without execution errors, but
its first B completeness review failed. The model defined full-score gap IDs
such as `G1_status_field_omission`, then referenced abbreviations `G1`, `G2`,
`G3` in its improvements, both initially and after the existing single repair.
The strict audit rejected `round_1/B_review_execution`; this was an unrecovered
contract error, unlike its earlier recovered Jury JSON syntax error. Claw was
interrupted with SIGINT and exited 130. Six completed receipts were hash-bound
at interruption and verified unchanged afterward. This attempt is retained as
a diagnostic failure; the stop was unrelated to native quality scores.

WorkBuddy v7 independently completed 27 construction checks without execution
errors and entered A/B/C. It continues on its original frozen source. The gap-ID
diagnostic repair does not modify either frozen run.

Commit `b382448` makes the improvement schema and repair diagnostics explicit
about complete, exact IDs from the model's own gap ledger. The existing single
repair now receives all invalid fields/indices and allowed IDs together. The
host supplies no alias mapping or replacement ID. The focused suite passed 548
tests; full regression passed **2,712 tests**, with nine dependency warnings,
no source drift and all 282 current file hashes matching. Independent bounded
review passed 37 focused tests, 93 acceptance comparisons and 24 offline Role
replays, retaining the same acceptance and retry counts. It remains same-family,
provisional source evidence. WorkBuddy's first three B review versions have no
execution errors and request further improvement; the round itself is not yet
complete.

Fresh Claw v9 started on `b382448`; 147 copied files match its 2,712-test
snapshot. WorkBuddy v7 continues on `1d6e90c` and its separately bound 2,697-test
snapshot. Their exact versions remain distinct; the gap-ID diagnostic change
is not retroactively attributed to WorkBuddy's execution. Both still require
three completed rounds and strict receipt audits.

WorkBuddy v7 subsequently completed round one and entered round two. The strict
audit snapshot `workbuddy_v7_after_round1.json` passes every first-round stage,
including two changed-artifact same-session B revisions, three original full
pack measurements, ranking/selection, and memory/checkpoint persistence. Overall
status remains pending because rounds two and three are unfinished.

Raw native first-round scores are **A 0.9213, B 0.1049, C 0.9082**; the Jury
selected B. B's native deterministic result is 8/61 and its native LLM component
is 0.0, despite its Reward Engine full measurement recording nine passes. This
disagreement was traced to the submission path: B wrote
`output/final_answer.md` in every delivery snapshot instead of the public Task's
required root-level `final_answer.md`. The native verifier received empty
declared-artifact text. Internal semantic checks accepted the wrong-location
file; their passes do not establish correct Task delivery. The low score is
retained, not normalized or used to trigger quality resampling; no
quality-improvement claim is supported. The separate read-only report is
`workbuddy_v7_round1_quality_diagnosis.md` under the local evidence directory.

That report also retains native-contract conflicts without altering grades:
three alleged input-file modifications are byte-identical across inspected
initial/final snapshots, and a rule penalizes A/C for creating the Task-requested
root file. Neither observation cancels B's missing required submission.

WorkBuddy v7 then exited 1 during round-two Jury construction. Its initial
definition lacked product-specific evidence for every affected product; the two
targeted repairs used an unsupported `add` operation and then a nonexistent path.
`construction_failure.json` retains `Repair path must already exist`. The final
strict audit fails; one completed round does not validate the three-round loop.

Claw v9 completed round one and entered round two, but its first-round strict
audit fails ranking/selection. One ranking member initially returned nested
`products` instead of top-level `groups`. Its single repair corrected the groups
but rewrote a valid observation and added the unavailable `product_3:source:8`
reference. Only two of three members completed, so ranking was unavailable and
the driver selected fallback A. That fallback is not validated ranking. All
three full delivered-pack measurements and two actual B revisions passed their
execution checks. Raw native scores remain A 0.916, B 0.958, C 0.840.

Claw v9 was interrupted for this confirmed execution failure and exited 130.
Its `diagnostic_stop.json` binds 19 completed receipts; hashes were unchanged
after termination. Targeted Jury repair diagnostics and protection of valid
ranking observations are now being repaired. No frozen code or historical
measurement is modified, and no native quality result prompted a rerun.

The new targeted-repair diagnostics passed 201 focused tests and 1,824 edit
acceptance/exception/atomicity comparisons with the prior implementation. A
separate hybrid probe then froze that diagnostic patch over `b382448`, replayed
one retained invalid round-two Jury draft, and called the configured real reducer
for its repair. The model supplied five explicit edits in one call; normal Jury
validation succeeded. The raw draft, edit reply and resulting definition are
retained under `live_jury_repair_probe_6b9dbb80`. All original and probe-source
hashes remained unchanged. This verifies the bounded repair path on that frozen
snapshot only; it is neither a fresh native run nor repaired historical evidence.

Ranking field corrections then passed 477 focused tests, including 40 new
controls. A second isolated hybrid probe reconstructed the original Claw ranking
context exactly, injected its retained malformed reply, and made two real model
calls. The first supplied only model-authored partition corrections; observations
and reason were preserved exactly in the separately recorded composition. The
second request attached 12 previously unread full evidence records, then returned
a valid full ranking. `live_ranking_field_probe_7c057913` retains all attempts and
provenance; original/source hashes remained unchanged. This separate Map-stage
probe does not replace the original failed ranking or validate the full loop.

Independent review of the ranking patch subsequently reproduced a further
defect through the public `jury.rank` path and offline production Role adapter:
a contract-repair response could request previously unread evidence and then
reverse an already valid partition. Four controls reproduce this with bounded
and unbounded Role modes, with and without an intervening malformed JSON reply.
These are scripted counterexamples, not native results. The original review
receipts remain under `source_integrity_jury_repair_ranking_review_evidence/`;
the successful narrow hybrid probe above does not establish that this bypass
was closed. A follow-up fix and independent verification are required.

The WorkBuddy public-path prerequisite is also under implementation and testing.
It binds an explicit executor translation of public instruction lines 1 and 21
to their exact text and prompt hash. It checks regular-file presence at the
specified workspace-relative path; it does not establish content quality or
automatically extract every natural-language requirement. Model-authored Jury
and Pack records are retained separately from the declared deterministic
projection. Every B review and final measurement must include the prerequisite,
and selection cannot accept a nonconforming fallback. The unchanged original
pipeline auditor and a separate byte/provenance audit must both pass. Neither
implementation nor final native validation is complete yet.

The combined RR1 repair and public-delivery implementation subsequently passed
the full unit/integration suite: **2,869 passed, zero skipped, nine existing
Pydantic deprecation warnings, 217.01 seconds**. The captured 291 source,
configuration and fixture hashes remained unchanged. The enabled Docker/OpenCode
integration uses a scripted provider, so this is controlled validation, not
native benchmark success. Metadata and log are retained as
`full_tests_before_claw_v10_workbuddy_v8.{json,log}`. Independent review 007 is
still in progress; no fresh native runs have been launched at this checkpoint.

Review 007 then reproduced **PD1**: jointly replacing `jury_semantic.json` and
the effective `jury.json` with an unauthored standard passes the added projection
audit. The original Jury/Builder substages also accept it, while the retained
reducer response and construction raw record remain unchanged. This is a
scripted counterexample, not a claimed complete native-audit pass. It shows that
two mutually consistent copied definitions do not establish model origin.
`PD1_counterexample.json` and the forged copies are retained in the review's
evidence directory. The additional auditor is being strengthened to reconstruct
the semantic definition from actual retained construction records; the original
pipeline auditor remains unchanged. The 2,869-test result predates this fix.

The PD1 repair now reconstructs semantic Jury and Pack contents with the shipped
construction code and an offline adapter that matches retained requests and
parses original provider output. It validates ordered model-authored edits,
atomic rejection of invalid batches, and full or partial reuse of previously
audited Packs. Only the generated Pack UUID is treated as host randomness;
semantic fields must reconstruct exactly. The original forged Jury and analogous
Pack/parsed-answer controls reject. **401 focused tests passed**, followed by
**2,873 full-suite tests passed**, zero skipped, nine existing deprecation warnings,
and no drift across the captured 291 files. The new suite is retained as
`full_tests_after_pd1_origin_binding.{json,log}`; independent review 010 is still
running. No live model calls, tools or historical-record rewrites occur in this
origin check. Final native validation is still pending.

Independent review 010 closes the original PD1 Jury/Pack rewrite counterexample
but still returns **FAIL** on two further source-consistency gaps. **PD1-F1**:
an invented successful checker trial can override independently retained failed
tool results during origin reconstruction. **PD1-F2**: a recorded syntax-failure
summary can authorize recovery despite contradictory original provider stream
bytes. Both are simulated counterexamples with their unchanged source receipts
retained in `source_integrity_pd1_origin_followup_evidence/`. The internal
sub-Pack UUID concern was withdrawn because those UUIDs are legitimately
independent. The additional auditor and its tests are being repaired; all
previous test results and FAIL reports remain bound to their original snapshots.
No fresh native task has started.

The subsequent two-file audit repair verifies physical provider payloads before
authorizing syntax recovery and replays the production checker-author protocol
against distinct, matching tool records in the same recorder scope. It retains
the failed source hashes and preserves legitimate streaming/nonstream recovery,
failed-trial correction and exhausted failure. **406 focused tests** and then
**2,878 full-suite tests** passed, with no skips or drift across the captured
291 files. The full run took 237.80 seconds with nine existing deprecation
warnings; metadata/log are `full_tests_after_origin_transport_trials.{json,log}`.
Independent review 011 subsequently passed its bounded F1/F2 closure checks.
It rejected the exact original attacks and verified legitimate authoring,
recovery and reuse in four three-round simulations; original construction,
measurement, B review and selection checks were exercised. This is still
controlled evidence, not native success. All 291 source hashes matched and
10,470 protected historical files remained unchanged. Original failed reviews,
counterexamples and the unchanged original strict auditor are retained. The
review is same-family/provisional; fresh native validation remains required.

The fresh native **Claw v10 / WorkBuddy v8** pair then ran on frozen commit
`fe1cc5286ee50c5d85e0df91803723bc8a6a36a3`. Both failed the unchanged original
strict audit in round 1 and were diagnostically interrupted with observed exit
130. Claw's mandatory evidence expansion produced a reply missing one claim's
`judgment`; the bounded repair changed the claim count and was correctly
rejected. Its repair request also retained an earlier provisional assessment and
an instruction to reassess it. WorkBuddy had two separate failures: a Jury seat
twice omitted literal standard text, leaving only two Map seats, and one
construction measurement twice nested the verdict under `response` instead of
providing the required top-level fields. These are execution failures, not
quality-based restarts. All completed receipts remain unchanged: ten recorded
Claw hashes and eighteen WorkBuddy hashes were verified after interruption.

The launch-binding check also found that WorkBuddy's manifest hashed only Python
files despite copying non-Python runtime resources. Its existing manifest is
preserved with that limitation. Prospective fixes address precise model-authored
structural repairs and complete frozen-resource manifests; they do not alter the
failed runs, increase retries, or weaken audit requirements. No three-round
native success has yet been established. The prior 2,878-test result binds the
pre-fix source and does not validate these subsequent changes.

The subsequent repair adds explicit model-authored missing-judgment corrections
and an optional, explicit operation to relocate an unchanged, unambiguous nested
measurement verdict. It preserves the original and composed responses with
field provenance, keeps all valid claim fields locked, and still requires cited
preview expansion. Structural retries no longer carry a conflicting instruction
to reassess an older provisional response. Jury diagnostics identify the exact
missing `standard`/clause fields without deriving standards from other text;
the requested preservation of draft identity is an instruction, not a new
cross-response validator. New WorkBuddy manifests bind all copied regular
source/resource files except transient bytecode caches, while legacy manifests
retain their original limitations. **2,981 tests passed**, zero skipped, with
nine existing warnings in 248.06 seconds; all 298 captured source/config/fixture
hashes remained unchanged. Metadata/log are
`full_tests_after_native_contract_repairs.{json,log}`. Independent review 012
found no additional scoped measurement/Jury issue, but returned **FAIL (NR1)**:
the new complete-file manifest includes MJS resources while the additional
public auditor still compares package entries against Python files alone.
A three-round simulated control reproduces that incompatibility. The additional
auditor is being corrected to honor the explicit manifest scope; the original
strict auditor and historical manifests remain unchanged. These controlled
results predate that follow-up and do not establish native success.

The NR1 follow-up now verifies the exact complete file set and byte hashes,
including external transport and package resources, while explicitly retaining
the legacy manifest limitation. Public pipeline controls use the actual
prepare/validate flow. **3,019 full-suite tests passed**, zero skipped, with nine
existing warnings in 276.87 seconds and no drift across 299 captured files
(`full_tests_after_complete_manifest_audit.{json,log}`). Independent review 013
passed the bounded NR1 closure: baseline failure versus fixed success, actual
prepare/validate integration, and file/scope rejection controls. No additional
blocking issue was found within that review. It remains same-family/provisional,
and the original strict audit is still independently required. Fresh native
three-round validation remains pending; all previous FAIL reports are retained.

The next real pair used frozen commit
`33d39e0da135f1eceebbbabecc955c83c47ac3f6`. Claw v11 completed naturally with
exit 0 and an original strict audit **PASS**: three fresh seeds, three completed
rounds, nine native actors and native grades, nine full delivered-pack
measurements, three Hypothesis proposals, and three validated same-session
changed-artifact revision rounds. Jury, Builder, ranking/selection, and
memory/checkpoint stages passed. This is real native pipeline evidence, separate
from the controlled suite. The native scores still do not establish quality
improvement or official leaderboard equivalence:

| Claw round | A | B | C | Selected |
| --- | ---: | ---: | ---: | --- |
| 1 | 0.916 | 1.000 | 0.916 | B |
| 2 | 0.930 | 0.972 | 0.868 | C |
| 3 | 0.916 | 0.972 | 0.868 | B |

In particular, the second-round selection has a lower native score than both
alternatives. All selections and grades are retained unchanged.

WorkBuddy v9 completed its first round with all strict stages passing, including
real B continuation and artifact changes. Its A/B/C native scores were
0.9213/0.9475/0.9082, and B was selected. The second round failed construction
measurement `language_format_usability_check`: a model-authored `unknown`
correction was correctly retained in the next evidence-expansion request, but
the model then returned `pass` while retaining an unverified claim with omitted
relevance (which means material). Prose calling that claim nonmaterial did not
satisfy the structured contract. The strict rejection and refusal of another
repair were correct; no lost-correction or retry-accounting defect was found.
The run was diagnostically stopped with observed exit 130. All 654 protected
completed receipts remained unchanged, and only its three owned orphan actor
containers were stopped, preserving their records.

A prospective reminder now repeats the existing claim/status consistency rules
after genuine preview expansion. It permits legitimate reassessment from new
evidence, requires explicit criterion-specific advisory relevance, and does not
prescribe a verdict, change acceptance, or increase any budget. The original
three replies still fail in controlled replay. **3,035 full-suite tests passed**,
zero skipped, with nine existing warnings in 275.68 seconds; all 300 captured
source/config/fixture hashes remained unchanged
(`full_tests_after_post_preview_consistency.{json,log}`). Independent review 015
passed its bounded source and offline controls, with same-family/provisional
acceptance. The new reminder's native reliability remains unproven. Claw v11's success belongs to the earlier
frozen source; WorkBuddy v9 remains failed. Fresh validation of the changed
shared request behavior is pending.

Prospective reporting-helper review 014 passed its bounded controls, but the
actual Claw terminal audit exposed an additional evidence-binding incompatibility:
the original auditor lists directories and expected-absent failure markers as
well as regular files. The initial command-receipt collector incorrectly assumed
all entries were regular files and stopped; the PASS audit output was retained.
Typed evidence binding was subsequently implemented and the frozen original
Claw audit was rerun with observed exit 0 and the identical PASS output. Its
236 file, 30 directory and 21 absent-path observations match before and after
that fresh invocation. Independent review 016 passed the bounded reporting
follow-up: 240 supplied controls and 28 independent cases, including the actual
287-path report shape. Acceptance remains same-family/provisional; the original
auditor remains unchanged. No native coverage has been exported or combined, and the
Claw v11 / WorkBuddy v9 pair is ineligible for a successful paired terminal report.

The next pair, Claw v12 / WorkBuddy v10, used frozen commit
`bf85cddf1d16d658e82a86bb2ef2637e562c6a25` and failed on upstream HTTP 403
responses, surfaced by the local role gateway as HTTP 502. WorkBuddy's Jury
passed, but its first-round Builder call failed before any answer was retained;
the driver exited 1 with zero completed rounds. Claw's original strict audit
failed first-round construction measurements after eleven rejected role
requests. It was diagnostically stopped with observed exit 130; all 680 protected
completed receipt hashes remained unchanged. Neither run establishes full native
pipeline success or the reminder's reliability.

The original provider rejection bodies were not retained, so the narrower cause
of those 403 responses is unknown. Same-model native requests with identical
options and larger bodies also succeeded. Separate access diagnostics later
returned 200 for a model-list request, a tiny generation request, and an exact
replay of the rejected Builder request. The exact replay was closed after SSE
headers without reading or using its response; this establishes request
acceptance at that time, not completed generation or a valid answer. These
diagnostics do not replace any native receipt. Fresh Claw v13 / WorkBuddy v11
validation has started on the unchanged source, budgets and strict gates;
the failed runs remain historical and ineligible.

Claw v13 subsequently failed the strict first-round Jury gate: only two of three
seats were usable. Its retained role transport includes GLM HTTP 401 and Qwen
HTTP 403 responses. It was diagnostically stopped with observed exit 130 and
243 protected completed receipts unchanged. A separate exact Qwen request
diagnostic at 2026-10-12 15:43:15 UTC retained the provider's explicit error
`pre_consume_token_quota_failed`: the token had about USD 0.312028 remaining,
while that request required about USD 0.915600 in quota reservation. This
confirms a current token-quota blocker; it does not retrospectively establish
the cause of every earlier 403 or the GLM 401, whose rejection bodies are absent.
The separate GLM replay received 200 headers and was closed without reading.

WorkBuddy v11 was then diagnostically stopped for the confirmed shared quota
blocker, with observed exit 130. Its last original strict audit was **pending**,
with three seeds and zero completed rounds, actors or grades; it was not a
completed strict FAIL or PASS. Its closed transport receipts also include
DeepSeek and Qwen 403 responses. All 306 protected completed receipts remained
unchanged. No native validation driver remains active. The user has been asked
to replenish the token's available quota before further real validation;
models, budgets, required Jury seats and acceptance rules have not been relaxed.

A bounded read-only investigation of Claw v11's second-round choice reproduced
C with the actual selector. Its Jury reducer preferred saved artifacts and
traceability, whereas the native grader preferred B's anomaly analysis and
actionable advice. All three arms passed the six pack checks, so the selector's
check-regression veto did not exclude C. No host selection/mapping bug was
reproduced. This is an observed assessment disagreement, not evidence of quality
improvement; native grades remain excluded from learning and selection inputs.

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
- `runs/pipeline_validation_20261012/full_tests_after_claim_diagnostic.json` and
  `.log`: 2,541-test passing suite and exact source hashes.
- `runs/pipeline_validation_20261012/pipeline_facts_continuation_1.json`:
  the first-round progress snapshot described above and input hashes.
- `runs/pipeline_validation_20261012/claim_repair_probe_af3f6306/`:
  separate prospective measurement, input binding and source provenance.
- `runs/pipeline_validation_20261012/workbuddy_v5_audit_material_claim_failure.json`:
  strict audit rejecting A's original full-pack measurement contradiction.
- `runs/pipeline_validation_20261012/{claw_v6,workbuddy_v5}_after_round1.json`:
  strict audits including intermediate review execution failures.
- `runs/pipeline_validation_20261012/pipeline_facts_round1_with_review_history.json`:
  progress snapshot including every retained review version.
- `runs/pipeline_validation_20261012/full_tests_after_claim_preservation_fix.json`
  and `.log`: 2,578-test passing suite and exact source hashes.
- `runs/pipeline_validation_20261012/source_integrity_claim_status_review.md`
  and `.json`: original CS-1 FAIL and reproductions, retained unchanged.
- `runs/pipeline_validation_20261012/source_integrity_claim_preservation_followup.md`
  and `.json`: bounded, provisional closure and independent control results.
- `runs/pipeline_validation_20261012/live_claw_v7_workbuddy_v6_handles.json`:
  new driver handles, code bindings and distinct coverage prefixes.

The earlier [2026-10-10 report](validation-2026-10-10.md) remains historical.
Successful native process exit, grader scores, and source-line execution will
not substitute for complete stage receipts or the final strict audit.
