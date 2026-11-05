# JiuwenSwarm runtime

This adapter runs workswarm 0.2.7 through Process CLI schema 0.1 in each benchmark's native Docker environment. Install the controller with `pip install '.[runtime,workbuddy]'` and the benchmark's native dependencies. JiuwenSwarm runs from an explicit bundle; OpenCode is needed only when `actor_backend=opencode` is selected.

## Immutable bundle

With Linux Python 3.12 and workswarm 0.2.7 installed in a dedicated interpreter, export its pinned dependencies and standard library:

```sh
python -m reward_engine.adapters.jiuwenswarm.bundle build _local/runtime/jiuwenswarm
python -m reward_engine.adapters.jiuwenswarm.bundle verify _local/runtime/jiuwenswarm
```

The interpreter must use `$ORIGIN/../lib` RPATH. The launcher unsets `LD_LIBRARY_PATH`, so task subprocesses use the container's native libraries. Experiment preparation and startup read the declared manifest identity without scanning installed runtime files; new configurations record `runtime_integrity_check=manifest_only`. Run `bundle verify` explicitly to check the complete file inventory, content hashes, dependency lock, and Process version. An existing compatible bundle can be copied with `bundle repackage TARGET --source SOURCE`; this discards bytecode caches and repairs the launcher without modifying the source. Keep the resulting bundle read-only.

## Configuration

Both benchmark entries accept `--runtime jiuwenswarm` (also spelled
`jiuwen-swarm`) or `--runtime opencode`. The previous `--actor-backend`
option remains supported. These options are mutually exclusive and write the
same canonical `actor_backend` field; saved configurations and checkpoint
identities keep their existing meaning.


| Field | New experiment default / requirement |
| --- | --- |
| `actor_backend` | `jiuwenswarm`; explicit `opencode` remains supported |
| `jiuwenswarm_mode` | `agent.code.normal`; `agent.work.normal` also supported |
| `jiuwenswarm_bundle` | Required explicit bundle directory |
| `generation_protocol` | `three_b_fixed_baselines_v1` |
| `rounds` | 3; accepts 1–3 |
| `task_images_manifest` | Optional Office image manifest |

Saved configurations lacking these fields retain their previous interpretation. Existing checkpoints are never upgraded automatically. To prepare the previous A/B/C protocol, explicitly select both `--runtime opencode --generation-protocol legacy_abcd`.

A new three B experiment freezes three independent baseline deliveries once. Each round builds and audits a RewardPack, measures it, and starts B1/B2/B3 in independent sessions. Each B can receive at most two critic revisions in its own session. The selector compares only the fixed baselines and that round's final Bs. Previous Bs can inform subsequent reward construction. Hypothesis generation is disabled. Native scoring runs separately and its results never enter construction, actor guidance, critic feedback, or selection.

Use the same entry's `run --output DIRECTORY` to resume. Completed stages are bound to task, runtime, image, Pack, session and delivery content. A started actor with no confirmed completion produces an explicit error requiring inspection. Independent pool selection, public delivery contracts, and legacy recovery/replay overrides cannot be combined with the new protocol.

New three B configurations request JSON objects from internal roles, disable their optional thinking mode, and use complete nonstream responses (`role_stream=false`). Raw responses and failures are retained; truncated answers cannot become verdicts. Saved configurations without this field keep streaming. With streaming enabled, `role_transport_recovery=true` allows one separate nonstream request after a conflicting stream identity or provider stream failure, using the same model and semantic parameters.

The gateway applies the configured `role_stream_timeout_s` deadline to complete internal role responses, including header retries and body reads. Streaming retains its idle-read timeout. Timeouts preserve a failed transport receipt and release gateway capacity.

The bounded correction protocol allows one syntax correction and one semantic correction per role stage. With `role_protocol_edits=true`, Jury map and reduce corrections use explicit model-authored edits bound to the rejected draft's hash. Untouched fields remain exact, and the reconstructed draft passes all original provenance, requirement and evidence checks. This does not add model calls or relax validation. New three B experiments enable this setting; saved configurations without it retain their previous correction protocol. Actor and native grading requests do not inherit these internal role settings.

Implementation updates use the [explicit revision workflow](../../../experiments/README.md#resume-with-an-explicit-implementation-revision). That guide defines preparation, frozen settings and completed-stage reuse for both native adapters.

## Tools and retained records

Each actor has its own authenticated gateway route. One policy drives both advertised tools and runtime permissions. Claw permits only the trial's registered MCP tools; Office permits file operations, shell and todo tools. Filtering happens before request identity and quota checks. Invalid `tool_choice` fails explicitly. Role and native grading routes remain separate.

Actor directories retain Process requests, raw JSONL, stderr, validated events/results, cleanup metadata, real session archives and delivery snapshots. Timeout and cancellation stop the actual actor container. Reports include B1/B2/B3, selection source, actual revisions, hypothesis state, tool errors, observed token receipts and private native scores. Missing usage or unknown tool outcomes remain missing.

Process execution receipts record `requested_timeout_s` and `host_timeout_s`. A timeout additionally records `timeout_source` as `native` when the child returns a valid timeout result, or `host` when the host deadline expires; host timeouts include `timeout_elapsed_s` before cleanup. The raised host `TimeoutError` identifies both limits and the execution receipt, so enclosing task failures retain an actionable reason. `duration_s` includes cleanup. These diagnostics do not extend execution budgets, turn partial output into a completed delivery, or authorize an Actor retry.

In the Claw three B protocol, a validated native HTTP 500, 502, 503, or 504 failure can continue once within the same live Actor turn. The container, services, workspace, and native session stay in place. Continuation requires a bound exported session and uses the remaining Actor execution time. Both Process executions, their usage, and their session archives are retained under distinct labels. A recovery receipt binds the failed execution and remaining deadline; checkpoint reuse verifies that provenance. Transport recovery does not add a critic revision. Another failure, missing session evidence, timeout, or cancellation ends the turn. Archived Actors whose containers and services have closed still require separately authorized recovery.

### Explicit terminated Actor retry

Claw's JiuwenSwarm three B entry supports an opt-in `one_terminated_actor_retry_v1` policy. It is disabled unless the operator supplies a matching approval document. It permits one additional attempt for a first execution that exhausted its frozen host deadline, has a nonzero child return code and no cleanup error, and has no prior completed delivery or session archive. An event-only stream is eligible. A complete stream ending in a native `timed_out` result is also eligible when the Process schema, request and session identities, exit code 124, native `TIMEOUT` error, and unchanged host deadline all agree. This native terminal is retained and verified again when the completed retry is reused. Completed results, other terminal states, ambiguous, interrupted, truncated, or previously retried executions are rejected. This policy changes the number of Actor attempts and requires an explicit experiment-policy decision; normal resume does not enable it.

Prepare an unapproved proposal without changing the experiment:

```sh
python -m reward_engine.adapters.claw_eval.actor_retry \
  --config CONFIG_JSON --task-id TASK_ID --stage round_1/B1 \
  --output PROPOSAL_JSON
```

Stages can be `seed_0` through `seed_2`, or `round_1/B1` through `round_3/B3`. The output contains `proposal` and `approval: null`. After explicit approval, an authorization document retains that exact proposal and sets `approval` to an object containing `decision: "approve"` and `proposal_sha256`: the value returned by `reward_engine.core.outcome.construction.fingerprint(proposal)`. Supply its absolute path to `experiments.claw_eval.campaign run` using `--actor-retry-authorization APPROVAL_JSON` and select the authorized task with `--only-task TASK_ID`. Multiple authorization paths are accepted; canary/probe-only modes are excluded. The selected implementation revision must contain this interface.

Before consuming approval, the controller checks the frozen experiment, original input identity, timeout, complete failed Actor file inventory, and stage failure history. The original Actor directory stays in place. An exclusive claim under the task's `actor_retries/` directory stores the authorization and original stage receipts, and the additional Actor uses a separate directory. A consumed claim cannot start another attempt, including after interruption. Completed stages continue to use their retained checkpoints. Retry records retain failed execution time (including cleanup), successful execution time, and their sum, and are verified against the saved result on reuse and included in reports. These records are not added to model construction, Actor guidance, or selection inputs. Model settings, per-execution timeout, critic revision limits, and internal role budgets stay fixed.

If that additional execution exits with a fully validated native upstream HTTP 500, 502, 503, or 504 failure, `propose_transport_retry(..., policy=UPSTREAM_POLICY)` can prepare a separate `one_native_upstream_actor_retry_v1` proposal. It requires a bound exported session, an unchanged native and host deadline, and no completed Actor, snapshot, or delivery state. Explicit approval permits one fresh Actor execution in a distinct claim directory while retaining the original Actor, the failed retry, its partial workspace and session, and both failed execution costs. The receipt identifies this as `fresh_actor_with_retained_failed_session`; it does not claim to restore the failed container or its services. A third failed execution cannot reuse this approval. The earlier first-output timeout policy retains its stricter session exclusion.

After that third execution has terminated, `propose_transport_retry(..., policy=INFRASTRUCTURE_POLICY)` can inspect a `one_terminated_infrastructure_actor_retry_v1` proposal. It accepts either the same validated native server failure with an exported failed session, or a host timeout whose Actor trace and gateway receipts prove that waiting for capacity consumed at least 90% of the native execution budget. It revalidates both prior authorizations and all three failed executions. A separate matching approval permits one fourth execution in a new directory. All failed workspaces, sessions, raw receipts and costs remain in the lineage, and each execution retains the original deadline. The completed retry records three additional attempts and verifies the entire lineage on reuse. This authorization cannot be reused for a fifth execution or a completed Actor.

## Office images

Office image records contain `task_id`, an immutable `image_id` (`sha256:...`), and `input_sha256` mapping paths relative to `/workspace` to SHA-256. Preparation verifies the image and task inputs and runs native Python, SSL, SQLite, dependency and file read/write checks before starting actors. Without a manifest, preparation builds and freezes the selected task images; those images must already satisfy the dependency preflight.

To add pinned Office dependencies to base images while preserving input bytes:

```sh
python -m reward_engine.adapters.jiuwenswarm.office \
  --source-manifest _local/images/base.json \
  --output _local/images/office \
  --tasks TASK_ID
```

The helper downloads pinned wheels once, records their hashes, installs with `--no-index --require-hashes` in an offline image build, and checks PPTX/XLSX reopen operations. Supply the resulting `manifest.json` to Office preparation. Task execution does not install these dependencies.

See the [Claw entry](../../../experiments/claw_eval/README.md) and [Office entry](../../../experiments/workbuddy/README.md) for commands.
