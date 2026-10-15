# Claw-Eval native grader hook

The hook runs the benchmark's own `tasks/<task>/grader.py` against a completed native trajectory. Model-assisted judgments use the configured judge, with the native formula `safety * (0.8 * completion + 0.2 * robustness)`. A custom judge configuration is not the official leaderboard configuration, and a single score is not Pass³.

## Required inputs

Provide the actual `trace.jsonl` and `environment-snapshot.json`, plus matching benchmark source and `task.yaml`. Collect environment snapshots after actor execution and before sandbox destruction. Do not substitute another candidate's snapshot or reconstruct an empty sandbox as the original terminal state.

Complete archives can be regraded without rerunning the actor. The hook does not deploy the sandbox or convert summaries into authentic native traces. Snapshot errors are retained and listed in `snapshot_error_keys`; a missing delivered file may be a product failure, so inspect collection receipts before classifying it as infrastructure failure.

Grading runs in a separate process. Results are private evaluation outputs and do not return to Jury, Hypothesis, or actor inputs. Requests, response chunks, returned model identity, and usage are retained without embedding API keys. Transport failures may be retried once; low scores are not resampled. Missing records, incomplete replies, and model identity errors are not recorded as zero scores.

## Standalone request

[The hook](native_grader.py) accepts a JSON request such as:

```json
{
  "benchmark_root": "/absolute/path/to/frozen-claw-eval",
  "task_yaml": "/absolute/path/to/frozen-claw-eval/tasks/T110_complaint_investigation/task.yaml",
  "task_id": "T110_complaint_investigation",
  "trace": "/absolute/path/to/native-trial/trace.jsonl",
  "environment_snapshot": "/absolute/path/to/native-trial/environment-snapshot.json",
  "judge": {
    "base_url": "https://yibuapi.com/v1",
    "model": "qwen3.8-max",
    "allowed_models": ["qwen3.8-max"],
    "api_key_env": "YIBU_API_KEY",
    "temperature": 0.6,
    "max_tokens": 32768,
    "read_timeout_s": 120
  }
}
```

Copy `task_id` from the actual YAML. Verify returned model aliases before adding them to the allowlist. Load the chosen key variable in the trusted parent process. The standalone example uses Yibu Qwen; campaign grading uses its own configured role model.

```sh
python -m reward_engine.adapters.claw_eval.native_grader \
  --request _local/request.private.json --output _local/runs/native_grade
```

`run_private` provides a process deadline (900 seconds by default); direct CLI execution does not supply that parent deadline. API read timeout defaults to 120 seconds. Use an interpreter with both the engine and native grader dependencies installed, and set an overall deadline for production callers.

The evaluator hook accepts `native_grader_trial`, or otherwise `runtime_directory`, pointing to the same candidate's archive. Missing native artifacts return `needs_native_artifacts`. The minimal text example does not produce native sandbox archives. Existing grades and traces are not overwritten.
