# WorkBuddy Bench Office

Follow [native setup](../README.md) to prepare WorkBuddy/Harbor, Docker, OpenCode, and the model service. Install native Python dependencies into the same interpreter used for the controller.

```sh
python -m experiments.workbuddy.experiment prepare --output _local/runs/new_office
python -m experiments.workbuddy.experiment run --output _local/runs/new_office --canary-only
```

Use `WORKBUDDY_ROOT` to select the runner root. The default is `_local/native/workbuddy/runner/`, containing `workbuddy_swarm/` and `upstream/`. The latter holds benchmark `src/`, `datasets/wb-bench-office-v1.0/`, and `configs/`.

The command above runs the first task as a canary. Omit `--canary-only` to run all prepared tasks. Use `prepare --help` for task selection, provider, and concurrency options before preparation freezes the configuration.

Supply credentials through explicit configuration or a private dotenv file. Choose a fresh output directory and retain valid zero scores, failed attempts, and original selection receipts. See the [adapter interface](../../reward_engine/adapters/workbuddy/README.md) for snapshots, same-session revisions, and task scope limitations.

## Optional model profile

`reward_engine/configs/bailian_token_plan.json` is an optional WorkBuddy profile for the `--model-config` argument of `python -m experiments.validate_pipeline`. It is not loaded by the default campaign commands. It binds explicit model names, token budgets, concurrency, and a separate native grader. Your provider must support those settings.

The profile contains WorkBuddy-only `native_grader`, `measurement_candidate_concurrency`, and `delivery_review_json_schema_models` fields. It is not a general Claw-Eval profile. Keep private credentials in the environment or an ignored dotenv file; the public profile contains no credentials.
