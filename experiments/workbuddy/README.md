# WorkBuddy Bench Office

Follow [native setup](../README.md) to prepare WorkBuddy/Harbor, Docker, OpenCode, and the model service. Install native Python dependencies into the same interpreter used for the controller.

```sh
python -m experiments.workbuddy.experiment prepare --output _local/runs/new_office
python -m experiments.workbuddy.experiment run --output _local/runs/new_office --canary-only
```

Use `WORKBUDDY_ROOT` to select the runner root. The default is `_local/native/workbuddy/runner/`, containing `workbuddy_swarm/` and `upstream/`. The latter holds benchmark `src/`, `datasets/wb-bench-office-v1.0/`, and `configs/`.

The command above runs the first task as a canary. Omit `--canary-only` to run all prepared tasks. Use `prepare --help` for task selection, provider, and concurrency options before preparation freezes the configuration.

Supply credentials through explicit configuration or a private dotenv file. Choose a fresh output directory and retain valid zero scores, failed attempts, and original selection receipts. See the [adapter interface](../../reward_engine/adapters/workbuddy/README.md) for snapshots and multi-step task handling.
