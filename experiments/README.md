# Native benchmark entry points

- [WorkBuddy Bench Office](workbuddy/README.md): prepare and run a new experiment.
- [Claw-Eval](claw_eval/README.md): prepare an initial round and continue eligible algorithm rounds.

Native adapters require Python 3.12, Docker Compose, Node.js, OpenCode 1.18.16, and an available model API. Claw also needs the benchmark's `claw-eval-agent:latest` image. The offline core example has none of these requirements.

## Prepare dependencies and data

Obtain the required archives from the project's [evidence release](https://github.com/TheLittleJimmy/reward-engine/releases/tag/evidence-v1.0.0), subject to access permission, and place them in `_local/zenodo/`. Restoration verifies archive hashes and restores data without replacing the current engine source:

```sh
python -m experiments.restore_native --archives _local/zenodo --root _local
python -m pip install '.[runtime,workbuddy]'
python -m pip install -r _local/data/claw_eval/benchmark/requirements.txt
```

Copy `.env.example` to a private `.env` and fill in the endpoint and credentials for your provider. Then install the native source layout:

```sh
python -m experiments.setup_native --credentials .env
```

`REWARD_ENGINE_CREDENTIALS_FILE`, `REWARD_ENGINE_KEY_ENV`, and `REWARD_ENGINE_BACKEND_URL` select the credential file, key variable, and API base URL. Model names must be supported by that service. Benchmark data, native images, and provider access are separate prerequisites; installing the core package alone does not provision them.

Use a new `_local/runs/<name>/` directory for each execution. Native grading is post-hoc evaluation and is excluded from reward construction and candidate selection.
