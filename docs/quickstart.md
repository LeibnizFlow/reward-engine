# Running the core loop

The package provides a runnable Reward Engine core and native WorkBuddy and Claw-Eval adapters. The core example runs independently of benchmark datasets, Docker, and GPUs.

## Install and run offline

Use Python 3.11 or newer from the repository root:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install .
.venv/bin/python -m reward_engine.examples.minimal --output _local/runs/minimal
```

Once installed, the module works from any directory. Output paths are relative to that directory. Always choose a new output directory; existing records are not overwritten.

The example calls `reward_engine.core.outcome.loop.run_task` with a small text task:

| Stage | Behavior |
|---|---|
| Jury | Derive criteria from the task and candidate work; compare candidates |
| Builder | Bind criteria to executable checks and measure products |
| Hypothesis | Propose guidance for subsequent candidates |
| Feedback and revision | Continue the same actor conversation using check feedback |
| Selection | Select a candidate using Jury comparisons and check results |
| Memory | Persist experience and pass it into the next round |

Offline roles are deterministic and explicitly marked synthetic. The example runs three rounds, with one deliberately needed B revision per round. It demonstrates controller execution; it does not establish model quality, reproduce paper metrics, or call a native benchmark grader.

## Outputs

- `summary.json`: mode, completed rounds, selections, revision counts, and memory count.
- `round_1/` through `round_3/`: Jury, RewardPack, checks, feedback, selection, and checkpoints.
- `actors/`: conversations before and after candidate revisions.
- `calls/`: role requests and responses.
- `scope.json`: execution scope. Failures retain `failure.json` instead of a successful summary.

A live model may satisfy the review on its first draft. Revision counts record what actually happened; a revision is not forced to manufacture branch coverage.

## Live model mode

Install API dependencies:

```sh
.venv/bin/python -m pip install '.[demo]'
```

Set `OPENAI_API_KEY` in the environment or in a private `.env`, then run:

```sh
.venv/bin/python -m reward_engine.examples.minimal --live \
  --model YOUR_MODEL --base-url https://your-provider.example/v1 \
  --key-env OPENAI_API_KEY --credentials .env --output _local/runs/minimal-live
```

The provider must support non-streaming Chat Completions and `response_format={"type":"json_object"}`. The example sends `temperature=0` and `max_tokens=8192`. Responses must satisfy the role contracts. HTTP or contract errors fail the run; they do not fall back to offline results.

`OPENAI_MODEL`, `OPENAI_BASE_URL`, and `OPENAI_API_KEY` are supported. Explicit model and URL arguments take precedence over environment variables, which take precedence over dotenv values. Without `--credentials`, an existing `.env` in the current directory is loaded. Environment-only configuration also works. Use `--key-env` to select a different credential variable, with the corresponding `--base-url`.

The same model occupies every role in this example. Do not interpret it as a validation of a multi-model jury. Keep credentials private; calls consume your provider's quota. See all options with:

```sh
.venv/bin/python -m reward_engine.examples.minimal --help
```

## Native runs and optional modules

[Native adapters](../experiments/README.md) require separate benchmark assets, Docker, and model configuration. They share the core construction, feedback, selection, and memory implementation.

Adversarial QC and counterexample repair are opt-in. Completing the example does not establish coverage or effectiveness of every optional algorithm combination. The manuscript's reinforcement-learning and cross-task meta-learning extensions are not evaluated by this example.
