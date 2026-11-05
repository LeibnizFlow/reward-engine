# Claw-Eval adapter

Set `CLAW_EVAL_ROOT` to the benchmark checkout and `CLAW_EVAL_TASK_MANIFEST` to its pinned 104-task manifest. Install the native `claw-eval-agent:latest` Docker image. Missing declared fixtures require an explicit verified overlay.

Install controller dependencies with `pip install '.[runtime,workbuddy]'`. Supply a private model profile using the [configuration interface](../../docs/configuration.md).

```sh
export OPENAI_API_KEY='YOUR_API_KEY'
export REWARD_ENGINE_BACKEND_URL=https://api.example.com/v1
python -m experiments.claw_eval.campaign prepare --output _local/runs/claw_eval-opencode \
  --runtime opencode --model-config /path/to/models.json
python -m experiments.claw_eval.campaign run --output _local/runs/claw_eval-opencode
```

For JiuwenSwarm, replace `--runtime opencode` with `--runtime jiuwenswarm --jiuwenswarm-bundle /path/to/immutable/bundle` and choose a new output directory. Both runtimes support the default `three_b_fixed_baselines_v1` protocol. Use `--generation-protocol legacy_abcd` for the legacy protocol explicitly. Native scores are recorded separately from RewardPack construction, feedback and selection.
