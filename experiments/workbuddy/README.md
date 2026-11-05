# Office adapter

Set `WORKBUDDY_ROOT` to the native runner directory containing `upstream/`, or pass `--upstream`. Install its `workbuddy_bench` Python package and the pinned Office 50 dataset. JiuwenSwarm preparation builds or verifies immutable task images.

Install controller dependencies with `pip install '.[runtime,workbuddy]'`. Supply a private model profile using the [configuration interface](../../docs/configuration.md).

```sh
export OPENAI_API_KEY='YOUR_API_KEY'
export REWARD_ENGINE_BACKEND_URL=https://api.example.com/v1
python -m experiments.workbuddy.experiment prepare --output _local/runs/workbuddy-opencode \
  --runtime opencode --model-config /path/to/models.json
python -m experiments.workbuddy.experiment run --output _local/runs/workbuddy-opencode
```

For JiuwenSwarm, replace `--runtime opencode` with `--runtime jiuwenswarm --jiuwenswarm-bundle /path/to/immutable/bundle` and choose a new output directory. Both runtimes support the default `three_b_fixed_baselines_v1` protocol. Use `--generation-protocol legacy_abcd` for the legacy protocol explicitly. Native scores are recorded separately from RewardPack construction, feedback and selection.
