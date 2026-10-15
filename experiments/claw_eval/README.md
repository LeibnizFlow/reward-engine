# Claw-Eval

Follow [native setup](../README.md) before running the OpenCode campaign:

```sh
python -m experiments.claw_eval.opencode.campaign prepare --output _local/runs/new_claw
python -m experiments.claw_eval.opencode.campaign run --output _local/runs/new_claw --canary-only
```

Configure `CLAW_EVAL_ROOT`, `CLAW_EVAL_TASK_MANIFEST`, and `OPENCODE_BINARY` for the benchmark checkout, task manifest, and OpenCode executable. Defaults use the restored local layout. Optional fixture overlays must pass their identity checks.

The command above runs the first task as a canary. Omit `--canary-only` to run the prepared campaign. Use `prepare --help` for provider options and `python -m experiments.claw_eval.opencode.continue_rounds --help` for subsequent eligible algorithm rounds. Each run freezes its configuration and source identities; native grading remains separate from online selection. See [native grader requirements](../../reward_engine/adapters/claw_eval/NATIVE_GRADER_HOOK.md).
