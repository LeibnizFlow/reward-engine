# Claw-Eval

公开实验覆盖 104 题、每题三个 seed 和三轮 A/B/C，累计 936 个有效评分。
最终结果与配置见 `data/claw_eval/experiment.json` 和 [最终报告](../../reports/claw_eval/report.html)。

```sh
python -m release.report --experiment claw_eval --data-root data --output /tmp/claw-report
```

新实验入口为 `python -m experiments.claw_eval.opencode.campaign prepare --output runs/new_claw`，随后使用 `run`。
第二、三轮入口见 `python -m experiments.claw_eval.opencode.continue_rounds --help`。
需要 `.[runtime]`、Docker、固定 `claw-eval-agent` 镜像、OpenCode 1.18.16 与可用模型服务。
使用 `CLAW_EVAL_ROOT`、`CLAW_EVAL_TASK_MANIFEST`、`OPENCODE_BINARY` 和 `REWARD_ENGINE_CREDENTIALS_FILE` 配置路径。
前两个默认指向公开 `data/claw_eval/`；OpenCode 二进制保存在 `data/claw_eval/environment/bin/opencode`。
完整附件已经随任务输入保存；只有缺少声明附件时才需要提供含来源验证信息的 `CLAW_EVAL_FIXTURE_OVERLAY`。

模型、联网工具限制、实际输出额度和必要替换尝试随任务保留。公开副本不用于恢复原机器的未完成会话。
