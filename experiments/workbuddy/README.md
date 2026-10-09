# WorkBuddy Bench Office

公开实验覆盖 50 题、每题三个 seed 和三轮 A/B/C。主结果、真实配置和代码快照见
`data/workbuddy_office/experiment.json` 与 [最终报告](../../reports/workbuddy_office/report.html)。

离线报告：

```sh
python -m release.report --experiment workbuddy_office --data-root data --output /tmp/workbuddy-report
```

新实验入口为 `python -m experiments.workbuddy.experiment prepare --output runs/new_office`，随后使用 `run`。
先安装 `.[runtime]`，并准备与冻结记录一致的 WorkBuddy/Harbor、Docker、OpenCode 和模型服务。
用 `WORKBUDDY_ROOT` 指定 runner 根目录，其下需要 `workbuddy_swarm/` 和 `upstream/`；后者包含 benchmark 的
`src/`、`datasets/wb-bench-office-v1.0/`、`configs/`。相关源码与配置在公开数据的 `environment/` 和 `benchmark/` 中。
原生 WorkBuddy Python 包须安装到使用的解释器。凭据由该 upstream 下的 `.env` 读取，使用 `.env.example` 中的变量名，自行填写。

公开结果中实际使用过多种模型配置；新入口默认设置并不代表所有历史任务的条件。复现实验条件时应参照逐任务配置和执行代码，
保留原生有效零分、失败及选择规则。整理与离线报告重建均不自动启动新实验。
