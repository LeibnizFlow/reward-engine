# 最小完整流程

这个仓库提供可运行的 Reward Engine 核心及两个原生 benchmark 适配器。
使用核心示例不需要恢复论文的历史数据，也不需要 Docker 或 GPU。

## 安装与离线运行

在仓库根目录使用 Python 3.11 或更新版本：

```sh
python3 -m venv .venv
.venv/bin/python -m pip install .
.venv/bin/python -m reward_engine.examples.minimal --output runs/minimal
```

安装后可从任意目录运行这个模块；输出路径相对当前目录。每次使用新的输出目录，避免覆盖先前记录。

示例调用实际 `reward_engine.core.outcome.loop.run_task` 控制器，以小型文本任务展示三轮构建与学习：

| 论文中的步骤 | 示例中的对应行为 |
|---|---|
| Jury construction | 依据任务与候选产物产生标准，并比较候选 |
| Reward Builder | 将标准转成可执行的检查，实际测量产物 |
| Hypothesis generation | 生成后续候选使用的改进建议 |
| Reward-guided critique/revision | 根据检查反馈继续修订候选 |
| Candidate selection | 按 Jury 比较和检查结果选择交付候选 |
| Learning / Memory | 保存本轮经验并传递到下一轮，共三轮 |

离线模式使用确定性角色和本地任务产物，输出会明确标记其模式。
它证明的是这些控制器路径可以执行，不证明语言模型能力、论文指标或原生 benchmark 成功。
原生 grader 与 Reward Engine Jury 是独立组件；该示例不调用 WorkBuddy 或 Claw 的 grader。

运行后查看输出目录中的：

- `summary.json`：运行模式、完成轮次、各轮选择、修订次数和 Memory 数量。
- `round_1/` 到 `round_3/`：Jury、reward pack、检查、反馈、选择和 checkpoint。
- `actors/`：同一候选修订前后的对话；`calls/`：角色请求与回复。
- `scope.json`：运行范围；失败时保留 `failure.json`，不会生成成功摘要。

默认离线示例特意生成需要修正的候选，每轮展示一次 B 修订。
真实模型若首稿已经满足要求，可以没有修订；摘要如实记录次数，不会强制制造失败。

## 真实模型

真实模型模式使用你提供的 OpenAI-compatible Chat Completions 服务，需要该服务的可用额度。
服务和所选模型必须支持非流式回复及 `response_format={"type":"json_object"}`；
本示例发送 `temperature=0` 和 `max_tokens=8192`。
模型回复必须满足角色协议；HTTP 错误或不合法的回复会使运行失败，不会回退到离线结果。
安装轻量 API 依赖，然后配置一个实际可用的模型。例如：

```sh
.venv/bin/python -m pip install '.[demo]'
# 在私有 .env 中填写 OPENAI_API_KEY，勿提交到 Git。
.venv/bin/python -m reward_engine.examples.minimal --live \
  --model YOUR_MODEL --base-url https://your-provider.example/v1 \
  --key-env OPENAI_API_KEY --credentials .env --output runs/minimal-live
```

也可设置 `OPENAI_MODEL`、`OPENAI_BASE_URL` 和 `OPENAI_API_KEY` 环境变量。
显式模型名称和地址参数优先于环境，环境优先于 `.env`。
省略 `--credentials` 时读取当前目录存在的 `.env`；也可完全使用环境变量。
通过 `--key-env YIBU_API_KEY` 等选项使用已有凭据变量，并显式传入相应的 `--base-url`。
同一模型用于本示例的所有角色席位，不能据此声称已验证多模型评审效果。
完整参数见 `.venv/bin/python -m reward_engine.examples.minimal --help`。
百炼 Token Plan 的端点与原生多模型配置见 [百炼接入](bailian.md)。

不要将填入密钥的 `.env` 提交到 Git。模型调用可能产生费用；模型名称应使用你的服务实际支持的名称。

## 原生 benchmark 与论文范围

需要真实工具环境和官方评分时，按 [原生环境安装](native-smoke.md) 恢复 benchmark 资产并配置运行环境，
再使用 [三轮 pipeline 验证](pipeline-validation.md)。Actor 加 native grader 的烟测只验证运行接口，不能代替 Reward Engine 完整流程。

默认三轮流程覆盖 Jury、Builder、Hypothesis、反馈修订、选择与 Memory，与论文的核心方法对应。
最小示例简化了任务环境和角色模型配置，不用于复现论文的实验数值。
论文描述的扩展对抗 QC、反例修复和跨轮元学习能力有独立或实验性路径；默认流程不保证执行这些可选模块。
不要将默认流程或本示例的完成解释为所有可选算法组合已通过真实模型验证。

历史结果与证据恢复见 [发布说明](../release/README.md)。它们不是运行最小示例的前置条件。
