# Reward Engine

从任务与候选产物构建 Jury 和可执行奖励检查，再通过反馈修订、候选选择和记忆更新运行三轮学习。仓库同时提供 WorkBuddy Office 与 Claw-Eval 的原生适配器及历史实验材料。

## 快速开始

Python 3.11+，无需 Docker、API key 或下载历史实验数据：

```sh
python3 -m venv .venv
.venv/bin/python -m pip install .
.venv/bin/python -m reward_engine.examples.minimal --output runs/minimal
```

该入口运行实际核心控制器的三轮流程，使用明确标记的确定性角色与本地文本任务。它用于理解流程和检查安装，**不是真实模型或 benchmark 验证，也不证明质量提升**。真实模型配置、输出说明与论文对应关系见 [使用指南](docs/quickstart.md)。每次运行使用新的输出目录。

- [最小完整流程与 API 配置](docs/quickstart.md)
- [三轮原生 pipeline 验证](docs/pipeline-validation.md)
- [原生任务烟测与环境安装](docs/native-smoke.md)
- [本次安装与流程验证记录](docs/usability-validation-2026-10-14.md)

## 历史实验

两个实验的最终结果、完整证据与离线复现入口。数据按实验、任务、seed、算法轮次和 A/B/C 组织；这些历史结果与上述最小示例分开记录。

| 实验 | 任务 | 轮次记录 | 有效分数 | 最终报告 |
|---|---:|---:|---:|---|
| WorkBuddy Bench Office | 50 | 150 | 450 | [HTML](reports/workbuddy_office/report.html) · [Markdown](reports/workbuddy_office/report.md) |
| Claw-Eval | 104 | 312 | 936 | [HTML](reports/claw_eval/report.html) · [Markdown](reports/claw_eval/report.md) |

主结果均为第三轮。真实模型、搜索设置和协议差异保留在逐任务配置中，整体均分不能解释为完全同质配置的对照实验。
关联分析是固定历史候选上的回顾性分析，原始联合评分、技术修复、官方分差子集与 midpoint 重计算分别保留。

- [发布及离线复现](release/README.md)
- [关联分析](experiments/analysis/README.md) · [报告](reports/analysis/README.md)
- [数据字典](release/DATA_DICTIONARY.md) · [第三方许可](release/LICENSES.md)
- [WorkBuddy 运行入口](experiments/workbuddy/README.md) · [Claw 运行入口](experiments/claw_eval/README.zh-CN.md)
- [架构](docs/architecture.md) · [测试](tests/README.md)
- [原生任务烟测与环境安装](docs/native-smoke.md)

GitHub 源码导出不包含大数据。附件在私有 [evidence release](https://github.com/TheLittleJimmy/reward-engine/releases/tag/evidence-v1.0.0) 中；下载到 `zenodo/` 后运行 `python -m release.restore --archives zenodo` 得到 `data/`，报告中的完整证据链接才能使用。恢复工具不会用历史源码包覆盖当前代码。
本机 `archive/` 保存历史材料及回滚记录，不属于发布内容；公开版本不依赖它。

```sh
python -m release.verify --data-root data
python -m release.report --experiment workbuddy_office --data-root data --output /tmp/workbuddy-report
python -m release.report --experiment claw_eval --data-root data --output /tmp/claw-report
```

代码沿用 Apache-2.0；自有数据和报告的许可证、正式署名尚待作者确定，见 [发布状态](release/metadata.json)。第三方材料遵循各自原许可。

仓库外独有材料也已纳入：[新增材料说明](release/EXTERNAL_MATERIALS.md) · [补充分析](reports/analysis/supplementary/README.md) · [150 份 seed 原生评分](reports/workbuddy_office/seed_grading.md)。
