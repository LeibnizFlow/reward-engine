# Reward Engine — WorkBuddy Office 与 Claw-Eval

两个实验的代码、最终结果、完整证据与离线复现入口。数据按实验、任务、seed、算法轮次和 A/B/C 组织。

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

GitHub 源码导出不包含大数据。将 Zenodo 材料包全部解压至源码根目录后得到 `data/`，报告中的完整证据链接才能使用。
本机 `archive/` 保存历史材料及回滚记录，不属于发布内容；公开版本不依赖它。

```sh
python -m release.verify --data-root data
python -m release.report --experiment workbuddy_office --data-root data --output /tmp/workbuddy-report
python -m release.report --experiment claw_eval --data-root data --output /tmp/claw-report
```

代码沿用 Apache-2.0；自有数据和报告的许可证、正式署名尚待作者确定，见 [发布状态](release/metadata.json)。第三方材料遵循各自原许可。

仓库外独有材料也已纳入：[新增材料说明](release/EXTERNAL_MATERIALS.md) · [补充分析](reports/analysis/supplementary/README.md) · [150 份 seed 原生评分](reports/workbuddy_office/seed_grading.md)。
