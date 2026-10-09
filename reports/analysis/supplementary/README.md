# 已完成的补充分析

这些分析来自同一组实验的冻结记录。按实际分析条件分别保留，不作为额外独立主实验，不改写第三轮主结果。

| 条件 | 入口或报告 | 口径 |
|---|---|---|
| 后续原生 seed 评分 | [150 份评分](../../workbuddy_office/seed_grading.md) | 只评分已有交付物，未生成新 actor；后续 grader 与环境差异保留 |
| 六次基线 | `--analysis six_baseline` | 三个历史 seed 与三轮 A 的均值；WB 阈值 80，Claw 阈值 75；两组三次的成功事件分别平均 |
| Claw 三轮与最终轮 | `--analysis claw_continuation` | 保留最终轮端点、三轮描述性汇总及已知暴露敏感性 |
| 联合评分离线选择与校准 | [报告](joint_offline.md) | 原始与修复版本分别分析 |
| 已知 criterion 分母 | [报告](known_reward.md) | P/(P+F)，unknown 不计零 |
| 冻结候选分差诊断 | [报告](frozen_gap.md) | ≥2/3/5/10；不替换早期主阈值 5 的分析 |
| 平局计正分 | [报告](tie_credit.md) | τ₊=(C+T−D)/(C+T+D)，不是标准 Kendall τ-b |
| Jury 分差诊断 | [报告](jury_gap.md) | 已记录 Jury 排序，按轮次与累计候选池分别统计 |
| 缓存评分与选择 | `reward_agreement`、`claw_agreement`、`task_local`、`utility` | 以各自冻结候选和缓存判断为准 |
| 胜平负与阈值敏感性 | `win_tie_loss`、`thresholds` | 保留历史端点与原有阈值定义 |

统一复现入口：

```sh
python -m experiments.analysis.supplementary --analysis six_baseline --data-root data --output /tmp/six-baseline
python -m experiments.analysis.supplementary --analysis tie_credit --data-root data --output /tmp/tie-credit
```

输出目录必须是新目录。工具先验证公开输入字节，再恢复其原始身份用于历史哈希断言；只在输出副本中执行冻结的统计代码。数值算法不变，论文版式文件及已停止研究的“文件未变化”检查不属于科学输入。独立复算已核对 13 项分析、15,093 个数值，见 [验证记录](../../../release/supplementary_validation.json)。

六次基线与三轮汇总仅在其自身分析中保留原作者披露的一处 WB 无效首轮 A 的零填充；主实验原生回执与 1,386 个分数保持不变。历史 WB 阈值 85 与后续阈值 80 的结果不能混成同一口径。

冻结输入、统计表及回执位于 `data/analysis/supplementary/`。五组完成的开发性 probe 位于 `data/analysis/probes/`，按协议分开，不与主实验合并计数。已停止的排名扩展、旧构造 probe 和论文中间稿只在本机 archive 留存；完成的 Jury 诊断只纳入其必要的候选身份与排序记录。
