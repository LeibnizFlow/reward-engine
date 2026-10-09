# 最终关联分析

保留固定 154 任务、每题三候选的联合评分及既有技术修复，以及主阈值 5 的分差子集分析和 midpoint 重计算。
这些分析冻结了 462 个历史候选及其 Pack，不使用后来新增的 Claw 候选。原始与修复版不能混成一个未修订协议。
已有最终 Pack 可能见过部分轨迹，因此结果仅属于回顾性诊断，不支持未见轨迹泛化结论。

```sh
python -m experiments.analysis --analysis joint --data-root data --output /tmp/joint-analysis
python -m experiments.analysis --analysis score_gap --data-root data --output /tmp/gap-analysis
python -m experiments.analysis --analysis midpoint --data-root data --output /tmp/midpoint-analysis
```

原始分数、评分回执、技术 unknown、语义 unknown、有效零分和缺失分别保留。
默认种子固定、20,000 次 bootstrap；Claw 按 scenario 聚类，不把双语版本当独立任务簇。
midpoint 是对已有 unknown 的事后赋值假设，不是新的模型评分。
主报告与分析分别核对自身冻结输入；主报告后续完成不改变分析候选集合。

原模型判读实现与其协议哈希见 `data/analysis/code/`；当前入口仅做离线统计，不调用模型或 grader。
