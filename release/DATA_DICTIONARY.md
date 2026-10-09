# 数据字典

| 字段 / 文件 | 含义 |
|---|---|
| `experiment.json` | 单个主实验的任务集合、全部轮次、协议说明和必要证据列表 |
| `task` | benchmark 原任务 ID；中英文任务版本保留原身份 |
| `round` | 算法轮次 1–3，不是运行批次 |
| `A`, `B`, `C` | 百分制原生评分；有效零分保留，缺失不填零 |
| `selected_arm`, `selected_score` | 运行时实际选择及其原生成绩，不按官方最高分重选 |
| `A_mean_round1` | Claw 第一轮 A 与三个 seed 的固定事后均值，不是第三轮候选 |
| `configuration`, `code` | 实际配置文件和执行代码目录，使用内容哈希标识 |
| `evidence.<arm>` | 对应评分、执行、轨迹和交付物的公开路径 |
| `required_evidence` | 重建主报告必须存在并通过哈希验证的文件 |
| `source_sha256` | 规范化前的原始内容身份 |
| `sha256` | 当前公开文件的实际 SHA-256 |
| `source_canonical_sha256` / `public_canonical_sha256` | 原始/公开 JSON 按固定序列化规则计算的身份，用于历史协议绑定 |
| `attempts/` | 已记录失败或身份修订的证据，不作为额外成功样本 |
| `transport/` | 请求、规范化流式响应、usage、错误和传输回执 |
| `analysis/joint_original` / `joint_repaired` | 同一固定 462 个候选的原始联合评分与既有技术修复 |
| `analysis/score_gap` | 相邻官方分差主阈值 5、敏感性阈值 2/10 的固定三候选诊断 |
| `analysis/midpoint` | 对已有 unknown 使用 `(P + 0.5 U) / K` 的事后重计算，不是新的测量 |

正文中的候选 ID、模型名、实际参数和输入内容不会因统一目录而合并。相同产物的独立执行仍有独立身份。
关联分析使用其冻结时的候选与 Pack，不用后来完成的主实验候选替换。
