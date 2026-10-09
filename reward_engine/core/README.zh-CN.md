# 算法核心

这里不启动 Agent，不读取实验目录，也不绑定模型服务商。
模型调用、采样、检查执行由调用者传入；算法返回结构化结果。

- [engine/](engine/README.zh-CN.md)：可训练 Reward Engine 的构建控制、版本账本、策略与外层训练协议。QC 的 SDK Agent 在 Adapter 中启动，按需调用 Jury、Builder、Quality Review、内部 Hypothesis、Search、Memory 和真实沙箱 Actor。Hypothesis 用于改进评价器；本轮不运行 A/B/C/D，也不要求各组件固定三轮。
- [outcome/](outcome/README.zh-CN.md)：Engine 复用的标准生成、Harness、反例核验、修复与测量算法。旧固定构建／交付实验入口仍保留，其中 `construction_rounds=3` 是旧实验配置，不能据此推断 QC 一定循环三轮。Task 定义目标，知识包仅补充证据与方法。
- `counterexample_validated` 路径使用 `outcome/validated_construction.py`：独立 Jury → Builder → 反例攻击、独立核验、必要修复与复查。`construct_reward` 可先单独产出 Reward Pack，完整 `construct` 复用同一函数后再构建 Hypothesis；模型服务商由 Adapter 注入。没有实际检查回执的草案判断只保留为待验证，不能自行变成已证实结论。
- `batch.py`：共享 MapReduce 并发与截止时间。
- `shared/`：JSON 快照、回复格式、证据过滤、Harness 数据定义与 Pack 编译基础件。
- `outcome/policy_critic.py`：既有运行时及终局 Outcome Critic。旧四臂实验协议保留以便读取历史结果，不是当前共享策略训练协议。

代码量随实验迭代变化，不能将早期约1,500行的统计当作当前规模。
旧 Criterion Loop 与旧 Runtime 协议已移到 `reward_engine/legacy/`。当前 Engine 复用 Outcome 算法，不依赖历史实现。
运行时接线见 [adapters](../adapters/README.zh-CN.md)。
