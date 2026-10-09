# 四臂：分别比较 Hypothesis 与 Reward 反馈

协议名：`hypothesis_feedback_v1`。实现：`arms.py`，两个原生入口和 `loop.py` 共用，不另写一套算法。

2026-09-11 新验证配置使用 `construction_strategy="counterexample_validated"`，不更换协议名。
它先验证 Reward/Hypothesis 的反例，再运行四臂；C/D 共用终局审查控制器，
D 保留运行中反馈且终局只允许一次修订。建议经过证据准入，版本分别保存并在匿名排序和关键检查后选用。
原生最后回复保持原样，`selected_candidate`／`delivery_selection` 记录选用结果。
详细规则和新配置见 [当前策略](README.zh-CN.md)。下文三轮与旧停止状态仅描述兼容配置。

| 臂 | 初始输入 | 额外反馈 |
|---|---|---|
| A | Task | 无：Original Actor |
| B | Task＋Hypothesis | 无 |
| C | Task | 交付前 Reward 检查 → 同会话修订 → 复测 |
| D | Task | 运行时 Critic（默认每 1 个根 Actor 回合）＋同一 Critic 审终稿、必要时续写一次并复查 |

仅 B 使用本次生成的 Hypothesis；C/D 共用冻结 RewardPack＋HarnessPack，不接收 Hypothesis 或历史 Battle Memory，保留自身当前会话的反馈与修订记录。
C/D **不额外注入静态 Reward Prompt**，D **不叠加 C 的交付修订循环**。
四臂分别生成独立作品，C 不接着修改那份 B 成品。主结论 `target_satisfied` 且零修订只说明 C 最终作品等于 C 自己的初稿，不代表等于独立生成的 B。
B−A、C−A、D−A 是相应处理组的样本成绩差；其中混有生成与评分波动，单次差异不能直接归因于假设或反馈。C−B、D−B 同时改变了 Hypothesis 与反馈条件。若要测同一初稿的修订收益，应保留 C 初稿／终稿及 D 终局修订前／后作品作配对评估，不能把独立四臂追认为配对实验。

## 配置与启动

在新的任务／模型配置中设置以下字段，其余沿用有效的原生配置，不改旧运行记录：

```json
{
  "arm_protocol": "hypothesis_feedback_v1",
  "arms": ["A", "B", "C", "D"],
  "rounds": 1,
  "construction_rounds": 3,
  "max_revisions": 2,
  "max_tests": 3,
  "actor_harness": {"enabled": false},
  "policy_critic": {"enabled": true, "every_k_turns": 1}
}
```

- 完整初始化入口：`reward_engine.adapters.claw_eval.outcome_experiment`。
- `construction_rounds=3` 仅扩展 Jury、Builder、Hypothesis 构建；搜索、Actor、C/D 审查频率及独立评分次数不乘三。未指定仍为单轮构建。
- 已有同题 Pack／Hypothesis：`reward_engine.adapters.claw_eval.outcome_pack_trial`，传 `--config --pack --hypothesis --output`。
- 新协议默认四臂、一轮、`every_k_turns=1`；显式填写的历史 `every_k_turns=3` 仍使用 3，旧协议默认也仍为 3。若包含 D 却显式关闭 Critic，启动前报错。
- C 默认最多两次修订，实际调用同一个 Actor 的 resume，不新开一次“重抽”。仍未解决就记录未解决，不能强称通过。
- C 的“最多 3 条主张”是输出简洁性指导。额外主张完整保留、逐条校验并记录数量提示；实际执行检查仍由 `max_tests` 限制，超出部分待测。字段／证据契约错误记录具体位置，不因检查故障要求 Actor 任意重写；主结论统一针对满分 Outcome；任何质量差距都强制归为 `needs_improvement`，重要疑问待测归为 `needs_verification`。事实是否有缺陷单独记录，不能替代满分判断。只有满分差距与优先改进均为空、重要疑问已处理时才能正常结束。JSON／输出契约失败带原证据重做一次。
- D 在第 k 个根模型回合确实调用工具后、下一次根模型请求前介入。无论运行中是否已有有效审查，终局都由同一 Critic 检查实际作品；有改进意见则最多一次同会话续写和复查，不强制工具次数。运行时和终局收据分别统计。
- Hypothesis 明确给出 keep 时，**只跳过 B**；C 继续 Actor＋交付前检查修订，D 继续 Actor＋运行时 Critic。记录 `hypothesis_applied=false`；此时比较 C−A、D−A，没有 B 成绩，不计算 C−B／D−B，四臂整体标 partial。完整初始化入口若因 Map／Reduce 调用失败返回 `availability=unavailable`，则在采样前停止并保存恢复状态，不能把调用失败当作语义 keep。手动已有 Pack 入口仍允许显式不提供 Hypothesis 来做 A/C/D 对照，不冒充完整四臂验收。
- 新协议不另给 Actor 主动 Harness 工具；测量由 C 交付检查／D Critic 按需选择。旧 `actor_harness` 实验保留原协议。
- 四路独立 fresh 环境，同模型／基础预算；反馈有额外成本，保存各自请求、检查、修订、耗时和原始轨迹，不能称等成本对比。
- 官方 Grader 只在采样关闭后独立验收，不进入 Hypothesis／Reward／Critic 输入。

旧配置缺省仍是 `legacy_abcd`：A 原始、B 静态 Reward＋交付检查、C Hypothesis、D 两者组合；旧的可选 Critic 开关不变。
新协议必须显式选择，历史 T068 分数不能按新四臂重命名。早期同名 `hypothesis_feedback_v1` 的 C/D 曾接收 Hypothesis；归档请求与源码保留，当前维护版已移除此输入，不增加 v2。

## 验证边界

`tests/integration/test_claw_outcome_feedback_arms.py` 覆盖新旧协议隔离、Hypothesis 输入隔离、四路并发、C 真解析失败后同会话修复并通过复测、D 独立运行 Critic 并调用真实解析器，以及无方法时不伪造处理组。
另由 `tests/integration/test_outcome_policy_critic.py` 验证 Jiuwen 回合计数／反馈注入／失败继续执行。
`feedback_execution` 在原生产物记录、轮次摘要和最终状态中分别记录 `delivery_review` 与 `policy_critic`。状态包括 `disabled`、`completed`、`degraded`、`unavailable`、`unknown`；D 另有 `not_triggered`。C 保留实际审查结论、有效／不可用审查数、修订请求数与诊断；D 保留根回合数、检查点数、失败检查点数和实际注入次数。`completed` 指模块执行完成，不表示作品正确；Critic 有效 `observe` 也可以零注入。只有完整日志和正常交付能证明 D 未到周期检查点，缺失日志标 `unknown`。

既有 `status`／`completed_rounds` 表示产物完成情况；反馈链路是否执行、是否有效需要查看 `feedback_execution`，不能由四份作品存在或 Critic 开关开启来推断。
这些是受控模型的离线测试，不提供提分结论；实际付费四臂的进度、恢复和结果见 [T068 完整验收](../../../reports/claw_eval/outcome/flash_thinking_full_20260910.zh-CN.md)。
