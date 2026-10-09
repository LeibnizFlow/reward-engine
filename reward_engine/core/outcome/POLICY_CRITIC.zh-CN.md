# Outcome Runtime Policy Critic

Critic 的明确目标是推动 Actor 交付满分 Outcome：目标作品 → 当前差距 → 优先动作 → 实际落实 → 再评估。共用 `improvement.py` 的 `outcome_goal`、`full_score_gaps`、`improvements`；每项行动包含稳定 ID、关联差距、具体改法、预期贡献、验证方式和证据句柄。正确性检查只是一部分，不能因为最低要求已满足就放行。每次优先最多三项，完整剩余差距保留。

JSON／结构错误重做当前操作一次，带同一任务及已补读证据；两轮补读累计最多六个来源，必要时一次收束。重试与补读共用检查总时限，初判不再被固定截成一半时限。


Actor 每推进几轮，Critic 看“高质量交付还差什么”，给能指导接下来几轮的建议，不替 Actor 干活。

阶段性说明仍使用以下两个字段，并与满分差距和行动清单一起输出：

- `global_assessment`：当前阶段要达成什么，还差哪个关键结果。
- `local_guidance`：接下来几步可做什么／该问什么，以及看到哪些结果应继续、调整方向或求助。

这三件事是模型的输出要求，不是代码强制的行动清单；每次介入都对照新状态更新建议，不能把旧计划当事实。无有用建议可 `observe`，不强迫发言。

## 数据与调用

`PolicyCritic(task, pack, registry, memory=..., config=...)` 接收当前 Outcome Pack；没有第二套 Criteria/Jury。调用 `await critic(candidate, turn=3, intent=...)` 返回结构化判断及 `feedback`。

输入是 Task、实际用户补充、Actor 近期发言、真实工具观察、按相关性检索的 Battle Memory，及可选探索意图。只读运行中证据，不读官方 Grader、隐藏答案或内部推理文本。

Rail 不再把最后一句旁白塞进 `artifact.final_response`：`actor_updates` 按回合记录本 checkpoint 期间的发言，明确只是陈述／计划。工具请求与真实结果保存在 `source_documents`，附回合；environment 只标观察范围与计数，不冒充独立工作区扫描。成功写入不等于内容正确；实际文件正文没被观察到时应读取／检查，而不是猜测。没有收集到最终答案时缺少 final_response 是正常现象，绑定这个字段的检查返回 unknown。

默认完整内联最近6条工具记录，总预算16 KiB；超过预算的记录仍有预览和完整证据句柄，沿用按需读取，不把 preview_only 改成假完整。所有原始记录仍落盘；不自动扫描整个工作区。一般 Harness／终局上下文不因此扩大内联范围。

Pack 中尚未覆盖的问题通过同一个 `question_context` 投影传入，与 Hypotheses／交付检查一致：保留问题、来源、适用性与未决原因；长解释可按句柄补读。它们是待核实线索，不是额外用户要求或已证实缺陷。Criteria 仅去掉提案修订账本，标准、anchors、适用条件及来源仍保留。

1. 一次模型调用：全局判断、局部建议/澄清问题，以及零到少量 test_id。
2. 有疑点才用已有注册表执行所选 Harness；默认最多三项，无疑点不测。
3. 有测试再用一次调用汇总实测结果。输出 observe / suggest / clarify；反馈只给同一 Actor。

不做运行时 A/B，不计算 Before/After Process Reward，不因缺口新增裁判层。建议不改用户权限，探索方法不是额外要求。测量收据供交付前检查参考，旧 pass 不替代终局检查。

## 轻量进展与超时

每个 Critic 实例只保留最近两次 checkpoint 的回合、阶段目标、建议、缺口和检查摘要，下次作为 `prior_feedback` 提供，并保存 `turn_N_progress` 日志。它不是第二套 Battle Memory，也不是自动完成清单。历史建议是否注入、采纳、完成不能从“已生成建议”推断；必须看新观察。新 trial 使用新实例，不继承状态；不新增跨进程恢复系统。

检索的 `query_extra` 显式携带当前工作／近期观察／探索意图；不再把草稿冒充 `context.gaps` 或在评价 Prompt 重复插入。

共用一个总截止时间：初次判断及其恢复使用剩余总时限，测量最多用随后剩余预算的一半，其余留给汇总。没有完成测量时出错，返回 unavailable；测量已完成而汇总超时／校验失败，返回 `degraded` 和程序整理的真实检查摘要。不沿用可能被测量推翻的初步建议，也不把错误改成成功。

缺陷引用 `runtime_check` 必须有无执行错误的主项 fail，或独立有效的局部 `inquiry_result.fail`。主项 pass／unknown 不抹掉有效局部失败；纯 unknown／执行错误不能证明产品有错。降级摘要保留这两种结论，Actor 收到真实事实及限制，不是编造的完整建议。

## 周期调度

请求默认上限仍为160,000字节，可显式设置 `policy_critic.max_request_bytes`；正式入口同步核心与传输层。Flash thinking 配置生成器设262,144字节，T068 full02 及后续恢复／cd01 配置显式设524,288字节，以各次冻结配置为准。容量拒绝记录实际 UTF-8 字节数、上限、阶段和各字段体积，明确该请求未调用模型，不混成模型能力失败。

汇总请求保留所有真实测量结果，只把重复的 `outcome_coverage` 草稿明细改成索引：预期、来源、优先级、状态与引用仍可见，完整账本通过 `outcome_coverage_details_ref` 沿用上述有界按需补读。返回值、检查回执及日志仍是原始完整账本。长未决问题同理：问题和重要性不删，重复处置理由放在现有 `details_ref`；没有额外模型总结或新增测试。

```json
{"policy_critic": {"enabled": true, "every_k_turns": 3, "max_tests": 3, "timeout_s": 120, "top_k": 3}}
```

上例显式使用 k=3。新 [四臂协议](ARMS.zh-CN.md) 的缺省频率已改为 k=1，显式 k=3 保留；仅 D 使用 Critic；其终局补查复用同一 Critic，不叠加 C 的 DeliveryReview。B/C/D 初始 Hypothesis 相同，各自生成独立作品。旧 `legacy_abcd` 配置缺省 `enabled=false`、k=3，显式启用作用于旧 B/D。Critic 模型默认 `learning_model`，可用 `critic_model` 独立配置；不写密钥。Embedding仍默认关闭。

根 Actor 完成的模型回合计数；一回合多个工具仍只计一次，Critic调用、模型失败和重复回调不计。第 k/2k/3k 轮有工具行动时，工具完成后、下一模型请求前介入；不撤销或覆盖工具结果。最终无工具答案不做周期检查；在原生交付桥中始终由同一 Critic 审查实际终稿，运行中的计划／局部审查不能替代这一步。有意见则最多续写一次并复查。是否进行交付检查由臂协议决定（C 使用 DeliveryReview；D 的终局补查仍使用 PolicyCritic）；日志必须如实记录。计数随同一会话持续。

Jiuwen 接线是 `reward_engine.adapters.jiuwen_outcome_rail.OutcomePolicyRail`。真实反馈通过 `model_context.add_messages` 写入，assistant 消息显式携带空 `reasoning_content`，标记为内部建议而非用户消息；只改预览 messages 不够。冻结的 Actor SDK 会在序列化时丢掉这个空字段，因此官方 DeepSeek thinking 请求还在 `claw_eval/actor_wire.py` 的 HTTP 出口补全缺失字段。已有推理内容不覆盖；补全前请求与最终 wire 均留档，原生回合预算仍执行。其他端点和 non-thinking 不应用此变换。其他 Runtime 可直接调用相同核心，尚不表示已适配验证。

## 容器与验收

原生 Claw容器使用可选 `rail_extension`：控制器创建每trial的认证回调，容器只得到临时trial token。provider密钥不进入Actor。需要外部 Claw Runtime/worker 的薄扩展，缺失会在启动付费试验前报错；补丁见 `reward_engine/adapters/claw_eval/companion/`。

`tests/integration/test_outcome_policy_critic.py` 验证计数、动态测量、证据/记忆引用、真实会话注入及错误降级；`tests/integration/test_claw_outcome_critic_progress.py` 验证输入真实性、完整记录预算／补读、两次状态、trial隔离、检索分离、超时保留事实和失败引用。详细运行记录见实验 state；接线通过不等于已证明 benchmark 提分。
