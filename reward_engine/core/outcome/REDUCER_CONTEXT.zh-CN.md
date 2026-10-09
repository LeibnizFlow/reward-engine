# Jury 修订与 Reducer context 协议

正常 Outcome loop、validated construction 和 Engine Jury 入口已启用
`revision_protocol=True`。低层 `jury.propose` / `_propose_once` 默认仍使用旧协议，
供显式历史恢复和既有调用方使用；新的调用方应明确传入该选项。

## 第二阶段输出

第一阶段继续只根据 Task、实际用户补充及允许的独立资料提出预期，不读取产物或历史方法。
第二阶段读取自己的 `independent_expectations` 与产物，返回完整修订稿及 `revision_decisions`：

- `retain`：原样保留；整条 criterion 可用 `{"inherit_from":"expectation_id"}` 继承。
- `rewrite` / `merge`：返回完整目标条款，同时给出理由与已存在的证据引用。
- `delete` / `defer`：保留原条款供 Reducer 审核；删除建议需要理由和证据。
- `add`：新条款需要说明依据。缺少声明的新增内容明确列为 `unclaimed_additions`。

每个 decision 用原 `source_parts` 和目标 `criterion_id` / `part_index` 建立对应。
代码逐项检查文字、父级语义、条件、优先级、来源及验证方式是否一致；改变这些内容不能使用
`retain`。第一阶段未得到处置的条款自动列为待决。即使所有产物都遗漏同一要求，也不能据此删除它。
第二阶段的删除、修改和新增均为成员建议，只有 Reduce 的校验后结果能成为标准。

## Reducer 输入

保留 Task、实际用户补充、当前标准、修订依据、争议、未决原要求、检查结果与不确定性。
已验证完全相同的保留项使用 selector alias；代码在输出中展开回每个原始 part ID。
未修改原文存于完整 checkpoint；改写、删除和未决项的原文仍在模型输入中供审查。

长字符串只做逐字相等的共享：`{"$reducer_text":"t0"}` 指向 `shared_texts.t0`。
父级范围、身份、条件及决策对象保持独立；不通过近义词、大小写归一化或 embedding 推断等价。
编码可逆且仅在包含说明和引用表的完整请求更小时采用。Adapter 的修订建议保持 literal。

历史 memory 的 `method` 正文改为可补读的索引，保留其字节数和 SHA-256。
适用范围、观测变化、状态、限制及未知字段仍内联。重复的 Task 前缀用原 Task 引用表达。
Reducer 可用既有 `need_evidence` 请求完整历史记录。这些本地导航引用不是最终标准的依据；
必须引用原 Task、product 或 check。导航记录补读返回全文，随后接受整个请求的字节上限检查；
不会截断后冒充全文，也不会提供外部 document.read 无法访问的假全文入口。

不增加 LLM 压缩调用，不增加证据补读轮数或引用数。现有普通修复和 Office 定点修复边界保留。
无法在预算内保留的请求明确报超限并保留 checkpoint，不暗中截断独有要求。

## 恢复、审计与兼容

Map 的缓存身份包含 `outcome-member-revision/2026-09-15-v1`，Engine 完成标准的身份还包含
`outcome-reducer-context/2026-09-15-v1`。第一阶段 prompt 和 scope 不变，允许复用旧的独立草稿缓存；
旧的完成标准不会被当作新协议结果复用。旧 checkpoint 没有 revision 标记时仍通过旧 Reduce 协议恢复，
可使用现有显式 `qc_recover_jury` 路径。已有 21 题产物与评分不改动。

完整 revision 稿保存在 `revisions/<seat>` 和 proposal/checkpoint 中。Reward Pack 原有 criteria、tests、
coverage 等下游 schema 保持兼容。`reducer_request_projections` 累计记录每次请求准备的大小、
字段大小、SHA-256、证据阶段、修复序号和是否在预算内；普通 Reduce 重试按 attempt 分文件，
Office 定点修复在同一个累计列表中保存。超限而未发送的条目由 `within_budget=False` 标识。

## 离线验证

离线工具：

```bash
python -m experiments.reward_engine_training.analyze_reducer_context \
  --manifest /path/to/combined21/results.json \
  --output /path/to/reducer-context-report
```

2026-09-15 对 21 题的 131 个已保存 Reduce/修复请求逐一投影：累计 JSON 字节从
28,169,825 降至 23,481,474，减少 16.64%；中位数从 203,103 降至 177,774，
P95 从 400,686 降至 302,631。包含历史请求链中的 4 次证据补读和 62 个带修复信息的请求。

该测量验证全文共享可逆、memory 原文可定位；不调用模型，不重新评分。
它只测确定性投影，不包含新修订稿的潜在收益，也不预测新请求链的补读次数。
字节下降不能直接推导 token、时延或 Reward Pack 质量提升。
