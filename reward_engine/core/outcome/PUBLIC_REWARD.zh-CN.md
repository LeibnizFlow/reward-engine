# 公开要求驱动的独立 Reward v1

这是显式启用的 `public-task-reward-v1`，供单轨迹独立评估。旧 Pack、历史结果和运行时选优逻辑不迁移。首版提供真实公开事实检查、可审查的要求绑定、分级聚合与离线入口；没有新增模型实验，也没有证明相关性提高。

## 接入与契约

1. 使用现有 `build_context(task, [candidate])` 建立单候选公开上下文。
2. 可信 Adapter 调用 `public_facts.bind_observations(context, evidence, descriptors)`，冻结实际观察。每条 descriptor 必须有来源句柄、JSON 路径、对象、属性、事件序号、采集时间、阶段、覆盖完整性和执行状态。数值从实际来源读取，不能提交一个期望值冒充观察。
3. 调用 `public_reward.with_reward_protocol(pack, requirements, context, evidence)`，返回新的 Pack。每个原 criterion 必须标为 `quality`、`gate` 或 `advisory`。函数不改输入，保留 parent Pack ID，并对新 Pack 内容生成身份哈希。
4. 正常调用 `harness.measure()`；新 Pack 自动注册 `public.fact`，并在原测量结构旁返回 `independent_reward`。原有 registry 不启用新协议时不会改变。

`requirements` 是**经审阅的显式绑定**，不会从 oracle 标签反推，也不会自动升级现有全部 criterion。每项计分要求有公开来源的原文引用和 obligations；每个 obligation 绑定同 criterion 的测试 ID，或提供一个自动编译为测试的 `fact`。可以同时绑定事实测试和语义测试，任何已执行的实质失败都不会被另一项 pass 覆盖。

公开来源仅接受原 Task、原 Task 附件和当前候选的真实用户补充。精确引用校验只能证明来源存在，**不能证明规则解释正确或引用蕴含所声称的限制**，这仍需审阅。额外建议标记 advisory，不计分。`gate` 必须给出 prohibition/delivery_prerequisite 分类及理由，不从“重要性”自动生成。

`scope.delivery_channel` 记录交付边界；实际验证必须通过对应对象的 obligation 实现，不能只填写文字。`scope.conditions` 是带公开来源的事实谓词；条件已知为假时不适用，条件未知时不强行计分。涉及日期时，`scope.reference_date` 保存 `{value, sources}`；缺失、引用冲突或值不一致保持 unknown，不取机器当前日期。任务附件或用户补充发生变化后，旧要求绑定变为 unknown，需重新审阅编译。

## 事实与时间

`public.fact` 支持结构化 equals、contains、forbids。成员比较精确匹配，不把 basename、子串或别名视为同一交付路径。描述中的 object/property 由可信 Adapter 明确映射到观测源；它们不是通用自然语言解析器。

- terminal：同对象/属性采用最新序号的终态。只有中间结果、终态之后又有中间观察或终态顺序冲突时保持 unknown。
- history：已观察到的禁止行动持续成立；后续删除或诚实披露无法撤销。证明“没有禁止行动”需要完整历史，且完整历史不能早于更新的观察。
- 部分清单可以证明某项存在，不能证明某项缺失。执行失败与证据不可用分别记为 execution/evidence unknown。
- 观察与完整公开输入一起绑定候选身份；更换来源或候选后禁止复用旧绑定。采集阶段、事件序号与完整性由可信 Adapter 声明，哈希不是对这些声明的真实性证明。

`adapters.public_reward_snapshot.inspect_snapshot()` 可检查精确文件位置、SQLite 表列。只读取指定公开快照及指定文件；已存在文件需匹配归档 SHA-256，拒绝路径穿越、符号链接和未 checkpoint 的数据库。SQLite 使用只读 immutable 连接，不创建 sidecar，不执行候选脚本。缺失路径只有在明确完整的快照中才能证明不存在。没有公开终态快照时保留 unknown，禁止借用 grader 私有快照或重跑候选后回填“原终态”。

## 标量、区间和训练

检查器仍输出 pass/fail/unknown。criterion 从明确列出的 obligations 得到：全满足 full=1；全违反 none=0；同时有满足与违反 partial=0.5。partial 输出实际 satisfied/missing 及证据，不从回答风格推断。未决 obligations 给出可能的分档区间；已知满足和违反同时存在时，剩余未知不改变 partial 档，但仍被记录。

普通 criterion 等权得到 Q。先在 criterion 内汇总，再计算 Q，增加测试或拆分 obligation 不增加该 criterion 的总权重。拒绝重复 ID、完全重复的 obligation 描述和同一测试重复计分；语义重复仍需审阅。

- 任一明确硬约束失败：R=0，已知失败可覆盖其他未知。
- 硬约束全部通过：R=0.5+0.5Q。
- 影响结果的 unknown：`reward=null`，保存 `r_min/r_max`，`training_mask=false`。未知适用性使用保守区间，不能填零或当 pass。
- 没有普通 criterion 但有效硬约束全部通过时 Q=1；全部要求都不适用时不产生满分，保持未定。

这是 v1 的排序编码，不是通过概率。结果保留每项 scope、来源、satisfied/missing/unresolved、原测试和证据。构造上可能真实同分，不人为打破平局。

## 零模型离线入口

仓库内提供一个**合成**的“只起草却发送”示例：

```sh
python -m reward_engine.adapters.offline_public_reward \
  --input tests/fixtures/public_reward/draft_only.json \
  --output /tmp/public-reward-draft-only-receipt.json
```

预期 `measurement.independent_reward.reward=0`。输出必须是新路径，已有回执禁止覆盖。入口只运行 public.fact 和 json.parse，不导入生成的检查器执行配方；未执行的语义测试列在 `unexecuted_test_ids`，保持 unknown。所有公开输入哈希、协议 Pack 和实际回执保留在输出中。

输入可选 `snapshot`：`root` 指向已归档公开工作区，`captured_at`/`sequence` 给出原始采集信息，`complete_inventory` 明示覆盖范围，`requests` 为 `{path, kind: exists|sqlite_columns, sha256, table?}` 列表。缺失路径的存在性检查无需 sha256；已存在文件必须匹配 sha256。采集身份与清单是否完整需由调用方核对，不能任意把当前宿主目录标成历史终态。

## 回归与效果验证边界

```sh
python -m pytest -q tests/unit/test_public_task_reward.py
```

测试使用公开要求的合成回归数据，覆盖路径精确性、只起草与发送、历史成功和终态回退、日期冲突、执行错误、部分证据、分档与权重、当前用户来源隔离、输入绑定以及只读 SQLite/CLI。测试替身不代表真实模型会判对。

已有八对审查案例作为开发问题清单：两例路径错误、schema migration 终态缺口、customer followup 日期/渠道差异、VIP 草稿越权、onboarding 权限范围差异、priority sync evaluator 异常和 contract extract 抽取分歧。其中 oracle 异常不能变成 reward 失败；权限含义和表格抽取争议仍需原始公开证据审查，不能写成“已修复”。

后续独立效果验证固定候选和模型，分别比较旧协议、仅更正证据/判定、新聚合三个阶段。主报公开要求的误放行、误拒绝与覆盖率；辅报 oracle 排序相关性、逆序/平局及选择损失，并单列异常 oracle、保持全样本结果。八对开发案例不得冒充留出集；历史 Pack 看过的候选也不能冒充未见泛化。首版未启动此类付费评估，不能声称这些指标已经提高。
