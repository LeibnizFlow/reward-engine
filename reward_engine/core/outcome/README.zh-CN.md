# Outcome-first 简化 Reward Engine

独立单轨迹 reward 的显式可选协议见 [公开要求驱动的 Reward v1](PUBLIC_REWARD.zh-CN.md)：公开要求绑定、真实终态检查、硬约束及分级区间聚合。默认运行时选择与历史 Pack 保持原协议。

原生多轮任务使用 [公开 Task 范围与跨步骤规则](MULTITURN.zh-CN.md)：真实追问更新要求，
内部修订不增加用户指令，旧验证结论不跨范围沿用，作品只在当前范围比较和回退。

## 默认基础审查，反例审查需显式开启

所有当前构建入口默认 `adversarial_review_enabled=false`；缺字段同样关闭。
只有用户明确要求使用反例审查时，才为指定运行设置 `true`。配置放在实验配置顶层，
QC manifest 则放在 `runtime` 中；`construction_strategy="counterexample_validated"` 本身不授予权限。
普通构建继续执行 Jury、Builder 和基础完整性审查，跳过 Attack、独立参考和关联修复复测。
检查器可照常测量；未经反例核验的标准／检查保持 `unverified`，不获得案例支持或关键保护权限。
专用反例诊断入口（含 `qc_recheck_probe`、`outcome_validation_run`）未开启时在模型调用前报错。

Hypothesis 继续生成、选择和投影。关闭反例审查时，新建议记录验证已跳过、
`actor_projection.validated=false`，Actor 收到明确的未验证、可选建议说明。
建议正文与哈希仍必须一致；失败或争议审查不能改标为跳过后放行。
显式开启后，恢复原独立反例验证和建议准入条件。旧验证结果保持原记录，不追溯改写。

## 显式开启后的验证策略

新实验显式使用 `construction_strategy="counterexample_validated"`、`construction_rounds=1`。
原 Task 决定要求，知识包只补充事实与适用条件。流程为：Task-only 预期 → 三席 Jury →
Flash Reduce → Builder → 三席 Attack → 执行对照案例并与独立公开参考判断比较 →
最多一次候选修复 → 原案例及新边界复查 → 选择候选或保留原 Pack → Hypothesis 提议、面向新 Actor 的投影、独立攻击与复查。
没有叠加固定三轮构建，也不读取官方评分标准或旧分数。

Jury、Attack、独立参考及终局排序 Map 共用 Qwen3.8-max、MiniMax-M3、Gemini-3.8-Flash。
Qwen/MiniMax 关闭 thinking，Gemini 使用 `reasoning_effort=medium`、非流式 JSON；Reduce、Builder、Hypothesis、Critic 保持官方 Flash-thinking。
每次最多验证三个高风险目标，保留正确对照、错误变体、边界、程序实际回执及内容哈希。
`supported_on_cases` 只表示这些案例提供支持；参考模型与检查结果冲突标为 `disputed`，不认定哪方必错。
候选必须在同一批对照上有改善、不损失已有支持，并通过新边界复查；失败、不明确或平局保留原 Pack。
攻击按三席轮流选择目标；标准／检查别名去重，局部条款案例不因缺少整篇作品而误判。
来源模型数量和多数意见都不构成正确性证明。

新 Jury 用结构化 `clauses` 保存每项的 `use`、原文出处、适用条件和等价替代组；
用户约束、专业质量判断、示例方法分别标注。父标准为 hard 不会让示例自动成为硬要求。
Reduce 可以显式合并同义表述，保留来源映射；Builder、Harness 和 Critic 使用同一条款定义。
旧 Pack 不自动补造这些语义，也不自动获得新权限。
结构化条款的最终文本是唯一正文；带明确条款 ID 和理由的合并保留原文谱系，标为模型提出的改写，
不声称语义等价已获验证。有效原文引号已明确引用的来源会同步到引用列表，不要求模型重复填写。
单个反例核验因补读、格式或传输预算失败时标为 inconclusive，其他目标继续，整体审计标为 partial；
失败目标不会获得案例支持或关键保护权限。

C/D 建议分为已证实缺陷、Outcome 增益、待核实和普通建议。推动修订需要定位当前作品、
引用 Task 原文、说明预期增益和验证方式。未完全验证的质量建议可作为 `eligible_for_trial` 尝试，
但仍明确列出未验证前提。若声称前提已获案例支持，须匹配具体条款、原文和条件，不能只挂一个有效 ID。
深度、推导、可执行方案仍可成为改进目标；软偏好和未核实事实不会自动变成必须修改的清单。

C/D 共用终局阶段控制器，D 保留每个符合条件根回合的运行中 Critic。
进入终局后关闭运行中通道。此次通用烟测 C/D 均最多一次终局修订；预算不足仍保存已交付作品。
记录每个完整版本，三席匿名排序只读原 Task 和实际作品／文件证据，不携带本次估计标准或反馈；
只在严格偏好且关键检查保留时采纳新版。只有有案例支持、纯用户约束且非语义判断的检查可作关键保护；
关键通过变为失败、unknown 或缺测均不能当成保留。原生最后回复与最终选用作品分别保存。
审查执行成功、实际注入、发生修订、选版回退和独立得分必须分开报告。

角色每阶段共享时限、最多两轮补读／六个引用、一次契约补正、一次空传输恢复，最多六次物理请求。
Reduce 上限 393,216 字节；超限先触发 Flash 导航压缩，Task、条款、案例和原始回执不靠截断塞入。
固定部分仍超过容量时明确失败，不冒充完成。

单次集成烟测入口为 `reward_engine.adapters.claw_eval.general_smoke`；WorkBuddy Office 适配器见
[实际文件与原生链路](../../adapters/workbuddy/README.zh-CN.md)。两边共用核心，不读取隐藏 grader。

正式完整入口及冻结 Pack 入口均支持此策略。此前两次独立 ABCD 的冻结编排入口为
`reward_engine.adapters.claw_eval.outcome_validation_run`，使用
[T068 配置](../../../experiments/claw_eval/outcome/counterexample_20260911.plan.json)。
同一套公开输入只构建一次，八个新 Actor 无跨次学习，各不同正文独立双评。
实施与真实结果见[记录](../../../experiments/claw_eval/outcome/COUNTEREXAMPLE_VALIDATION_20260910.zh-CN.md)。

## 历史兼容：Task 优先的固定三轮构建

下面保留旧配置行为，便于复现已有实验；新反例验证策略不叠加这一循环。

`construction_rounds=3` 让 Jury、Builder、Hypothesis 各完成三轮，和 Actor 外层 `rounds=1` 分开；旧配置未指定时保持单轮构建。原始 Task 和按作品归属保留的真实用户补充决定目标，知识仅提供可错的补充证据、方法及适用条件。Jury 首次独立要求提议不接收知识包或历史作品，不能把知识覆盖率当成 Outcome 质量。

第一轮保留各模块的原有构建方式。第二、三轮各有三位独立评审审核同一当前产物，然后一次 Reduce 修订；第三轮检查改动是否带来错误和遗漏。Jury 关注目标与标准，Builder 关注满分判据及检查是否准确落实标准，Hypothesis 关注方法是否正确、可执行并解决主要 Outcome 缺口。无修改也完成全部轮次，审核一致不证明正确。

Jury 三轮结束后，Builder 与 Hypothesis 并行各跑三轮。冷启动主要逻辑调用分别为 15、9、12 次，共 36 次；补读、格式补正及检查器 Author 另计。Builder 仅在最后定稿后安装检查器一次；旧单轮 Pack 不能跳过审核。每轮输入、意见、当前条款、修订、覆盖关系和未解决项单独保存，退役旧条款不再作为当前要求回流。JSON／契约和无效补读引用在每个证据阶段最多同上下文补正一次，补读预算保持有界，不因低分重抽。

`initialization="archived_seeds"` 加 `seed_source` 可使用同题三份已归档原始种子，逐份核对实际公开 Task；知识与三个构建模块重新运行，旧 Hypothesis、Battle Memory 和私有评分不复用。新试验仍分别运行完整 ABCD，C/D 自身修订前后另作配对评价。

## 满分 Outcome 导向的 C/D

当前主线只维护一套反馈实现。C/D 从 Task、已有知识及 Outcome Reward 的满分标准反推目标作品；核心解释正确、没有事实错误，不足以自动结束改进。`improvement.py` 共用目标、差距及行动契约：`outcome_goal` 描述满分结果，`full_score_gaps` 保存完整剩余差距，`improvements` 给出最多三项优先动作及预期贡献、验证方式、稳定 ID 和依据。改进机会可以是深度、完整性和可执行性的提升，不必先构造成事实缺陷。

C 的主结论评估整个满分 Outcome：`needs_improvement`（需要改进）、`needs_verification`（需要核实）、`target_satisfied`（当前审查目标满足）或 `unknown`。只要存在满分差距或改进动作，主结论必为 `needs_improvement`，请求同会话修订，默认最多两次；再次检查当前实际作品并保留未解决项。正常结束记录 `full_score_target_satisfied`，仅表示当前审查认为目标满足，实际满分由独立评分验证。预算用完记录 `budget_exhausted_unresolved`。

C/D 的 JSON 解析／输出契约错误由 `ask` 在每个证据阶段保留同任务和当前证据补正一次；原始失败仍在模型记录中。补正请求的 `response_recovery` 携带本次被拒的对象或可取得的 JSON 原文，以及具体错误，要求仅修正输出契约问题、保留不受影响的分析；不能串入上个阶段的回复。有效语义结论不因得分低而重抽。补读最多两轮、累计六个来源，累计原文不丢失；一次收束只能使用已读依据或提出待验证行动。所有操作共用原有时限和容量。

D 每个配置到期的工具回合后介入。无论运行中是否审查过，都在交付前用同一 Critic 审查实际终稿，最多一次续写并复查；计划或草稿审查不能代替终稿审查。内部 assistant 反馈显式携带空 `reasoning_content`，原 Actor 推理字段完整保留；不伪造推理。运行时审查与终局补查分别记录，Actor 失败不能被反馈回调成功掩盖。

新增独立 `outcome-artifact-quality` 评分入口：Actor 全部关闭后只评完整作品正文，支持 C 初稿／终稿同口径比较；模型请求不含臂标签、Critic 意见或工具努力项。原生 Grader 和综合分保留，两种口径分列。


**发现真实结果缺口 → 制作标准与检查 → 指导并修订交付 → 相对排名检验改善。**

当前 [四臂协议](ARMS.zh-CN.md)：A 原始、B Hypothesis、C Task＋交付前 Reward 检查修订、D Task＋运行时 Critic（新配置默认每1回合，历史显式3保留）。仅 B 接收 Hypothesis；C/D 不接收它或历史 Battle Memory，但保留自身当前会话的测量与修订历史。四臂分别生成独立作品；C 的零修订表示保留自己的初稿，不代表等于 B 的作品。C/D 不另加静态 Reward Prompt，D 始终由同一 Critic 审查实际终稿，最多一次同会话续写和复查。`arms.py` 统一入口路由，仍只维护 `hypothesis_feedback_v1`；过去同名协议的实际请求以各运行冻结源码为准，不回写旧结果。缺省旧配置保持 `legacy_abcd`，以下历史 B/D 说明只适用于旧协议。无运行时 Before/After Battle。

## 默认剪枝（2026-09-08）

2026-09-10 当前实验统一官方 Flash＋thinking，包括搜索、来源Map和最终Reduce；独立Grader不改。见[统一配置与验收](../../../reports/claw_eval/agentic_search_v13_20260909/v39_thinking_implementation.zh-CN.md)。v39保留知识类型但修正越界：同来源条件先判断相关性，不自动全变硬前提；显式非法use不冒充方法，旧unclassified仍兼容；复用／Author后重新绑定。Skill目录的拒绝标签不再代替一次真实正文试读，模型仍拒读则明确失败。

长未决问题保留每项文本／重要性，重复处置明细按原句柄取回。Critic汇总同样保留全部检查事实，重复覆盖草稿按需读；拒绝超限记录实际字节与限额，不报成模型能力差。测量完成后的降级仍传真实结果。具体见[Critic合同](POLICY_CRITIC.zh-CN.md)。

2026-09-09 Builder／反馈细化：合同错误最多修一次，整包校验后才试跑 Author；拒绝模板绑定历史用户消息序号。单项检查不重复携带提案账本，保留标准／适用性／来源。确定性检查依赖不变才复用，语义检查不冒险缓存。

2026-09-10 知识先验按角色消费（task-review-v38）：Task Review 保留组带封闭 `use` 与 `source_class`；Builder 的 `knowledge_roles` 把先验分为 method／reference／condition，Jury 标准仍是每个检查的依据。method 可决定怎么测，reference 只作参考，condition 由代码挂到引用同来源方法的检查（`test.knowledge.applicability_refs`），`evidence.review` 先核条件；`measurement_priors` 不过滤。研究控制器在尚未读到任何 SKILL.md 时对"带着可读 Skill 入口 stop"提醒一次，放弃需封闭 `reason_kind`。测试：`test_claw_outcome_builder_knowledge_roles.py`、`test_search_source_map_reduce.py`、`test_search_selection_contract.py`。
交付检查只剩执行错误时保留作品和未核实项，不驱动盲目重写；真实缺陷仍需修订与复测。Critic 输出阶段目标、后续动作、调整条件，复用既有字段与待核实问题投影；不加角色或运行时 Battle。测试：`test_claw_outcome_builder_critic_refinement.py`。

Critic进展补齐：近期真实工具记录取代“旧旁白冒充成品”，同trial最近两次反馈／测量明确标为历史；检索query_extra不污染gaps。分配总时间给初判、测量、汇总，已完成测量在汇总失败后仍以degraded摘要注入Actor。缺陷型runtime_check引用须有有效主项或局部fail。见[当前合同](POLICY_CRITIC.zh-CN.md)及`test_claw_outcome_critic_progress.py`。

2026-09-09 停止策略：默认一轮结束。终局Ranking的同一次Jury Reduce只有给出有证据的重要新缺口及下一轮具体方向，才能请求继续；Runner检查选中作品、采样完整性和轮数上限。字段无效不抹掉有效排名，只拒绝额外轮次。`rounds` 是1–3的上限，不是固定执行数；Grader评分不参与继续决定。每轮保存 `loop_decision.json`。

- Jury：同一次 run 内复用未变的三席独立预期；作品判断仍每轮三 Map＋Reduce。
- Builder：不变的标准／尺子复用，只生成变化部分；不按历史产品数量扩展测试。构建后仅合成夹具收据 smoke，不跑旧作品×全检查。
- Hypotheses：预载真实需求、重要缺口、知识与 Memory；完整作品按现有证据句柄补读。内部提案／映射账本留档，前台只给待核实问题、来源与验证方法；映射失败不是产品缺陷。B 的短复核共用这一投影。
- B：[delivery_triage.py](delivery_triage.py) 先短问“达成 Outcome Reward 还差什么？哪些主张值得核实？”，再选少量 Harness，修订后复测未解决项。
- 可选 Actor Harness 在 adapters 层：原生 B/D 显式开启 `actor_harness.enabled` 后，Actor 可自己提交草稿、选择已有测试，复用本核心的 `measure`，不新增裁判；默认关闭。真实工具结果保存，检查自身评语不作为环境事实。交付回调不因主动检查而跳过。
- 草稿测量通过 `DeliveryReview.note_draft_measurement` 传给终局短复核；按测试＋具体问题保留最近收据，附“正文是否已变化”。旧pass不是新通过，旧失败需要结合当前作品判断；纯超时不自动返工。旧疑问也不能被另一个窄标准pass覆盖。不把旧作品引用混成当前环境事实。测试：`test_claw_outcome_draft_findings.py`、`test_claw_native_trial.py`。
- `claim_checks`保留最多3条关键主张及其依据（不要求凑满）。已承认需要核实→真实test或unknown，不能被放行标签抹掉。引用存在不代表事实成立；不增加工具配额。测试：`test_delivery_claim_routing.py`。
- 具体疑问随所选检查传入；模型辅助检查分开回答原标准和 `inquiry_result`。问题中的缺陷不能被窄标准pass吞掉；结果返回Actor，也随作品证据进入下一轮Jury。同一问题的已知前后结果可以写Memory，不同问题／unknown不能冒充改善。测试：`test_claw_outcome_inquiries.py`。
- 两种结果各自校验：主张审查字段错误只让该审查unknown，不能抹掉引用有效的局部inquiry；反过来，inquiry引用无效也不能伪造局部fail。未解析JSON、模型身份不符等整份回复问题仍不能使用。保留原始模型判断，不自动猜枚举值或补引用。测试：`test_claw_outcome_claim_audit.py`、`test_claw_native_actor_harness.py`。
- 同一次检查兼顾遗漏与错误，优先审查容易出错的重要主张；这是判断方法，不保证模型一定发现问题。专业文档只有标题／链接时，不能当成已经阅读的支持证据，引用后补读存档片段并重判。
- 单项语义Harness通过 `measurement_prior_view` 预载该标准明确引用的研究知识；其他研究方向只列适用范围和验证方法，完整原文仍可按原handle读取。Task、用户补充、作品、环境、标准不裁剪；普通／未知格式Skills保持原样。目录引用必须经过现有一次补读，不把目录当证据，当前 C/D 最多两轮补读、累计六个来源，必要时一次收束；其他调用保持既有预算。测试：`test_claw_outcome_prior_view.py`。不改Jury、Hypotheses或终局全局检查的知识输入。
- 未选检查保持 unknown、非自动失败；`no_material_gap_found` 不是全检查通过或官方成功。用户待回复、真实缺口、请求核验未解决仍保留。
- 显式诊断才用 `DeliveryReview(audit_mode="full")`／`run_task(full_audit=True)`；默认 `targeted`，`max_tests=3` 可配置。

专项测试：`tests/unit/test_claw_outcome_pruning.py`；三轮默认入口联调：`tests/legacy/test_claw_outcome_experiment.py`。不以减少请求数冒充已实测提速或提分。

这里是 2026-09-08 新简化实现，不是历史 T110/C17 已运行版本。
Runtime Policy Critic 已接当前 Outcome 主线，详见[说明](POLICY_CRITIC.zh-CN.md)；原生 Grader Hook 已移至 `reward_engine/adapters/claw_eval/`，已在187完成T068真实评分，自定义Judge配置不等同默认排行榜。
已单独运行一次 [真实 B 冒烟](../../../reports/claw_eval/outcome/live_b_smoke_20260908.zh-CN.md)：Harness 实际调用，但初稿全获放行，未触发修订；人工复核发现漏判，不宣称实模型闭环已验证。

## 从哪里读

可选知识前置：[task_review.py](task_review.py)，[模块说明](../../../docs/design/outcome/docs/task_review/README.zh-CN.md)。默认关闭，Task-only 搜索/原文提取后沿 Skills 通道进入 Jury，无新增裁判层。
选材/补搜/长文补读由 [agentic_search.py](agentic_search.py) 的模型决策循环负责，代码只执行并发工具和预算。未解决的知识缺口也随包传给 Jury，不因汇总完成而消失。

先看 [简化版算法](../../../docs/design/outcome/SIMPLIFIED_ALGORITHM.zh-CN.md)，再按模块阅读：

| 模块 | 代码 | README | 单元测试 |
|---|---|---|---|
| 共享输入、Jury | [context.py](context.py)、[jury.py](jury.py) | [Jury](../../../docs/design/outcome/docs/jury/README.zh-CN.md) | `test_claw_outcome_context_roles.py`、`test_claw_outcome_gap_signals.py` |
| 联合 Builder、开放 Harness | [builder.py](builder.py)、[harness.py](harness.py) | [Builder](../../../docs/design/outcome/docs/builder/README.zh-CN.md) | `test_claw_outcome_builder_harness.py` |
| Hypotheses | [hypotheses.py](hypotheses.py) | [Hypotheses](../../../docs/design/outcome/docs/hypotheses/README.zh-CN.md) | `test_claw_outcome_context_roles.py`、`test_claw_outcome_infra.py` |
| Battle Memory／Attention | [memory.py](memory.py) | [Memory](../../../docs/design/outcome/docs/battle_attention/README.zh-CN.md) | `test_claw_outcome_memory_runner.py`、`test_claw_outcome_infra.py` |
| 单次交付、修订、当前任务清单 | [runner.py](runner.py)、[task_progress.py](task_progress.py) | [Runner](../../../docs/design/outcome/docs/runner/README.zh-CN.md) | `test_claw_outcome_task_progress.py`、`test_claw_outcome_memory_runner.py`、`test_claw_outcome_native.py` |
| 默认一轮、按需续跑、原生入口 | [loop.py](loop.py)、[outcome_experiment.py](../../adapters/claw_eval/outcome_experiment.py) | [Loop](../../../docs/design/outcome/docs/loop/README.zh-CN.md) | `test_claw_outcome_loop.py`、`test_t068_native_entry.py` |

MapReduce 复用已有 [batch.py](../batch.py)，不增加一层模型指挥官。
模型传输、原生 Jiuwen 运行器和任务适配复用仓库已有实现，不复制 SDK。

## 输入和输出

Jury 的最终 Reduce 校验失败最多纠错一次，不重采 Map；失败后的 Loop 保留已有成果并明确停止。

Jury 定义和终局排名的 Reduce 请求默认上限为 160,000 个 UTF-8 字节。原生实验配置可显式设置 `jury_reduce_max_bytes`（例如 `262144`，即 256 KiB）；入口同步设置 Core 检查和 Reduce 模型传输限制，纠错请求也受同一上限约束。此项不改变 Map 或其他模块的限制，不截断或改写输入；thinking/non-thinking 对比应使用相同值。

其他 `ask` 调用可从注入 callable 的 `max_request_bytes` 读取显式容量，未声明时仍为 160,000 字节；调用处显式 `max_bytes` 优先。JSON Role 把配置中的同一上限同时提供给 Core 和传输层，避免新版知识已超过 160 KB、传输上限提高后仍被 Core 旧默认拦截。补读同样受限，Batch 收据保留具体字节诊断。T068 full01 的 185,260 字节前置预期请求已验证此问题，见[完整验收记录](../../../reports/claw_eval/outcome/flash_thinking_full_20260910.zh-CN.md)。

官方 Flash-thinking 配置的搜索 `selection_model.max_request_bytes` 显式为 **393,216 字节（384 KiB）**，正式配置生成器与搜索测试入口共用常量；历史配置不回写。`TaskReviewRunner` 在 selection 请求超过此值时触发 `task_review_selection_compress`，使用同一 selection 模型及 thinking 设置，将变长的导航预览、最近正文、重复引文分片摘要；每个压缩请求自身也须满足同一上限。Task、用户要求、未解决问题、候选 ID 和来源偏移保持原样，完整源文仍供 Map/Reduce 使用。摘要明确标注为可能出错的阅读导航，不能当作原文引证。单轮压缩最多 32 次操作，无质量重抽；压缩后重新计算 UTF-8 JSON 字节数，仍超限或摘要非法则停止，并在 `request_compactions` / `request_budgets` 留下收据。这不改变其他角色的容量或历史实验结果。
原输入／两次回复可单独回放。只对原文和作用域一致的条款合并选择号，全部来源仍可回查。
提案 Prompt 集中于 [jury_prompts.py](jury_prompts.py)，带版本与文本记录。
细节及兼容边界见 [Jury README](../../../docs/design/outcome/docs/jury/README.zh-CN.md)。

Jury 草稿容错和小点去向见 [jury_drafts.py](jury_drafts.py)。Reduce 按短编号成组保留原条款，代码原样带入Criterion并生成父子映射；确需改写时要求新文本、理由和依据。漏项保持 unknown，贯穿 Builder、Harness 覆盖报告和任务清单。保留文字不是证明文字正确，无新增裁判调用。单测：`tests/unit/test_claw_outcome_jury_details.py`、`test_claw_outcome_clause_selection.py`。

Jury、Hypotheses 输入共享一份公开 Task、按作品归属的实际用户补充、作品／终局环境、来源、简短检查、可选经验。
原始轨迹完整归档，不默认进入模型输入；官方评分和隐藏用户设定不能进入这些上下文。

新输出是 `criteria + gaps + tests + typed results + relative ranking`，没有主观绝对总分。
`callable` 仅指代码可调用；`pass` 只指单项检查通过；`coverage` 只指测量覆盖。
`measurement_kind=model_assisted` 明确表示检查仍依赖模型，不冒充确定性程序测试。
真实测量工具与一次检查器补写已接入 Builder／交付检查，见 [测量能力](MEASUREMENT_TOOLS.zh-CN.md)。知识包原文随 Pack 保存，不只在 Jury 阶段可见。

语义检查现在先输出 `claims`：主张及位置 → 对应证据及实际观察 → supported／contradicted／unverified，
再汇总 pass／fail／unknown。必须覆盖本标准相关的重要主张和实际交付渠道，不只听 Actor 的总结。
程序能发现引用不存在、缺审查记录、未证实主张却判 pass 等结构问题；不能代替模型判断证据是否真正支持主张。
一次 [固定作品复测](../../../reports/claw_eval/outcome/claim_audit_recheck_20260908.zh-CN.md) 找到了真实缺口，但也误把未展开的源记录列为证据不足；尚不能声明检查已可靠。
现增加引用预览补读：语义结论引用 preview-only 记录时先补全文，再由同一检查模型定稿。
与模型主动 `need_evidence` 共用一次补读预算，不增加新裁判；超预算保留 unknown。
第二次真实复测确认补读能执行，但最终仍漏判，详见上述诊断记录；不要据此宣称检查质量已通过。
Hypotheses 可提出少量主动澄清问题，C 的 Actor 通过现有会话询问；不预填答案，不修改 A/B 的提示。
方法投影标明来源：其他采样中的版本／分歧／检查不是fresh Actor的会话历史、亲自执行结果或用户确认。保留方法内容，不把历史说法直接改写成已验证事实；这个提示边界不等于模型服从已被证明。

## 运行配置

独立终局验收见 [原生 Grader Hook · Yibu Qwen](../../adapters/claw_eval/NATIVE_GRADER_HOOK.zh-CN.md)：
使用 Claw-Eval 原生评分器＋Yibu `qwen3.8-max`，不是官方榜单模型配置。
187 Jiuwen原生Actor已打通完整轨迹、最终环境快照和付费评分。各轮成绩、配置差异与最新运行入口集中在 [T068试验索引](../../../experiments/claw_eval/studies/t068_outcome_20260908/README.zh-CN.md)；算法README不再复制可能过时的分数。历史B92存在修订超时／工具项差异，尚不能称稳定复现。

入口读取已有任务／模型配置，但在**新输出目录**保存新协议和真实 A/B/C 开关，不改历史结果。
例如 `c17_simple_flash_20260908/experiment.json` 提供已有 Task、DeepSeek 官方 Flash 角色和流式参数；
它不是新的实验成绩，也不代表重新选择了 C17 作为后续难题。

可在配置中添加以下小节；省略时使用相同默认值：

```json
{
  "outcome_loop": {
    "rounds": 3,
    "max_revisions": 2,
    "top_k": 3,
    "batch": {"concurrency": 3, "deadline_s": 240},
    "skills": [],
    "checker_plugins": []
  }
}
```

- `rounds` 为最大轮数1–3；默认先一轮，只有终局Jury批准重要新缺口才可继续。初始化三条 Original 不计入轮数。
- `max_revisions=2`：B 初稿检查一次，最多修两次、再各查一次。
- `top_k=0` 关闭经验检索；不调用 Embedding。库保留可注入的 Embedding 接口。
- `checker_plugins` 是可信配置中的 `module:function`，函数接收 Registry 注册真实检查器；不执行模型返回的代码字符串。
- Actor、Jury 三 Map／Reduce、Hypotheses 三 Map／Reduce、Builder、语义检查分别按 `roles` 查模型；同款模型也用独立上下文。
- 关键参数和模型角色缺失在采样前报错。输出目录存在时拒绝覆盖，不自动重抽质量差的作品。

以下命令会开始模型调用；本轮没有执行它：

```bash
python -m experiments.claw_eval.opencode.campaign prepare --output runs/new_claw
```

每题保留独立配置、池和失败状态。当前运行入口与环境说明见 [Claw-Eval](../../../experiments/claw_eval/README.zh-CN.md)；离线复现入口见 [发布说明](../../../release/README.md)。

## 日志

```text
NEW_EXPERIMENT_DIRECTORY/<task_id>/
  experiment.json / state.json             实际配置、状态
  seeds/                                  三条独立种子及原生轨迹
  round_N/models/                         每次模型请求、回复、流式记录
  round_N/pack.json / construction_checks.json
  round_N/actors/{A,B,C}/runtime/          真实 Agent 工具与最终作品
  round_N/actors/B/checks/                 每次交付、检查、修订反馈
    task_progress.json                    当前要求、已验证／未完成／未知项、下一步
    task_progress_N.json                  各次检查后的清单快照
  round_N/check_sources.json              原检查文件与作品对应关系
  round_N/ranking.json / selected_delivery.json / memory.json
  result.private.json                    汇总，不作为学习输入
```

终局排名包括 Original A。只有严格优于 A 且没让 A 已过的检查退步时才选择 B/C/D；否则保留 A。
这保护选择流程，不保证官方成绩单调上升。每条作品和失败都保留，不只保存赢家。
各模型原始 usage 和耗时落盘；未提供价格时不编造费用。检查和修订的额外开销不计作免费提升。
各臂独立Grader现在并发执行，上限沿用 `BatchPolicy.concurrency`（未提供时3）；单路异常只记录该路failed，不取消其他成绩，不将失败记成0，也不触发重抽。私有成绩仍在选择完成后生成，不进入下一轮学习。单次Grader自己的超时／重试协议不变。此优化在run18冻结后完成，不能声称已加速run18。

## 测试与边界

历史 [收尾记录](../../../reports/claw_eval/outcome/test_report_20260908.zh-CN.md) 不代表当前测试总数。2026-09-09最新统一离线回归494项通过；记录入口与实际命令见 [Infra说明](../../../experiments/claw_eval/outcome/INFRA_REPRODUCTION.zh-CN.md)。
当前 outcome 核心＋共享batch为3885行（2026-09-09 `wc -l`），包含知识研究／数值测量能力、注释和空行。
计数不含复用的 SDK／传输／历史框架，不是整个仓库的总代码量。

离线回归不需要付费Key；统一使用Infra说明中的指定venv＋`record_run.py`命令，保留日志，不另维护一份容易漏组的测试命令。

覆盖：需求隔离、共同缺口、真实引用、Map 并发／超时、开放 Registry、错误为 unknown、
同会话修订、经验检索、Original 回退、三条种子＋三轮 ABC、日志和配置预检。
原生测试运行真实 Jiuwen SDK，但模型回复受控；不是实模型能力评测或官方验收。

目前新交付回调只接 Jiuwen Single；OpenCode 和动作级 Policy Critic 留在原路径。
Task Adapter 仍是已接入的少数 Claw-Eval 任务；多模态及其他任务需要真实适配／检查器，不宣称全 benchmark 已完成。
缺能力只记录 unknown，不能靠一个通用语义检查器宣布所有环境都可测。
原生 Grader 已有独立 Hook，T068原生入口已完成真实归档和付费评分；通用核心不强制启用Hook。缺沙箱原生归档时返回 `needs_native_artifacts`，未执行时为 `not_run`。当前真实成绩见T068索引，不能据此宣称所有Task／Runtime均已接通。
