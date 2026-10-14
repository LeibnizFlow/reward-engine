# 百炼真实调用验证记录

本记录区分真实模型执行、离线回归测试和原生 benchmark 验证，不将连通性或原生 grader 当作 Reward Engine Jury。
连接配置与命令见 [Bailian 接入](bailian.md)。密钥仅存放于本机被 Git 忽略的凭据文件。

## 小型真实示例：三轮执行完成

`minimal_live_v6` 使用源码提交 `6e2327c`、真实 `qwen3.8-max`，没有脚本回复、模型回退或传输重试。
显式参数为：HTTP 超时 600 秒、role stage 超时 1800 秒、并发 1、关闭思考、Jury `targeted` 协议。
`targeted` 使用现有核心最多两次外层修复；内层调用预算与各自阶段超时仍然适用。

- 3 个真实种子、3 轮、9 次 Actor 执行，共 96 组模型请求与可解析 JSON 回复，无传输失败，进程退出码 0。
- 每轮 Jury、Hypothesis 和 ranking 的 Map 记录均为三个席位完成，Builder 的检查可调用，三路产物的测量与候选选择完成。
- 中间协议并非全程无错：含 9 次结构修复请求，9 个 Jury proposal 中有 8 个保留了草稿或 revision ledger 失败标记；核心保留原有要求与失败记录，随后由 Reduce 生成可接受的定义。不能把 Map 完成解释为所有中间回复均通过协议校验。
- 每轮均选择 A；B 的评审均为 `satisfied_under_estimator`，实际修订次数为 0。
- Memory 新增条目为 0。因此不声称该任务验证了实际产物修订、经验写入或质量提升。
- 这是计算公开数量并交付 JSON 的基础路径示例，所有席位使用同一模型，没有原生评分，也不是全部分支覆盖。

本机原始记录：`runs/bailian_validation_20261014/minimal_live_v6/`。
`summary.json`、应用层请求与解析后的 JSON 回复及完整结果均保留，另有 355 个 JSON 文件的 SHA-256 清单。
该示例未保存完整 HTTP wire、provider 返回的 model 字段或 token usage；模型名来自显式请求配置。
这些运行数据默认不进入发布仓库。

## 保留的失败尝试

未删除失败尝试，也未把失败回复改成成功。后续配置与提示变更只用于新运行。

| 运行 | 结果 |
| --- | --- |
| minimal v1 / qwen3.8-flash | Actor 返回任务 JSON，缺少传输层要求的 `final_response`，严格拒绝。随后明确了提示中的两层格式。 |
| minimal v2 / qwen3.8-flash | 完成一轮，随后出现 provider/超时失败；原诊断不足，不能追溯确定每次 HTTP 原因。随后增加安全失败原因与显式运行参数。 |
| minimal v3 / qwen3.8-flash | 无传输失败；Jury reducer 将字符串字段写成对象，协议修复后仍不合格。 |
| minimal v4 / qwen3.8-max | Jury 完成；Builder 请求不可读取的未来产物，修复回复又多包一层 `response` 且缺少标准覆盖。 |
| minimal v5 / qwen3.8-max | Jury reducer 重复提交 canonical 条目及其 alias，被严格拒绝。 |
| native WorkBuddy v1 | 第一轮 GLM Jury 席位耗尽 32,768 输出 tokens，其中 32,766 为思考 tokens，最终 JSON 截断。严格审计失败；正常中断，退出码 130，保留原始记录。 |
| native WorkBuddy v2 | 三个 Jury 席位均完成；Reduce 两次协议修复后仍把非空 criterion_ids 与 deferred/excluded 状态组合使用，严格拒绝。完成 0 轮，退出码 1。 |
| native WorkBuddy v3 | 三个 Jury 席位均完成；一个 gap 的任务要求依据仅引用环境观察，缺少 Task 引用。诊断未定位 expected_refs 字段，模型两次提交非法编辑路径。完成 0 轮，两项审计失败，退出码 1。 |
| native WorkBuddy v4 | 第一轮 Jury、Builder、Hypothesis 和三路 Actor 已执行；C 路独立原生 grader 首次与修复回复均缺少要求的 verdict 字段，严格审计失败。正常中断，退出码 130，完成 0 轮。 |
| native WorkBuddy v5 | 第一轮构建完成；种子检查的根级引用错误只有泛化诊断，模型修复时改动了原本有效的 claims 字段，被严格拒绝。C 路新的独立 grader 正常 scored；随后中断，退出码 130，完成 0 轮，两项完整审计失败。 |
| native WorkBuddy v6 | 初始检查无执行错误，A/B/C 原生评分均有效；B 路补充证据后的评审再次漏掉 outcome_goal，一次协议修复额度已用完，严格拒绝并标记 check_unavailable。随后中断，退出码 130，完成 0 轮，两项审计失败。 |

WorkBuddy v1 的核心允许按两席可用结果继续，但完整验收要求三个席位均成功。
其审计失败原因为 `Map contains failed/inactive seats`，不是完整流程通过。
随后为 GLM 显式配置 `reasoning_effort=low`，并将其两个内部席位的输出预算增至 65,536。
真实参数预检返回 HTTP 200、正确模型、`finish_reason=stop` 和完整 JSON；这不等同于完整任务通过。

## 原生 WorkBuddy

新的 `native_workbuddy_v2` 在提交 `37351ea` 上冻结，从三个新种子启动。
任务为 `api-usage-explain-cli-l3-001`：交叉核对示例仓库，生成带路径证据的中文 CLI/Python API 手册。
模型配置 SHA-256：`2e467647c1dfb5fe06b532d4540a28a11ddd3ecadbf40a5e924ae469914a3a82`。

该运行在第一轮 Jury Reduce 终止，完成 0 轮；pipeline 与 public-delivery 审计均退出 1。
它证明 GLM 新配置完成了本次真实 Jury 席位，不能证明原生三轮或全部模块已通过。
验收继续使用原有 `experiments.pipeline_audit` 与独立 public-delivery 审计；没有放宽判定规则。

`a1a7c51` 随后细化了 Jury 修复诊断：对已选 deferred/excluded 的条目，定位到
`criterion_ids` 字段并指出协议要求的空列表，由模型明确提交编辑；不自动修复、不增加预算。
`native_workbuddy_v3` 在该提交冻结，从新种子开始，仍在第一轮失败，见上表。

`740bb7e` 补齐了 gap 的 `expected_refs` 字段诊断：区分要求来源与观察证据，
仍由模型依据实际 Task 明确提交编辑，不自动补引用、不放宽校验。
同时将 Qwen Max 内部角色配置为低强度思考、65,536 输出预算。
`native_workbuddy_v4` 在该提交冻结，重新生成三个种子；第一轮 Jury Reduce、检查包构建、三个 Hypothesis 席位及三路 Actor 均完成，但 C 路原生评分格式错误使严格审计失败。
模型配置 SHA-256：`ac8fa6746b9eab300292a77a61a5b00ef9ed6a1f6e6be2ba4092f309068bd86f`。

本次检查还定位到 Map 回复格式遗漏 `why`：Expectations 要求该字段，严格 retain 校验也要求保留，
但下一步的格式示例没有列出它。后续源码补齐该字段与精确差异诊断，保留原校验和修复次数。
该变更未注入已冻结的 v4，也未修改历史回复；v4 三个 proposal 均保留中间失败记录。

v4 的初始候选检查共 39 项：38 pass、1 unknown；unknown 保留未读到相关文件导致的不确定性。
第一轮 B 的真实评审为 `full_score_target_satisfied`，只有一个版本、没有实际修订、Memory 新增 0。
A 的后续完整检查为 12 pass、1 fail，其中结构检查把模型自己提出的反面假设当成 material claim，
其否定判定与 pass 冲突；模型在修复中明确返回 fail。历史结果保留，后续只补充初次评估的提示，
说明失败锚点与调查假设不等于 Task 或产物的主张，不改判定、不放宽修复。提示能否改善真实可靠性仍待验证。

C 的原生 grader 使用 `qwen3.8-flash`，首次返回 `result`、修复返回 `decision`，均不符合原生 `verdict` 字段要求。
虽然原生工具生成了数字总分，独立评分契约仍将它标记为 failed；不得把这个总分当作有效评分。
v4 原生 grader 的实际请求只包含 model/messages；worker 设置的预算环境变量未出现在上游请求中，不能据此声称实际使用了 32,768 上限。
原有 pipeline 与 public-delivery 审计均退出 1；正常中断后保留 1,800 个 JSON 文件的 SHA-256 清单。
尚无原生三轮、候选选择、Memory 写入或真实修订分支的完整通过结论。

后续新增可选的独立原生 grader 冻结配置，Bailian profile 使用 `qwen3.8-max`、
`reasoning_effort=low` 和实际请求中的 `max_tokens=32768`。Actor 仍为 Flash；
原生 grader 的 prompt、解析器、修复次数与评分规则不变。`native_workbuddy_v5` 在 `24c8d5e` 冻结，从三个新种子开始。
配置 SHA-256：`1be03ca75186124ee7ac0594639fa818187f91456f283019b7734a22ca962bf5`。
C 路真实原生评分状态为 scored、scoring_errors 为空，实际返回模型为 Qwen Max，
接收请求与发出请求验证了预算和 low 参数进入上游；这仅验证该次原生评分路径。

v5 第一轮初始候选检查仍未完整通过：38 pass、1 unknown，其中 unknown 带 error=true，
属于引用协议修复失败，不能算作普通证据不确定。原始根级引用仅有工具观察与 Task，
模型在修复中又改动 claims 的有效引用和观察文本；保留校验正确拒绝。
后续补齐根级 `evidence_refs` 与 `inquiry_result.evidence_refs` 的结构化诊断，启用已有的定点修复协议。
只允许模型修正对应引用列表；claims、观察、判断、summary 与判定均受原有保留约束。
直接产物引用和合格 workspace.read 的接受条件不变，workspace.inspect 单独引用仍不被接受。
该运行在确认 C 路原生评分完成后正常中断，保留 1,247 个 JSON 文件的 SHA-256 清单；
没有完整轮次、候选选择或 Memory 写入的通过结论。

`native_workbuddy_v6` 在 `c81987c` 冻结，重新生成三个种子。第一轮 33 项初始检查为
32 pass、1 unknown、执行错误 0；unknown 是缺少两个文件的直接读取证据，未硬判通过。
三路 Actor 及独立原生评分均完成，scoring_errors 均为空。B 评审首先漏掉 outcome_goal，
原有一次协议修复补齐；强制扩展公开源码证据后的新评审又遗漏该字段，剩余修复额度为 0。
字段在四次请求的回复格式中均明确列出，属于模型输出遗漏，系统没有自动沿用旧值。
严格审计以 B_review_execution unavailable 失败；只有一个版本，Memory 新增 0。
正常中断后保留 1,790 个 JSON 文件的 SHA-256 清单，尚无完整轮次通过。

另用实际 Token Plan 端点对 Qwen Max 的 strict JSON Schema 做了小型能力检查：
HTTP 200、返回模型正确、finish_reason=stop，支持必填字段、非空字符串与分支 Schema。
此项只是格式约束能力预检，不是原生流程验证；功能依据见[官方结构化输出文档](https://help.aliyun.com/zh/model-studio/qwen-structured-output)。

后续为 WorkBuddy 交付评审增加显式选择的 strict JSON Schema，保留完整评审与请求更多证据的分支。
它约束八个必填字段，不替模型填值；原有语义校验、证据扩展和共享修复预算不变。
配置和 Schema 内容摘要均冻结，实际请求格式进入缓存标识；端点拒绝时直接记录失败，不自动回退。
`b_review_schema_regression_v1` 在 `b7962b2` 上对 v6 的同一公开 B 候选做独立真实模型回归，
输入指纹与历史评审一致，不读取独立私有评分，不调用 Actor 或原生 grader。
三次实际 Qwen Max 请求均使用 JSON Schema：一次请求证据、一次完整评审、一次强制源码证据扩展后的完整评审。
两次完整评审均包含八个必填字段，没有协议修复；原有验证器接受 `needs_improvement`，
所选一项检查执行正常，退出 0。源码及输入哈希前后一致，独立网关已关闭。
这验证了该历史候选的失败路径修复，不是新种子实验，也不能证明三轮闭环或真实修订已完成。

`native_workbuddy_v7` 在 `97a7cd9` 冻结，从三个新种子开始。第一轮 Jury、Builder、
三路 Hypothesis 已完成，初始 42 项检查全部 pass，执行错误 0。
B 的第一次真实评审为 `needs_improvement`，Actor 根据反馈在同一会话修改 `final_answer.md`：
文件从 16,285 bytes 变为 16,608 bytes，SHA-256 不同；第二版评审为 `target_satisfied`，
保留两个交付版本和一条 Memory。A/B/C 均完成，三路独立原生评分均为 scored、scoring_errors 为空。
原有阶段审计已验证一次评审、续作和真实修订，整体仍为 pending，完整轮次为 0。

2026-10-15 05:47 CST 后检查发现 v7 主进程已不存在、网关连接被拒绝，
原运行状态文件仍为 running。没有留下可确认退出原因或退出码的记录，不能称为正常运行或完成。
中断时正在做交付后的完整检查，保留 1,478 个 JSON 文件的哈希与压缩归档。
该轮还没有完整 checkpoint，直接重新执行会重新生成 Jury Memory 标识，不能安全作为原轮次恢复。
因此保持 v7 不变，另以独立后台 supervisor 启动 `native_workbuddy_v8`，使用同一冻结代码与三个新种子，
显式记录进程、启动与退出状态。v7 已验证的分支不能替代 v8 的三轮完成证据。

`native_workbuddy_v8` 的后台进程持续正常运行，三个新种子、第一轮 Jury、Builder 与 Hypothesis 完成。
检查包包含 12 个 criterion、18 项检查；三个种子的 54 项初始检查共享两个模型通道。
各候选的批次预算同时开始计时，在 1,800 秒到期时得到 41 pass、13 个超时错误，
后者不是普通证据不足。严格阶段审计在 construction_measurements 失败，完整轮次为 0。
确认失败后对主进程发送 SIGINT，supervisor 记录退出码 -2；两个原有审计均退出 1，
保留 1,382 个 JSON 文件的 SHA-256 清单。没有修改历史判定或把超时项记为通过。

后续新增显式冻结的 `measurement_candidate_concurrency`，百炼 profile 设为 1。
完整初始检查先取得候选执行位置，再启动该候选原有的 1,800 秒批次；批次内部排队仍计入预算。
所有候选、全部检查和原顺序保留，实际超时仍返回错误，不重试或重置已开始的预算。
每个候选检查完成后立即记录结果，最终仍生成完整汇总；此记录不提供未完成轮次的恢复能力。
省略配置时保持旧调度。新的真实三轮运行仍待验证。

## 回归测试边界

- `ff4c87b` 全量回归：3205 passed、0 skipped，9 条已有警告；测试期间源码未变化。
- 后续示例运行参数与原生 Jury 协议选项：113 项相关测试通过。
- GLM 请求参数与冻结模型配置：269 项相关测试通过，包括截断回复仍被拒绝。
- `a1a7c51`：3210 项单元测试、77 项常规集成测试通过；补齐本机环境变量后，剩余 Docker/OpenCode 同会话修订集成用例单独通过，合计 3288 项，无未执行的跳过项。Docker 用例使用本地脚本 provider，并非真实 LLM。
- Jury 诊断变更的 229 项相关测试通过；未改动验收逻辑或两次外层修复上限。

- `740bb7e`：单元与集成测试合并执行，3299 passed、0 skipped，9 条已有 Pydantic 警告；测试期间源码未变化。包含 Docker/OpenCode 同会话修订用例，其 provider 仍为本地脚本。

- 后续 retain 格式与诊断修复：296 项相关测试通过，包含 14 个新增回归用例；该子集与全量测试重叠，不作累计总数。

- 后续初次评估提示澄清：393 项现有 harness/claim/recovery 测试通过；未改验收、修复规则或预算，也不据此声称模型误判已解决。

- 独立原生 grader 路由与配置：339 项相关测试通过，涵盖实际两字段请求的参数补充、原始请求保留、Actor 隔离和冻结配置篡改拒绝。

- `24c8d5e` 全量单元与集成回归：3374 passed、0 skipped，9 条已有 Pydantic 警告；源码无漂移。Docker/OpenCode 修订集成仍使用本地脚本 provider。

- 根级／inquiry 产物引用诊断：412 项相关测试通过，包含 19 个新增离线回归；未改接受条件、修复次数或审计规则。

- `c81987c` 全量单元与集成回归：3393 passed、0 skipped，9 条已有 Pydantic 警告；源码无漂移。

- 交付评审 Schema 及相关配置、请求、预算、冻结绑定回归：432 passed、0 skipped；为离线测试，不代表真实原生流程通过。

- `b7962b2` 全量单元与集成回归：3443 passed、0 skipped，9 条已有 Pydantic 警告；源码无漂移。

- 候选检查调度、配置冻结及相关模型参数回归：94 passed。覆盖 3/6/9 候选、全部检查及顺序、共享队列超时、真实超时保留与取消；为离线测试。

上述测试包含模拟 provider 的用例，不能当作真实模型结果。真实调用结论仅来自单独保存的运行记录。
