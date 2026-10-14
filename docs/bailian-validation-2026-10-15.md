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

## 回归测试边界

- `ff4c87b` 全量回归：3205 passed、0 skipped，9 条已有警告；测试期间源码未变化。
- 后续示例运行参数与原生 Jury 协议选项：113 项相关测试通过。
- GLM 请求参数与冻结模型配置：269 项相关测试通过，包括截断回复仍被拒绝。

上述测试包含模拟 provider 的用例，不能当作真实模型结果。真实调用结论仅来自单独保存的运行记录。
