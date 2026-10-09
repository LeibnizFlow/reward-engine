# WorkBuddy Office Outcome 适配

共用 `core/outcome` 的 Jury、Attack、Builder、C/D 审查和匿名选版。
适配层只负责原生 Office 会话、公开文件观察和真实文件恢复，不读取隐藏 tests 或 grader。

`OutcomeOfficeAgent` 继承冻结的 `office_jiuwen_agent.JiuwenOfficeAgent`。
部署时通过 `PYTHONPATH` 提供该原生模块；不改 benchmark 数据集或个人默认配置。
`worker_hook.patch_worker` 严格匹配原生 worker 的启动锚点，有源码漂移就停止。
原有启动、模型、默认工具、内部子代理、记忆隔离和 Actor 结束后停止进程的流程继续使用。

四组采用 `hypothesis_feedback_v1`：A 原生基线，B 仅 Hypothesis，C 终局检查，D 每个符合条件的根模型回合 Critic 加终局检查。
C/D 最多一次同会话修订，内部反馈通过 AssistantMessage 注入；恢复请求不增加公开用户指令。
控制器保留模型凭据；Actor 只有本次回调地址和临时 token。

原生 `[[steps]]` 使用 Harbor 的 `MultiStepTrial`，保留步骤顺序、逐步 verifier、
`min_reward` 提前停止和原生最终/均值计分。Jury/Attack 初建只接收第一步已公开指令及输入；
后续步骤到达时才加入该步指令和 `workdir`，不向算法提供未来步骤或私有测试。
共享评分目录在下一步准备前清空；上一轮真实工作区保留。

原生 Worker 在每步评分前退出。适配器在退出前保存实际 user/assistant/tool 消息，
下一步恢复这些历史及工具调用关联，继续使用当前原生系统消息和新的真实用户指令。
选版恢复会留下内部交付回执，使历史中的被放弃修订不被误认为当前文件。
每步分别记录 Task 范围、文件快照和审查；C/D 每步最多一次内部修订，遵守原生该步预算。

新步骤不重新执行整套 Jury/Attack。旧标准降为未验证参考，清除旧验证支持和需求覆盖结论；
完整当前 Task 驱动审查和选版。B 的既有 Hypothesis 也是早期可选方法，不覆盖新指令。
新步骤的作品只能在该步内比较、恢复；这不意味着已证明多轮题提分。

每次终局暂停原生请求，控制器核对传输前后文件清单、保存所有文件字节、大小、模式和 SHA256。
Office XML 检查保留正文、结构、单元格地址、公式和缓存值；未执行公式、未看渲染或解析不全会明确标注。
这些观察不等于 Office 内容正确或视觉合格。文件内容有补读索引，二进制原件始终保存。

比较版本后先恢复选中快照的真实文件并复核哈希，再释放 worker。
无明确改善、选版失败或修订失败时保留首份完整交付。原生 verifier 仍在 Actor 进程停止后才运行。
运行期 Critic 读取真实工具观察；它不把 Actor 的计划陈述当成交付物。

仅在已有 187 WorkBuddy 独立 Docker daemon 和 Python 3.12 控制器上执行。
单题烟测的执行记录与原生评分分开保存，不能推断 Office 50 题或两个 benchmark 已整体提升。
