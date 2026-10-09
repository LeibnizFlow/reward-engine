# 当前架构

实验入口调用 benchmark 适配器，适配器调用 `reward_engine.core.outcome`。WorkBuddy 与 Claw 共用构建、评审、记忆和选择逻辑，
各自保留任务输入、交付物采集及原生评分实现。原生分数只用于事后评估，不作为生成和运行时选择输入。

新运行的冻结代码与公开证据是不同用途：运行器维护在线 checkpoint；公开导出维护可携带的路径、证据身份和评分绑定。
`release.report` 与 `experiments.analysis` 只读取公开数据；它们不依赖运行服务、机器凭据或历史 archive。

逐文件索引同时记录原始和公开哈希。公开文件字节先通过实际 SHA-256 校验，再用于验证原始回执的身份绑定。
