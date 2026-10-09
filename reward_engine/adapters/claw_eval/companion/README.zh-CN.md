# Claw-Eval 原生容器的可选 Rail 接口

此补丁仅改外部仓库 `experiments/scripts/jiuwenswarm_container_runtime.py` 和 `jiuwenswarm_actor_worker.py`。新增 `rail_extension={source, config}` 可选参数：stage 一份受信任的 Rail 源文件，并在根 Agent 注册；不改 JiuwenSwarm 内核、默认工具或原始A基线。当前本机副本已应用。

在 **Claw-Eval 仓库**检查并应用 `optional_rail.patch`；若反向检查通过，表示已应用，不要重复打补丁。新部署的 controller bundle 必须包含这两个更新文件及 Reward Engine 新源文件。它们此前为实验本地文件，不应只依赖外部仓库 commit 来恢复。

不修改已冻结 release 或已结束实验。配置缺省不传扩展；开启 Critic 但接口缺失时，`run_trial` 在启动容器前报错。provider凭据留在控制器，容器只收到临时trial callback token。Root Rail 的实现唯一位于 `reward_engine/adapters/jiuwen_outcome_rail.py`，不维护第二份。

## 原生轨迹归档依赖

`native_trace.patch` 保存当前 Claw-Eval 的 `jiuwenswarm_trace.py` 与 `jiuwenswarm_schema_replay.py` 两个实验文件，作为外部依赖快照；这些文件原先未被外部仓库跟踪。缺文件的 checkout 可先用 `git apply --check` 检查后应用。已有文件应先反向检查确认版本一致，不能覆盖不同版本。

默认 `native_trial` 和事后 `reexport` 都使用其中的 `argument_formatter` 接口。实际 MCP 参数不匹配原始 wire 时，回放原 Actor 镜像的 SchemaUtils 转换，验证镜像及源码哈希，再精确匹配实际派发参数。回放容器无网络、无任务执行、无凭据、零模型调用；原始参数、实际参数与回放证明均留档。正常 wire 匹配不启动回放。

完整部署包同时包含这两个文件；仅推送 Reward Engine 的 Git commit 不会自动更新外部 Claw-Eval checkout。2026-09-10 已用旧 T068 D 的真实 `Edit` 失败轨迹离线验证：导出 59 条事件，新增 Actor／Grader 调用均为 0。
