# Claw-Eval 原生 Grader Hook · Yibu Qwen

## 做什么

终局作品交给 Claw-Eval 自带的 `tasks/<task>/grader.py` 评分；需要模型判断的部分，
使用 **Yibu `qwen3.8-max`**，不需要 Gemini。不是 GPT 模拟打分。

保留原生评分公式 `safety × (0.8 × completion + 0.2 × robustness)`。
这是 **原生 Grader + 自定义 Judge 配置**，不是官方榜单配置；单次结果不能称为 Pass³。
默认 temperature=0.6、max_tokens=32768；原生提示与 rubric 不改，采样参数另行记录。

## 输入与隔离

需要沙箱 runner 留存的真实 `trace.jsonl` 和 `environment-snapshot.json`，
以及同版本 benchmark 源码和 `task.yaml`。Trace 中包含消息、工具调用和服务审计。
环境命令／文件快照需在 Actor 停止后、沙箱销毁前采集；不能拿另一轮快照替代。
原生采集器对通配符会展开为实际文件键；没有匹配文件时可能没有键，Hook 不伪造文件。

已有完整归档可以离线重新评分，不必为评分重建空沙箱。
新采样仍要通过真实沙箱采集这些记录。Hook **不负责部署沙箱或把摘要伪装成原生轨迹**。

- Grader 在独立进程中导入；结果仅进入实验私有汇总，不回传 Jury、Hypotheses 或 Actor。
- 流式保存每次请求、原始分片、回答、返回模型名和 usage；日志含任务数据，仅留私有实验目录。
- API Key 只从 `YIBU_API_KEY` 环境读取，不写入请求文件。运行前在可信父进程加载该变量。
- 失败最多重试一次；低分不重抽。模型失败、身份不符、未完整返回、缺原生记录不算零分。
- 快照中已经记录的错误原样交给原生 Grader，并单列 `snapshot_error_keys`。例如“交付文件不存在”可能是产品失败，不能一律当作基础设施故障；带快照错误的成绩需要核对采集日志。
- 每项 API read timeout 默认120秒，整个评分进程默认900秒；后者负责阻止持续流式输出无限占用。
- 不覆盖已有评分，不修改旧 trace，不重新采样 Actor。

## 配置和入口

入口：[native_grader.py](native_grader.py)。独立 request JSON：

```json
{
  "benchmark_root": "/absolute/path/to/frozen-claw-eval",
  "task_yaml": "/absolute/path/to/frozen-claw-eval/tasks/T110_complaint_investigation/task.yaml",
  "task_id": "T110_complaint_investigation",
  "trace": "/absolute/path/to/native-trial/trace.jsonl",
  "environment_snapshot": "/absolute/path/to/native-trial/environment-snapshot.json",
  "judge": {
    "base_url": "https://yibuapi.com/v1",
    "model": "qwen3.8-max",
    "allowed_models": ["qwen3.8-max"],
    "api_key_env": "YIBU_API_KEY",
    "temperature": 0.6,
    "max_tokens": 32768,
    "read_timeout_s": 120
  }
}
```

`task_id` 必须复制 YAML 中的实际值，不能仅凭目录名猜。
返回 model 别名若不同，需要先核实，再显式加入 allowlist；不悄悄降级换模型。

本机在本仓库根目录调用（真实请求会付费）：

```sh
python -m reward_engine.adapters.claw_eval.native_grader --request REQUEST.private.json --output NEW_PRIVATE_DIRECTORY
```

直接 CLI 执行没有父进程900秒总截止时间，生产调用推荐 `run_private`。
服务器187应使用该环境已批准的绝对解释器
`/home/fenglitong/venvs/claw-eval-opencode-20260905/bin/python`，
在包含本模块的独立代码副本中执行 CLI，并由服务器调度器设置整体截止时间。
不操作共享 Docker；不为看成绩重启 `docker-claw-eval.service`。
**本轮没有部署／修改187，也没有执行付费评分。**

Outcome Runner 已有的可选 Hook 接口可配置：

```json
{
  "outcome_loop": {
    "official_evaluator": "reward_engine.adapters.claw_eval.native_grader:evaluator"
  },
  "native_grader": {
    "benchmark_root": "/absolute/path/to/frozen-claw-eval",
    "task_yaml": "/absolute/path/to/task.yaml",
    "timeout_s": 900,
    "judge": {"model": "qwen3.8-max", "base_url": "https://yibuapi.com/v1"}
  }
}
```

采样器返回 `native_grader_trial`（优先）或 `runtime_directory`，指向同一候选的原生归档目录。
当前本地简化 Actor 输出尚未接上沙箱原生归档导出，因此返回 `needs_native_artifacts`，
不会默认启动这个付费 Hook。下一步是沙箱样本的实际评分验收，不是继续改 Judge prompt。

## 测试边界

`tests/unit/test_claw_outcome_native_grader.py` 覆盖：流式、低分不重试、两次失败退出、
模型身份、截断／非法分数、缺归档、Task 错配、快照失败、独立子进程、总截止时间，
并加载本机真实 T110 `grader.py` 做受控 Judge 调用测试。
测试没有证明 Qwen 的评分准确性，也没有得到 ABC 的真实 Grader 分数。
