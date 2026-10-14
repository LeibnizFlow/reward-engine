# 百炼 Token Plan 接入

使用百炼 Token Plan 的 OpenAI-compatible 端点：
`https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1`。
它与 Coding Plan、按量计费端点不同，见[官方快速开始](https://help.aliyun.com/zh/model-studio/token-plan-personal-quick-start)。

将密钥写入本机私有 `.env.bailian`，权限设为 `600`；该文件已被 Git 忽略。文件中的变量为：

```dotenv
BAILIAN_TOKEN_PLAN_KEY=填入你自己的密钥
REWARD_ENGINE_KEY_ENV=BAILIAN_TOKEN_PLAN_KEY
REWARD_ENGINE_BACKEND_URL=https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
```

## WorkBuddy 完整三轮

先完成[原生环境安装](native-smoke.md)中的环境依赖步骤。
百炼用户使用本页的 `.env.bailian`，将该页默认读取 `.env` 的安装命令替换为：

```sh
.venv/bin/python -m experiments.setup_native --credentials .env.bailian
```

无需另备 YIBU 凭据。安装完成后，从源码仓库根目录使用新的输出目录：

```sh
.venv/bin/python -m experiments.validate_pipeline \
  --benchmark workbuddy --task-id api-usage-explain-cli-l3-001 \
  --credentials .env.bailian --key-env BAILIAN_TOKEN_PLAN_KEY \
  --backend-url https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1 \
  --model-config reward_engine/configs/bailian_token_plan.json \
  --role-thinking disabled \
  --public-delivery-contract configs/public_delivery/api-usage-explain-cli-l3-001.v1.json \
  --output runs/workbuddy_bailian
```

这个显式模型配置使用 `qwen3.8-flash` 运行 actor 和独立的原生 grader；
Jury 与 Hypothesis 的三个席位分别使用 `qwen3.8-max`、`deepseek-v4.1-flash`、`glm-5.3`。
Reducer、Builder 等角色使用 `qwen3.8-max`。Qwen Max 和 GLM 席位输出上限为 65,536 tokens，DeepSeek 内部角色为 32,768 tokens，
全局和单模型并发上限均为 2。角色模型、参数和配置原文在调用前冻结，不会在失败后静默换模型。

GLM-5.3 在当前端点拒绝 `enable_thinking=false`，因此配置为它使用 `reasoning_effort=low`，保留思考。
默认思考曾在真实 Jury 请求中耗尽 32,768 tokens，导致最终 JSON 截断；该失败记录不能算作完整流程通过。
推理强度的支持范围见[官方 GLM 文档](https://help.aliyun.com/zh/model-studio/glm)。
Qwen Max 也显式使用 `reasoning_effort=low`，以保留有限推理来处理复杂协议；
该参数及预算经真实端点连通性检查，完整流程仍须单独验证。参数范围见[官方 Chat Completions 文档](https://help.aliyun.com/zh/model-studio/qwen-api-via-openai-chat-completions/)。
DeepSeek 内部角色使用命令中的关闭思考请求。是否实际产生思考 tokens，以回执用量为准。
未知配置字段、不支持的角色名以及执行时覆盖模型配置都会报错。

这些模型由[官方模型列表](https://help.aliyun.com/zh/model-studio/token-plan-personal-overview)列出；
可用性和套餐权限可能变化。原生 grader 的模型与历史实验不同，因此分数不能直接当作论文复现数值。
官方评分不进入 Jury 学习或候选选择。

运行结束后按 [pipeline 验证指南](pipeline-validation.md)执行严格审计与 public-contract 审计。
只有实际阶段记录通过审计，才能宣称对应完整路径已验证；模型连通性检查不能替代它。

## 小型真实模型流程

不需要 Docker 或 benchmark 数据的入口：

```sh
.venv/bin/python -m pip install '.[demo]'
.venv/bin/python -m reward_engine.examples.minimal --live \
  --model qwen3.8-max \
  --base-url https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1 \
  --key-env BAILIAN_TOKEN_PLAN_KEY --credentials .env.bailian \
  --request-timeout 600 --stage-timeout 1800 \
  --role-concurrency 1 --thinking disabled \
  --jury-repair-mode targeted \
  --output runs/minimal_bailian
```

这是真实模型调用，但不是 WorkBuddy 原生测试。它不会在 provider 或协议失败时切换成脚本结果。
这些选项显式降低并发，并给模型更长的响应时间；实际参数记录在运行目录中。
`targeted` 显式选择原生适配器使用的 Jury coverage/edits 协议，最多两次外层修复；
默认 `full` 协议最多一次。内层证据读取、协议修复和阶段超时仍由核心控制，实际调用可能多于外层次数。
失败记录保留 HTTP 状态或超时类型，不保存密钥或未经筛选的错误响应。
