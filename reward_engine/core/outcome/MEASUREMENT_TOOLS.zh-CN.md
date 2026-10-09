# 真实测量：工具先执行，模型再解释

过去只有 JSON 解析与文本审查，现在把测量能力直接接进 Builder 和交付前检查，没有新增一层裁判。

| 工具 | 当前能力 | 不代表什么 |
|---|---|---|
| `calculate` | 实际计算数值表达式 | 表达式本身一定符合作品主张 |
| `code.run` | 解释执行受限数值 Python 子集：输入 JSON、赋值、循环、条件、断言、算术 | 任意 Python、依赖安装或 CUDA 执行 |
| `document.read` | 读取传入的真实证据；可配置 Tavily 补读已观察到的 HTTPS URL | 搜索摘要自动是真值、可读取隐藏 Grader／任意本机文件 |
| `cuda.run` | 调用可信配置绑定的隔离 GPU 后端 | 默认已经有 GPU，或模型说跑过就算跑过 |

## 执行流程

1. Builder 看真实能力清单，为标准选择检查方法；可附 `probes`。
2. 缺失 checker 时默认调用一次 Author，产出工具 recipe。真正试跑后才绑定；失败保持未实现，不递归修理。
3. recipe 与知识来源随 Pack 保存。新进程加载同一 Pack，即可恢复生成的 checker，不依赖旧进程内存。
4. 检查模型可以返回 `{"need_tools":[{"tool":"calculate","arguments":{"expression":"2+3"}}]}`。
5. 程序执行、保存原始收据，把 `tool:<test_id>:<序号>` 交回模型。模型输出“主张→证据→判断”，必须引用真实作品，不能只引用标准或计算结果。

工具收据是 observed／unknown，检查结论是 pass／fail／unknown，两者不同。工具失败不等于作品错误；计算正确不等于表达式选对；调用成功不等于官方 Grader 通过。
最多两轮工具请求、总计四次测量。每份工具收据单独保存，不被下一次调用覆盖。最终语义判断仍标 `model_assisted`。

## 入口配置

Outcome 实验入口、T068 构建与交付检查使用同一 `adapters/outcome_measurement.py`：

```json
{"measurement": {"external_documents": true}}
```

原文补读按需读取 `TAVILY_API_KEY`；只向读取服务发送已观察的 URL，不发送整题或交付物。默认关闭外部读取时，已保存证据仍可读取。T068 的下一次计划已开启；历史冻结配置不改。

CUDA 由操作者提供可信异步适配器 `async run(source, inputs)`：

```json
{"measurement": {"cuda_runner": "trusted_gpu_adapter:run", "cuda_isolated": true}}
```

适配器负责真实隔离、设备与执行，返回 `executed=true`、实际 `device`、`exit_code` 及测量数据。该配置不是模型输出，也不是凭这个标志就建立了隔离。当前未配置／未验证 GPU 后端，不能声明 CUDA 性能已实测。

数值代码不用 `eval/exec`，无 import、文件、网络、属性访问；源码／输入／输出与运算步数有界。它不是通用 Python sandbox。更广的执行能力需可信隔离后端，不能悄悄在 Mac 主进程运行生成代码。

测试入口：`tests/unit/test_claw_outcome_measurement_tools.py`，覆盖真实 CPU 计算、代码限制、来源读取、工具失败、伪收据拒绝、自动补写、新进程恢复和入口接线。模型与外部服务采用测试替身；不把这些单元测试称为模型质量／GPU 测试。
