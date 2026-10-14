# 发布收尾验证 — 2026-10-14

目标是可安装、可运行核心三轮流程，并与论文核心方法对应；不是复现论文全部实验数字。
本轮补齐公开任务配置的源码导出，统一原生入口的 provider 配置，并新增无需 benchmark 数据的最小完整示例。

## 已验证

- 全量 Python unit/integration：**3,096 passed，0 skipped**，9 条已有 Pydantic 弃用警告，163.09 秒；运行期间源码哈希无变化。
- 上述全量回归后，将根目录 `configs/` 的导出进一步收紧为两份公开任务文件的白名单；**11 项发布相关测试再次通过**，实际检查 GitHub 源码目录和 tar 均不包含私有配置样本。
- Node 适配器：**29 passed**。Ruff lint、format 和 Git whitespace 检查通过。
- 使用发布导出文件集合构建 wheel，在全新虚拟环境安装基础包，从仓库外运行最小示例成功；不依赖本机 editable install、Docker、历史任务数据或第三方 Python 包。
- 示例完成三轮、9 个 A/B/C 候选运行、每轮一次 B 反馈修订，Memory 条目依次为 1、2、3。角色、检查、候选、修订和选择记录均保留。
- `demo` extras 安装、包外导入和 `pip check` 通过。公开任务 contract 与 instruction 均进入源码导出；原生指南中的相对路径以源码 checkout 为工作目录。
- 自定义 provider、环境变量凭据、dotenv 优先级、冻结配置及 actor/role/grader 网关路由均有受控测试。真实模型示例另有 MockTransport 三轮及失败路径测试。

全量回归包含真实 Docker/OpenCode 与本地脚本 provider 的集成测试；它仍属于受控测试。
导出白名单修复之后没有再次运行整套测试；最终回归差异仅为该导出器和对应发布测试，运行代码保持不变。

## 验证边界

最小示例的默认角色是明确标记的确定性脚本，它调用实际核心控制器和检查器。
其成功不是模型能力、质量提升或 native benchmark 成功的证据。真实模型模式的接口已测试，
本轮未完成真实模型运行：配置中的 Yibu token 只读查询返回 `total_available=-3097`、`unlimited_quota=false`。

历史 native 记录仍为：Claw v11 在冻结版本 `33d39e0` 上通过一次三轮严格审计；
WorkBuddy 尚无干净完整严格通过。没有把这些历史结果算成当前源码的新验证。
可选的对抗 QC 等模块不属于默认三轮示例的已验证范围，详见 [pipeline 文档](pipeline-validation.md)。

## 使用与本机记录

从 [快速开始](quickstart.md) 安装并运行。私有 GitHub 仓库的可见性保持 private。
本机记录位于 `runs/release_readiness_20261014_final/`，包含全量测试日志与源码哈希、
最终安装验证、wheel 及最小示例完整输出；这些生成物不提交到源码仓库。
