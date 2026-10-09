# 仓库外材料的整理

公开材料沿用原结构。新增 seed 评分放在 `data/workbuddy_office/tasks/<task>/seeds/<n>/native_grading/`，重复的工作区与 verifier 文件通过 `data/workbuddy_office/seed_grading/evidence_index.json` 引用已有文件。原始包中的重复成员不重复计为新产物。

Claw 独立 verifier 的原始、技术修复和引用修复条件放在 `data/claw_eval/verifier_analysis/`。完成的补充分析与代码放在 `data/analysis/supplementary/`；GitHub 同时提供冻结代码副本 `experiments/analysis/supplementary_sources/`。统一离线入口为 `python -m experiments.analysis.supplementary`，不启动在线实验。Jury 依赖模块只保留所需的纯函数，已停止研究的执行器不公开为活动入口。

上游 WorkBuddy Office 下载包有 678 个文件，其中 627 个与已保留身份相同，51 个是不同版本；不同版本置于 `benchmark/distribution/`，不替换当时实际执行的输入。Claw 下载包中与 104 个任务相关的 277 个附件都已有副本。两者的 `benchmark/distribution_index.json` 给出逐成员绑定。Claw 的上游排行榜快照不属于本次实验结果。

原始下载目录已迁入本机 `archive/external_cleanup_20261006/originals/benchmarks/`，旧路径通过相对软链接继续可用。论文仓库中的实验材料、代码和数据已冻结到同一 archive 下的 `originals/manuscript/`，原论文工作区保留。已停止的排名扩展和被替代的 probe、下载缓存以及其他 benchmark 子集均只归档，不混入公开主实验。迁移、哈希比对和原始到公开路径映射留在本机归档。

署名和自有数据/报告许可仍待作者填写；没有上传、推送、运行模型或恢复已停止的研究。
