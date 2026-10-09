# 公开材料与复现

`data/workbuddy_office` 和 `data/claw_eval` 各自只有一个实验入口 `experiment.json`。
`tasks/<task>/` 包含 seed、三轮产物及必要失败尝试，`configurations/` 和 `code/` 记录实际配置与执行代码。
`objects/` 保存与当前任务文件同名但身份不同、仍被评分或历史分析引用的证据，按原始内容哈希寻址。

## GitHub 与 Zenodo

GitHub 导出包含当前代码、测试、轻量最终报告及说明，不包含 `.git`、`archive`、大数据或运行凭据。
Zenodo 包含相同代码以及完整规范化数据。分卷编号仅用于打包，不表示实验批次。
从私有 [evidence release](https://github.com/TheLittleJimmy/reward-engine/releases/tag/evidence-v1.0.0) 下载附件到 `zenodo/`。在当前源码根目录运行 `python -m release.restore --archives zenodo`，该命令先验证校验和，再恢复所有数据卷，不覆盖源码。`source-and-reports.tar.gz` 是原始历史快照，不应覆盖当前修复版。

```sh
python -m release.verify --data-root data --reports reports
python -m release.report --experiment workbuddy_office --data-root data --output /tmp/workbuddy-report
python -m release.report --experiment claw_eval --data-root data --output /tmp/claw-report
```

主报告重建仅依赖 Python 标准库。默认分析需要 Python 3.11+、NumPy 和 SciPy；图表需要 Matplotlib。
安装方式：`python -m pip install -e '.[analysis,dev]'`。环境准备完成后，下面的重建不访问网络、不调用模型或 grader。
输出目录必须尚不存在，不能覆盖冻结输入。

```sh
python -m experiments.analysis --analysis joint --data-root data --output /tmp/joint-analysis
python -m experiments.analysis --analysis score_gap --data-root data --output /tmp/gap-analysis
python -m experiments.analysis --analysis midpoint --data-root data --output /tmp/midpoint-analysis
```

正式分析默认固定随机种子和 20,000 次 bootstrap。降低 `--bootstrap-iterations` 仅适合测试，不是正式置信区间的替代。
WorkBuddy 按任务、Claw 按 scenario 聚类，保留双语版本间依赖。不同分析条件和候选集合不混合。

合并与候选保留规则见 [MERGE_POLICY.md](MERGE_POLICY.md)。

## 文件身份与规范化

`data/provenance/files.jsonl` 逐文件给出公开路径、公开 SHA-256、原始身份 SHA-256、大小及变换说明。
新生成的实验索引同样列入该清单；清单自身由 `release/derived_files.json` 记录校验和。
原始哈希是身份标签，不是脱敏文件的校验和。读取历史回执前先校验公开字节，再核对原始身份。
路径、来源字段与凭据规范化不会改变分数或选择。历史目录映射和未经修改的原始记录仅留在本机归档。
流式响应先拼接 delta，再执行脱敏，保留完成原因、模型返回、usage 及错误事件；不保留逐 token 的传输封装。
数据库缓存由导出的轨迹与回执替代；暂态进度文件不进入发布包。真实失败和实际替换尝试仍在任务证据中。
公开证据用于查验及离线重建，不能作为未经修改的在线续跑 checkpoint 使用。

## 重新导出

```sh
python -m release.package --root . --output dist/publication
```

发布采用白名单导出。不要直接推送当前本机仓库的历史 Git 数据，也不要把 archive 打包。
导出前要求四类完整数据及非空来源清单，并逐文件校验；缺失或损坏会在创建输出前报错。
默认分卷目标为 4 GB；工具同时检查 Zenodo 默认单记录 50 GB、100 文件限制，并保留至少 10 GiB 磁盘空间。
作者、正式仓库地址和自有数据/报告许可由作者补齐。本工具不上传、创建 DOI 或推送 GitHub。

## 仓库外补充材料

见 [整理范围与去向](EXTERNAL_MATERIALS.md) 和 [补充分析入口](../reports/analysis/supplementary/README.md)。13 项离线分析均保留各自条件，原主实验分数不变。
