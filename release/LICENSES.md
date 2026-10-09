# 许可与第三方材料

| 内容 | 许可 / 来源 |
|---|---|
| 当前项目自有代码 | 根目录 `LICENSE`，Apache-2.0 |
| 自有实验数据、产物与报告 | 作者尚未指定公开许可证；正式上传前必须确定 |
| WorkBuddy Bench 输入和上游代码 | `data/workbuddy_office/benchmark/LICENSE`，Tencent 自定义 WorkBuddy Bench 许可；包含地域限制，不能改标为 Apache 或 CC-BY |
| OpenCode 1.18.16 可执行文件与插件 | [MIT 文本](third_party/OpenCode_LICENSE.txt)，补附自 [官方 v1.18.16](https://raw.githubusercontent.com/anomalyco/opencode/v1.18.16/LICENSE)；取得日期 2026-10-06 |
| Claw-Eval 上游 | `data/claw_eval/benchmark/LICENSE`；上游 README 与官方许可证为 MIT |
| 附件、抓取页面、引用文档及内嵌软件 | 保留各自来源与原许可；不因进入实验包而改变权利归属 |

WorkBuddy 官方来源：https://github.com/Tencent/workbuddy-bench

Claw 官方来源：https://github.com/claw-eval/claw-eval 与 https://huggingface.co/datasets/claw-eval/Claw-Eval

Claw 本机冻结副本未附 LICENSE，也未记录可确认的上游 commit；已依据官方上游补附 MIT 文本并记录取得来源。
实际实验使用的文件身份以逐文件 SHA-256 为准，不声称补附许可证时的 main 分支就是当时执行版本。
冻结配置中的镜像身份与二进制身份保留；重新运行需要用户自行准备相应环境和模型访问权限。
