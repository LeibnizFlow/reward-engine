# 历史 joint reward：unknown=0.5 重计算

本报告重算已完成的原始版与修复版历史 joint 判分，不是正在运行的真实 Harness 结果。
每条轨迹的 midpoint=(pass+0.5×unknown)/criterion_count；所有 unknown 包括技术错误均参与。
这是事后赋值假设，不是新增测量证据。原分数、区间、unknown 比例保留，未调用模型。

## 修复版主结果

| 子集 | Benchmark | 算法 | 总题数/完整标量/相关性有定义 | τ-b | ρ | Top-1 | Regret | 配对准确率 | 配对覆盖率 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| all | workbuddy | original | 50/20/12 | 0.110 | 0.114 | 0.308 | 10.361 | 0.221 | 0.520 |
| all | workbuddy | midpoint | 50/50/42 | 0.192 | 0.198 | 0.443 | 10.279 | 0.412 | 1.000 |
| all | claw_eval | original | 104/89/53 | 0.243 | 0.261 | 0.489 | 8.604 | 0.312 | 0.884 |
| all | claw_eval | midpoint | 104/104/68 | 0.246 | 0.259 | 0.490 | 8.824 | 0.354 | 1.000 |
| gap_5 | workbuddy | original | 15/3/2 | 0.000 | 0.000 | 0.278 | 26.596 | 0.111 | 0.400 |
| gap_5 | workbuddy | midpoint | 15/15/14 | -0.024 | -0.036 | 0.422 | 19.051 | 0.378 | 1.000 |
| gap_5 | claw_eval | original | 33/29/20 | 0.262 | 0.285 | 0.431 | 13.873 | 0.363 | 0.919 |
| gap_5 | claw_eval | midpoint | 33/33/24 | 0.274 | 0.300 | 0.470 | 12.379 | 0.384 | 1.000 |
| gap_2 | workbuddy | original | 37/14/7 | 0.233 | 0.247 | 0.310 | 10.352 | 0.211 | 0.514 |
| gap_2 | workbuddy | midpoint | 37/37/30 | 0.176 | 0.182 | 0.428 | 11.935 | 0.396 | 1.000 |
| gap_2 | claw_eval | original | 61/52/31 | 0.290 | 0.307 | 0.423 | 9.868 | 0.301 | 0.891 |
| gap_2 | claw_eval | midpoint | 61/61/40 | 0.320 | 0.335 | 0.451 | 8.971 | 0.350 | 1.000 |
| gap_10 | workbuddy | original | 4/2/1 | -0.816 | -0.866 | 0.167 | 37.616 | 0.000 | 0.667 |
| gap_10 | workbuddy | midpoint | 4/4/3 | -0.605 | -0.622 | 0.208 | 53.394 | 0.083 | 1.000 |
| gap_10 | claw_eval | original | 14/14/7 | 0.276 | 0.281 | 0.417 | 18.361 | 0.238 | 1.000 |
| gap_10 | claw_eval | midpoint | 14/14/7 | 0.276 | 0.281 | 0.417 | 18.361 | 0.238 | 1.000 |

相关性为逐题系数的宏平均，排除不可定义项但报告其数量；没有把所有轨迹混合计算。
Top-1 在三条标量均完整的任务上计算，并列最高分按均匀选择；官方同分的随机基线见 JSON。
严格配对准确率只针对官方严格有序配对，reward 同分算未排对；分母为标量均可用的配对。

## 新增相关性样本与区间

| 子集 | Benchmark | 原相关性任务 | 新增相关性任务 | 新增任务 τ-b | 新增任务 ρ | midpoint τ 95% CI | midpoint ρ 95% CI |
|---|---|---:|---:|---:|---:|---|---|
| all | workbuddy | 12 | 30 | 0.226 | 0.232 | [-0.00986194017302526, 0.3891255705805787] | [-0.014486327244805591, 0.4059060669876685] |
| all | claw_eval | 53 | 15 | 0.253 | 0.249 | [0.1016348629190641, 0.3878314867117111] | [0.10367584915629484, 0.4118635403919143] |
| gap_5 | workbuddy | 2 | 12 | -0.028 | -0.042 | [-0.35235681392502244, 0.3076923076923077] | [-0.3832875288417456, 0.3154700538379252] |
| gap_5 | claw_eval | 20 | 4 | 0.333 | 0.375 | [0.04544852675006655, 0.5005821498083155] | [0.04928203230275509, 0.5482808102950486] |
| gap_2 | workbuddy | 7 | 23 | 0.158 | 0.162 | [-0.0636674124262279, 0.40777598932801734] | [-0.06958898246424819, 0.42684993139281524] |
| gap_2 | claw_eval | 31 | 9 | 0.424 | 0.430 | [0.13736440282807108, 0.4952712529904793] | [0.13650291258125907, 0.5247945386895281] |
| gap_10 | workbuddy | 1 | 2 | -0.500 | -0.500 | [-1.0, 0.0] | [-1.0, 0.0] |
| gap_10 | claw_eval | 7 | 0 | — | — | [-0.2833900285796469, 0.8164965809277259] | [-0.32243597651629824, 0.8660254037844387] |

原来有确定分数的候选没有 unknown，midpoint 与原分数逐条相同。因此同一历史版本在共同支持上的
τ、ρ、Top-1、regret 和严格配对准确率差值全部为零（无法定义则保持空值）。总体均值变化来自新增样本，不能解释为共同样本排序改善。
新增可计算标量任务与新增相关性任务分开统计：全部 unknown 会成为 0.5，同分任务依然没有相关性。

## 两个历史版本与官方运行异常敏感性

| 子集 | Benchmark | 异常处理 | 判分版本 | 算法 | 任务/相关性有定义 | τ-b | ρ |
|---|---|---|---|---|---:|---:|---:|
| all | workbuddy | all_selected | original | original | 50/7 | 0.233 | 0.247 |
| all | workbuddy | all_selected | original | midpoint | 50/43 | 0.142 | 0.153 |
| all | workbuddy | all_selected | repaired | original | 50/12 | 0.110 | 0.114 |
| all | workbuddy | all_selected | repaired | midpoint | 50/42 | 0.192 | 0.198 |
| all | workbuddy | without_confirmed_runtime_errors | original | original | 49/7 | 0.233 | 0.247 |
| all | workbuddy | without_confirmed_runtime_errors | original | midpoint | 49/42 | 0.134 | 0.145 |
| all | workbuddy | without_confirmed_runtime_errors | repaired | original | 49/11 | 0.074 | 0.079 |
| all | workbuddy | without_confirmed_runtime_errors | repaired | midpoint | 49/41 | 0.185 | 0.191 |
| all | claw_eval | all_selected | original | original | 104/43 | 0.216 | 0.239 |
| all | claw_eval | all_selected | original | midpoint | 104/70 | 0.210 | 0.223 |
| all | claw_eval | all_selected | repaired | original | 104/53 | 0.243 | 0.261 |
| all | claw_eval | all_selected | repaired | midpoint | 104/68 | 0.246 | 0.259 |
| all | claw_eval | without_confirmed_runtime_errors | original | original | 104/43 | 0.216 | 0.239 |
| all | claw_eval | without_confirmed_runtime_errors | original | midpoint | 104/70 | 0.210 | 0.223 |
| all | claw_eval | without_confirmed_runtime_errors | repaired | original | 104/53 | 0.243 | 0.261 |
| all | claw_eval | without_confirmed_runtime_errors | repaired | midpoint | 104/68 | 0.246 | 0.259 |
| gap_2 | workbuddy | all_selected | original | original | 37/5 | 0.327 | 0.346 |
| gap_2 | workbuddy | all_selected | original | midpoint | 37/31 | 0.106 | 0.120 |
| gap_2 | workbuddy | all_selected | repaired | original | 37/7 | 0.233 | 0.247 |
| gap_2 | workbuddy | all_selected | repaired | midpoint | 37/30 | 0.176 | 0.182 |
| gap_2 | workbuddy | without_confirmed_runtime_errors | original | original | 37/5 | 0.327 | 0.346 |
| gap_2 | workbuddy | without_confirmed_runtime_errors | original | midpoint | 37/31 | 0.106 | 0.120 |
| gap_2 | workbuddy | without_confirmed_runtime_errors | repaired | original | 37/7 | 0.233 | 0.247 |
| gap_2 | workbuddy | without_confirmed_runtime_errors | repaired | midpoint | 37/30 | 0.176 | 0.182 |
| gap_2 | claw_eval | all_selected | original | original | 61/27 | 0.290 | 0.321 |
| gap_2 | claw_eval | all_selected | original | midpoint | 61/42 | 0.265 | 0.283 |
| gap_2 | claw_eval | all_selected | repaired | original | 61/31 | 0.290 | 0.307 |
| gap_2 | claw_eval | all_selected | repaired | midpoint | 61/40 | 0.320 | 0.335 |
| gap_2 | claw_eval | without_confirmed_runtime_errors | original | original | 61/27 | 0.290 | 0.321 |
| gap_2 | claw_eval | without_confirmed_runtime_errors | original | midpoint | 61/42 | 0.265 | 0.283 |
| gap_2 | claw_eval | without_confirmed_runtime_errors | repaired | original | 61/31 | 0.290 | 0.307 |
| gap_2 | claw_eval | without_confirmed_runtime_errors | repaired | midpoint | 61/40 | 0.320 | 0.335 |
| gap_5 | workbuddy | all_selected | original | original | 15/0 | — | — |
| gap_5 | workbuddy | all_selected | original | midpoint | 15/14 | 0.058 | 0.062 |
| gap_5 | workbuddy | all_selected | repaired | original | 15/2 | 0.000 | 0.000 |
| gap_5 | workbuddy | all_selected | repaired | midpoint | 15/14 | -0.024 | -0.036 |
| gap_5 | workbuddy | without_confirmed_runtime_errors | original | original | 15/0 | — | — |
| gap_5 | workbuddy | without_confirmed_runtime_errors | original | midpoint | 15/14 | 0.058 | 0.062 |
| gap_5 | workbuddy | without_confirmed_runtime_errors | repaired | original | 15/2 | 0.000 | 0.000 |
| gap_5 | workbuddy | without_confirmed_runtime_errors | repaired | midpoint | 15/14 | -0.024 | -0.036 |
| gap_5 | claw_eval | all_selected | original | original | 33/17 | 0.221 | 0.255 |
| gap_5 | claw_eval | all_selected | original | midpoint | 33/25 | 0.203 | 0.233 |
| gap_5 | claw_eval | all_selected | repaired | original | 33/20 | 0.262 | 0.285 |
| gap_5 | claw_eval | all_selected | repaired | midpoint | 33/24 | 0.274 | 0.300 |
| gap_5 | claw_eval | without_confirmed_runtime_errors | original | original | 33/17 | 0.221 | 0.255 |
| gap_5 | claw_eval | without_confirmed_runtime_errors | original | midpoint | 33/25 | 0.203 | 0.233 |
| gap_5 | claw_eval | without_confirmed_runtime_errors | repaired | original | 33/20 | 0.262 | 0.285 |
| gap_5 | claw_eval | without_confirmed_runtime_errors | repaired | midpoint | 33/24 | 0.274 | 0.300 |
| gap_10 | workbuddy | all_selected | original | original | 4/0 | — | — |
| gap_10 | workbuddy | all_selected | original | midpoint | 4/3 | -0.444 | -0.500 |
| gap_10 | workbuddy | all_selected | repaired | original | 4/1 | -0.816 | -0.866 |
| gap_10 | workbuddy | all_selected | repaired | midpoint | 4/3 | -0.605 | -0.622 |
| gap_10 | workbuddy | without_confirmed_runtime_errors | original | original | 4/0 | — | — |
| gap_10 | workbuddy | without_confirmed_runtime_errors | original | midpoint | 4/3 | -0.444 | -0.500 |
| gap_10 | workbuddy | without_confirmed_runtime_errors | repaired | original | 4/1 | -0.816 | -0.866 |
| gap_10 | workbuddy | without_confirmed_runtime_errors | repaired | midpoint | 4/3 | -0.605 | -0.622 |
| gap_10 | claw_eval | all_selected | original | original | 14/6 | 0.378 | 0.411 |
| gap_10 | claw_eval | all_selected | original | midpoint | 14/7 | 0.276 | 0.281 |
| gap_10 | claw_eval | all_selected | repaired | original | 14/7 | 0.276 | 0.281 |
| gap_10 | claw_eval | all_selected | repaired | midpoint | 14/7 | 0.276 | 0.281 |
| gap_10 | claw_eval | without_confirmed_runtime_errors | original | original | 14/6 | 0.378 | 0.411 |
| gap_10 | claw_eval | without_confirmed_runtime_errors | original | midpoint | 14/7 | 0.276 | 0.281 |
| gap_10 | claw_eval | without_confirmed_runtime_errors | repaired | original | 14/7 | 0.276 | 0.281 |
| gap_10 | claw_eval | without_confirmed_runtime_errors | repaired | midpoint | 14/7 | 0.276 | 0.281 |

主结果保留全部选定任务；敏感性只排除既有审查已确认的官方运行错误，不因评分分歧临时排除任务。
所有标准指标使用固定种子、20,000 次聚类 bootstrap（默认）；WorkBuddy 按 task，claw-eval 按 scenario。
完整区间、分母、有效 bootstrap 次数、同分率、完整排序成功率和逐题分类保存在 summary.json 与 CSV。
输入凭据、Pack、manifest、原始评分及既有分析报告的哈希都在 protocol.json；原评分统计已与既有报告校验（若配置参考报告）。
