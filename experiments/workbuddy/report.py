"""Scores and usage remain reporting-only; missing results retain denominators."""

import csv
import statistics
import time
from collections import defaultdict
from pathlib import Path

from experiments.workbuddy.scoring import scoring_errors
from reward_engine.core.outcome.pool_selector import VERSION as POOL_SELECTION

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reward_engine.adapters.paths import report_directory
from reward_engine.adapters.workbuddy.io import read, write


def report(root, report_dir=None):
    root = Path(root)
    config = read(root / "config.json")
    destination = report_directory("workbuddy", root, report_dir or config.get("report_dir"))
    rows, groups, usage = [], defaultdict(list), defaultdict(lambda: {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0})
    requested = len(config["task_ids"])
    selections = {}
    final_deliveries = {}
    for task in config["task_ids"]:
        task_root = root / "tasks" / task
        for number in range(1, 4):
            selected = task_root / f"round_{number}/selected_delivery.json"
            selection = read(selected).get("arm") if selected.exists() else None
            for arm in "ABC":
                path = task_root / f"round_{number}/actors/{arm}/OFFICIAL.private.json"
                result = read(path) if path.exists() else {"status": "not_run", "overall": None}
                native = Path(result["native_score"]) if result.get("native_score") else None
                errors = scoring_errors(read(native)) if native and native.exists() else result.get("scoring_errors", [])
                status = "judge_error" if errors else result["status"]
                row = {"task_id": task, "round": number, "arm": arm, "selected": arm == selection,
                       "reused": bool(result.get("reused_from")),
                       "status": status, "overall": result.get("overall") if not errors else None,
                       "raw_overall": result.get("raw_overall", result.get("overall")),
                       "judge_errors": len(errors), "trial": result.get("trial")}
                rows.append(row)
                if status == "scored":
                    groups[f"round_{number}_{arm}"].append(row["overall"])
                    if arm == selection:
                        groups[f"round_{number}_selected"].append(row["overall"])
                        if number == 3:
                            selections[task] = row
        if config.get('selection_policy') == POOL_SELECTION:
            from experiments.workbuddy.pool_selection_report import selected_score
            path = task_root / 'final_selection/selected_delivery.json'
            selected = read(path) if path.exists() else {'status': 'pending'}
            task_result = task_root / 'result.private.json'
            if task_result.exists():
                selected = read(task_result).get('selected_delivery') or {'status': 'pending'}
            try:
                final = selected_score(task_root, selected)
            except (ValueError, KeyError, FileNotFoundError):
                final = {'selection_status': 'invalid_source', 'status': 'invalid_source', 'overall': None}
            final_deliveries[task] = final
            selections.pop(task, None)
            if final['status'] == 'scored':
                selections[task] = {'task_id': task, **final}
                groups['final_pool_selected'].append(final['overall'])
    for category in ("transport", "official.private"):
        for path in (root / category).glob("*/*/receipt.json"):
            record = read(path)
            key = category + "/" + record["model"]
            usage[key]["requests"] += record.get("attempts", 1)
            billed = [a.get("usage") for a in record["stream_attempts"]] if record.get("stream_attempts") else [record.get("usage")]
            for item in billed:
                for name in ("prompt_tokens", "completion_tokens"):
                    usage[key][name] += (item or {}).get(name, 0) or 0
    summary = {"requested_tasks": requested, "valid_final_selected": len(selections),
               "valid_scores": sum(row["status"] == "scored" for row in rows), "expected_scores": requested * 9, "expected_actors": requested * 12,
               "means": {key: {"mean": statistics.mean(values), "n": len(values), "requested_n": requested}
                         for key, values in groups.items()}, "usage": dict(usage), "final_selected": selections}
    summary["scores_with_judge_errors"] = sum(row["judge_errors"] > 0 for row in rows)
    if config.get('selection_policy') == POOL_SELECTION:
        summary['final_deliveries'] = final_deliveries
        summary['final_selection_policy'] = POOL_SELECTION
    # Paired changes use exactly the tasks with both scores, never a different denominator.
    pairs = {}
    for number in range(1, 4):
        by_task = {task: {r["arm"]: r["overall"] for r in rows
                         if r["task_id"] == task and r["round"] == number and r["status"] == "scored"}
                   for task in config["task_ids"]}
        for arm in "BC":
            differences = [scores[arm] - scores["A"] for scores in by_task.values() if arm in scores and "A" in scores]
            pairs[f"round_{number}_{arm}-A"] = {"mean_difference": statistics.mean(differences) if differences else None,
                                               "n": len(differences)}
    summary["reused_scores"] = sum(row["reused"] for row in rows)
    summary["reused_actors"] = len(list((root / "tasks").glob("**/REUSE.json")))
    summary["reuse_first_round_from"] = config.get("reuse_first_round_from")
    summary["paired_changes"] = pairs
    state = read(root / "status.json") if (root / "status.json").exists() else {}
    summary["task_status"] = state.get("tasks", {})
    summary["wall_s"] = time.time() - state["started_at"] if state.get("started_at") else None
    resources = [read(p).get("resources", {}) for p in (root / "tasks").glob("**/ACTOR_RESULT.json")]
    summary["actor_resources"] = {key: sum(r.get(key, 0) for r in resources)
                                  for key in ("actor_execution_s", "actor_and_review_s", "turns")}
    summary["usage_missing"] = 0
    for category in ("transport", "official.private"):
        for path in (root / category).glob("*/*/receipt.json"):
            receipt = read(path)
            attempts = receipt.get("stream_attempts") or [{"usage": receipt.get("usage")}]
            summary["usage_missing"] += sum(not a.get("usage") for a in attempts)
            summary["usage_missing"] += max(0, receipt.get("attempts", 1) - len(attempts))
    summary["recovery_runs"], seen, previous = [], {str(root.resolve())}, root
    combined_usage = {key: dict(value) for key, value in usage.items()}
    while (previous / "RECOVERY.json").exists():
        source = read(previous / "RECOVERY.json").get("previous_run")
        if not source or str(Path(source).resolve()) in seen:
            break
        previous = Path(source)
        seen.add(str(previous.resolve()))
        if not (previous / "report.json").exists():
            break
        prior = read(previous / "report.json")
        summary["recovery_runs"].append({"output": str(previous), "usage": prior.get("usage", {}),
                                         "usage_missing": prior.get("usage_missing")})
        for key, values in prior.get("usage", {}).items():
            target = combined_usage.setdefault(key, {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0})
            for field in target:
                target[field] += values.get(field, 0)
    summary["usage_including_recovery"] = combined_usage
    write(destination / "report.json", summary)
    with (destination / "scores.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [f"# WorkBuddy Office · Outcome（{requested} 题）", "",
             f"最终选中作品有效评分：{len(selections)}/{requested}；逐轮 A/B/C 有效评分：{summary['valid_scores']}/{summary['expected_scores']}。",
             "缺失评分不计为零，均分必须结合有效样本数阅读。", "",
             f"含逐项评审错误的评分：{summary['scores_with_judge_errors']}；保留原始值，但不计入有效评分和均分。", "",
             f"| 组别 | 均分 | 有效题数 / {requested} |", "|---|---:|---:|"]
    lines.extend(f"| {key} | {value['mean']:.4f} | {value['n']} |" for key, value in summary["means"].items())
    if config.get('selection_policy') == POOL_SELECTION:
        lines += ['', '全池终评与逐轮选择分开统计；获选 seed 未评分时不记零，也不自动补评分。', '',
                  '| 题目 | 终评状态 | 来源 | 官方评分 |', '|---|---|---|---:|']
        for task, final in final_deliveries.items():
            label = (f"{final.get('source_kind')}/{final.get('source_round')}/{final.get('source_slot')}"
                     if final.get('source') else '未选择')
            score = final['overall'] if final['overall'] is not None else '未评分（' + final['status'] + '）'
            lines.append(f"| {task} | {final['selection_status']} | {label} | {score} |")
    if config.get("reuse_first_round_from"):
        lines += ["", f"复用旧种子/作品：{summary['reused_actors']}；沿用原生评分：{summary['reused_scores']}。",
                  "第一轮 A/B/C 为旧作品，由当前 Jury 重新评审；B/C 生成时使用原运行的指导。",
                  "复用评分不是新增评分调用；本轮调用量不含旧作品的历史生成成本。", ""]
    lines += ["", "逐题状态与失败见 status.json；逐轮分数及交付目录见 scores.csv；调用量见 report.json。",
              "评分使用固定的本机 WorkBuddy 原生评分器及 Yibu DeepSeek V4 Flash；不声称复现官方榜单模型配置。"]
    (destination / "report.md").write_text("\n".join(lines) + "\n")
    if (root / "status.json").exists():
        write(destination / "status.json", read(root / "status.json"))
    return summary
