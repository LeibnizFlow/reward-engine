"""Claw-Eval reporting for both runtime backends and retained protocols."""

import csv
import statistics
import time
from collections import defaultdict
from pathlib import Path

from reward_engine.adapters.workbuddy.io import read, write


def report(config):
    if config.get("generation_protocol") == "three_b_fixed_baselines_v1":
        from reward_engine.adapters.jiuwenswarm.report import report as three_b_report
        from reward_engine.adapters.paths import report_directory
        return three_b_report(config, report_directory("claw_eval", config["output"]))
    root = Path(config["output"])
    expected_tasks = len(config["task_ids"])
    expected_actors = expected_tasks * (config["seeds"] + len("ABC"))
    expected_scores = expected_tasks * len("ABC")
    tasks = {}
    rows = []
    for tid in config["task_ids"]:
        folder = root / "tasks" / tid
        state = (
            read(folder / "state.json")
            if (folder / "state.json").exists()
            else {"status": "pending"}
        )
        scores = {}
        for arm in "ABC":
            path = folder / "round_1/actors" / arm / "OFFICIAL.private.json"
            value = read(path) if path.exists() else {}
            score = value.get("task_score") if value.get("status") == "graded" else None
            scores[arm] = score
            rows.append(
                {
                    "task_id": tid,
                    "arm": arm,
                    "status": value.get("status", "pending"),
                    "score": score,
                }
            )
        selected = None
        result_file = folder / "result.private.json"
        if result_file.exists():
            selected = read(result_file).get("selected_delivery")
        tasks[tid] = {**state, "scores": scores, "selected_delivery": selected}
    actors = sum(
        1
        for p in (root / "tasks").glob("*/seeds/*/RESULT.json")
        if read(p)["status"] == "completed"
    )
    actors += sum(
        1
        for p in (root / "tasks").glob("*/round_1/actors/*/RESULT.json")
        if read(p)["status"] == "completed"
    )
    data = {
        "expected_tasks": expected_tasks,
        "expected_actors": expected_actors,
        "expected_scores": expected_scores,
        "completed_actors": actors,
        "valid_scores": sum(r["score"] is not None for r in rows),
        "completed_tasks": sum(t["status"] == "completed" for t in tasks.values()),
        "tasks": tasks,
        "at": time.time(),
    }
    data["arm_means"] = {
        arm: {"n": len(values), "mean": statistics.mean(values) if values else None}
        for arm in "ABC"
        for values in [
            [
                row["score"]
                for row in rows
                if row["arm"] == arm and row["score"] is not None
            ]
        ]
    }
    data["paired_differences"] = {}
    for arm in "BC":
        differences = [
            t["scores"][arm] - t["scores"]["A"]
            for t in tasks.values()
            if t["scores"][arm] is not None and t["scores"]["A"] is not None
        ]
        data["paired_differences"][arm + "-A"] = {
            "n": len(differences),
            "mean": statistics.mean(differences) if differences else None,
        }
    usage = defaultdict(
        lambda: {
            "attempts": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "missing_usage": 0,
        }
    )
    for section in ("transport", "official.private"):
        for path in (root / section).glob("*/*/receipt.json"):
            receipt = read(path)
            totals = usage[section + "/" + receipt["model"]]
            attempts = receipt.get("stream_attempts") or [
                {"usage": receipt.get("usage")}
            ]
            totals["attempts"] += receipt.get("attempts", 1)
            totals["missing_usage"] += max(
                0, receipt.get("attempts", 1) - len(attempts)
            )
            for attempt in attempts:
                item = attempt.get("usage") or {}
                totals["missing_usage"] += not bool(item)
                for key in ("prompt_tokens", "completion_tokens"):
                    totals[key] += item.get(key, 0) or 0
    data["usage"] = dict(usage)
    data["monetary_cost"] = {
        "status": "not_available",
        "reason": "Provider usage retained; no verified account pricing supplied",
    }
    write(root / "report.json", data)
    with (root / "scores.csv").open("w") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["task_id", "arm", "status", "score"]
        )
        writer.writeheader()
        writer.writerows(rows)
    (root / "REPORT.md").write_text(
        f"# Claw-Eval {expected_tasks} tasks · OpenCode\n\nActors {actors}/{expected_actors}; valid scores {data['valid_scores']}/{expected_scores}; completed tasks {data['completed_tasks']}/{expected_tasks}.\n\n"
        + "| Task | Status | A | B | C |\n|---|---|---|---|---|\n"
        + "".join(
            f"| {tid} | {v['status']} | "
            + " | ".join(
                "—" if v["scores"][a] is None else f"{v['scores'][a]:.4f}"
                for a in "ABC"
            )
            + " |\n"
            for tid, v in tasks.items()
        )
    )
    return data
