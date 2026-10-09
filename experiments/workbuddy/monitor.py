"""Refresh read-only progress and reports while a detached campaign runs."""

import argparse
import time
from pathlib import Path

from experiments.workbuddy.report import report
from reward_engine.adapters.workbuddy.io import read, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    while True:
        state = read(root / "status.json")
        summary = report(root)
        task = state.get("current_task") or next(reversed(state.get("tasks", {})), None)
        progress = root / "tasks" / str(task) / "progress.json"
        actors = [read(p) for p in (root / "tasks").glob("**/RESULT.json")]
        current = state.get("current_tasks") or ([task] if state.get("current_task") else [])
        phases = {}
        for name in current:
            path = root / "tasks" / name / "progress.json"
            phases[name] = read(path) if path.exists() else {"phase": "seeds"}
        write(root / "monitor.json", {
            "updated_at": time.time(), "campaign_status": state["status"], "current_task": task,
            "phase": read(progress) if progress.exists() else {"phase": "seeds"},
            "current_tasks": current, "phases": phases,
            "completed_tasks": sum(row.get("status") == "completed" for row in state.get("tasks", {}).values()),
            "completed_actors": sum(row.get("status") == "completed" for row in actors),
            "reused_actors": summary["reused_actors"], "reused_scores": summary["reused_scores"],
            "failed_actors": sum(row.get("status") == "failed" for row in actors),
            "expected_actors": summary["expected_actors"], "valid_scores": summary["valid_scores"], "expected_scores": summary["expected_scores"],
        })
        if state["status"] not in {"prepared", "running"}:
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
