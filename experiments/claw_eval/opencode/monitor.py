"""Read-only campaign observer; does not generate or retry any model work."""

import argparse
import time
from pathlib import Path

from experiments.claw_eval.opencode.campaign import report
from reward_engine.adapters.workbuddy.io import read, write


def campaign_process_state(output, proc_root=Path("/proc")):
    process = output / "process.json"
    if (output / "acceptance.json").exists():
        return {"status": "completed", "pid": read(process).get("pid") if process.exists() else None}
    if not process.exists():
        return {"status": "not_started", "pid": None}
    pid = read(process).get("pid")
    cmdline = proc_root / str(pid) / "cmdline" if type(pid) is int and pid > 0 else None
    try:
        command = cmdline.read_bytes().replace(b"\0", b" ").decode(errors="replace")
        alive = "experiments.claw_eval.opencode.campaign" in command and str(output) in command
    except (FileNotFoundError, NotADirectoryError, PermissionError, ProcessLookupError):
        alive = False
    return {"status": "running" if alive else "exited", "pid": pid}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = read(args.output / "config.json")
    while True:
        summary = report(config)
        phases = {}
        for task in config["task_ids"]:
            progress = args.output / "tasks" / task / "progress.json"
            if progress.exists():
                phases[task] = read(progress)
        write(
            args.output / "live_status.json",
            {
                "updated_at": time.time(),
                "campaign_process": campaign_process_state(args.output),
                "completed_actors": summary["completed_actors"],
                "completed_tasks": summary["completed_tasks"],
                "valid_scores": summary["valid_scores"],
                "phases": phases,
            },
        )
        if (args.output / "acceptance.json").exists():
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
