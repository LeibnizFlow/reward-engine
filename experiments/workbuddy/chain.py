"""Launch a prepared follow-on campaign only after audited predecessor completion."""

import argparse
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path

from experiments.workbuddy.experiment import verify
from experiments.workbuddy.resume import completed_rounds
from experiments.workbuddy.scoring import scoring_errors
from reward_engine.adapters.workbuddy.io import file_hash, read, write
from reward_engine.core.outcome.pool_selector import VERSION as POOL_SELECTION


def campaign_alive(root, pid):
    if type(pid) is not int:
        return False
    path = Path(f"/proc/{pid}/cmdline")
    try:
        args = path.read_bytes().split(b"\0")
    except FileNotFoundError:
        return False
    return (
        b"experiments.workbuddy.experiment" in args
        and b"run" in args
        and str(root).encode() in args
    )


def audit_completed(root):
    config, state = read(root / "config.json"), read(root / "status.json")
    if state.get("status") != "completed" or campaign_alive(root, state.get("pid")):
        raise ValueError("Predecessor has not completed and exited")
    verify(config)
    if set(state["tasks"]) != set(config["task_ids"]):
        raise ValueError("Predecessor task coverage differs")
    evidence, summaries = {}, []

    def remember(path):
        evidence[str(path)] = file_hash(path)
        return read(path)

    remember(root / "status.json")
    remember(root / "config.json")
    remember(root / "manifest.json")
    for task_id in config["task_ids"]:
        task_root = root / "tasks" / task_id
        task = {
            "task_id": task_id,
            "prompt": (
                Path(config["tasks_root"]) / task_id / "instruction.md"
            ).read_text(),
        }
        seeds = []
        for number in range(1, 4):
            seed = remember(task_root / f"seeds/{number}/RESULT.json")
            if seed.get("status") != "completed":
                raise ValueError("Predecessor seed incomplete")
            seeds.append(seed["candidate"])
        history, checkpoint = completed_rounds(task_root, task, seeds)
        result = remember(task_root / "result.private.json")
        if (
            len(history) != 3
            or len(checkpoint["pool"]) != 12
            or result.get("status") != "completed"
            or result.get("completed_rounds") != 3
            or result["rounds"] != history
            or result["memory"] != checkpoint["memory"]
        ):
            raise ValueError("Predecessor checkpoint or final result incomplete")
        independent = config.get("selection_policy") == POOL_SELECTION
        if independent:
            from reward_engine.adapters.workbuddy.pool_selector import (
                bind_source,
                collect_candidates,
            )
            from reward_engine.core.outcome.pool_selector import decide, prepare

            final_root = task_root / "final_selection"
            candidates, sources = collect_candidates(task_root)
            frozen = remember(final_root / "inputs.json")
            prepared = prepare(
                task, candidates, runtime=frozen["runtime"], previous=frozen
            )
            decision = remember(final_root / "decision.json")
            if (
                remember(final_root / "sources.json") != sources
                or decision != decide(decision["ranking"], prepared)
                or decision["candidate_key"] is None
                or result["selected_delivery"] != bind_source(decision, sources)
                or remember(final_root / "selected_delivery.json")
                != result["selected_delivery"]
            ):
                raise ValueError("Independent final selection inconsistent")
        elif result["selected_delivery"] != history[-1]["selection"]:
            raise ValueError("Final selection differs from completed round")
        scores = 0
        for number, summary in enumerate(history, 1):
            phase = task_root / f"round_{number}"
            remember(phase / "checkpoint.json")
            expected_ranking = "deferred" if independent and number == 3 else "compared"
            if summary["ranking"].get("status") != expected_ranking:
                raise ValueError("Predecessor ranking unavailable")
            if remember(phase / "selected_delivery.json") != summary["selection"]:
                raise ValueError("Predecessor selection inconsistent")
            for arm in "ABC":
                actor = remember(phase / f"actors/{arm}/RESULT.json")
                receipt = remember(phase / f"actors/{arm}/OFFICIAL.private.json")
                native = remember(Path(receipt["native_score"]))
                if (
                    actor.get("status") != "completed"
                    or actor.get("official_verifier_observed") is not False
                    or receipt.get("status") != "scored"
                    or receipt.get("exception")
                    or native.get("overall") != receipt.get("overall")
                    or scoring_errors(native)
                    or native.get("test_status") in {"judge_error", "build_error"}
                ):
                    raise ValueError(
                        f"Predecessor actor or score incomplete: {task_id}/{number}/{arm}"
                    )
                scores += 1
        summaries.append(
            {
                "task_id": task_id,
                "rounds": 3,
                "actors": 12,
                "clean_scores": scores,
                "selected_arm": result["selected_delivery"]["arm"],
            }
        )
    return {
        "status": "verified",
        "predecessor": str(root),
        "audited_at": time.time(),
        "tasks": summaries,
        "evidence_hashes": evidence,
    }


def launch_next(root, config):
    if read(root / "status.json").get("status") != "prepared":
        raise ValueError(
            "Follow-on campaign is not freshly prepared; inspect existing state"
        )
    verify(config)
    env = {**os.environ, "PYTHONPATH": config["code"], "PYTHONUNBUFFERED": "1"}
    processes = {}
    for role, module in (
        ("campaign", "experiments.workbuddy.experiment"),
        ("monitor", "experiments.workbuddy.monitor"),
    ):
        command = (
            [sys.executable, "-m", module]
            + (["run"] if role == "campaign" else [])
            + ["--output", str(root)]
        )
        with (root / f"{role}.log").open("a") as log:
            process = subprocess.Popen(
                command,
                cwd=config["code"],
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        processes[role] = {"pid": process.pid, "command": command}
        write(root / "processes.json", processes)
    return processes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    root, previous = args.output.resolve(), args.after.resolve()
    config = read(root / "config.json")
    if set(config["task_ids"]) & set(read(previous / "config.json")["task_ids"]):
        raise ValueError("Follow-on campaign repeats predecessor tasks")
    with (root / "chain.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            state = read(previous / "status.json")
            live = campaign_alive(previous, state.get("pid"))
            record = {
                "predecessor": str(previous),
                "predecessor_status": state.get("status"),
                "predecessor_alive": live,
                "updated_at": time.time(),
                "pid": os.getpid(),
            }
            if state.get("status") != "completed" or live:
                record["status"] = "waiting" if live else "predecessor_needs_attention"
            else:
                try:
                    audit = audit_completed(previous)
                    verify(config)
                except (ValueError, KeyError, FileNotFoundError) as error:
                    record.update(
                        status="predecessor_needs_attention", reason=str(error)
                    )
                else:
                    write(root / "PREDECESSOR_AUDIT.json", audit)
                    record["status"] = "ready"
                    if not args.check_only:
                        record.update(
                            status="launched", processes=launch_next(root, config)
                        )
                    write(root / "CHAIN_STATUS.json", record)
                    return
            write(root / "CHAIN_STATUS.json", record)
            if args.check_only:
                return
            time.sleep(30)


if __name__ == "__main__":
    main()
