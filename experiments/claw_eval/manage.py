"""Small, offline-first experiment record/launch helper; never a Reward Builder."""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

if not __package__:  # Historical launch.sh invokes this file from arbitrary cwd.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reward_engine.adapters.support import PYTHON, REPO, read_json, write_json


def load_config(path: Path):
    config = read_json(path)
    if not isinstance(config.get("python"), str) or not Path(config["python"]).is_absolute():
        raise ValueError("Declare an absolute interpreter path in the experiment config.")
    for role, profile_id in config["roles"].items():
        if profile_id not in config["model_profiles"]:
            raise ValueError(f"Unknown model profile for {role}: {profile_id}")
        profile = config["model_profiles"][profile_id]
        if profile["provider"] not in config["providers"]:
            raise ValueError(f"Unknown provider for {role}")
    # Credentials are names only. This helper never reads the env file or API keys.
    for provider in config["providers"].values():
        if any(key in provider for key in ("api_key", "token", "authorization")):
            raise ValueError("Do not store credential values in experiment config.")
    mode = config.get("role_protocol", {}).get("jury_comparison", "pairwise")
    if mode not in {"pairwise", "listwise-semantic-v1", "listwise-joint-v1"}:
        raise ValueError("Unsupported jury_comparison protocol")
    if mode != "pairwise":
        protocol = config["role_protocol"]
        members, reducer = protocol["overall_jury_roles"], protocol["listwise_reducer_role"]
        if (len(members) != 3 or len(set(members)) != 3 or reducer in members
                or any(role not in config["roles"] for role in [*members, reducer])):
            raise ValueError("Listwise needs three members and a separate configured reducer role")
    return config


def resolved_plan(config):
    roles = {}
    for role, profile_id in config["roles"].items():
        profile = config["model_profiles"][profile_id]
        roles[role] = {**profile, "provider_config": config["providers"][profile["provider"]]}
    return {"experiment_id": config["experiment_id"], "status": "planned_not_executed",
            "runtime": config["active_runtime"], "critic_enabled": config["critic_enabled"],
            "six_arm_comparison_enabled": config["six_arm_comparison_enabled"],
            "roles": roles, "algorithm": config["algorithm"],
            "actual_model_calls": None, "actual_trajectories": None}


def readiness(config, root: Path):
    issues = []
    if config["six_arm_comparison_enabled"] or config["active_runtime"] not in {"jiuwen_single", "opencode_single"}:
        issues.append("Select one Jiuwen/OpenCode Single; six-arm launches are deferred.")
    if not config["critic_enabled"]:
        issues.append("The full-algorithm pilot requires Runtime Policy Critic enabled.")
    driver = config.get("driver_path")
    if not driver or not (root / driver).is_file():
        issues.append("Full-pipeline driver is not bound; handwritten smoke is not a substitute.")
    for name, relative in config["inputs"].items():
        if not (root / relative).is_file():
            issues.append(f"Missing {name}: {relative}")
    pool_path = root / config["inputs"]["pool_manifest"]
    if pool_path.is_file():
        pool = read_json(pool_path)
        if pool.get("task_id") != config["task_id"]:
            issues.append("Seed pool belongs to a different Task.")
        if len(pool.get("candidates", [])) < 2:
            issues.append("Fewer than two real seed candidates registered; no A/B evidence yet.")
    return issues


def _git(*args):
    result = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def update_index(root: Path, target: Path, state):
    path = root / "runs" / "INDEX.json"
    if path.is_file():
        index = read_json(path)
        runtime = target.parent.name.removeprefix("pilot_")
        traces = list(target.parent.glob("attempt-*/trace.jsonl"))
        entry = {"runtime": runtime, "status": state["status"], "latest_record": str(target.relative_to(root)),
            "actual_trajectory_count": sum(p.stat().st_size > 0 for p in traces)}
        index.setdefault("single_pilots", {})[runtime] = entry
        # Preserve the original Jiuwen alias; an OpenCode attempt must not relabel it.
        if runtime == "jiuwen_single":
            index.setdefault("pilot", {}).update(entry)
        write_json(path, index)


def prepare(config, root: Path):
    runtime = config["active_runtime"]
    if runtime not in {"jiuwen_single", "opencode_single"}:
        raise ValueError("Select one supported Single runtime")
    parent = root / "runs" / f"pilot_{runtime}"
    parent.mkdir(parents=True, exist_ok=True)
    number = 1
    while True:
        target = parent / f"attempt-{number:03d}"
        try:
            target.mkdir()  # Exclusive creation: keep every previous attempt, including failures.
            break
        except FileExistsError:
            number += 1
    write_json(target / "CONFIG_SNAPSHOT.json", config)
    write_json(target / "RESOLVED_PLAN.json", resolved_plan(config))
    write_json(target / "CODE_CONTEXT.json", {
        "commit": _git("rev-parse", "HEAD"), "branch": _git("branch", "--show-current"),
        "dirty_files": _git("status", "--short"), "python": os.environ.get("REWARD_ENGINE_PYTHON", config["python"]),
        "cwd": str(REPO), "argv": None,
        "note": "Dirty filenames are recorded; this is not a snapshot of uncommitted source contents."})
    state = {"status": "prepared_not_started",
        "prepared_at": datetime.now(timezone.utc).isoformat(), "model_calls_started": False,
        "quality_status": "not_evaluated"}
    write_json(target / "STATUS.json", state)
    update_index(root, target, state)
    return target


def watchdog_budgets(config):
    """The native Actor allowance does not include its later terminal Jury."""
    from reward_engine.core.batch import BatchPolicy

    resource = config["resource_policy"]
    budgets = {"native_s": resource["watchdog_timeout_s"],
        "terminal_jury_s": BatchPolicy(**config.get("map_reduce", {}).get("jury", {})).deadline_s,
        "setup_cleanup_s": resource.get("driver_setup_cleanup_allowance_s", 60),
        "termination_grace_s": resource.get("driver_termination_grace_s", 10)}
    if config.get("role_protocol", {}).get("jury_comparison") in {"listwise-semantic-v1", "listwise-joint-v1"}:
        budgets["terminal_jury_s"] += BatchPolicy(**config.get("map_reduce", {}).get("jury_reduce",
            config.get("map_reduce", {}).get("jury", {}))).deadline_s
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or value <= 0 for value in budgets.values()):
        raise ValueError("Driver watchdog budgets must be finite positive seconds")
    budgets["driver_s"] = budgets["native_s"] + budgets["terminal_jury_s"] + budgets["setup_cleanup_s"]
    return budgets


def _run_driver(command, *, stdout, stderr, timeout_s, termination_grace_s):
    """Own exactly this child session; permit cleanup before killing its tree."""
    process = subprocess.Popen(command, cwd=REPO, stdout=stdout, stderr=stderr,
        start_new_session=True)
    try:
        return process.wait(timeout=timeout_s)
    except BaseException:
        # The CLI handles TERM by cancelling its async task, which lets native
        # runtime and pilot finally blocks save actual receipts. Descendants
        # remain in this run-owned process group, not the manager's/user's group.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=termination_grace_s)
        except subprocess.TimeoutExpired:
            pass
        finally:
            # A direct child may exit while an OpenCode descendant ignores TERM.
            # Kill the owned group even when the driver's wait already returned.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        raise


def run(config, root: Path):
    issues = readiness(config, root)
    if issues:
        raise ValueError("\n".join(issues))
    budgets = watchdog_budgets(config)
    target = prepare(config, root)
    # Future driver must consume the snapshot, not a config being edited during the run.
    command = [os.environ.get("REWARD_ENGINE_PYTHON", config["python"]), str((root / config["driver_path"]).resolve()),
               "--experiment-config", str(target / "CONFIG_SNAPSHOT.json"),
               "--experiment-root", str(root), "--output-root", str(target)]
    context = read_json(target / "CODE_CONTEXT.json")
    write_json(target / "CODE_CONTEXT.json", {**context, "argv": command, "watchdog_budgets": budgets})
    state = read_json(target / "STATUS.json")
    state.update(status="running", started_at=datetime.now(timezone.utc).isoformat())
    write_json(target / "STATUS.json", state)
    update_index(root, target, state)
    try:
        with (target / "stdout.log").open("w") as stdout, (target / "stderr.log").open("w") as stderr:
            returncode = _run_driver(command, stdout=stdout, stderr=stderr,
                timeout_s=budgets["driver_s"], termination_grace_s=budgets["termination_grace_s"])
        required = ("RUN_RECEIPT.json", "EFFECTIVE_CONFIG.json", "MODEL_CALLS.jsonl", "trace.jsonl")
        missing = [name for name in required if not (target / name).is_file() or not (target / name).stat().st_size]
        receipt = read_json(target / "RUN_RECEIPT.json") if not missing else {}
        status = ("failed" if returncode else "completed" if
                  not missing and receipt.get("completed") is True else "incomplete_receipt")
        state.update(status=status, returncode=returncode, missing_records=missing)
    except KeyboardInterrupt:
        state.update(status="interrupted")
    except subprocess.TimeoutExpired:
        state.update(status="timeout", error_type="TimeoutExpired")
    except Exception as error:
        state.update(status="failed", error_type=type(error).__name__)
    finally:
        state.update(finished_at=datetime.now(timezone.utc).isoformat())
        # Entry invocation alone cannot prove an API request or a successful Task.
        state["model_calls_started"] = None
        write_json(target / "STATUS.json", state)
        update_index(root, target, state)
    return target, state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("command", choices=("plan", "check", "prepare", "run"), nargs="?", default="plan")
    args = parser.parse_args()
    path = args.config.resolve()
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    if args.command == "run":
        def interrupt_run(*_):
            raise KeyboardInterrupt("Manager received SIGTERM")
        signal.signal(signal.SIGTERM, interrupt_run)
    try:
        config, root = load_config(path), path.parent
        if args.command == "plan":
            result = resolved_plan(config)
        elif args.command == "check":
            issues = readiness(config, root)
            print(json.dumps({"ready_to_launch": not issues, "issues": issues}, ensure_ascii=False, indent=2))
            return 2 if issues else 0
        elif args.command == "prepare":
            result = {"prepared_record": str(prepare(config, root)), "executed": False}
        else:
            target, state = run(config, root)
            print(json.dumps({"record": str(target), **state}, ensure_ascii=False, indent=2))
            return 0 if state["status"] == "completed" else 1
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(json.dumps({"status": "not_launched", "reason": str(error)}, ensure_ascii=False, indent=2))
        return 2
    finally:
        if args.command == "run":
            signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    raise SystemExit(main())
