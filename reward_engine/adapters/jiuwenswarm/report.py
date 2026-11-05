"""Three B reporting from retained public checkpoints and separate native scores."""

from pathlib import Path

from reward_engine.adapters.workbuddy.io import read, write
from reward_engine.adapters.workbuddy.role_audit import request_audit, task_failures
from reward_engine.core.outcome.authored_json import LIMITS
from reward_engine.core.outcome.construction import fingerprint
from reward_engine.core.outcome.three_b import ARMS, BASELINES, PROTOCOL

from .checkpoints import verify_actor


def report(config, destination, *, campaign_accounting=True):
    root, destination = Path(config["output"]), Path(destination)
    rows, completed_actors, scored, finished = [], 0, 0, 0
    for task in config["task_ids"]:
        task_root = root / "tasks" / task
        actor_sessions, failures, grades = set(), [], {}
        actor_retries = {}
        stage_paths = [p for pattern in (
            "seed_*/stage.json", "round_*/*/stage.json",
            "round_*/construction_checks/*/stage.json", "round_*/B*/measurement/stage.json",
            "grades.private/*/stage.json",
        ) for p in task_root.glob(pattern)]
        for path in stage_paths:
            stage = read(path)
            if stage.get("status") == "completed":
                if fingerprint(stage["value"]) != stage.get("value_sha256"):
                    raise ValueError("Report stage content changed")
                relative = path.parent.relative_to(task_root).parts
                actor_stage = (len(relative) == 1 and relative[0] in BASELINES) or (
                    len(relative) == 2 and relative[0].startswith("round_") and relative[1] in ARMS)
                if actor_stage:
                    verify_actor(stage["value"])
                    actor_sessions.add(stage["value"]["session"])
                    if "actor_retry" in stage["value"]:
                        actor_retries[str(path.parent.relative_to(task_root))] = stage["value"]["actor_retry"]
                elif relative[0] == "grades.private":
                    grades[relative[1]] = stage["value"]
            elif (path.parent / "failure.json").exists():
                failure = read(path.parent / "failure.json")
                if failure.get("attempt") == stage.get("attempt"):
                    failures.append({"stage": str(path.parent.relative_to(task_root)), **failure})
        result_path = task_root / "result.private.json"
        result = read(result_path) if result_path.exists() else {}
        protocol_path = task_root / "protocol.json"
        protocol = read(protocol_path) if protocol_path.exists() else {}
        rounds, unique = [], {}
        for number in range(1, config["rounds"] + 1):
            phase = task_root / f"round_{number}"
            path = phase / "summary.private.json"
            if not path.exists():
                rounds.append({"round": number, "status": "failed" if any(
                    f["stage"].startswith(f"round_{number}/") for f in failures) else "pending"})
                continue
            summary = read(path)
            checkpoint = read(phase / "checkpoint.json")
            if checkpoint["generation_protocol"] != PROTOCOL:
                raise ValueError("Report protocol mismatch")
            for key in (*BASELINES, *ARMS):
                state = (
                    read(task_root / (key + ".json"))
                    if key in BASELINES
                    else summary["arms"][key]
                )
                if fingerprint(state["candidate"]) != checkpoint["candidates"][key]:
                    raise ValueError("Report delivery checkpoint mismatch")
                stage = read(
                    (task_root / key if key in BASELINES else phase / key)
                    / "stage.json"
                )
                if (
                    fingerprint(stage["value"]) != stage["value_sha256"]
                    or stage["value"]["candidate"] != state["candidate"]
                ):
                    raise ValueError("Report actor stage mismatch")
                verify_actor(state)
                grade_key = fingerprint({"task": protocol["task"], "candidate": state["candidate"],
                    "session": state["session"], "native": state.get("runtime_directory"), "runtime": protocol["runtime"]})
                grade = read(task_root / "grades.private" / grade_key / "stage.json")
                if (grade["value"] != summary["official"].get(key)
                        or fingerprint(grade["value"]) != grade["value_sha256"]):
                    raise ValueError("Report native grade checkpoint mismatch")
                unique[state["session"]] = summary["official"].get(key, {})
            selection = summary["selection"]
            key = selection["arm"]
            expected = (
                (
                    f"{key}.json"
                    if key in BASELINES
                    else f"round_{number}/{key}/delivery.json"
                )
                if key
                else None
            )
            if selection["source"] != expected or selection != read(
                phase / "selected_delivery.json"
            ):
                raise ValueError("Report selection source mismatch")
            repetitions = {}
            for arm, state in summary["arms"].items():
                repetitions[arm] = {
                    "session": state["session"],
                    "revision_turns": state.get("resources", {}).get("revision_turns"),
                    "tool_errors": state.get("tool_errors"),
                    "usage": state.get("usage"),
                    "score": summary["official"].get(arm),
                }
            rounds.append(
                {
                    "round": number,
                    "status": summary["status"],
                    "selected_arm": key,
                    "selected_source": expected,
                    "selected_score": summary.get("selected_official"),
                    "hypothesis_applied": False,
                    "B_repetitions": repetitions,
                    "scores": summary["official"],
                }
            )
        completed_actors += len(actor_sessions)
        scored += sum(s.get("status") in {"scored", "graded"} for s in grades.values())
        finished += result.get("status") == "completed"
        rows.append(
            {
                "task_id": task,
                "status": result.get("status", "failed" if failures else "pending"),
                "completed_actors": len(actor_sessions),
                "failures": failures,
                "role_failures": task_failures(task_root),
                **({"actor_retries": actor_retries} if actor_retries else {}),
                "rounds": rounds,
            }
        )
    usage = _campaign_usage(root) if campaign_accounting else None
    expected = len(config["task_ids"])
    result = {
        "generation_protocol": PROTOCOL,
        **{key: config[key] for key in ("role_output_protocol", "role_recovery_policy", "role_input_projection", "role_contract_policy", "failure_policy", "implementation_revision") if key in config},
        **({"role_recovery_limits": LIMITS} if config.get("role_output_protocol") else {}),
        "hypothesis_enabled": False,
        "expected_tasks": expected,
        "completed_tasks": finished,
        "expected_actors": expected * (3 + 3 * config["rounds"]),
        "completed_actors": completed_actors,
        "expected_scores": expected * (3 + 3 * config["rounds"]),
        "valid_scores": scored,
        "usage": usage,
        "role_request_audit": request_audit(root) if campaign_accounting else None,
        "tasks": rows,
    }
    if not campaign_accounting:
        result["campaign_accounting_scope"] = "Not collected for this task-only report; full campaign audit remains required"
    write(destination / "report.json", result)
    return result


def _campaign_usage(root):
    usage = {}
    for category in ("transport", "official.private"):
        values, missing, requests = [], 0, 0
        for path in (root / category).glob("*/*/receipt.json"):
            row = read(path)
            attempts = row.get("stream_attempts") or [{"usage": row.get("usage")}]
            requests += row.get("attempts", 1)
            missing += max(0, row.get("attempts", 1) - len(attempts))
            for attempt in attempts:
                value = attempt.get("usage")
                if not value or not all(
                    type(value.get(k)) is int
                    for k in ("prompt_tokens", "completion_tokens")
                ):
                    missing += 1
                else:
                    values.append(value)
        usage[category] = {
            "requests": requests,
            "usage_missing": missing,
            "observed_prompt_tokens": sum(v["prompt_tokens"] for v in values)
            if values
            else None,
            "observed_completion_tokens": sum(v["completion_tokens"] for v in values)
            if values
            else None,
        }
    return usage
