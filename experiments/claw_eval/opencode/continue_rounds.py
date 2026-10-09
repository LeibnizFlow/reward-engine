"""Resume accepted Claw round one through the WorkBuddy checkpoint protocol."""

import argparse
import asyncio
import fcntl
import os
import shutil
import sys
import time
from pathlib import Path
from uuid import uuid4

import uvicorn

from experiments.claw_eval.opencode.campaign import grade, probe, verify_freeze
from experiments.claw_eval.opencode.runtime import Runtime, free_port, public_task
from experiments.workbuddy.resume import completed_rounds
from reward_engine.adapters.workbuddy.gateway import Capacity, application
from reward_engine.adapters.workbuddy.io import digest, file_hash, read, write
from reward_engine.adapters.workbuddy.models import Role, registry
from reward_engine.core.batch import BatchPolicy
from reward_engine.core.outcome.loop import run_task
from reward_engine.core.outcome.memory import Memory
from reward_engine.core.outcome.runner import DeliveryReview


def import_first_round(source, target, task):
    """Copy durable metadata, retain original artifact paths, audit later score repair."""
    original = source / "tasks" / task["task_id"]
    target.mkdir(parents=True)
    hashes = {}

    def copy(path):
        relative = path.relative_to(original)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        hashes[str(path)] = file_hash(path)

    for path in (original / "round_1").glob("*.json"):
        copy(path)
    for number in range(1, 4):
        copy(original / "seeds" / str(number) / "RESULT.json")
    summary_path = target / "round_1/summary.private.json"
    summary = read(summary_path)
    repairs = []
    for arm in "ABC":
        slot = original / "round_1/actors" / arm
        for filename in ("RESULT.json", "OFFICIAL.private.json"):
            copy(slot / filename)
        official = read(slot / "OFFICIAL.private.json")
        actor = read(slot / "RESULT.json")
        trial = Path(actor["native_grader_trial"])
        for filename in ("trace.jsonl", "environment-snapshot.json"):
            hashes[str(trial / filename)] = file_hash(trial / filename)
        if (
            official.get("status") != "graded"
            or official["binding"]["trace"] != file_hash(trial / "trace.jsonl")
            or official["binding"]["snapshot"]
            != file_hash(trial / "environment-snapshot.json")
        ):
            raise ValueError("Unbound or ungraded source receipt: " + str(slot))
        if summary["official"][arm] != official:
            if summary["official"][arm].get("status") != "failed":
                raise ValueError("Refusing to replace a successful historical score")
            repairs.append(
                {
                    "arm": arm,
                    "original": summary["official"][arm],
                    "accepted_receipt_sha256": file_hash(
                        slot / "OFFICIAL.private.json"
                    ),
                }
            )
            summary["official"][arm] = official
    if repairs:
        shutil.copy2(summary_path, target / "source_summary.private.json")
        write(summary_path, summary)
    seeds = [
        read(target / "seeds" / str(i) / "RESULT.json")["candidate"]
        for i in range(1, 4)
    ]
    history, checkpoint = completed_rounds(target, task, seeds)
    if len(history) != 1 or len(checkpoint["pool"]) != 6:
        raise ValueError(
            "Source must contain exactly one full ABC round and three seeds"
        )
    write(
        target / "source_reuse.json",
        {
            "source": str(original),
            "files": hashes,
            "official_receipt_repairs": repairs,
            "learning_state_changed": False,
        },
    )
    write(target / "state.json", {"status": "pending", "completed_rounds": 1})
    return hashes


def prepare(
    source,
    output,
    task_ids=None,
    task_concurrency=2,
    model_concurrency=4,
    docker_min_free_gib=20,
):
    source, output = source.resolve(), output.resolve()
    cfg = read(source / "config.json")
    verify_freeze(cfg)
    if not read(source / "acceptance.json").get("accepted"):
        raise ValueError("Source campaign has not passed acceptance")
    selected = task_ids or cfg["task_ids"]
    if (
        not selected
        or len(set(selected)) != len(selected)
        or not set(selected) <= set(cfg["task_ids"])
    ):
        raise ValueError("Task selection must be a unique source subset")
    output.mkdir(parents=True, exist_ok=False)
    # Use each cohort's actual frozen runtime, model roles, search and tool limits.
    shutil.copytree(
        source / "code",
        output / "code",
        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"),
    )
    module = Path("experiments/claw_eval/opencode/continue_rounds.py")
    shutil.copy2(Path(__file__), output / "code" / module)
    gateway, tool = free_port(), free_port()
    cfg.update(
        output=str(output),
        code=str(output / "code"),
        campaign_id=output.name,
        task_ids=selected,
        rounds=3,
        source_campaign=str(source),
        continuation_protocol="workbuddy_checkpoint_v1",
        task_concurrency=task_concurrency,
        model_concurrency=model_concurrency,
        max_task_concurrency=task_concurrency,
        max_model_concurrency=model_concurrency,
        per_model_concurrency=model_concurrency,
        docker_min_free_gib=docker_min_free_gib,
        auth_file=str(output / "campaign.auth"),
        gateway_port=gateway,
        tools_port=tool,
        gateway_local=f"http://127.0.0.1:{gateway}",
        gateway_container=f"http://host.docker.internal:{gateway}",
        tools_container=f"http://host.docker.internal:{tool}",
        canaries=[selected[0]],
        recover_changed_actor_identity=False,
    )
    cfg.pop("identity", None)
    cfg["identity"] = digest(cfg)
    token = output / "campaign.auth"
    token.write_text(uuid4().hex + uuid4().hex)
    token.chmod(0o600)
    sys.path.insert(0, str(Path(cfg["benchmark_root"]) / "src"))
    bindings = {}
    for tid in selected:
        _, task = public_task(cfg, tid)
        bindings.update(import_first_round(source, output / "tasks" / tid, task))
    bindings[str(source / "manifest.json")] = file_hash(source / "manifest.json")
    bindings[str(source / "config.json")] = file_hash(source / "config.json")
    files = {
        str(p.relative_to(output)): file_hash(p)
        for p in (output / "code").rglob("*")
        if p.is_file()
    }
    files.update(
        {
            str(p.relative_to(output)): file_hash(p)
            for p in (output / "tasks").rglob("*.json")
            if p.name != "state.json"
        }
    )
    write(output / "config.json", cfg)
    write(
        output / "manifest.json",
        {"files": files, "source_files": bindings, "config_identity": cfg["identity"]},
    )
    write(
        output / "protocol.json",
        {
            "source_campaign": str(source),
            "preserved_rounds": [1],
            "new_rounds": [2, 3],
            "round_updates": [
                "pool",
                "memory",
                "checks",
                "pack",
                "expectation_cache",
                "known_gaps",
                "decisions",
            ],
            "official_scores": "evaluation only, never learning or selection inputs",
            "source_code": "exact source freeze plus continuation driver",
            "actor_resampling": False,
            "docker_min_free_gib": docker_min_free_gib,
        },
    )
    report(cfg)
    return cfg


def restore(root, task):
    seeds = [
        read(root / "seeds" / str(i) / "RESULT.json")["candidate"] for i in range(1, 4)
    ]
    return completed_rounds(root, task, seeds)


async def one_task(config, runtime, tid):
    root = Path(config["output"]) / "tasks" / tid
    _, task = public_task(config, tid)
    history, checkpoint = restore(root, task)
    if not history:
        raise ValueError("Continuation requires an accepted first-round checkpoint")
    memory = Memory(checkpoint["memory"])
    policy = BatchPolicy(**config["batch"])

    def record(name, value):
        write(root / (name + ".json"), value)
        write(root / "progress.json", {"phase": name, "at": time.time()})

    async def sample(arm, number, prompt, pack):
        slot = root / f"round_{number}" / "actors" / arm
        review = (
            DeliveryReview(
                task,
                pack,
                registry(config, slot / "checks"),
                memory=memory,
                max_revisions=config["max_revisions"],
                max_tests=config["max_tests"],
                audit_mode="targeted",
                policy=policy,
                source_prefix=str(slot / "checks"),
                record=lambda n, d: write(slot / "checks" / (n + ".json"), d),
            )
            if arm == "B"
            else None
        )
        return await runtime.sample(tid, slot, guidance=prompt, review=review)

    async def evaluate(_task, row):
        return await grade(config, task, row, Path(row["runtime_directory"]))

    record("state", {"status": "running", "completed_rounds": len(history)})
    try:
        if len(history) == 3:
            result = {
                "status": "completed",
                "completed_rounds": 3,
                "rounds": history,
                "selected_delivery": history[-1]["selection"],
                "memory": memory.rows,
            }
        else:
            record(
                "resume",
                {
                    "completed_rounds": len(history),
                    "next_round": len(history) + 1,
                    "source_checkpoint_sha256": file_hash(
                        root / f"round_{len(history)}/checkpoint.json"
                    ),
                },
            )
            result = await run_task(
                task,
                checkpoint["pool"],
                rounds=3 - len(history),
                start_round=len(history) + 1,
                construction_rounds=1,
                construction_strategy="legacy",
                adversarial_review_enabled=False,
                memory=memory,
                initial_checks=checkpoint["checks"],
                initial_pack=checkpoint["previous_pack"],
                initial_expectation_cache=checkpoint["expectation_cache"],
                initial_known_gaps=checkpoint.get("known_gaps", []),
                initial_decisions=checkpoint.get("decisions", []),
                roles=lambda role, prefix: Role(
                    config, role, root / prefix / "models" / role
                ),
                registry=registry(config, root / "measurement"),
                sample=sample,
                evaluate=evaluate,
                policy=policy,
                jury_reduce_max_bytes=Role.request_budget_bytes,
                enabled_arms=("A", "B", "C"),
                arm_protocol="legacy_abcd",
                continuation_policy="fixed_rounds",
                record=record,
            )
            result["rounds"] = history + result["rounds"]
            result["completed_rounds"] += len(history)
            result["round_limit"] = 3
        if result["completed_rounds"] != 3:
            result["status"] = "partial"
        record("result.private", result)
        record(
            "state",
            {
                "status": result["status"],
                "completed_rounds": result["completed_rounds"],
            },
        )
        return result
    except BaseException as error:
        record(
            "state",
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "reason": str(error),
                "completed_rounds": len(restore(root, task)[0]),
            },
        )
        raise


def report(config):
    root = Path(config["output"])
    rows, states = [], {}
    for tid in config["task_ids"]:
        task = root / "tasks" / tid
        states[tid] = (
            read(task / "state.json")
            if (task / "state.json").exists()
            else {"status": "pending"}
        )
        for number in range(1, 4):
            for arm in "ABC":
                slot = task / f"round_{number}/actors" / arm
                path = slot / "OFFICIAL.private.json"
                receipt = read(path) if path.exists() else {}
                rows.append(
                    {
                        "task_id": tid,
                        "round": number,
                        "arm": arm,
                        "score": receipt.get("task_score")
                        if receipt.get("status") == "graded"
                        else None,
                        "status": receipt.get("status", "pending"),
                        "actor_completed": (slot / "RESULT.json").exists()
                        and read(slot / "RESULT.json").get("status") == "completed",
                    }
                )
    data = {
        "at": time.time(),
        "expected_tasks": len(states),
        "expected_new_actors": len(states) * 6,
        "expected_scores": len(states) * 9,
        "completed_tasks": sum(
            v.get("completed_rounds") == 3 and v["status"] == "completed"
            for v in states.values()
        ),
        "completed_new_actors": sum(
            r["actor_completed"] for r in rows if r["round"] > 1
        ),
        "valid_scores": sum(r["score"] is not None for r in rows),
        "tasks": states,
        "scores": rows,
    }
    write(root / "report.json", data)
    return data


def scheduling(config):
    path = Path(config["output"]) / "scheduling.json"
    value = read(path) if path.is_file() else {}
    limit = value.get("task_concurrency", config["task_concurrency"])
    if type(limit) is not int or not 1 <= limit <= 8:
        raise ValueError("Continuation task concurrency must be 1 to 8")
    dispatch = value.get("dispatch", "canary_first")
    if dispatch not in ("canary_first", "all_tasks"):
        raise ValueError("Unknown continuation dispatch mode")
    return limit, dispatch


def dispatch_tasks(config):
    path = Path(config["output"]) / "scheduling.json"
    value = read(path) if path.is_file() else {}
    selected = value.get("resume_task_ids", config["task_ids"])
    if (
        not isinstance(selected, list)
        or any(not isinstance(t, str) for t in selected)
        or len(selected) != len(set(selected))
        or not set(selected) <= set(config["task_ids"])
    ):
        raise ValueError("Resume task selection must be a unique campaign subset")
    return selected


class ContinuationRuntime(Runtime):
    def check_resources(self):
        limits = (
            (self.config["output"], 20),
            (
                self.config["docker_data_root"],
                self.config.get("docker_min_free_gib", 20),
            ),
        )
        for path, floor in limits:
            if shutil.disk_usage(path).free < floor * 1024**3:
                raise RuntimeError("Storage admission floor reached: " + path)
        available = next(
            int(line.split()[1]) * 1024
            for line in Path("/proc/meminfo").read_text().splitlines()
            if line.startswith("MemAvailable:")
        )
        if available < 20 * 1024**3:
            raise RuntimeError("Memory admission floor reached")


async def serve(config, probe_only=False):
    sys.path.insert(0, str(Path(config["benchmark_root"]) / "src"))
    runtime = ContinuationRuntime(config)
    servers = [
        uvicorn.Server(
            uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning")
        )
        for app, port in (
            (application(config), config["gateway_port"]),
            (runtime.app, config["tools_port"]),
        )
    ]
    futures = [asyncio.create_task(s.serve()) for s in servers]
    task_limit, dispatch = scheduling(config)
    gate = Capacity(task_limit)

    async def monitor():
        while True:
            report(config)
            limit, _ = scheduling(config)
            if limit != gate.limit:
                await gate.resize(limit)
            write(
                Path(config["output"]) / "active_scheduling.json",
                {
                    "task_concurrency": gate.limit,
                    "active_tasks": gate.active,
                    "dispatch": dispatch,
                    "at": time.time(),
                },
            )
            await asyncio.sleep(30)

    monitor_task = None
    try:
        for _ in range(100):
            if all(s.started for s in servers):
                break
            await asyncio.sleep(0.1)
        if not all(s.started for s in servers):
            raise RuntimeError("Gateway startup failed")
        await probe(config)
        if probe_only:
            return
        runtime.check_resources()
        monitor_task = asyncio.create_task(monitor())
        write(
            Path(config["output"]) / "active_scheduling.json",
            {"task_concurrency": task_limit, "dispatch": dispatch, "at": time.time()},
        )

        async def launch(tid):
            await gate.acquire()
            try:
                runtime.check_resources()
                try:
                    return await one_task(config, runtime, tid)
                except Exception as error:
                    write(
                        Path(config["output"]) / "task_failures" / (tid + ".json"),
                        {"error_type": type(error).__name__, "reason": str(error)},
                    )
                    return None
            finally:
                await gate.release()

        selected = dispatch_tasks(config)
        if dispatch == "canary_first" and selected:
            canary = await launch(selected[0])
            if not canary or canary.get("completed_rounds") != 3:
                raise RuntimeError(
                    "Continuation canary incomplete; inspect retained work"
                )
            remaining = selected[1:]
        else:
            remaining = selected
        await asyncio.gather(*(launch(tid) for tid in remaining))
        data = report(config)
        write(
            Path(config["output"]) / "acceptance.json",
            {
                "accepted": data["completed_tasks"] == data["expected_tasks"]
                and data["valid_scores"] == data["expected_scores"]
                and data["completed_new_actors"] == data["expected_new_actors"],
                **{k: v for k, v in data.items() if k not in ("tasks", "scores")},
            },
        )
    finally:
        if monitor_task:
            monitor_task.cancel()
            await asyncio.gather(monitor_task, return_exceptions=True)
        for server in servers:
            server.should_exit = True
        await asyncio.gather(*futures)
        report(config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run", "report"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--exclude-source", type=Path)
    parser.add_argument("--task-concurrency", type=int, default=2)
    parser.add_argument("--model-concurrency", type=int, default=4)
    parser.add_argument("--docker-min-free-gib", type=int, default=20)
    parser.add_argument("--probe-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.action == "prepare":
        tasks = read(args.source / "config.json")["task_ids"]
        if args.exclude_source:
            excluded = set(read(args.exclude_source / "config.json")["task_ids"])
            tasks = [t for t in tasks if t not in excluded]
        prepare(
            args.source,
            output,
            tasks,
            args.task_concurrency,
            args.model_concurrency,
            args.docker_min_free_gib,
        )
        print(output)
        return
    config = read(output / "config.json")
    if args.action == "report":
        report(config)
        return
    if Path(__file__).resolve().parents[3] != Path(config["code"]):
        os.chdir(config["code"])
        forwarded = [args.action, "--output", str(output)]
        if args.probe_only:
            forwarded.append("--probe-only")
        os.execv(
            config["python"],
            [
                config["python"],
                "-m",
                "experiments.claw_eval.opencode.continue_rounds",
                *forwarded,
            ],
        )
    with (output / "campaign.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify_freeze(config)
        for path, expected in read(output / "manifest.json")["source_files"].items():
            if file_hash(Path(path)) != expected:
                raise ValueError("Source evidence changed: " + path)
        write(output / "process.json", {"pid": os.getpid(), "started_at": time.time()})
        asyncio.run(serve(config, args.probe_only))


if __name__ == "__main__":
    main()
