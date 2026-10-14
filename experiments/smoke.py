"""Run one fresh native benchmark actor and its official grader.

This is an installation smoke test, not a multi-round reward-engine experiment.
No historical candidate, score, or synthetic model response is reused.
"""

import argparse
import asyncio
import json
import os
import shutil
import sys
import time
from pathlib import Path

import uvicorn

from reward_engine.adapters.provider_config import (
    add_provider_arguments,
    application,
    prepare_options,
    provider_arguments_supplied,
)
from reward_engine.adapters.workbuddy.io import digest, file_hash, read, write

REPO = Path(__file__).resolve().parents[1]


def prepare(args):
    output = args.output.resolve()
    provider = prepare_options(args, default_credentials=REPO / ".env")
    if args.benchmark == "claw_eval":
        from experiments.claw_eval.opencode.campaign import prepare as prepare_claw

        os.environ.setdefault(
            "OPENCODE_BINARY", str(REPO / "data/claw_eval/environment/bin/opencode")
        )
        config = prepare_claw(output, task_ids=[args.task_id], **provider)
        if args.task_id not in config["task_ids"]:
            raise ValueError("Task is not in the pinned Claw-Eval manifest")
        config["task_ids"] = [args.task_id]
        config["canaries"] = [args.task_id]
    else:
        from experiments.workbuddy.experiment import prepare as prepare_office

        config = prepare_office(
            output,
            upstream=Path(
                getattr(args, "workbuddy_root", None)
                or os.environ.get("WORKBUDDY_ROOT", REPO / "local/workbuddy/runner")
            )
            .expanduser()
            .resolve()
            / "upstream",
            task_ids=[args.task_id],
            **provider,
        )
    config.update(
        smoke_benchmark=args.benchmark,
        smoke_task_id=args.task_id,
        smoke_scope="one fresh native actor followed by the official grader",
        model_concurrency=2,
        per_model_concurrency=2,
    )
    # Record the explicit smoke protocol before launching or making model calls.
    config["identity"] = digest({k: v for k, v in config.items() if k != "identity"})
    code = Path(config["code"])
    shutil.copy2(Path(__file__), code / "experiments/smoke.py")
    manifest = read(output / "manifest.json")
    manifest["config_identity"] = config["identity"]
    if args.benchmark == "workbuddy":
        manifest["actor_slots"] = manifest["grading_slots"] = 1
        manifest["code_hashes"]["experiments/smoke.py"] = file_hash(
            code / "experiments/smoke.py"
        )
    else:
        manifest["files"]["code/experiments/smoke.py"] = file_hash(
            code / "experiments/smoke.py"
        )
    write(output / "config.json", config)
    write(output / "manifest.json", manifest)
    return config


async def run(config):
    root = Path(config["output"])
    benchmark = config["smoke_benchmark"]
    task_id = config["smoke_task_id"]
    servers = [
        uvicorn.Server(
            uvicorn.Config(
                application(config),
                host="0.0.0.0",
                port=config["gateway_port"],
                log_level="warning",
            )
        )
    ]
    if benchmark == "claw_eval":
        from experiments.claw_eval.opencode.campaign import grade, verify_freeze
        from experiments.claw_eval.opencode.runtime import Runtime, public_task

        verify_freeze(config)
        sys.path.insert(0, str(Path(config["benchmark_root"]) / "src"))
        runtime = Runtime(config)
        servers.append(
            uvicorn.Server(
                uvicorn.Config(
                    runtime.app,
                    host="0.0.0.0",
                    port=config["tools_port"],
                    log_level="warning",
                )
            )
        )
    else:
        from experiments.workbuddy.experiment import launch, verify

        verify(config)
    pending = [asyncio.create_task(server.serve()) for server in servers]
    started = time.time()
    write(root / "smoke_status.json", {"status": "running", "started_at": started})
    try:
        for _ in range(100):
            if all(server.started for server in servers):
                break
            if any(task.done() for task in pending):
                raise RuntimeError("Native gateway failed to start")
            await asyncio.sleep(0.1)
        else:
            raise TimeoutError("Native gateway startup timed out")
        output = root / "tasks" / task_id / "actor"
        if benchmark == "claw_eval":
            _, task = public_task(config, task_id)
            row = await runtime.sample(task_id, output)
            if row.get("status") != "completed":
                raise RuntimeError("Native Claw actor did not complete")
            official = await grade(config, task, row, output)
            accepted = official.get("status") == "graded"
            score = official.get("task_score")
        else:
            task = {
                "task_id": task_id,
                "prompt": (
                    Path(config["tasks_root"]) / task_id / "instruction.md"
                ).read_text(),
            }
            row = await launch(config, task, output, grade=True)
            official_path = output / "OFFICIAL.private.json"
            official = read(official_path) if official_path.exists() else {}
            accepted = (
                row.get("status") == "completed" and official.get("status") == "scored"
            )
            score = official.get("overall")
        result = {
            "status": "completed" if accepted else "failed",
            "accepted": accepted,
            "benchmark": benchmark,
            "task_id": task_id,
            "scope": config["smoke_scope"],
            "actor_status": row.get("status"),
            "grader_status": official.get("status"),
            "score": score,
            "source_root": str(REPO),
            "frozen_code": config["code"],
            "config_identity": config["identity"],
            "actor_receipt": str(output / "RESULT.json"),
            "official_receipt": str(output / "OFFICIAL.private.json"),
            "elapsed_s": time.time() - started,
        }
        write(root / "smoke_status.json", result)
        print(json.dumps(result), flush=True)
        return 0 if accepted else 1
    except BaseException as error:
        write(
            root / "smoke_status.json",
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "reason": str(error),
                "elapsed_s": time.time() - started,
            },
        )
        raise
    finally:
        for server in servers:
            server.should_exit = True
        await asyncio.gather(*pending)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("claw_eval", "workbuddy"))
    parser.add_argument("--task-id")
    parser.add_argument("--output", type=Path, required=True)
    add_provider_arguments(parser)
    parser.add_argument("--workbuddy-root", type=Path)
    parser.add_argument("--execute", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.execute and (provider_arguments_supplied(args) or args.workbuddy_root):
        parser.error("Provider and native paths are frozen at prepare time")
    if args.execute:
        return asyncio.run(run(read(args.output.resolve() / "config.json")))
    if not args.benchmark or not args.task_id:
        parser.error("--benchmark and --task-id are required")
    config = prepare(args)
    os.chdir(config["code"])
    os.execv(
        sys.executable,
        [
            sys.executable,
            "-m",
            "experiments.smoke",
            "--execute",
            "--output",
            config["output"],
        ],
    )


if __name__ == "__main__":
    raise SystemExit(main())
