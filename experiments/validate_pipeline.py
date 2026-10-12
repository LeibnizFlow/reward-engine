"""Fresh three-round native pipeline validation, with retained stage evidence.

Uses the production benchmark task runners. No historical samples or scores are
reused. A process exit alone is not a validation verdict; inspect stage receipts.
"""

import argparse
import asyncio
import os
import shutil
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import uvicorn

from reward_engine.adapters.workbuddy.gateway import application
from reward_engine.adapters.workbuddy.io import digest, file_hash, read, write

REPO = Path(__file__).resolve().parents[1]


def prepare(args):
    output = args.output.resolve()
    if args.benchmark == "claw_eval":
        from experiments.claw_eval.opencode.campaign import prepare as freeze

        os.environ.setdefault(
            "OPENCODE_BINARY", str(REPO / "data/claw_eval/environment/bin/opencode")
        )
        config = freeze(output, task_ids=[args.task_id])
    else:
        from experiments.workbuddy.experiment import prepare as freeze

        config = freeze(
            output,
            upstream=REPO / "local/workbuddy/runner/upstream",
            task_ids=[args.task_id],
        )
    config.update(
        validation_benchmark=args.benchmark,
        validation_task_id=args.task_id,
        rounds=3,
        full_audit=True,
        credentials_file=str(REPO / ".env"),
        backend_key_env="YIBU_API_KEY",
        backend_url="https://yibuapi.com/v1",
        model_concurrency=6,
        per_model_concurrency=6,
        quality_resampling=False,
        role_bounded_stage=True,
        role_stage_timeout_s=1800,
        role_response_format="json_object",
    )
    if getattr(args, "role_thinking", "provider_default") == "disabled":
        config["role_request_options"] = {
            "thinking": {"type": "disabled"},
            "enable_thinking": False,
        }
    config["validation_role_limits"] = {
        "role_stream_timeout_s": 1800,
        "role_max_tokens": {
            role: 131072 if role == "jury_reducer" else 65536
            for role in config["roles"]
            if role != "actor"
        },
    }
    config["identity"] = digest({k: v for k, v in config.items() if k != "identity"})
    code = Path(config["code"])
    shutil.copy2(Path(__file__), code / "experiments/validate_pipeline.py")
    manifest = read(output / "manifest.json")
    manifest["config_identity"] = config["identity"]
    manifest["actor_slots"] = 12
    manifest["grading_slots"] = 9
    key = "code_hashes" if args.benchmark == "workbuddy" else "files"
    relative = "experiments/validate_pipeline.py"
    manifest[key][relative if key == "code_hashes" else "code/" + relative] = file_hash(
        code / relative
    )
    write(output / "config.json", config)
    write(output / "manifest.json", manifest)
    write(
        output / "role_limits.json",
        config["validation_role_limits"],
    )
    write(
        output / "validation_protocol.json",
        {
            "scope": "Three fresh seeds, three fixed learning rounds, A/B/C each round",
            "full_harness_audit": True,
            "native_grading": "Separate from learning inputs; all nine arm outputs",
            "historical_samples_reused": False,
            "quality_resampling": False,
            "role_protocol": {
                "bounded_stage": True,
                "response_format": "json_object",
                "contract_repairs_per_stage": 1,
                "syntax_repairs_per_stage": 1,
                "evidence_expansion_rounds": 2,
                "stage_timeout_s": 1800,
                "output_tokens": 65536,
                "reducer_output_tokens": 131072,
                "repair_scope": "Same operation and evidence; structural correction only",
            },
            "optional_paths": "Require complementary tests; not implied by this run",
        },
    )
    return config


async def run(config):
    root = Path(config["output"])
    if (
        "validation_role_limits" in config
        and read(root / "role_limits.json") != config["validation_role_limits"]
    ):
        raise ValueError("Validation role limits changed after preparation")
    servers = [
        uvicorn.Server(
            uvicorn.Config(
                application(config),
                host="0.0.0.0",
                port=config["gateway_port"],
                log_level="warning",
                timeout_graceful_shutdown=10,
            )
        )
    ]
    if config["validation_benchmark"] == "claw_eval":
        from experiments.claw_eval.opencode.campaign import task_run, verify_freeze
        from experiments.claw_eval.opencode.runtime import Runtime

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
                    timeout_graceful_shutdown=10,
                )
            )
        )
    else:
        from experiments.workbuddy.experiment import one_task, verify

        verify(config)
    # asyncio.Runner owns interruption of the task, not either embedded server.
    # Competing Uvicorn signal handlers can leave a stopped gateway's work alive.
    for server in servers:
        server.capture_signals = nullcontext
    pending = [asyncio.create_task(server.serve()) for server in servers]
    started = time.time()
    write(root / "validation_status.json", {"status": "running", "started_at": started})
    try:
        for _ in range(100):
            if all(server.started for server in servers):
                break
            if any(task.done() for task in pending):
                raise RuntimeError("Validation gateway failed to start")
            await asyncio.sleep(0.1)
        else:
            raise TimeoutError("Validation gateway startup timed out")
        if config["validation_benchmark"] == "claw_eval":
            result = await task_run(config, runtime, config["validation_task_id"])
        else:
            await one_task(config, config["validation_task_id"])
            result = read(
                root / "tasks" / config["validation_task_id"] / "result.private.json"
            )
        completed = (
            result.get("status") == "completed" and result.get("completed_rounds") == 3
        )
        write(
            root / "validation_status.json",
            {
                "status": "execution_completed" if completed else "failed",
                "completed_rounds": result.get("completed_rounds"),
                "elapsed_s": time.time() - started,
                "stage_audit": "pending",
            },
        )
        return 0 if completed else 1
    except BaseException as error:
        write(
            root / "validation_status.json",
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
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--role-thinking",
        choices=("provider_default", "disabled"),
        default="provider_default",
        help="Explicit role-only provider request; the provider may still report reasoning tokens",
    )
    args = parser.parse_args()
    if args.execute:
        return asyncio.run(run(read(args.output.resolve() / "config.json")))
    if not args.benchmark or not args.task_id:
        parser.error("--benchmark and --task-id are required for a fresh run")
    config = prepare(args)
    os.chdir(config["code"])
    os.execv(
        sys.executable,
        [
            sys.executable,
            "-m",
            "experiments.validate_pipeline",
            "--execute",
            "--output",
            config["output"],
        ],
    )


if __name__ == "__main__":
    raise SystemExit(main())
