"""Freeze and run Claw-Eval campaigns with either supported actor runtime."""

import argparse
import asyncio
import fcntl
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

import httpx
import uvicorn

from experiments.claw_eval.opencode.runtime import (
    Runtime,
    free_port,
    public_task,
)
from experiments.claw_eval.report import report as report
from reward_engine.adapters.integrity import verification_progress
from reward_engine.adapters.provider_config import (
    add_provider_arguments,
    provider_arguments_supplied,
    provider_key,
    resolve_provider,
)
from reward_engine.adapters.scheduling import batch_policy
from reward_engine.adapters.workbuddy.io import auth, digest, file_hash, read, write
from reward_engine.adapters.workbuddy.models import Role, registry
from reward_engine.core.outcome.loop import run_task
from reward_engine.core.outcome.memory import Memory
from reward_engine.core.outcome.runner import DeliveryReview

REPO = Path(__file__).resolve().parents[3]
SOURCE = Path(
    os.environ.get("CLAW_EVAL_ROOT", REPO / "_local/data/claw_eval/benchmark")
)
MANIFEST = Path(
    os.environ.get(
        "CLAW_EVAL_TASK_MANIFEST", REPO / "_local/data/claw_eval/task_manifest.json"
    )
)


def install_public_fixtures(benchmark, task_ids, output):
    """Supplement only declared missing inputs from the verified official archive."""
    import yaml

    overlay = (
        Path(os.environ["CLAW_EVAL_FIXTURE_OVERLAY"])
        if os.environ.get("CLAW_EVAL_FIXTURE_OVERLAY")
        else None
    )
    provenance = None
    added = {}
    for tid in task_ids:
        folder = benchmark / "tasks" / tid
        spec = yaml.safe_load((folder / "task.yaml").read_text())
        paths = (
            spec.get("sandbox_files", [])
            + spec.get("environment", {}).get("fixtures", [])
            + spec.get("prompt", {}).get("attachments", [])
            + spec.get("sandbox_grader_files", [])
            + spec.get("local_grader_files", [])
        )
        # Services run with benchmark as their cwd. A declared input referenced
        # verbatim by a service environment variable must exist at that root,
        # even when its archive entry is supplied through a task overlay.
        service_inputs = {
            value
            for service in spec.get("services", [])
            for value in (service.get("env") or {}).values()
            if isinstance(value, str)
        }
        for relative in paths:
            if Path(relative).is_absolute() or ".." in Path(relative).parts:
                raise ValueError("Unsafe declared fixture path")
            service_input = relative in service_inputs
            destination = (benchmark if service_input else folder) / relative
            if destination.exists() or (not service_input and (benchmark / relative).exists()):
                continue
            if overlay is None:
                raise FileNotFoundError(
                    f"Missing benchmark fixture {tid}/{relative}; set CLAW_EVAL_FIXTURE_OVERLAY to a verified upstream overlay"
                )
            if provenance is None:
                provenance = read(overlay / "SOURCE.json")
                if (
                    file_hash(overlay.parent / "fixtures.tar.gz")
                    != provenance["archive_sha256"]
                ):
                    raise ValueError("Official fixture archive hash differs")
            source = overlay / "tasks" / tid / relative
            if not source.is_file():
                raise FileNotFoundError(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            added[str(destination.relative_to(output))] = file_hash(destination)
    write(output / "fixture_overlay.json", {"source": provenance, "files": added})


def verify_freeze(config):
    if config.get("implementation_revision"):
        from experiments.revision import verify_revision
        return verify_revision(config, verify_freeze)
    from reward_engine.adapters.jiuwenswarm.config import verify
    verify(config)
    root = Path(config["output"])
    from reward_engine.adapters.model_config import verify_profile
    verify_profile(config)
    manifest = read(root / "manifest.json")
    if (
        config["identity"] != manifest["config_identity"]
        or digest({k: v for k, v in config.items() if k != "identity"})
        != config["identity"]
    ):
        raise ValueError("Frozen configuration identity changed")
    for relative, expected in manifest["files"].items():
        if file_hash(root / relative) != expected:
            raise ValueError("Frozen file changed: " + relative)


def prepare(output, *, task_ids=None, credentials=None, key_env=None, backend_url=None,
            actor_backend="jiuwenswarm", jiuwenswarm_mode="agent.code.normal", jiuwenswarm_bundle=None,
            generation_protocol="three_b_fixed_baselines_v1", rounds=3, task_images_manifest=None, role_models=None, model_config=None, native_grader_model=None,
            per_model_concurrency=None, minimum_free_gib=None):
    from reward_engine.adapters.fresh_config import fresh_roles, freeze_settings
    from reward_engine.adapters.jiuwenswarm.config import options
    runtime_options = options(actor_backend=actor_backend, jiuwenswarm_mode=jiuwenswarm_mode,
        jiuwenswarm_bundle=jiuwenswarm_bundle, generation_protocol=generation_protocol, rounds=rounds)
    if task_images_manifest is not None:
        raise ValueError("task_images_manifest is Office-only")
    from experiments.claw_eval.opencode.model_settings import (
        configure_roles,
        validate_model,
    )
    if native_grader_model is not None:
        validate_model(native_grader_model)
    if per_model_concurrency is not None and (type(per_model_concurrency) is not int or not 1 <= per_model_concurrency <= 128):
        raise ValueError("per_model_concurrency must be in 1..128")
    if minimum_free_gib is not None and (type(minimum_free_gib) is not int or not 5 <= minimum_free_gib <= 1024):
        raise ValueError("minimum_free_gib must be an integer in 5..1024")
    roles = fresh_roles(model_config, role_models, generation_protocol)
    manifest = read(MANIFEST)
    if len(manifest["task_ids"]) != 104 or len(set(manifest["task_ids"])) != 104:
        raise ValueError("Expected exactly 104 unique task IDs")
    selected = list(manifest["task_ids"] if task_ids is None else task_ids)
    if (
        not selected
        or len(set(selected)) != len(selected)
        or not set(selected) <= set(manifest["task_ids"])
    ):
        raise ValueError("Selected tasks must be unique members of the pinned Claw 104")
    provider = resolve_provider(
        credentials=credentials,
        key_env=key_env,
        backend_url=backend_url,
        default_credentials=REPO / ".env",
        default_key_env="OPENAI_API_KEY",
        default_backend_url=None,
    )
    provider_key(provider)
    output.mkdir(parents=True, exist_ok=False)
    code = output / "code"
    code.mkdir()
    for name in ("reward_engine", "experiments"):
        shutil.copytree(
            REPO / name,
            code / name,
            ignore=shutil.ignore_patterns("__pycache__", "output", ".pytest_cache"),
        )
    benchmark = output / "benchmark"
    shutil.copytree(
        SOURCE, benchmark, ignore=shutil.ignore_patterns(".git", "__pycache__", ".venv")
    )
    install_public_fixtures(benchmark, selected, output)
    if actor_backend == "opencode":
        binary = Path(
            os.environ.get("OPENCODE_BINARY", shutil.which("opencode") or "opencode")
        )
        if not binary.is_file():
            raise FileNotFoundError(
                "Install the pinned OpenCode binary and set OPENCODE_BINARY"
            )
        (output / "bin").mkdir()
        shutil.copy2(binary, output / "bin/opencode")
    import subprocess

    image = subprocess.check_output(
        ["docker", "image", "inspect", "claw-eval-agent:latest", "--format", "{{.Id}}"],
        text=True,
    ).strip()
    docker_root = subprocess.check_output(
        ["docker", "info", "--format", "{{.DockerRootDir}}"], text=True
    ).strip()
    port, tool_port = free_port(), free_port()
    cfg = {
        "schema": "claw_eval.office50_opencode.v1",
        "campaign_id": output.name,
        "output": str(output),
        "code": str(code),
        "benchmark_root": str(benchmark),
        "task_ids": selected,
        "groups": manifest["groups"],
        "rounds": 1,
        "seeds": 3,
        "arm_protocol": "legacy_abcd",
        "adversarial_review_enabled": False,
        "selection_policy": "legacy",
        "quality_resampling": False,
        "roles": roles,
        "actor_timeout_s": 1800,
        "max_revisions": 2,
        "max_tests": 3,
        "batch": {"concurrency": 3, "group_concurrency": 2, "deadline_s": 1800},
        "role_stream_timeout_s": 3600,
        "model_concurrency": 8,
        "task_concurrency": 3,
        "max_model_concurrency": 128,
        "per_model_concurrency": 128,
        **provider,
        "auth_file": str(output / "campaign.auth"),
        "gateway_port": port,
        "gateway_local": f"http://127.0.0.1:{port}",
        "gateway_container": f"http://host.docker.internal:{port}",
        "tools_port": tool_port,
        "tools_container": f"http://host.docker.internal:{tool_port}",
        "image": image,
        "docker_data_root": docker_root,
        "opencode_binary": str(output / "bin/opencode"),
        "python": sys.executable,
        "model_aliases": {},
        "canaries": [
            manifest["task_ids"][0],
            manifest["groups"]["fresh_sandbox_25"][0],
            "T091_pinbench_humanize_blog",
            "T096_pinbench_business_metrics_summary",
        ],
    }
    cfg.update(runtime_options)
    cfg["validation_benchmark"] = "claw_eval"
    if native_grader_model is not None:
        cfg["native_grader_model"] = native_grader_model
    if per_model_concurrency is not None:
        cfg["per_model_concurrency"] = per_model_concurrency
    if minimum_free_gib is not None:
        cfg["minimum_free_gib"] = minimum_free_gib
        cfg["minimum_free_bytes"] = minimum_free_gib * 1024**3
    freeze_settings(cfg, model_config, role_models, generation_protocol)
    cfg["canaries"] = [task for task in cfg["canaries"] if task in selected]
    cfg["identity"] = digest(cfg)
    (output / "campaign.auth").write_text(uuid4().hex + uuid4().hex)
    (output / "campaign.auth").chmod(0o600)
    write(output / "config.json", cfg)
    write(
        output / "search_backend.json",
        {
            "provider": "bing_rss",
            "reason": "Native Novada search credentials unavailable; explicit public search adapter",
            "native_tool_contract": "unchanged",
            "model_provider": cfg["backend_url"],
        },
    )
    hashes = {
        str(p.relative_to(output)): file_hash(p)
        for root in (code, benchmark, output / "bin")
        for p in root.rglob("*")
        if p.is_file()
    }
    write(
        output / "manifest.json",
        {
            "files": hashes,
            **({"model_config": cfg["model_config"]} if "model_config" in cfg else {}),
            "task_manifest": manifest,
            "source_manifest_sha256": file_hash(MANIFEST),
            "config_identity": cfg["identity"],
            "image": image,
            "runtime_identity": cfg.get("runtime_identity"),
            "generation_protocol": cfg.get("generation_protocol"),
        },
    )
    return cfg


def prepare_web_rerun(output, source):
    """Create a fresh run of the source campaign's public-web tasks."""
    import yaml

    source = source.resolve()
    original = read(source / "config.json")
    frozen = read(source / "manifest.json")
    verify_freeze(original)
    if not read(source / "acceptance.json").get("accepted"):
        raise ValueError("Source campaign is not accepted")
    if original["output"] != str(source):
        raise ValueError("Source campaign path changed")
    task_ids = []
    for task_id in original["task_ids"]:
        spec = yaml.safe_load(
            (source / "benchmark/tasks" / task_id / "task.yaml").read_text()
        )
        if any(
            service.get("name") == "web_real" for service in spec.get("services", [])
        ):
            task_ids.append(task_id)
    if len(task_ids) != 19:
        raise ValueError(f"Expected 19 public-web tasks, found {len(task_ids)}")
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Rerun output and source must be separate")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source / "code", output / "code")
    shutil.copy2(
        Path(__file__), output / "code/experiments/claw_eval/opencode/campaign.py"
    )
    shutil.copytree(source / "benchmark", output / "benchmark")
    shutil.copytree(source / "bin", output / "bin")
    shutil.copy2(source / "fixture_overlay.json", output / "fixture_overlay.json")
    shutil.copy2(source / "search_backend.json", output / "search_backend.json")
    image = subprocess.check_output(
        ["docker", "image", "inspect", "claw-eval-agent:latest", "--format", "{{.Id}}"],
        text=True,
    ).strip()
    if image != original["image"]:
        raise ValueError("Docker image changed since source campaign")
    cfg = {**original}
    cfg.update(
        campaign_id=output.name,
        output=str(output),
        code=str(output / "code"),
        benchmark_root=str(output / "benchmark"),
        task_ids=task_ids,
        groups={
            name: [tid for tid in group if tid in task_ids]
            for name, group in original["groups"].items()
        },
        canaries=[],
        task_concurrency=8,
        max_task_concurrency=8,
        model_concurrency=8,
        max_model_concurrency=8,
        per_model_concurrency=8,
        auth_file=str(output / "campaign.auth"),
    )
    cfg["gateway_port"], cfg["tools_port"] = free_port(), free_port()
    cfg["gateway_local"] = f"http://127.0.0.1:{cfg['gateway_port']}"
    cfg["gateway_container"] = f"http://host.docker.internal:{cfg['gateway_port']}"
    cfg["tools_container"] = f"http://host.docker.internal:{cfg['tools_port']}"
    cfg.pop("identity", None)
    cfg["identity"] = digest(cfg)
    (output / "campaign.auth").write_text(uuid4().hex + uuid4().hex)
    (output / "campaign.auth").chmod(0o600)
    write(output / "config.json", cfg)
    hashes = dict(frozen["files"])
    hashes["code/experiments/claw_eval/opencode/campaign.py"] = file_hash(
        output / "code/experiments/claw_eval/opencode/campaign.py"
    )
    write(
        output / "manifest.json",
        {
            **frozen,
            "files": hashes,
            "config_identity": cfg["identity"],
            "task_manifest": {"task_ids": task_ids, "groups": cfg["groups"]},
            "source_campaign": str(source),
            "source_config_identity": original["identity"],
        },
    )
    verify_freeze(cfg)
    return cfg


async def grade(config, task, row, output):
    from reward_engine.adapters.claw_eval.native_grader import evaluator

    target = output / "OFFICIAL.private.json"
    binding = {
        "trace": file_hash(Path(row["native_grader_trial"]) / "trace.jsonl"),
        "snapshot": file_hash(
            Path(row["native_grader_trial"]) / "environment-snapshot.json"
        ),
        "identity": config.get("checkpoint_config_identity", config["identity"]),
    }
    if target.exists():
        saved = read(target)
        if saved.get("binding") != binding:
            raise ValueError("Score artifact binding changed")
        if config.get("generation_protocol") != "three_b_fixed_baselines_v1" or saved.get("status") == "graded":
            return saved
        # Original worker receipts remain immutable; retry only native grading.
        write(output / "official.private" / (str(time.time_ns()) + ".failed.json"), saved)
    os.environ["CLAW_CAMPAIGN_AUTH"] = auth(config)
    grader_model = config.get("native_grader_model", config["roles"]["actor"])
    settings = {
        "benchmark_root": config["benchmark_root"],
        "task_yaml": str(
            Path(config["benchmark_root"]) / "tasks" / task["task_id"] / "task.yaml"
        ),
        "python": config["python"],
        "timeout_s": 3600,
        "judge": {
            "model": grader_model,
            "base_url": config["gateway_local"] + "/grade/v1",
            "api_key_env": "CLAW_CAMPAIGN_AUTH",
            "allowed_models": [grader_model]
            + config["model_aliases"].get(grader_model, []),
            "temperature": 0.6,
            "max_tokens": 32768,
            "read_timeout_s": 1860,
        },
    }
    result = await evaluator({"native_grader": settings}, output / "official.private")(
        task, row
    )
    result["binding"] = binding
    write(target, result)
    return result


def substantive_recovery_count(task):
    """Count model/infrastructure retries without charging operator restarts."""
    count = 0
    for event_path in (Path(task) / "recovery_events").glob("*.json"):
        event = read(event_path)
        state_path = Path(event.get("archived_to", "")) / "state.json"
        state = read(state_path) if state_path.is_file() else {}
        operational = state.get("error_type") == "CampaignInterrupted" or (
            state.get("error_type") == "ValueError"
            and state.get("reason") == "Cannot reuse a changed or failed Actor slot"
        )
        if not operational:
            count += 1
    return count


def corrected_failure_retry_limit(task):
    """Return the bounded retry limit for an audited corrected failure."""
    task = Path(task)
    result_path = task / "result.private.json"
    state_path = task / "state.json"
    result = read(result_path) if result_path.exists() else {}
    state = read(state_path) if state_path.exists() else {}
    failure = result.get("construction_failure") or {}
    messages = (failure.get("error", ""), state.get("reason", ""))
    if "User criterion must cite an actual user statement" in messages:
        # The prior provenance repair can expose this sibling legacy path.
        return 5
    if any(
        message
        in {
            "User expectation requires an actual user statement, not Actor narration",
            "Fewer than two usable independent expectations; keep raw failures",
        }
        for message in messages
    ):
        return 3
    for slot in (task / "round_1" / "actors").glob("*"):
        failure_path = slot / "FAILURE.json"
        if not failure_path.exists():
            continue
        row = read(failure_path)
        if row.get("error_type") == "RuntimeError" and row.get("reason", "").startswith(
            "docker exited "
        ):
            return 3
    return 2


def corrected_failure_retry_available(task):
    return corrected_failure_retry_limit(task) > 2


async def three_b_task(config, runtime, task_id):
    from reward_engine.adapters.claw_eval.recovery import recover_prelaunch
    from reward_engine.adapters.jiuwenswarm.checkpoints import verify_actor
    from reward_engine.adapters.scheduling import check_stage_admission
    from reward_engine.core.outcome import three_b
    root = Path(config["output"]) / "tasks" / task_id
    root.mkdir(parents=True, exist_ok=True)
    _, task = public_task(config, task_id)
    policy = batch_policy(config)
    def record(name, value):
        write(root / (name + ".json"), value)
    def restore(name):
        path = root / (name + ".json")
        return read(path) if path.exists() else None
    retries = getattr(runtime, "actor_retries", None)
    def actor_slot(name, slot):
        return retries.slot(task_id, name, slot) if retries else slot
    def recover_actor(name, stage):
        return recover_prelaunch(root, name, stage) or bool(
            retries and retries.recover(task_id, name, stage)
        )
    async def baseline(index):
        slot = actor_slot(f"seed_{index}", root / "seeds" / str(index + 1))
        return await runtime.sample(task_id, slot)
    async def sample(arm, number, prompt, pack):
        slot = actor_slot(f"round_{number}/{arm}", root / f"round_{number}" / "actors" / arm)
        review = DeliveryReview(task, pack, registry(config, slot / "checks"), memory=Memory(),
            max_revisions=2, audit_mode="full", policy=policy, source_prefix=str(slot / "checks"),
            record=lambda n, v: write(slot / "checks" / (n + ".json"), v))
        return await runtime.sample(task_id, slot, guidance=prompt, review=review)
    async def evaluate(task, row):
        return await grade(config, task, row, Path(row["runtime_directory"]))
    try:
        result = await three_b.run_task(task, rounds=config["rounds"],
            runtime={"config_identity": config.get("checkpoint_config_identity", config["identity"]), "runtime_identity": config.get("runtime_identity"), "image": config["image"]},
            implementation=config["identity"],
            baseline=baseline, sample=sample, evaluate=evaluate,
            roles=lambda role, prefix: Role(config, role, root / prefix / "models" / role),
            registry=registry(config, root / "measurement"), policy=policy,
            jury_reduce_max_bytes=Role.request_budget_bytes, record=record, restore=restore, verify_actor=verify_actor,
            admit_stage=lambda name: check_stage_admission(config, task_id, name),
            recover_actor=recover_actor
            if config.get("actor_backend") == "jiuwenswarm" else None)
        record("state", {"status": result["status"], "completed_rounds": result["completed_rounds"]})
        return result
    except BaseException as error:
        record("state", {"status": "failed", "error_type": type(error).__name__, "reason": str(error)})
        raise


async def task_run(config, runtime, task_id):
    from reward_engine.core.outcome.three_b import PROTOCOL
    if config.get("generation_protocol") == PROTOCOL:
        return await three_b_task(config, runtime, task_id)
    root = Path(config["output"]) / "tasks" / task_id
    root.mkdir(parents=True, exist_ok=True)
    recovery_count = substantive_recovery_count(root)
    recovery_limit = corrected_failure_retry_limit(root)
    if (root / "result.private.json").exists():
        previous = read(root / "result.private.json")
        failure = previous.get("construction_failure") or {}
        recoverable_construction = (
            previous.get("status") in {"failed", "partial"}
            and previous.get("completed_rounds") == 0
            and (
                previous.get("stop_reason") in {"jury_unavailable", "incomplete_arms"}
                or (
                    failure.get("error_type") == "ValueError"
                    and "Incomplete or mismatched role reply: finish=length"
                    in failure.get("error", "")
                )
                or failure.get("error_type")
                in {
                    "HTTPStatusError",
                    "ProviderStreamError",
                    "ReadTimeout",
                    "TimeoutError",
                }
            )
            and recovery_count < recovery_limit
        )
        if recoverable_construction:
            recover_failed_canary(Path(config["output"]), task_id)
        else:
            return previous
    elif (root / "state.json").exists():
        state = read(root / "state.json")
        recoverable_runtime = state.get("error_type") == "RuntimeError" and state.get(
            "reason", ""
        ).startswith("docker exited ")
        if (
            state.get("status") == "failed"
            and (
                state.get("error_type")
                in {
                    "CampaignInterrupted",
                    "CancelledError",
                    "HTTPStatusError",
                    "ProviderStreamError",
                    "ReadTimeout",
                    "TimeoutError",
                }
                or recoverable_runtime
            )
            and recovery_count < recovery_limit
        ):
            recover_failed_canary(Path(config["output"]), task_id)
    _, task = public_task(config, task_id)

    def record(name, data):
        write(root / (name + ".json"), data)
        write(root / "progress.json", {"phase": name, "at": time.time()})

    record("state", {"status": "running"})
    try:
        seeds = await asyncio.gather(
            *(runtime.sample(task_id, root / "seeds" / str(i)) for i in range(1, 4)),
            return_exceptions=True,
        )
        failures = [r for r in seeds if isinstance(r, BaseException)]
        if failures:
            raise failures[0]
        record("seeds", seeds)
        policy = batch_policy(config)
        memory = Memory()

        async def sample(arm, number, prompt, pack):
            slot = root / f"round_{number}" / "actors" / arm
            review = (
                DeliveryReview(
                    task,
                    pack,
                    registry(config, slot / "checks"),
                    memory=memory,
                    max_revisions=2,
                    max_tests=3,
                    audit_mode="targeted",
                    policy=policy,
                    source_prefix=str(slot / "checks"),
                    record=lambda n, d: write(slot / "checks" / (n + ".json"), d),
                )
                if arm == "B"
                else None
            )
            # Review objects create no output until the actual Actor slot starts.
            row = await runtime.sample(task_id, slot, guidance=prompt, review=review)
            return row

        async def evaluate(_task, row):
            return await grade(config, task, row, Path(row["runtime_directory"]))

        result = await run_task(
            task,
            [row["candidate"] for row in seeds],
            rounds=config.get("rounds", 1),
            construction_rounds=1,
            construction_strategy="legacy",
            adversarial_review_enabled=False,
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
            full_audit=config.get("full_audit", False),
            memory=memory,
            record=record,
        )
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
            },
        )
        raise


def recover_failed_canary(root, task_id):
    """Archive a zero-round canary failure so cached successful work can resume."""
    task = Path(root) / "tasks" / task_id
    result_path = task / "result.private.json"
    result = read(result_path) if result_path.exists() else None
    state_path = task / "state.json"
    state = read(state_path) if state_path.exists() else None
    campaign_interrupted = (
        state is not None and state.get("error_type") == "CampaignInterrupted"
    )
    interrupted_actor_slots = {
        arm
        for round_data in (result or {}).get("rounds", [])
        for arm, row in (round_data.get("arms") or {}).items()
        if row.get("error_type") == "RuntimeError"
        and row.get("reason")
        == "Interrupted Actor slot needs inspection; not resampling"
    }
    if result is not None and result.get("status") not in {"failed", "partial"}:
        return False
    if result is not None and result.get("completed_rounds") != 0:
        raise RuntimeError(
            "Refusing to recover a canary after a completed round: " + task_id
        )
    failed_slots = []
    for slot in (task / "round_1/actors").glob("*"):
        if not slot.is_dir():
            continue
        if (slot / "RESULT.json").exists():
            continue
        failure = slot / "FAILURE.json"
        files = {p.name for p in slot.iterdir()}
        row = read(failure) if failure.exists() else {}
        dependency_failure = (
            files <= {"STARTED.json", "FAILURE.json"}
            and row.get("session") is None
            and row.get("error_type") == "ModuleNotFoundError"
            and row.get("reason") == "No module named 'swarm_reward'"
        )
        transport_failure = row.get("error_type") in {
            "CancelledError",
            "HTTPStatusError",
            "ProviderStreamError",
            "ReadTimeout",
            "TimeoutError",
        }
        recoverable_runtime = row.get("error_type") == "RuntimeError" and row.get(
            "reason", ""
        ).startswith("docker exited ")
        if (
            campaign_interrupted
            or slot.name in interrupted_actor_slots
            or dependency_failure
            or transport_failure
            or recoverable_runtime
        ):
            failed_slots.append(slot)
        else:
            raise RuntimeError(
                "Refusing to recover an inspected or unknown Actor failure: "
                + str(slot)
            )
    failed_seed_slots = []
    for slot in (task / "seeds").glob("*"):
        if not slot.is_dir() or (slot / "RESULT.json").exists():
            continue
        failure = slot / "FAILURE.json"
        row = read(failure) if failure.exists() else {}
        recoverable_runtime = row.get("error_type") == "RuntimeError" and row.get(
            "reason", ""
        ).startswith("docker exited ")
        if (
            campaign_interrupted
            or row.get("error_type")
            in {
                "CancelledError",
                "HTTPStatusError",
                "ProviderStreamError",
                "ReadTimeout",
                "TimeoutError",
            }
            or recoverable_runtime
        ):
            failed_seed_slots.append(slot)
        else:
            raise RuntimeError(
                "Refusing to recover an unknown seed failure: " + str(slot)
            )
    if (
        result is None
        and not failed_slots
        and not failed_seed_slots
        and (state is None or state.get("status") != "failed")
    ):
        return False
    recovery_id = str(time.time_ns()) + "_" + uuid4().hex
    archive = (
        Path(root)
        / "implementation_revisions"
        / recovery_id
        / "task_recovery"
        / task_id
    )
    archived = []
    for relative in (
        "result.private.json",
        "state.json",
        "progress.json",
        "round_1/construction_failure.json",
        "round_1/loop_decision.json",
    ):
        source = task / relative
        if source.exists():
            destination = archive / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, destination)
            archived.append(relative)
    for slot in failed_slots + failed_seed_slots:
        relative = slot.relative_to(task)
        destination = archive / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(slot, destination)
        archived.append(str(relative))
    write(
        task / "recovery_events" / (recovery_id + ".json"),
        {
            "task_id": task_id,
            "at": time.time(),
            "reason": "retry zero-round failed canary after infrastructure correction",
            "archived_to": str(archive),
            "archived_files": archived,
            "reuse_policy": "reuse exact completed Actor and role caches; retry only missing work",
        },
    )
    return True


def recover_orphaned_tasks(config):
    """Recover work left running when the previous campaign process vanished."""
    root = Path(config["output"])
    recovered = []
    for task_id in config["task_ids"]:
        task = root / "tasks" / task_id
        state_path = task / "state.json"
        if not state_path.exists():
            continue
        state = read(state_path)
        if state.get("status") != "running":
            continue
        write(
            task / "interruption.json",
            {
                "at": time.time(),
                "reason": "Campaign process was absent at resume startup",
                "previous_state": state,
            },
        )
        write(
            state_path,
            {
                "status": "failed",
                "error_type": "CampaignInterrupted",
                "reason": "Campaign process was absent at resume startup",
            },
        )
        if recover_failed_canary(root, task_id):
            recovered.append(task_id)
    if recovered:
        write(
            root / "orphaned_task_recovery.json",
            {"at": time.time(), "tasks": recovered},
        )
    return recovered


async def probe(config):
    async with httpx.AsyncClient(timeout=240, trust_env=False) as client:
        for model in dict.fromkeys(config["roles"].values()):
            previous = Path(config["output"]) / "preflight" / (model + ".json")
            if (
                previous.exists()
                and read(previous).get("choices", [{}])[0].get("finish_reason")
                == "stop"
            ):
                continue
            response = await client.post(
                config["gateway_local"] + "/probe/v1/chat/completions",
                headers={"Authorization": "Bearer " + auth(config)},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "Reply with exactly OK."}],
                    "max_tokens": 1024,
                    "thinking": {"type": "disabled"},
                    "stream": False,
                },
            )
            response.raise_for_status()
            result = response.json()
            write(Path(config["output"]) / "preflight" / (model + ".json"), result)
            if (
                not result.get("choices")
                or result["choices"][0].get("finish_reason") != "stop"
            ):
                raise RuntimeError("Preflight reply incomplete: " + model)


async def run_three_b_tasks(config, runtime, names):
    from reward_engine.adapters.claw_eval.services import ServiceStartError
    from reward_engine.adapters.scheduling import run_bounded
    from reward_engine.adapters.task_admission import TaskAdmission, check_space

    admission = TaskAdmission(config, check_space)

    async def one(tid):
        admission.admit()
        for attempt in range(3):
            try:
                return await task_run(config, runtime, tid)
            except ServiceStartError:
                if attempt == 2:
                    raise

    results = await run_bounded(config, names, one)
    report(config)
    errors = [result for result in results if isinstance(result, BaseException)]
    if errors:
        raise errors[0]


async def wait_for_servers(servers, tasks, timeout=60):
    """Wait for bound listeners, surfacing startup failures before any model call."""
    async with asyncio.timeout(timeout):
        while True:
            for name, task in zip(servers, tasks, strict=True):
                if task.done():
                    await task
                    raise RuntimeError(f"{name} server exited before readiness")
            if all(server.started for server in servers.values()):
                return
            await asyncio.sleep(0.05)


async def serve(config, canary_only=False, probe_only=False, *, only_tasks=None,
                actor_retry_authorizations=()):
    from reward_engine.adapters.provider_config import application

    root = Path(config["output"])
    sys.path.insert(0, str(Path(config["benchmark_root"]) / "src"))
    runtime = Runtime(config)
    if actor_retry_authorizations:
        from reward_engine.adapters.claw_eval.actor_retry import ActorRetries
        documents = [read(path) for path in actor_retry_authorizations]
        runtime.actor_retries = ActorRetries(config, documents)
        if (only_tasks is None or canary_only or probe_only
                or any(task_id not in only_tasks for task_id, _ in runtime.actor_retries.entries)):
            raise ValueError("Actor retry authorizations require explicit matching task selection")
    gateway = uvicorn.Server(
        uvicorn.Config(
            application(config),
            host="0.0.0.0",
            port=config["gateway_port"],
            log_level="warning",
        )
    )
    tool_server = uvicorn.Server(
        uvicorn.Config(
            runtime.app, host="0.0.0.0", port=config["tools_port"], log_level="warning"
        )
    )
    servers = [
        asyncio.create_task(gateway.serve()),
        asyncio.create_task(tool_server.serve()),
    ]
    ready = False
    try:
        await wait_for_servers({"gateway": gateway, "tools": tool_server}, servers)
        ready = True
        await probe(config)
        if config.get("generation_protocol") == "three_b_fixed_baselines_v1":
            if probe_only:
                return
            names = only_tasks if only_tasks is not None else config["canaries"] if canary_only else config["task_ids"]
            if len(set(names)) != len(names) or not set(names) <= set(config["task_ids"]):
                raise ValueError("Run subset must name unique frozen tasks")
            await run_three_b_tasks(config, runtime, names)
            return
        recover_orphaned_tasks(config)
        if probe_only:
            return

        # A failed zero-round canary may be resumed after an implementation or
        # transport correction. Exact completed Actor and role caches remain in place.
        for task_id in config["canaries"]:
            recover_failed_canary(root, task_id)

        # First task gates basic operation; then three independent capability canaries fill the initial task limit.
        async def canary(tid):
            await task_run(config, runtime, tid)
            report(config)
            result = read(root / "tasks" / tid / "result.private.json")
            if result.get("completed_rounds") != 1:
                raise RuntimeError(
                    "Canary construction failed: "
                    + tid
                    + ": "
                    + str(result.get("construction_failure"))
                )
            scores = [
                read(
                    root
                    / "tasks"
                    / tid
                    / "round_1/actors"
                    / a
                    / "OFFICIAL.private.json"
                )
                for a in "ABC"
            ]
            if result.get("completed_rounds") != 1 or any(
                s.get("status") != "graded" for s in scores
            ):
                raise RuntimeError("Canary not accepted: " + tid)

        if config["canaries"]:
            await canary(config["canaries"][0])
            canary_results = await asyncio.gather(
                *(canary(tid) for tid in config["canaries"][1:]), return_exceptions=True
            )
            errors = [r for r in canary_results if isinstance(r, BaseException)]
            if errors:
                raise errors[0]
        if canary_only:
            return
        from experiments.claw_eval.opencode.capacity import run_remaining

        await run_remaining(config, runtime, task_run, report)
        result = report(config)
        write(
            root / "acceptance.json",
            {
                "accepted": result["completed_tasks"] == result["expected_tasks"]
                and result["completed_actors"] == result["expected_actors"]
                and result["valid_scores"] == result["expected_scores"],
                **{k: v for k, v in result.items() if k != "tasks"},
            },
        )
    finally:
        gateway.should_exit = tool_server.should_exit = True
        if not ready:
            for task in servers:
                task.cancel()
        await asyncio.gather(*servers, return_exceptions=not ready)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "run", "report"])
    parser.add_argument("--output", type=Path, required=True)
    add_provider_arguments(parser)
    parser.add_argument("--source-campaign", type=Path)
    parser.add_argument("--canary-only", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--only-task", action="append", help="Run only these frozen task IDs; leaves all other tasks pending")
    parser.add_argument("--actor-retry-authorization", action="append", type=Path,
                        help="Explicit operator approval for one evidence-bound terminated Actor retry")
    from reward_engine.adapters.jiuwenswarm.config import add_arguments, arguments
    add_arguments(parser)
    parser.add_argument("--task-id", action="append")
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--role-model", action="append", default=[])
    parser.add_argument("--native-grader-model")
    parser.add_argument("--per-model-concurrency", type=int)
    parser.add_argument("--minimum-free-gib", type=int, help="Frozen storage reserve for fresh campaigns; default 20 GiB")
    parser.add_argument("--revision", type=Path)
    args = parser.parse_args()
    if args.revision and args.action == "prepare":
        parser.error("Use experiments.revision to prepare an explicit code revision")
    if (arguments(args) or args.task_id or args.model_config or args.role_model or args.native_grader_model or args.per_model_concurrency is not None or args.minimum_free_gib is not None) and (args.action != "prepare" or args.source_campaign):
        parser.error("Runtime and task settings are frozen during fresh preparation")
    if provider_arguments_supplied(args) and (
        args.action != "prepare" or args.source_campaign
    ):
        parser.error("Provider settings must be supplied during fresh preparation")
    if args.only_task and (args.action != "run" or args.canary_only or args.probe_only):
        parser.error("--only-task requires run without canary/probe-only")
    if args.actor_retry_authorization and not args.only_task:
        parser.error("--actor-retry-authorization requires --only-task")
    if args.actor_retry_authorization:
        if any(not path.is_absolute() for path in args.actor_retry_authorization):
            parser.error("--actor-retry-authorization requires absolute paths")
        args.actor_retry_authorization = [path.resolve() for path in args.actor_retry_authorization]
    output = args.output.resolve()
    if args.action == "prepare":
        if args.source_campaign:
            prepare_web_rerun(output, args.source_campaign)
        else:
            prepare(
                output,
                credentials=args.credentials,
                key_env=args.key_env,
                backend_url=args.backend_url,
                task_ids=args.task_id,
                role_models=args.role_model, model_config=args.model_config,
                native_grader_model=args.native_grader_model,
                per_model_concurrency=args.per_model_concurrency,
                minimum_free_gib=args.minimum_free_gib,
                **arguments(args),
            )
        print(output)
        return
    from experiments.revision import load, verify_execution
    config = load(output, args.revision)
    if args.action == "report":
        report(config)
        return
    # Every launch uses its immutable source tree, including uncommitted fixes.
    if Path(__file__).resolve().parents[3] != Path(config["code"]):
        os.chdir(config["code"])
        os.execv(
            config["python"],
            [
                config["python"],
                "-m",
                "experiments.claw_eval.opencode.campaign",
                *sys.argv[1:],
            ],
        )
    with (output / "campaign.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(output / "process.json", {"pid": os.getpid(), "started_at": time.time()})
        with verification_progress(output / "verification-progress.json", "startup_verification"):
            verify_execution(config, verify_freeze)
        asyncio.run(serve(config, args.canary_only, args.probe_only,
                          **({"only_tasks": args.only_task} if args.only_task else {}),
                          **({"actor_retry_authorizations": args.actor_retry_authorization}
                             if args.actor_retry_authorization else {})))


if __name__ == "__main__":
    main()
