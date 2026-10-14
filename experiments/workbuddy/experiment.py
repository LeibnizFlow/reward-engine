"""Prepare, run and resume the pinned 50-task Outcome Office experiment."""

import argparse
import asyncio
import fcntl
import importlib.metadata
import os
import secrets
import shutil
import signal
import socket
import stat
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import httpx

from reward_engine.adapters.paths import WORKBUDDY_ROOT
from reward_engine.adapters.provider_config import (
    add_provider_arguments,
    provider_arguments_supplied,
    provider_key,
    resolve_provider,
)
from reward_engine.adapters.workbuddy.io import (
    digest,
    file_hash,
    read,
    role_model,
    write,
)
from reward_engine.adapters.workbuddy.models import Role, registry
from reward_engine.core.batch import BatchPolicy
from reward_engine.core.outcome.loop import run_task
from reward_engine.core.outcome.pool_selector import VERSION as POOL_SELECTION

REPO = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


UPSTREAM = WORKBUDDY_ROOT / "upstream"
PANEL = ["MiniMax-M3", "qwen3.8-max", "gemini-3.8-flash"]
_CODE_HASH_SCOPE = "all-regular-files-except-__pycache__-v1"


def _frozen_code_hashes(code):
    """Bind copied resources as well as Python; never follow frozen symlinks."""
    if not stat.S_ISDIR(code.lstat().st_mode):
        raise ValueError("Frozen implementation root must be a regular directory")
    hashes = {}

    def visit(directory):
        for path in sorted(directory.iterdir()):
            mode = path.lstat().st_mode
            name = path.relative_to(code).as_posix()
            if stat.S_ISDIR(mode):
                if path.name != "__pycache__":
                    visit(path)
            elif stat.S_ISREG(mode):
                hashes[name] = file_hash(path)
            else:
                raise ValueError(f"Frozen implementation is not a regular file: {name}")

    visit(code)
    return hashes


def role_mapping(*, include_final=False):
    roles = {
        r: "deepseek-v4-flash"
        for r in (
            "actor",
            "jury_reducer",
            "hypothesis_chooser",
            "harness_designer",
            "reward_calibration_judge",
        )
    }
    for family in ("jury", "hypothesis"):
        roles.update({f"{family}_member_{i}": m for i, m in enumerate(PANEL, 1)})
    if include_final:
        roles["final_selector"] = "deepseek-v4-flash"
    return roles


def slots(tasks):
    return [(task, "seed", i) for task in tasks for i in range(1, 4)] + [
        (task, f"round_{number}", arm)
        for task in tasks
        for number in range(1, 4)
        for arm in "ABC"
    ]


def prepare(
    output,
    upstream=UPSTREAM,
    recover_from=None,
    task_ids=None,
    reuse_first_round_from=None,
    *,
    task_concurrency=1,
    model_concurrency=2,
    per_model_concurrency=2,
    canary_from=None,
    selection_policy="legacy",
    public_delivery_contract=None,
    credentials=None,
    key_env=None,
    backend_url=None,
):
    if selection_policy not in {"legacy", POOL_SELECTION}:
        raise ValueError("Unknown selection policy")
    from workbuddy_bench.runner.prepare_tasks import (
        ensure_apt_mirror,
        ensure_composite_verifier_contract,
        ensure_host_gateway_compose,
        ensure_task_user,
    )

    if recover_from and reuse_first_round_from:
        raise ValueError("Choose seed recovery or explicit first-round replay")
    for value, maximum in (
        (task_concurrency, 3),
        (model_concurrency, 8),
        (per_model_concurrency, 8),
    ):
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError(f"Concurrency must be an integer from 1 to {maximum}")
    if output.exists():
        raise FileExistsError("Preparation requires a fresh output directory")
    tasks_root = upstream / "datasets/wb-bench-office-v1.0/tasks"
    tasks = sorted(p.name for p in tasks_root.iterdir() if (p / "task.toml").is_file())
    if len(tasks) != 50 or len(set(tasks)) != 50:
        raise ValueError("Expected the pinned complete Office 50")
    if task_ids is not None:
        if (
            not task_ids
            or len(set(task_ids)) != len(task_ids)
            or not set(task_ids) <= set(tasks)
        ):
            raise ValueError(
                "Selected task IDs must be unique members of the pinned Office 50"
            )
        tasks = list(task_ids)
    provider = resolve_provider(
        credentials=credentials,
        key_env=key_env,
        backend_url=backend_url,
        default_credentials=upstream / ".env",
    )
    provider_key(provider)
    output.mkdir(parents=True)
    code = output / "code"
    code.mkdir()
    for name in ("reward_engine",):
        shutil.copytree(
            REPO / name, code / name, ignore=shutil.ignore_patterns("__pycache__")
        )
    (code / "experiments").mkdir()
    shutil.copy2(REPO / "experiments/__init__.py", code / "experiments/__init__.py")
    shutil.copytree(
        REPO / "experiments/workbuddy",
        code / "experiments/workbuddy",
        ignore=shutil.ignore_patterns("__pycache__", "local"),
    )
    # Pin reused OpenCode transport too; concurrent work in the sibling checkout cannot change it.
    shutil.copytree(
        upstream.parent / "workbuddy_swarm",
        code / "workbuddy_swarm",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    dataset = output / "dataset/wb-bench-office-v1.0"
    shutil.copytree(
        tasks_root.parent, dataset, ignore=shutil.ignore_patterns("__pycache__")
    )
    original_hashes = {
        str(p.relative_to(tasks_root.parent)): file_hash(p)
        for p in tasks_root.parent.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    ensure_task_user(dataset / "tasks", agent_user="dev", verifier_user="root")
    ensure_host_gateway_compose(dataset / "tasks")
    ensure_composite_verifier_contract(dataset / "tasks")
    mirrors = {
        "apt": "http://mirrors.tuna.tsinghua.edu.cn",
        "pip": "https://pypi.tuna.tsinghua.edu.cn/simple",
        "npm": "https://registry.npmmirror.com",
    }
    ensure_apt_mirror(
        dataset / "tasks",
        mirrors["apt"],
        pip_index_url=mirrors["pip"],
        npm_registry=mirrors["npm"],
    )
    with socket.socket() as listener:
        listener.bind(("0.0.0.0", 0))
        port = listener.getsockname()[1]
    token = output / "campaign.auth"
    token.write_text(secrets.token_urlsafe(32))
    token.chmod(0o600)
    from reward_engine.adapters import paths as locations

    config = {
        "schema": "workbuddy.outcome.office50.v1",
        "code_hash_scope": _CODE_HASH_SCOPE,
        "repo": str(REPO),
        "code": str(code),
        "output": str(output),
        "experiment_id": locations.experiment_id(output),
        "report_dir": str(
            locations.REPORTS / "workbuddy" / locations.experiment_id(output)
        ),
        "upstream": str(upstream),
        "tasks_root": str(dataset / "tasks"),
        "task_ids": tasks,
        "rounds": 3,
        "seeds": 3,
        "max_revisions": 2,
        "max_tests": 3,
        "roles": role_mapping(include_final=selection_policy == POOL_SELECTION),
        "model_concurrency": model_concurrency,
        "task_concurrency": task_concurrency,
        "per_model_concurrency": per_model_concurrency,
        "model_aliases": {
            "deepseek-v4-flash": ["deepseek-v4-flash-0731", "deepseek-flash"]
        },
        "batch": {"concurrency": 3, "group_concurrency": 2, "deadline_s": 1800},
        **provider,
        "auth_file": str(token),
        "gateway_port": port,
        "gateway_local": f"http://127.0.0.1:{port}",
        "gateway_container": f"http://host.docker.internal:{port}",
        "harness_mount": str(
            upstream / "configs/harnesses/opencode-swarm/docker/rootfs"
        ),
        "minimum_free_bytes": 5 * 1024**3,
        "quality_resampling": False,
        "package_mirrors": mirrors,
        "official_protocol": "pinned WorkBuddy CompositeVerifier with deepseek-v4-flash",
    }
    if selection_policy != "legacy":
        config.update(selection_policy=selection_policy)
    if canary_from:
        from experiments.workbuddy.scoring import scoring_errors

        previous = read(canary_from / "config.json")
        verify(previous)
        old_manifest = read(canary_from / "manifest.json")
        effective_roles = {
            role: role_model(previous, role) for role in previous["roles"]
        }
        if (
            effective_roles != config["roles"]
            or previous["official_protocol"] != config["official_protocol"]
            or old_manifest["original_dataset_hashes"] != original_hashes
        ):
            raise ValueError(
                "Canary must use the same dataset, role models and scoring protocol"
            )
        state = read(canary_from / "status.json")
        if state.get("status") != "completed":
            raise ValueError("Canary campaign has not completed")
        task_id = previous["task_ids"][0]
        task_root = canary_from / "tasks" / task_id
        result_path = task_root / "result.private.json"
        result = read(result_path)
        if result.get("status") != "completed" or result.get("completed_rounds") != 3:
            raise ValueError("Canary requires three completed rounds")
        if (
            previous.get("selection_policy") == POOL_SELECTION
            and result.get("selected_delivery", {}).get("status")
            not in {"selected", "selected_with_unknowns"}
            or previous.get("selection_policy") != POOL_SELECTION
            and result["rounds"][-1]["ranking"].get("status") != "compared"
        ):
            raise ValueError("Canary final ranking unavailable")
        evidence = [canary_from / "status.json", result_path]
        if (canary_from / "role_models.json").exists():
            evidence.append(canary_from / "role_models.json")
        for arm in "ABC":
            path = task_root / f"round_3/actors/{arm}/OFFICIAL.private.json"
            receipt = read(path)
            native = Path(receipt["native_score"])
            if receipt.get("status") != "scored" or scoring_errors(read(native)):
                raise ValueError(
                    "Canary final round requires three scores without judge errors"
                )
            evidence.extend([path, native])
        config["validated_canary"] = {
            "source": str(canary_from),
            "task_id": task_id,
            "scope": "Completed pipeline and clean final-round grading; no products or memory reused",
            "evidence_hashes": {str(p): file_hash(p) for p in evidence},
        }
    if recover_from:
        previous = read(recover_from / "config.json")
        verify(previous)
        old_manifest = read(recover_from / "manifest.json")
        if (
            previous["roles"] != config["roles"]
            or old_manifest["original_dataset_hashes"] != original_hashes
        ):
            raise ValueError("Recovery source models or original tasks differ")
        for name, expected in old_manifest["code_hashes"].items():
            if name.startswith("reward_engine/core/") or name in {
                "reward_engine/adapters/workbuddy/agent.py",
                "reward_engine/adapters/workbuddy/evidence.py",
                "experiments/workbuddy/worker.py",
            }:
                if file_hash(code / name) != expected:
                    raise ValueError(
                        f"Recovery changes actor or learning semantics: {name}"
                    )
        sources = {}
        for task_id in tasks:
            for number in range(1, 4):
                source = recover_from / "tasks" / task_id / "seeds" / str(number)
                if (source / "RESULT.json").exists() and read(
                    source / "RESULT.json"
                ).get("status") == "completed":
                    if read(source / "spec.json")["grade"]:
                        raise ValueError("Seed recovery source was graded")
                    sources[f"{task_id}/{number}"] = {
                        "path": str(source),
                        "sha256": file_hash(source / "RESULT.json"),
                    }
        config["seed_sources"] = sources
        config["recovery_from"] = str(recover_from)
    if reuse_first_round_from:
        from experiments.workbuddy.reuse import freeze_sources

        previous = read(reuse_first_round_from / "config.json")
        verify(previous)
        old_manifest = read(reuse_first_round_from / "manifest.json")
        if (
            old_manifest["original_dataset_hashes"] != original_hashes
            or previous["roles"]["actor"] != config["roles"]["actor"]
            or previous["official_protocol"] != config["official_protocol"]
        ):
            raise ValueError(
                "Replay requires the same dataset, actor model and native scoring protocol"
            )
        config["reused_slots"] = freeze_sources(reuse_first_round_from, tasks)
        config["reuse_first_round_from"] = str(reuse_first_round_from)
    if public_delivery_contract is not None:
        if selection_policy != "legacy" or recover_from or reuse_first_round_from:
            raise ValueError("Public delivery v1 requires a fresh legacy run")
        from reward_engine.adapters.workbuddy.public_delivery import freeze_contract

        config["public_delivery_contract"] = freeze_contract(
            public_delivery_contract, output, config["tasks_root"], tasks
        )
    config["identity"] = digest(config)
    write(output / "config.json", config)
    versions = {
        d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
    }
    sources = _frozen_code_hashes(code)
    write(
        output / "manifest.json",
        {
            "task_ids": tasks,
            "actor_slots": len(slots(tasks)),
            "grading_slots": len(tasks) * 9,
            "branch": subprocess.check_output(
                ["git", "branch", "--show-current"], cwd=REPO, text=True
            ).strip(),
            "base_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
            ).strip(),
            "code_hashes": sources,
            "code_hash_scope": _CODE_HASH_SCOPE,
            "original_dataset_hashes": original_hashes,
            "prepared_dataset_hashes": {
                str(p.relative_to(dataset)): file_hash(p)
                for p in dataset.rglob("*")
                if p.is_file()
            },
            "opencode_sha256": file_hash(
                Path(config["harness_mount"]) / "bin/opencode"
            ),
            "versions": versions,
            "config_identity": config["identity"],
            **(
                {"public_delivery_contract": config["public_delivery_contract"]}
                if config.get("public_delivery_contract")
                else {}
            ),
        },
    )
    write(
        output / "status.json",
        {"status": "prepared", "requested": len(tasks), "finished": 0, "tasks": {}},
    )
    return config


def check_space(config):
    for directory in (config["output"], "/var/lib/docker"):
        if shutil.disk_usage(directory).free < config["minimum_free_bytes"]:
            raise OSError(
                f"Insufficient free space to dispatch another trial: {directory}"
            )


def verify(config):
    manifest = read(Path(config["output"]) / "manifest.json")
    if (
        digest({k: v for k, v in config.items() if k != "identity"})
        != config["identity"]
    ):
        raise ValueError("Campaign configuration changed after preparation")
    if config.get("public_delivery_contract"):
        binding = config["public_delivery_contract"]
        if manifest.get("public_delivery_contract") != binding:
            raise ValueError("Public contract manifest mismatch")
        from reward_engine.adapters.workbuddy.public_delivery import read_bound_contract

        read_bound_contract(
            binding,
            {
                "task_id": binding["task_id"],
                "prompt": (
                    Path(config["tasks_root"]) / binding["task_id"] / "instruction.md"
                )
                .read_bytes()
                .decode("utf-8"),
            },
        )
    hashes = manifest["code_hashes"]
    code = Path(config["code"])
    if "code_hash_scope" in config or "code_hash_scope" in manifest:
        if (
            config.get("code_hash_scope") != _CODE_HASH_SCOPE
            or manifest.get("code_hash_scope") != _CODE_HASH_SCOPE
        ):
            raise ValueError("Frozen implementation hash scope missing or unsupported")
        if manifest.get("config_identity") != config["identity"]:
            raise ValueError("Frozen manifest configuration identity mismatch")
        if not isinstance(hashes, dict):
            raise ValueError("Frozen implementation hashes must be a mapping")
        actual = _frozen_code_hashes(code)
        if set(actual) != set(hashes):
            raise ValueError(
                "Frozen implementation file set changed: "
                f"missing={sorted(set(hashes) - set(actual))}; "
                f"unexpected={sorted(set(actual) - set(hashes))}"
            )
    else:
        # Historical manifests only bind listed files, usually Python. Do not
        # retrospectively claim complete resource coverage for those runs.
        actual = {name: file_hash(code / name) for name in hashes}
    for name, expected in hashes.items():
        if actual[name] != expected:
            raise ValueError(f"Frozen implementation changed: {name}")
    dataset = Path(config["tasks_root"]).parent
    for name, expected in manifest["prepared_dataset_hashes"].items():
        if file_hash(dataset / name) != expected:
            raise ValueError(f"Frozen dataset changed: {name}")
    if (
        file_hash(Path(config["harness_mount"]) / "bin/opencode")
        != manifest["opencode_sha256"]
    ):
        raise ValueError("Pinned OpenCode binary changed")
    for name, expected in (
        config.get("validated_canary", {}).get("evidence_hashes", {}).items()
    ):
        if file_hash(Path(name)) != expected:
            raise ValueError(f"Canary evidence changed: {name}")


async def launch(config, task, output, *, prompt="", pack=None, grade=True):
    task_path = Path(config["tasks_root"]) / task["task_id"]
    limit = (
        2 * tomllib.loads((task_path / "task.toml").read_text())["agent"]["timeout_sec"]
    )
    identity = digest(
        {
            "campaign": config["identity"],
            "task": task,
            "slot": str(output),
            "prompt": prompt,
            "pack": pack,
            "grade": grade,
        }
    )
    if (output / "RESULT.json").exists():
        row = read(output / "RESULT.json")
        if row.get("spec_identity") != identity:
            raise ValueError("Existing sample belongs to different inputs")
        return row
    if output.exists():
        raise ValueError(
            "Interrupted sample retained; explicit recovery required, no fresh task retry"
        )
    check_space(config)
    output.mkdir(parents=True)
    spec = {
        "campaign_config": str(Path(config["output"]) / "config.json"),
        "output": str(output),
        "candidate_id": "wb-" + identity[:20],
        "task": task,
        "task_path": str(task_path),
        "actor_timeout_s": limit,
        "prompt": prompt,
        "pack": pack,
        "grade": grade,
        "spec_identity": identity,
    }
    phase = "seed" if not grade else output.parent.parent.name
    reused = config.get("reused_slots", {}).get(
        f"{task['task_id']}/{phase}/{output.name}"
    )
    if reused:
        from experiments.workbuddy.reuse import replay

        spec["candidate_id"] = reused["candidate_id"]
        spec["generation"] = (
            "reused product; prompt and pack below are current assessment context"
        )
        spec["source_spec"] = str(Path(reused["path"]) / "spec.json")
        write(output / "spec.json", spec)
        return replay(reused, spec, output)
    write(output / "spec.json", spec)
    source = (
        config.get("seed_sources", {}).get(f"{task['task_id']}/{output.name}")
        if not grade
        else None
    )
    if source:
        original = Path(source["path"])
        if (
            file_hash(original / "RESULT.json") != source["sha256"]
            or read(original / "spec.json")["task"] != task
        ):
            raise ValueError("Recovered seed evidence changed")
        row = read(original / "RESULT.json")
        row["recovered_source_identity"] = row["spec_identity"]
        row["spec_identity"], row["reused_from"] = identity, str(original)
        write(output / "RESULT.json", row)
        write(output / "ACTOR_RESULT.json", row)
        return row
    command = [
        sys.executable,
        "-m",
        "experiments.workbuddy.worker",
        "--spec",
        str(output / "spec.json"),
    ]
    with (output / "worker.log").open("w") as log:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=config["code"],
            env={**os.environ, "PYTHONPATH": config["code"], "PYTHONUNBUFFERED": "1"},
            stdout=log,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        write(output / "process.json", {"pid": process.pid, "command": command})
        try:
            await process.wait()
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            await process.wait()
            raise
    if not (output / "RESULT.json").exists():
        write(
            output / "RESULT.json",
            {
                "status": "failed",
                "exit_code": process.returncode,
                "spec_identity": identity,
                "log": str(output / "worker.log"),
            },
        )
    return read(output / "RESULT.json")


async def one_task(config, task_id):
    root = Path(config["output"]) / "tasks" / task_id
    root.mkdir(parents=True, exist_ok=True)
    task = {
        "task_id": task_id,
        "prompt": (Path(config["tasks_root"]) / task_id / "instruction.md").read_text(),
    }

    binding = config.get("public_delivery_contract")
    if binding and binding["task_id"] == task_id:
        from reward_engine.adapters.workbuddy.public_delivery import read_bound_contract

        task["public_delivery_contract"] = read_bound_contract(binding, task)

    def record(name, value):
        write(root / (name + ".json"), value)
        write(root / "progress.json", {"phase": name, "updated_at": time.time()})

    seed_rows = await asyncio.gather(
        *(
            launch(config, task, root / "seeds" / str(i), grade=False)
            for i in range(1, 4)
        ),
        return_exceptions=True,
    )
    if any(
        isinstance(row, BaseException) or row.get("status") != "completed"
        for row in seed_rows
    ):
        raise RuntimeError(
            "Initial sample collection incomplete; inspect retained seed logs"
        )

    async def sample(arm, number, prompt, pack):
        return await launch(
            config,
            task,
            root / f"round_{number}" / "actors" / arm,
            prompt=prompt,
            pack=pack,
        )

    async def evaluate(task, row):
        # Only this callback reads private scores. Core passes only candidate to learning.
        cid = row["candidate"]["candidate_id"]
        paths = [
            p
            for p in root.glob("round_*/actors/*/spec.json")
            if read(p)["candidate_id"] == cid
        ]
        if len(paths) != 1:
            raise ValueError("Ambiguous official result binding")
        return read(paths[0].parent / "OFFICIAL.private.json")

    from experiments.workbuddy.resume import completed_rounds
    from reward_engine.core.outcome.memory import Memory

    seeds = [row["candidate"] for row in seed_rows]
    history, checkpoint = completed_rounds(root, task, seeds)
    independent = config.get("selection_policy", "legacy") == POOL_SELECTION
    if len(history) == 3:
        result = {
            "status": "completed",
            "completed_rounds": 3,
            "rounds": history,
            "selected_delivery": history[-1]["selection"],
            "memory": checkpoint["memory"],
            "official_evaluation": "separate_from_outcome_estimator",
        }
    else:
        restored = (
            {
                "start_round": len(history) + 1,
                "memory": Memory(checkpoint["memory"]),
                "initial_checks": checkpoint["checks"],
                "initial_pack": checkpoint["previous_pack"],
                "initial_expectation_cache": checkpoint["expectation_cache"],
                "initial_known_gaps": checkpoint.get("known_gaps", []),
                "initial_decisions": checkpoint.get("decisions", []),
            }
            if checkpoint
            else {}
        )
        if checkpoint:
            record(
                "resume",
                {"completed_rounds": len(history), "next_round": len(history) + 1},
            )
        selection_kwargs = {}
        if independent:
            selection_kwargs["defer_final_selection"] = True
        result = await run_task(
            task,
            checkpoint["pool"] if checkpoint else seeds,
            rounds=3 - len(history),
            roles=lambda role, prefix: Role(
                config, role, root / prefix / "models" / role
            ),
            registry=registry(config, root / "measurement"),
            sample=sample,
            evaluate=evaluate,
            policy=BatchPolicy(**config["batch"]),
            jury_reduce_max_bytes=Role.request_budget_bytes,
            continuation_policy="fixed_rounds",
            full_audit=config.get("full_audit", False),
            **(
                {
                    "measurement_candidate_concurrency": config[
                        "measurement_candidate_concurrency"
                    ]
                }
                if "measurement_candidate_concurrency" in config
                else {}
            ),
            record=record,
            **restored,
            **selection_kwargs,
        )
        result["rounds"] = history + result["rounds"]
        result["completed_rounds"] += len(history)
    if independent and result["completed_rounds"] == 3:
        from reward_engine.adapters.workbuddy.pool_selector import finalize

        # Round checkpoints are already durable. A failure below resumes ONLY this stage.
        result["selected_delivery"] = {"status": "pending", "arm": None, "source": None}
        result["status"] = "partial"
        record("result.private", result)
        result["selected_delivery"] = await finalize(config, root, task)
        result["status"] = (
            "completed"
            if result["selected_delivery"]["candidate_key"] is not None
            else "partial"
        )
    record("result.private", result)
    return {
        "status": result["status"],
        "completed_rounds": result["completed_rounds"],
        "scored": sum(
            s.get("status") == "scored"
            for r in result["rounds"]
            for s in r["official"].values()
        ),
    }


async def campaign(config, *, canary_only=False):
    from experiments.workbuddy.report import report

    root = Path(config["output"])
    state = read(root / "status.json")
    verify(config)
    names = config["task_ids"][:1] if canary_only else config["task_ids"]
    with (root / "campaign.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (root / "gateway.log").open("a") as log:
            gateway = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "reward_engine.adapters.provider_config",
                "--config",
                str(root / "config.json"),
                cwd=config["code"],
                env={**os.environ, "PYTHONPATH": config["code"]},
                stdout=log,
                stderr=asyncio.subprocess.STDOUT,
            )
            try:
                async with httpx.AsyncClient(trust_env=False) as client:
                    for _ in range(100):
                        if gateway.returncode is not None:
                            raise RuntimeError("Campaign proxy failed to launch")
                        try:
                            response = await client.get(
                                config["gateway_local"] + "/health", timeout=1
                            )
                            if response.status_code == 200:
                                break
                        except httpx.HTTPError:
                            pass
                        await asyncio.sleep(0.1)
                    else:
                        raise TimeoutError("Campaign proxy readiness timeout")
                from experiments.workbuddy.schedule import run_tasks

                await run_tasks(
                    config,
                    state,
                    names,
                    one_task=one_task,
                    check_space=check_space,
                    record=lambda value: write(root / "status.json", value),
                )
                state.update(
                    status="completed"
                    if len(state["tasks"]) == len(config["task_ids"])
                    and all(
                        r.get("status") == "completed" and r.get("scored") == 9
                        for r in state["tasks"].values()
                    )
                    else "partial",
                    current_task=None,
                    current_tasks=[],
                )
                write(root / "status.json", state)
            except BaseException as error:
                state.update(
                    status="interrupted",
                    error_type=type(error).__name__,
                    reason=str(error),
                )
                write(root / "status.json", state)
                raise
            finally:
                if gateway.returncode is None:
                    gateway.terminate()
                    await gateway.wait()
                report(root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run", "report"))
    parser.add_argument("--output", type=Path, required=True)
    native_paths = parser.add_mutually_exclusive_group()
    native_paths.add_argument("--upstream", type=Path)
    native_paths.add_argument("--workbuddy-root", type=Path)
    add_provider_arguments(parser)
    parser.add_argument("--canary-only", action="store_true")
    parser.add_argument("--recover-from", type=Path)
    parser.add_argument("--reuse-first-round-from", type=Path)
    parser.add_argument(
        "--task-id",
        action="append",
        help="Prepare only these Office task IDs; repeatable",
    )
    parser.add_argument(
        "--task-concurrency", type=int, choices=range(1, 4), default=None
    )
    parser.add_argument(
        "--model-concurrency", type=int, choices=range(1, 9), default=None
    )
    parser.add_argument(
        "--per-model-concurrency", type=int, choices=range(1, 9), default=None
    )
    parser.add_argument(
        "--canary-from",
        type=Path,
        help="Use a completed prior canary as pipeline validation only",
    )
    parser.add_argument("--selection-policy", choices=("legacy", POOL_SELECTION))
    parser.add_argument("--public-delivery-contract", type=Path)
    args = parser.parse_args()
    if (
        args.task_id
        or args.reuse_first_round_from
        or args.canary_from
        or args.task_concurrency
        or args.model_concurrency
        or args.per_model_concurrency
        or args.selection_policy
        or args.public_delivery_contract
        or provider_arguments_supplied(args)
        or args.upstream
        or args.workbuddy_root
    ) and args.action != "prepare":
        parser.error(
            "Task, provider, native paths and replay sources are frozen at prepare time"
        )
    output = args.output.resolve()
    if args.action == "prepare":
        prepare(
            output,
            (
                args.upstream
                or (
                    args.workbuddy_root / "upstream"
                    if args.workbuddy_root
                    else UPSTREAM
                )
            )
            .expanduser()
            .resolve(),
            args.recover_from.resolve() if args.recover_from else None,
            task_ids=args.task_id,
            reuse_first_round_from=args.reuse_first_round_from.resolve()
            if args.reuse_first_round_from
            else None,
            task_concurrency=args.task_concurrency or 1,
            model_concurrency=args.model_concurrency or 2,
            per_model_concurrency=args.per_model_concurrency or 2,
            canary_from=args.canary_from.resolve() if args.canary_from else None,
            selection_policy=args.selection_policy or "legacy",
            public_delivery_contract=args.public_delivery_contract,
            credentials=args.credentials,
            key_env=args.key_env,
            backend_url=args.backend_url,
        )
        print(output / "config.json")
    elif args.action == "run":
        config = read(output / "config.json")
        if REPO != Path(config["code"]):
            os.chdir(config["code"])
            os.execve(
                sys.executable,
                [
                    sys.executable,
                    "-m",
                    "experiments.workbuddy.experiment",
                    *sys.argv[1:],
                ],
                {**os.environ, "PYTHONPATH": config["code"]},
            )
        asyncio.run(campaign(config, canary_only=args.canary_only))
    else:
        from experiments.workbuddy.report import report

        report(output)


if __name__ == "__main__":
    main()
