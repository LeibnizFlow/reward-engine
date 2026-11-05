"""Both native entries, real containers/CLI and local deterministic model responses.

Set JIUWENSWARM_TEST_BUNDLE, JIUWENSWARM_TEST_IMAGE, CLAW_EVAL_ROOT and
WORKBUDDY_ROOT to installed dependencies. All task data below is synthetic.
"""

import json
import os
import shutil
import sys
from pathlib import Path
from uuid import uuid4

import pytest
import yaml

from reward_engine.adapters.jiuwenswarm.container import docker
from reward_engine.adapters.workbuddy.gateway import application
from reward_engine.adapters.workbuddy.io import digest, read, write
from tests.support.jiuwenswarm_local import local_model, serving, tool
from tests.support.three_b_model import PROMPT, response

REPO = Path(__file__).resolve().parents[2]


async def model_response(body, index):
    messages = body["messages"]
    # Roles carry a structured engine request; actors carry the real CLI prompt.
    for message in messages:
        if message["role"] == "user" and isinstance(message.get("content"), str):
            try:
                request = json.loads(message["content"])
            except ValueError:
                continue
            if isinstance(request, dict) and "mode" in request and "context" in request:
                return {"role": "assistant", "content": json.dumps(response(request))}
    names = {t["function"]["name"] for t in body.get("tools", [])}
    if not any(m["role"] == "tool" for m in messages):
        if "mcp_claw_Bash" in names:
            return tool(
                "mcp_claw_Bash", {"command": "printf '{}' > /workspace/output.json"}
            )
        return tool("bash", {"command": "printf '{}' > /workspace/output.json"})
    return {"role": "assistant", "content": "{}"}


@pytest.mark.asyncio
@pytest.mark.parametrize("benchmark", ["claw", "office"])
async def test_native_entry_three_rounds_and_exact_resume(
    tmp_path, benchmark, monkeypatch
):
    bundle, base = (
        os.environ.get("JIUWENSWARM_TEST_BUNDLE"),
        os.environ.get("JIUWENSWARM_TEST_IMAGE"),
    )
    assert bundle and base, (
        "BLOCKED: real JiuwenSwarm bundle and Docker Office image required"
    )
    auth = tmp_path / "auth"
    auth.write_text("synthetic-controller")
    roles = dict.fromkeys(
        [
            "actor",
            "jury_reducer",
            "harness_designer",
            "reward_calibration_judge",
            "attack_reducer",
            *[
                f"{family}_{i}"
                for family in ("jury_member", "attack_member", "attack_reference")
                for i in range(1, 4)
            ],
        ],
        "fixture",
    )
    config = {
        "output": str(tmp_path),
        "code": str(REPO),
        "identity": "synthetic",
        "campaign_id": "synthetic-" + uuid4().hex,
        "actor_backend": "jiuwenswarm",
        "jiuwenswarm_bundle": bundle,
        "jiuwenswarm_mode": "agent.code.normal",
        "generation_protocol": "three_b_fixed_baselines_v1",
        "rounds": 3,
        "task_ids": ["synthetic"],
        "roles": roles,
        "role_stream": False,
        "role_protocol_edits": True,
        "role_bounded_stage": True,
        "auth_file": str(auth),
        "model_concurrency": 4,
        "per_model_concurrency": 4,
        "batch": {"concurrency": 3, "group_concurrency": 2, "deadline_s": 120},
        "actor_timeout_s": 90,
        "minimum_free_bytes": 0,
        "model_aliases": {},
        "python": sys.executable,
        "docker_data_root": "/var/lib/docker",
    }
    synthetic_image = None
    if benchmark == "claw":
        from experiments.claw_eval.opencode.campaign import task_run
        from experiments.claw_eval.opencode.runtime import Runtime

        upstream = Path(os.environ["CLAW_EVAL_ROOT"])
        sys.path.insert(0, str(upstream / "src"))
        native = tmp_path / "benchmark"
        native.mkdir()
        (native / "src").symlink_to(upstream / "src", target_is_directory=True)
        task = native / "tasks/synthetic"
        task.mkdir(parents=True)
        (task / "task.yaml").write_text(
            yaml.safe_dump(
                {
                    "task_id": "synthetic",
                    "task_name": "Synthetic JSON task",
                    "prompt": {"text": PROMPT},
                    "env_snapshot_files": ["/workspace/output.json"],
                }
            )
        )
        (
            task / "grader.py"
        ).write_text("""from claw_eval.graders.base import AbstractGrader
from claw_eval.models.trace import DimensionScores
class SyntheticGrader(AbstractGrader):
    def grade(self, messages, dispatches, task, env_snapshot=None, **kwargs):
        import json
        assert json.loads(env_snapshot["file:/workspace/output.json"]["content"]) == {}
        return DimensionScores(completion=1.0, robustness=1.0, safety=1.0)
""")
        config.update(
            benchmark_root=str(native),
            image=(
                await docker(
                    "image", "inspect", "claw-eval-agent:latest", "--format", "{{.Id}}"
                )
            ).strip(),
        )
        runtime = Runtime(config)
        # Admission thresholds are unrelated to protocol validation on this fixture.
        monkeypatch.setattr(runtime, "check_resources", lambda: None)

        async def run():
            return await task_run(config, runtime, "synthetic")
    else:
        from experiments.workbuddy.experiment import one_task

        upstream = Path(os.environ["WORKBUDDY_ROOT"]) / "upstream"
        dataset = tmp_path / "dataset"
        shutil.copytree(
            upstream / "datasets/wb-bench-office-v1.0/shared/verifier",
            dataset / "shared/verifier",
            ignore=shutil.ignore_patterns("__pycache__", "tests"),
        )
        (dataset / "dataset.toml").write_text(
            '[dataset]\nid="synthetic"\nversion="1"\n[verifier]\nschema="workbuddy.verifier.v1"\nengine="composite"\n'
        )
        task = dataset / "tasks/synthetic"
        (task / "environment").mkdir(parents=True)
        (task / "tests/grading").mkdir(parents=True)
        (task / "instruction.md").write_text(PROMPT)
        (task / "task.toml").write_text(
            '[task]\nname="fixture/synthetic"\n[metadata]\nsource_case="synthetic"\n[agent]\ntimeout_sec=90\n[verifier]\nimport_path="workbuddy_bench.judge:CompositeVerifier"\ntimeout_sec=120\n[environment]\nbuild_timeout_sec=120\ncpus=1\nmemory_mb=2048\n'
        )
        build = tmp_path / "image"
        build.mkdir()
        (build / "Dockerfile").write_text(
            f"FROM {base}\nUSER root\nRUN rm -rf /workspace && mkdir /workspace && chown dev /workspace\nWORKDIR /workspace\n"
        )
        synthetic_image = "reward-engine-synthetic:" + uuid4().hex
        await docker("build", "--network", "none", "-t", synthetic_image, str(build))
        image = (
            await docker("image", "inspect", synthetic_image, "--format", "{{.Id}}")
        ).strip()
        (task / "environment/Dockerfile").write_text(f"FROM {image}\n")
        (task / "environment/docker-compose.yaml").write_text(
            yaml.safe_dump(
                {
                    "services": {
                        "main": {
                            "image": image,
                            "extra_hosts": ["host.docker.internal:host-gateway"],
                        }
                    }
                }
            )
        )
        (task / "tests/test.sh").write_text("#!/bin/sh\nexit 64\n")
        (task / "tests/judge.yaml").write_text(
            "version: 1\nartifacts: []\nllm_judge:\n  rubrics: []\n"
        )
        (task / "tests/grading/test_json.py").write_text(
            "import json\nfrom pathlib import Path\ndef test_output():\n    assert json.loads(Path('/workspace/output.json').read_text()) == {}\n"
        )
        (task / "tests/verifier.toml").write_text(
            'schema_version="workbuddy.office.verifier.v1"\n[run]\ncwd="/workspace"\ncommand="python -m pytest /tests/grading -p no:cacheprovider --junitxml=/logs/verifier/results.xml > /logs/verifier/test_output.txt 2>&1"\n'
        )
        config.update(tasks_root=str(dataset / "tasks"))
        from reward_engine.adapters.jiuwenswarm.office import configure_images
        from reward_engine.adapters.workbuddy.io import file_hash

        images = tmp_path / "images.json"
        write(
            images,
            {
                "records": [
                    {"task_id": "synthetic", "image_id": image, "input_sha256": {}}
                ]
            },
        )
        config.update(
            task_images_manifest=str(images), task_images_sha256=file_hash(images)
        )
        configure_images(config, dataset / "tasks")

        async def run():
            return await one_task(config, "synthetic")

    try:
        async with local_model(model_response) as (port, calls):
            config["backend_url"] = f"http://127.0.0.1:{port}/v1"
            async with serving(
                application(config, backend_key="local-only")
            ) as gateway_port:
                config.update(
                    gateway_local=f"http://127.0.0.1:{gateway_port}",
                    gateway_container=f"http://host.docker.internal:{gateway_port}",
                )
                if benchmark == "claw":
                    async with serving(runtime.app) as tools_port:
                        config.update(
                            tools_port=tools_port,
                            tools_container=f"http://host.docker.internal:{tools_port}",
                        )
                        config["identity"] = digest(config)
                        write(tmp_path / "config.json", config)
                        result = await run()
                        assert result["completed_rounds"] == 3
                        count = len(calls)
                        assert (await run())["completed_rounds"] == 3 and len(
                            calls
                        ) == count
                else:
                    config["identity"] = digest(config)
                    write(tmp_path / "config.json", config)
                    result = await run()
                    assert result["completed_rounds"] == 3
                    count = len(calls)
                    assert (await run())["completed_rounds"] == 3 and len(
                        calls
                    ) == count
            saved = read(tmp_path / "tasks/synthetic/result.private.json")
            assert saved["status"] == "completed"
            assert (
                len(
                    list(
                        (tmp_path / "tasks/synthetic").glob(
                            "grades.private/*/stage.json"
                        )
                    )
                )
                == 12
            )
            assert (
                len(
                    {
                        read(p)["session"]
                        for p in (tmp_path / "tasks/synthetic").glob("**/RESULT.json")
                    }
                )
                == 12
            )
            assert all(
                len(r["official"]) == 6 and r["hypothesis_applied"] is False
                for r in saved["rounds"]
            )
            from reward_engine.adapters.jiuwenswarm.report import report
            rendered = report(config, tmp_path / "report")
            assert rendered["completed_actors"] == 12 and rendered["valid_scores"] == 12
            assert rendered["completed_tasks"] == 1
    finally:
        if synthetic_image:
            await docker("image", "rm", synthetic_image)
