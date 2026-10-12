"""Validation execution is not a stage verdict; exercise both frozen runners offline."""

import asyncio
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from experiments import validate_pipeline as validation
from experiments.claw_eval.opencode import campaign, runtime
from experiments.workbuddy import experiment
from reward_engine.adapters.workbuddy.io import digest, file_hash, read, write


@pytest.fixture(params=["claw_eval", "workbuddy"])
def frozen(request, tmp_path, monkeypatch):
    """Replace benchmark provisioning only; retain real config/hash verification."""
    benchmark = request.param
    output = tmp_path / "validation"
    code = output / "code"
    binary = output / "harness/bin/opencode"
    original = {}

    def freeze(destination, **kwargs):
        assert destination == output
        assert kwargs["task_ids"] == ["task"]
        if benchmark == "workbuddy":
            assert (
                kwargs["upstream"]
                == validation.REPO / "local/workbuddy/runner/upstream"
            )
        (code / "experiments").mkdir(parents=True)
        binary.parent.mkdir(parents=True)
        binary.write_text("offline binary fixture")
        config = {
            "output": str(output),
            "code": str(code),
            "tasks_root": str(output / "dataset/tasks"),
            "benchmark_root": str(output / "benchmark"),
            "harness_mount": str(binary.parent.parent),
            "gateway_port": 12345,
            "tools_port": 12346,
            "task_ids": ["task"],
            "seeds": 3,
            "rounds": 1,
            "roles": {"actor": "offline-fixture"},
        }
        config["identity"] = digest(config)
        original.update(deepcopy(config))
        write(
            output / "manifest.json",
            {
                "config_identity": config["identity"],
                "files": {"harness/bin/opencode": file_hash(binary)},
                "code_hashes": {},
                "prepared_dataset_hashes": {},
                "opencode_sha256": file_hash(binary),
                "provenance": "retain-me",
            },
        )
        return config

    module = campaign if benchmark == "claw_eval" else experiment
    monkeypatch.setattr(module, "prepare", freeze)
    monkeypatch.setenv("OPENCODE_BINARY", "/unused/offline-opencode")
    config = validation.prepare(
        SimpleNamespace(output=output, benchmark=benchmark, task_id="task")
    )
    verify = campaign.verify_freeze if benchmark == "claw_eval" else experiment.verify
    return SimpleNamespace(
        config=config, output=output, original=original, verify=verify
    )


@pytest.fixture
def offline_runner(monkeypatch):
    servers = []

    class Server:
        def __init__(self, config):
            self.started = True
            self.should_exit = False
            self.stopped = False
            servers.append(self)

        async def serve(self):
            while not self.should_exit:
                await asyncio.sleep(0)
            self.stopped = True

    monkeypatch.setattr(validation.uvicorn, "Server", Server)
    monkeypatch.setattr(validation, "application", Mock(return_value=object()))
    fake_runtime = SimpleNamespace(app=object())
    monkeypatch.setattr(runtime, "Runtime", Mock(return_value=fake_runtime))
    monkeypatch.setattr(validation.sys, "path", list(validation.sys.path))
    claw_run = AsyncMock()
    buddy_run = AsyncMock()
    monkeypatch.setattr(campaign, "task_run", claw_run)
    monkeypatch.setattr(experiment, "one_task", buddy_run)
    return SimpleNamespace(
        servers=servers, claw=claw_run, buddy=buddy_run, runtime=fake_runtime
    )


def test_prepare_rebinds_validation_settings_and_freezes_driver(frozen):
    config = frozen.config
    assert config["rounds"] == 3
    assert config["full_audit"] is True
    assert config["seeds"] == 3
    assert config["validation_task_id"] == "task"
    assert config["task_ids"] == ["task"]
    assert config["quality_resampling"] is False
    assert config["role_bounded_stage"] is True
    assert config["role_response_format"] == "json_object"
    assert config["role_stage_timeout_s"] == 1800
    assert read(frozen.output / "role_limits.json") == config["validation_role_limits"]
    assert config["roles"] == frozen.original["roles"]
    assert config["identity"] != frozen.original["identity"]
    assert config["identity"] == digest(
        {k: v for k, v in config.items() if k != "identity"}
    )
    assert read(frozen.output / "config.json") == config
    manifest = read(frozen.output / "manifest.json")
    assert manifest["config_identity"] == config["identity"]
    assert manifest["provenance"] == "retain-me"
    assert (manifest["actor_slots"], manifest["grading_slots"]) == (12, 9)
    relative = "experiments/validate_pipeline.py"
    frozen_code = Path(config["code"]) / relative
    assert frozen_code.read_bytes() == Path(validation.__file__).read_bytes()
    hashes = (
        manifest["files"]
        if config["validation_benchmark"] == "claw_eval"
        else manifest["code_hashes"]
    )
    key = (
        "code/" + relative
        if config["validation_benchmark"] == "claw_eval"
        else relative
    )
    assert hashes[key] == file_hash(frozen_code)
    protocol = read(frozen.output / "validation_protocol.json")
    assert protocol["full_harness_audit"] is True
    assert protocol["historical_samples_reused"] is False
    assert protocol["quality_resampling"] is False
    assert protocol["role_protocol"]["contract_repairs_per_stage"] == 1
    frozen.verify(config)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tamper", ["rounds", "full_audit", "validation_task_id", "code", "role_limits"]
)
async def test_changed_frozen_inputs_are_rejected_before_dispatch(
    frozen, offline_runner, tamper
):
    config = frozen.config
    if tamper == "code":
        (Path(config["code"]) / "experiments/validate_pipeline.py").write_text(
            "changed"
        )
    elif tamper == "role_limits":
        write(frozen.output / "role_limits.json", {"role_max_tokens": {}})
    else:
        config[tamper] = {
            "rounds": 1,
            "full_audit": False,
            "validation_task_id": "other",
        }[tamper]
    with pytest.raises(ValueError, match="changed"):
        await validation.run(config)
    offline_runner.claw.assert_not_awaited()
    offline_runner.buddy.assert_not_awaited()
    assert not (frozen.output / "validation_status.json").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result, expected_exit",
    [
        ({"status": "completed", "completed_rounds": 3}, 0),
        ({"status": "completed", "completed_rounds": 2}, 1),
        ({"status": "completed", "completed_rounds": 4}, 1),
        ({"status": "partial", "completed_rounds": 3}, 1),
        ({"status": "failed", "completed_rounds": 3}, 1),
        ({"status": "completed"}, 1),
        ({"completed_rounds": 3}, 1),
        ({}, 1),
    ],
)
async def test_runner_exit_never_certifies_stage_completion(
    frozen, offline_runner, result, expected_exit
):
    config = frozen.config
    before = deepcopy(config)
    offline_runner.claw.return_value = result

    async def workbuddy(config, task_id):
        write(frozen.output / "tasks" / task_id / "result.private.json", result)
        # A successful function return must not override the retained task receipt.
        return {"status": "completed", "completed_rounds": 3}

    offline_runner.buddy.side_effect = workbuddy
    assert await validation.run(config) == expected_exit
    status = read(frozen.output / "validation_status.json")
    assert status["status"] == (
        "execution_completed" if expected_exit == 0 else "failed"
    )
    assert status["stage_audit"] == "pending"
    assert status["completed_rounds"] == result.get("completed_rounds")
    if config["validation_benchmark"] == "claw_eval":
        offline_runner.claw.assert_awaited_once_with(
            config, offline_runner.runtime, "task"
        )
        offline_runner.buddy.assert_not_awaited()
        assert len(offline_runner.servers) == 2
    else:
        offline_runner.buddy.assert_awaited_once_with(config, "task")
        offline_runner.claw.assert_not_awaited()
        assert len(offline_runner.servers) == 1
    assert all(s.should_exit and s.stopped for s in offline_runner.servers)
    assert config == before
    frozen.verify(config)


@pytest.mark.asyncio
async def test_runner_failure_retains_error_and_shuts_down_servers(
    frozen, offline_runner
):
    error = RuntimeError("missing native grading receipt")
    offline_runner.claw.side_effect = error
    offline_runner.buddy.side_effect = error
    with pytest.raises(RuntimeError, match="missing native grading receipt"):
        await validation.run(frozen.config)
    status = read(frozen.output / "validation_status.json")
    assert status["status"] == "failed"
    assert status["error_type"] == "RuntimeError"
    assert status["reason"] == str(error)
    assert all(s.should_exit and s.stopped for s in offline_runner.servers)


def test_main_reexecutes_only_the_prepared_frozen_code(frozen, monkeypatch):
    config = frozen.config
    prepare = Mock(return_value=config)
    monkeypatch.setattr(validation, "prepare", prepare)
    monkeypatch.setattr(
        validation.sys,
        "argv",
        [
            "validate_pipeline",
            "--benchmark",
            config["validation_benchmark"],
            "--task-id",
            "task",
            "--output",
            str(frozen.output),
        ],
    )
    events = []
    monkeypatch.setattr(
        validation.os, "chdir", lambda path: events.append(("chdir", path))
    )
    monkeypatch.setattr(
        validation.os, "execv", lambda *args: events.append(("execv", *args))
    )
    validation.main()
    prepare.assert_called_once()
    assert events == [
        ("chdir", config["code"]),
        (
            "execv",
            validation.sys.executable,
            [
                validation.sys.executable,
                "-m",
                "experiments.validate_pipeline",
                "--execute",
                "--output",
                config["output"],
            ],
        ),
    ]
