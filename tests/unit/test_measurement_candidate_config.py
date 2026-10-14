"""WorkBuddy-only candidate admission: actual prepare, freeze and runner wiring."""

import asyncio
import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from experiments import validate_pipeline as validation
from experiments.workbuddy import experiment, resume
from reward_engine.adapters import model_config as profiles
from reward_engine.adapters.workbuddy.io import digest, file_hash, read, write
from tests.unit import test_validation_model_config as profile_tests

offline = profile_tests.offline
launcher = profile_tests.launcher
prepare_tree = profile_tests.prepare_tree
KEY = "measurement_candidate_concurrency"
REPO = Path(__file__).resolve().parents[2]
PUBLIC_PROFILE = REPO / "reward_engine/configs/bailian_token_plan.json"
SOURCES = (
    "reward_engine/core/outcome/loop.py",
    "reward_engine/adapters/model_config.py",
    "experiments/workbuddy/experiment.py",
    "reward_engine/configs/bailian_token_plan.json",
)


@pytest.fixture
def frozen(prepare_tree, tmp_path):
    for relative in SOURCES:
        target = tmp_path / "repo" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((REPO / relative).read_bytes())
    source = tmp_path / "profile.json"
    source.write_bytes(PUBLIC_PROFILE.read_bytes())
    return prepare_tree(with_launcher=True, model_config=source)


def test_real_prepare_binds_policy_bytes_sources_effective_settings_and_identity(
    frozen,
):
    config = frozen.config
    assert config[KEY] == 1
    assert config["batch"] == {
        "concurrency": 3,
        "group_concurrency": 2,
        "deadline_s": 1800,
    }
    assert config["model_concurrency"] == config["per_model_concurrency"] == 2
    assert config["model_config"]["effective"][KEY] == 1
    assert config["model_config"]["sha256"] == file_hash(PUBLIC_PROFILE)
    assert (
        frozen.output / "model_config.json"
    ).read_bytes() == PUBLIC_PROFILE.read_bytes()
    assert frozen.manifest["model_config"] == config["model_config"]
    assert config["identity"] == digest(
        {k: v for k, v in config.items() if k != "identity"}
    )
    assert frozen.manifest["config_identity"] == config["identity"]
    for relative in SOURCES:
        assert frozen.manifest["code_hashes"][relative] == file_hash(REPO / relative)
        assert file_hash(frozen.code / relative) == file_hash(REPO / relative)
    assert read(frozen.output / "validation_protocol.json")["role_protocol"][KEY] == 1
    experiment.verify(config)
    profiles.verify_profile(config)


@pytest.mark.parametrize("value", [None, True, False, 0, -1, 1.5, "1", 129, [], {}])
def test_invalid_profile_value_rejected_before_prepare(launcher, value):
    with pytest.raises(ValueError, match=KEY):
        launcher.prepare({"schema": profiles.SCHEMA, KEY: value})
    launcher.freeze.assert_not_called()


@pytest.mark.parametrize("value", [1, 2, 128])
def test_profile_policy_is_workbuddy_only_and_does_not_change_other_budgets(
    launcher, value
):
    profile = {"schema": profiles.SCHEMA, KEY: value}
    if launcher.benchmark == "claw_eval":
        with pytest.raises(ValueError, match="WorkBuddy-only"):
            launcher.prepare(profile)
        return
    config = launcher.prepare(profile)
    assert config[KEY] == value
    assert config["batch"]["deadline_s"] == config["role_stage_timeout_s"] == 1800
    assert config["model_concurrency"] == config["per_model_concurrency"] == 6
    profiles.verify_profile(config)


def test_absent_preserves_old_config_effective_settings_and_binding(launcher):
    config = launcher.prepare({"schema": profiles.SCHEMA})
    assert KEY not in config
    assert KEY not in profiles.effective_settings(config)
    assert KEY not in config["model_config"]["effective"]
    assert (
        KEY not in read(launcher.output / "validation_protocol.json")["role_protocol"]
    )
    profiles.verify_profile(config)
    config[KEY] = 1
    profiles.apply_profile(config, {"schema": profiles.SCHEMA})
    assert KEY not in config


@pytest.mark.parametrize(
    "tamper",
    [
        "value",
        "remove",
        "raw",
        "effective",
        "manifest",
        "missing_binding",
        "rebound_effective",
    ],
)
def test_tamper_rejected_before_gateway(frozen, monkeypatch, tamper):
    config, root = frozen.config, frozen.output
    if tamper == "value":
        config[KEY] = 2
    elif tamper == "remove":
        del config[KEY]
    elif tamper == "raw":
        path = root / "model_config.json"
        profile = read(path)
        profile[KEY] = 2
        write(path, profile)
    elif tamper == "effective":
        config["model_config"]["effective"][KEY] = 2
    elif tamper == "manifest":
        manifest = read(root / "manifest.json")
        manifest["model_config"]["effective"][KEY] = 2
        write(root / "manifest.json", manifest)
    elif tamper == "missing_binding":
        del config["model_config"]
    else:
        # Even synchronized effective metadata must still match the raw profile.
        config[KEY] = config["model_config"]["effective"][KEY] = 2
        config["identity"] = digest(
            {k: v for k, v in config.items() if k != "identity"}
        )
        manifest = read(root / "manifest.json")
        manifest["config_identity"] = config["identity"]
        manifest["model_config"] = deepcopy(config["model_config"])
        write(root / "manifest.json", manifest)
    gateway = Mock(side_effect=AssertionError("Gateway must not start"))
    monkeypatch.setattr(validation, "application", gateway)
    with pytest.raises(ValueError):
        asyncio.run(validation.run(config))
    gateway.assert_not_called()


def test_frozen_source_drift_is_rejected_by_existing_freeze_verifier(frozen):
    with (frozen.code / SOURCES[0]).open("a") as stream:
        stream.write("\n# source drift\n")
    with pytest.raises(ValueError, match="Frozen implementation"):
        experiment.verify(frozen.config)


def test_unbound_policy_is_not_accepted_as_a_legacy_config(tmp_path):
    write(tmp_path / "manifest.json", {})
    with pytest.raises(ValueError, match="binding is missing"):
        profiles.verify_profile({"output": str(tmp_path), KEY: 1})


def test_invalid_profile_application_is_atomic():
    config = {"validation_benchmark": "claw_eval", "untouched": True}
    before = deepcopy(config)
    with pytest.raises(ValueError, match="WorkBuddy-only"):
        profiles.apply_profile(config, {"schema": profiles.SCHEMA, KEY: 1})
    assert config == before


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [None, 1, 2])
async def test_workbuddy_one_task_forwards_only_explicit_policy(
    tmp_path, monkeypatch, limit
):
    instruction = tmp_path / "dataset/task/instruction.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_text("Return valid JSON")
    config = {
        "output": str(tmp_path),
        "tasks_root": str(instruction.parent.parent),
        "batch": {"concurrency": 3, "group_concurrency": 2, "deadline_s": 1800},
    }
    if limit is not None:
        config[KEY] = limit
    config["identity"] = digest(config)
    before = deepcopy(config)
    monkeypatch.setattr(resume, "completed_rounds", lambda *_: ([], None))
    monkeypatch.setattr(experiment, "registry", lambda *_: object())

    async def launch(config, task, slot, **kwargs):
        assert kwargs == {"grade": False}
        return {"status": "completed", "candidate": {"candidate_id": slot.name}}

    monkeypatch.setattr(experiment, "launch", launch)

    async def core(task, seeds, **kwargs):
        assert len(seeds) == 3
        if limit is None:
            assert KEY not in kwargs
        else:
            assert kwargs[KEY] == limit
        assert kwargs["policy"].deadline_s == 1800
        assert kwargs["policy"].concurrency == 3
        assert kwargs["policy"].group_concurrency == 2
        return {"status": "partial", "completed_rounds": 0, "rounds": []}

    run = AsyncMock(side_effect=core)
    monkeypatch.setattr(experiment, "run_task", run)
    await experiment.one_task(config, "task")
    run.assert_awaited_once()
    assert config == before


def test_duplicate_and_nested_policy_fields_are_not_alternate_configuration_paths():
    with pytest.raises(ValueError, match="duplicate"):
        profiles.parse_profile(
            (f'{{"schema":"{profiles.SCHEMA}","{KEY}":1,"{KEY}":2}}').encode()
        )
    with pytest.raises(ValueError, match="concurrency field"):
        profiles.parse_profile(
            json.dumps({"schema": profiles.SCHEMA, "concurrency": {KEY: 1}}).encode()
        )
