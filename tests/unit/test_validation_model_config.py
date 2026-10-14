"""Explicit prospective model profiles; temporary fixtures and offline Role calls only."""

import asyncio
import hashlib
import json
import socket
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from experiments import validate_pipeline as validation
from experiments.claw_eval.opencode import campaign
from experiments.workbuddy import experiment
from reward_engine.adapters import model_config as profiles
from reward_engine.adapters.workbuddy import models
from reward_engine.adapters.workbuddy.gateway import application
from reward_engine.adapters.workbuddy.io import digest, file_hash, read, write
from tests.unit import test_workbuddy_frozen_manifest as freeze_tests

prepare_tree = freeze_tests.prepare_tree


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No provider or native connections in profile tests")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    for name in (
        "REWARD_ENGINE_CREDENTIALS_FILE",
        "REWARD_ENGINE_KEY_ENV",
        "REWARD_ENGINE_BACKEND_URL",
        "WORKBUDDY_ROOT",
    ):
        monkeypatch.delenv(name, raising=False)


def profile_value():
    roles = {role: "qwen3.8-max" for role in profiles.ROLES}
    roles["actor"] = "qwen3.8-flash"
    for family in ("jury", "hypothesis"):
        roles.update(
            {
                f"{family}_member_{i}": model
                for i, model in enumerate(
                    ("qwen3.8-max", "deepseek-v4-flash-0731", "glm-5.3"), 1
                )
            }
        )
    return {
        "schema": profiles.SCHEMA,
        "roles": roles,
        "role_max_tokens": {role: 32768 for role in profiles.ROLES - {"actor"}},
        "model_aliases": {},
        "concurrency": {"model_concurrency": 2, "per_model_concurrency": 1},
        "role_request_options_by_model": {"glm-5.3": {}},
    }


@pytest.fixture(params=["workbuddy", "claw_eval"])
def launcher(request, tmp_path, monkeypatch):
    benchmark = request.param
    output = tmp_path / "run"
    credentials = tmp_path / "fixture.env"
    credentials.write_text(
        "YIBU_API_KEY=offline-fixture\nYIBU_BASE_URL=http://offline.invalid/v1\n"
    )

    def freeze(destination, **kwargs):
        assert destination == output
        code = output / "code"
        (code / "experiments").mkdir(parents=True)
        target = code / "reward_engine/adapters/model_config.py"
        target.parent.mkdir(parents=True)
        target.write_bytes(Path(profiles.__file__).read_bytes())
        binary = output / "harness/bin/opencode"
        binary.parent.mkdir(parents=True)
        binary.write_text("offline binary fixture")
        config = {
            "output": str(output),
            "code": str(code),
            "tasks_root": str(output / "dataset/tasks"),
            "harness_mount": str(binary.parent.parent),
            "roles": experiment.role_mapping(),
            "model_aliases": {
                "deepseek-v4-flash": ["deepseek-v4-flash-0731", "deepseek-flash"]
            },
            "model_concurrency": 2,
            "per_model_concurrency": 2,
            "batch": {"concurrency": 3, "group_concurrency": 2, "deadline_s": 1800},
            "gateway_port": 12345,
            "auth_file": str(output / "auth"),
            "gateway_local": "http://offline",
            "seeds": 3,
            "backend_url": "http://provider.invalid/v1",
        }
        (output / "auth").write_text("offline-fixture")
        config["identity"] = digest(config)
        write(
            output / "manifest.json",
            {
                "config_identity": config["identity"],
                "files": {
                    "code/reward_engine/adapters/model_config.py": file_hash(target)
                },
                "code_hashes": {
                    "reward_engine/adapters/model_config.py": file_hash(target)
                },
                "prepared_dataset_hashes": {},
                "opencode_sha256": file_hash(binary),
            },
        )
        return config

    frozen = Mock(side_effect=freeze)
    monkeypatch.setattr(
        experiment if benchmark == "workbuddy" else campaign, "prepare", frozen
    )

    def prepare(value=None, raw=None, role_thinking="provider_default"):
        source = tmp_path / "profile.json"
        if value is not None or raw is not None:
            source.write_bytes(
                raw if raw is not None else json.dumps(value, indent=3).encode() + b"\n"
            )
        args = SimpleNamespace(
            output=output,
            benchmark=benchmark,
            task_id="fixture",
            credentials=credentials,
            model_config=source if source.exists() else None,
            role_thinking=role_thinking,
        )
        return validation.prepare(args)

    return SimpleNamespace(
        prepare=prepare,
        freeze=frozen,
        output=output,
        source=tmp_path / "profile.json",
        benchmark=benchmark,
    )


def test_prepare_freezes_exact_bytes_and_effective_protocol(launcher):
    value = profile_value()
    config = launcher.prepare(value)
    root = launcher.output
    raw = launcher.source.read_bytes()
    assert (root / "model_config.json").read_bytes() == raw
    assert config["model_config"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert read(root / "manifest.json")["model_config"] == config["model_config"]
    assert config["identity"] == digest(
        {k: v for k, v in config.items() if k != "identity"}
    )
    assert config["roles"] == value["roles"]
    if launcher.benchmark == "workbuddy":
        assert (
            config["official_protocol"]
            == "pinned WorkBuddy CompositeVerifier with qwen3.8-flash"
        )
    assert config["model_aliases"] == {}
    assert config["rounds"] == 3 and config["full_audit"] is True
    assert config["quality_resampling"] is False and config["seeds"] == 3
    limits = read(root / "role_limits.json")
    assert limits == config["validation_role_limits"]
    assert limits["role_max_tokens"] == value["role_max_tokens"]
    protocol = read(root / "validation_protocol.json")["role_protocol"]
    assert protocol["role_max_tokens"] == value["role_max_tokens"]
    assert protocol["roles"] == value["roles"]
    assert protocol["concurrency"] == value["concurrency"]
    assert (
        protocol["actor_output_tokens"]
        == protocol["native_grader_output_tokens"]
        == 32768
    )
    assert "output_tokens" not in protocol and "reducer_output_tokens" not in protocol
    assert protocol["contract_repairs_per_stage"] == 1
    verify = (
        experiment.verify
        if launcher.benchmark == "workbuddy"
        else campaign.verify_freeze
    )
    verify(config)
    profiles.verify_profile(config)
    launcher.source.unlink()  # Execute consumes the frozen bytes, never the mutable input.
    profiles.verify_profile(config)


def test_default_configuration_and_protocol_remain_unchanged(launcher):
    config = launcher.prepare()
    assert "model_config" not in config
    assert not (launcher.output / "model_config.json").exists()
    assert config["roles"] == experiment.role_mapping()
    assert config["model_aliases"]["deepseek-v4-flash"] == [
        "deepseek-v4-flash-0731",
        "deepseek-flash",
    ]
    assert config["model_concurrency"] == config["per_model_concurrency"] == 6
    limits = config["validation_role_limits"]["role_max_tokens"]
    assert limits["jury_reducer"] == 131072 and limits["harness_designer"] == 65536
    protocol = read(launcher.output / "validation_protocol.json")["role_protocol"]
    assert (
        protocol["output_tokens"] == 65536
        and protocol["reducer_output_tokens"] == 131072
    )
    profiles.verify_profile(config)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "unknown"),
        ("unknown", {}),
        ("roles", []),
        ("roles", {"jury_member_4": "model"}),
        ("roles", {"native_grader": "model"}),
        ("roles", {"actor": ""}),
        ("roles", {"actor": True}),
        ("role_max_tokens", {"actor": 32768}),
        ("role_max_tokens", {"jury_reducer": True}),
        ("role_max_tokens", {"jury_reducer": 1.5}),
        ("role_max_tokens", {"jury_reducer": 131073}),
        ("role_max_tokens", {"jury_reducer": 0}),
        ("concurrency", {"model_concurrency": 129}),
        ("concurrency", {"per_model_concurrency": False}),
        ("concurrency", {"task_concurrency": 1}),
        ("concurrency", {"group_concurrency": 0}),
        ("model_aliases", {"model": "alias"}),
        ("model_aliases", {"model": ["alias", "alias"]}),
        (
            "role_request_options_by_model",
            {"glm-5.3": {"thinking": {"type": "disabled"}}},
        ),
        (
            "role_request_options_by_model",
            {"glm-5.3": {"thinking": {"type": "disabled"}, "enable_thinking": 0}},
        ),
        ("role_request_options_by_model", {"glm-5.3": {"extra": True}}),
    ],
)
def test_invalid_profiles_rejected_before_benchmark_preparation(launcher, field, value):
    profile = {"schema": profiles.SCHEMA, field: value}
    with pytest.raises(ValueError):
        launcher.prepare(profile)
    launcher.freeze.assert_not_called()
    assert not launcher.output.exists()


@pytest.mark.parametrize(
    "raw",
    [
        b"[]",
        b"{",
        b'{"schema":"reward_engine.model_config.v1","roles":{"actor":"a","actor":"b"}}',
        b'{"schema":"reward_engine.model_config.v1","role_max_tokens":{"jury_reducer":NaN}}',
        b" " * 65537,
    ],
)
def test_malformed_or_ambiguous_json_rejected(launcher, raw):
    with pytest.raises(ValueError):
        launcher.prepare(raw=raw)
    launcher.freeze.assert_not_called()


@pytest.mark.parametrize(
    "tamper",
    [
        "bytes",
        "missing",
        "symlink",
        "roles",
        "tokens",
        "alias",
        "concurrency",
        "request_options",
        "manifest",
        "missing_binding",
        "role_limits",
        "role_models",
        "transport_limits",
    ],
)
def test_tampered_profile_rejected_before_gateway(launcher, monkeypatch, tamper):
    config = launcher.prepare(profile_value())
    root = launcher.output
    if tamper == "bytes":
        (root / "model_config.json").write_text("{}")
    elif tamper in {"missing", "symlink"}:
        (root / "model_config.json").unlink()
        if tamper == "symlink":
            (root / "model_config.json").symlink_to(launcher.source)
    elif tamper == "roles":
        config["roles"]["actor"] = "different-model"
    elif tamper == "tokens":
        config["validation_role_limits"]["role_max_tokens"]["jury_reducer"] = 1
    elif tamper == "alias":
        config["model_aliases"] = {"qwen3.8-flash": ["different-model"]}
    elif tamper == "concurrency":
        config["model_concurrency"] = 4
    elif tamper == "request_options":
        config["role_request_options_by_model"] = {}
    elif tamper == "manifest":
        manifest = read(root / "manifest.json")
        manifest["model_config"]["sha256"] = "0" * 64
        write(root / "manifest.json", manifest)
    elif tamper == "missing_binding":
        del config["model_config"]
    else:
        write(root / (tamper + ".json"), {})
    gateway = Mock(side_effect=AssertionError("Gateway must not start"))
    monkeypatch.setattr(validation, "application", gateway)
    with pytest.raises(ValueError):
        asyncio.run(validation.run(config))
    gateway.assert_not_called()


def test_aliases_are_explicit_and_cannot_collapse_requested_models(launcher):
    config = launcher.prepare(profile_value())
    original = deepcopy(config)
    for aliases in (
        {"unrequested": ["alias"]},
        {"qwen3.8-max": ["glm-5.3"]},
        {"qwen3.8-max": ["shared"], "glm-5.3": ["shared"]},
    ):
        profile = profiles.parse_profile(
            json.dumps({"schema": profiles.SCHEMA, "model_aliases": aliases}).encode()
        )
        with pytest.raises(ValueError):
            profiles.apply_profile(config, profile)
        assert config == original
    profiles.apply_profile(
        config, {"model_aliases": {"qwen3.8-max": ["explicit-version"]}}
    )
    assert config["model_aliases"] == {"qwen3.8-max": ["explicit-version"]}


def test_model_options_and_concurrency_respect_configured_scope(launcher):
    config = launcher.prepare(profile_value())
    before = deepcopy(config)
    for profile in (
        {"role_request_options_by_model": {"unconfigured-model": {}}},
        {"concurrency": {"model_concurrency": 9}},
    ):
        with pytest.raises(ValueError):
            profiles.apply_profile(config, profile)
        assert config == before
    config["max_model_concurrency"] = 128
    profiles.apply_profile(
        config,
        {"concurrency": {"model_concurrency": 128, "per_model_concurrency": 128}},
    )
    assert config["model_concurrency"] == config["per_model_concurrency"] == 128


@pytest.mark.asyncio
async def test_glm_provider_default_overrides_global_disabled_only_for_glm(
    launcher, monkeypatch
):
    config = launcher.prepare(profile_value(), role_thinking="disabled")
    calls = []

    def reply(request):
        body = json.loads(request.content)
        calls.append(body)
        assert body["max_tokens"] == 32768
        if body["model"] == "glm-5.3":
            assert "thinking" not in body and "enable_thinking" not in body
        else:
            assert body["model"] == "qwen3.8-max"
            assert body["thinking"] == {"type": "disabled"}
            assert body["enable_thinking"] is False
        event = {
            "model": body["model"],
            "choices": [
                {"delta": {"content": '{"fixture":true}'}, "finish_reason": "stop"}
            ],
        }
        return httpx.Response(
            200, text="data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n"
        )

    client = httpx.AsyncClient
    monkeypatch.setattr(
        models.httpx,
        "AsyncClient",
        lambda **kw: client(transport=httpx.MockTransport(reply), **kw),
    )
    for name in ("jury_member_3", "jury_reducer"):
        role = models.Role(config, name, launcher.output / name)
        assert await role({"mode": "offline-fixture"}) == {"fixture": True}
    assert len(calls) == 2
    protocol = read(launcher.output / "validation_protocol.json")["role_protocol"]
    assert protocol["role_request_options_by_model"] == {"glm-5.3": {}}
    assert protocol["role_request_options"] == {
        "thinking": {"type": "disabled"},
        "enable_thinking": False,
    }
    profiles.verify_profile(config)


@pytest.mark.asyncio
@pytest.mark.parametrize("actual_model", ["qwen3.8-max", "invented-alias"])
async def test_actual_role_request_uses_frozen_model_and_32k_budget(
    launcher, monkeypatch, actual_model
):
    config = launcher.prepare(profile_value())
    calls = []

    def reply(request):
        body = json.loads(request.content)
        calls.append(body)
        assert body["model"] == "qwen3.8-max" and body["max_tokens"] == 32768
        event = {
            "model": actual_model,
            "choices": [
                {"delta": {"content": '{"fixture":true}'}, "finish_reason": "stop"}
            ],
        }
        return httpx.Response(
            200, text="data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n"
        )

    client = httpx.AsyncClient
    monkeypatch.setattr(
        models.httpx,
        "AsyncClient",
        lambda **kw: client(transport=httpx.MockTransport(reply), **kw),
    )
    role = models.Role(config, "jury_reducer", launcher.output / "role")
    if actual_model == "qwen3.8-max":
        assert await role({"mode": "offline-fixture"}) == {"fixture": True}
    else:
        with pytest.raises(ValueError, match="mismatch"):
            await role({"mode": "offline-fixture"})
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_profile_concurrency_reaches_real_gateway(launcher):
    config = launcher.prepare(profile_value())
    app = application(config, backend_key="offline-fixture")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://offline"
    ) as client:
        health = (await client.get("/health")).json()
    assert health["limit"] == 2 and health["per_model_limit"] == 1


def test_real_workbuddy_freeze_includes_new_module_and_profile(prepare_tree, tmp_path):
    module = tmp_path / "repo/reward_engine/adapters/model_config.py"
    module.write_bytes(Path(profiles.__file__).read_bytes())
    path = tmp_path / "profile.json"
    write(path, profile_value())
    frozen = prepare_tree(with_launcher=True, model_config=path)
    experiment.verify(frozen.config)
    profiles.verify_profile(frozen.config)
    assert frozen.manifest["code_hashes"][
        "reward_engine/adapters/model_config.py"
    ] == file_hash(module)
    assert frozen.manifest["model_config"] == frozen.config["model_config"]
    assert frozen.config["roles"]["actor"] == "qwen3.8-flash"


def test_execute_cannot_override_model_profile(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_pipeline",
            "--execute",
            "--output",
            str(tmp_path / "absent"),
            "--model-config",
            str(tmp_path / "absent.json"),
        ],
    )
    with pytest.raises(SystemExit) as error:
        validation.main()
    assert error.value.code == 2
