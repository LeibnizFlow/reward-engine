"""Offline launch/transport regressions; all credentials and benchmark inputs are fixtures."""

import json
import socket
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from experiments import smoke, validate_pipeline
from experiments.claw_eval.opencode import campaign
from experiments.workbuddy import experiment
from reward_engine.adapters import provider_config as provider
from reward_engine.adapters.workbuddy.io import file_hash, read, write
from tests.unit import test_workbuddy_frozen_manifest as freeze_tests

prepare_tree = freeze_tests.prepare_tree


@pytest.fixture(autouse=True)
def offline_environment(monkeypatch):
    for name in (
        "REWARD_ENGINE_CREDENTIALS_FILE",
        "REWARD_ENGINE_KEY_ENV",
        "REWARD_ENGINE_BACKEND_URL",
        "YIBU_API_KEY",
        "YIBU_BASE_URL",
        "OPENLUX_API_KEY",
        "OPENLUX_BASE_URL",
        "CUSTOM_API_KEY",
        "CUSTOM_BASE_URL",
        "WORKBUDDY_ROOT",
    ):
        monkeypatch.delenv(name, raising=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("Provider tests must never connect to a network")

    monkeypatch.setattr(socket.socket, "connect", forbidden)


def test_settings_and_secret_precedence(tmp_path, monkeypatch):
    selected = tmp_path / "selected.env"
    selected.write_text(
        "CUSTOM_API_KEY=dotenv-fixture\n"
        "REWARD_ENGINE_KEY_ENV=CUSTOM_API_KEY\n"
        "REWARD_ENGINE_BACKEND_URL=https://dotenv.invalid/api/v2/\n"
    )
    other = tmp_path / "unused.env"
    other.write_text("YIBU_API_KEY=wrong-fixture\n")
    monkeypatch.setenv("REWARD_ENGINE_CREDENTIALS_FILE", str(other))
    config = provider.resolve_provider(credentials=selected, default_credentials=other)
    assert config["credentials_file"] == str(selected)
    assert config["backend_key_env"] == "CUSTOM_API_KEY"
    assert config["backend_url"] == "https://dotenv.invalid/api/v2"
    assert provider.provider_key(config) == "dotenv-fixture"

    # Process environment beats both generic and provider-specific dotenv settings.
    monkeypatch.setenv("CUSTOM_BASE_URL", "https://environment.invalid/v3/")
    monkeypatch.setenv("CUSTOM_API_KEY", "environment-fixture")
    config = provider.resolve_provider(credentials=selected, default_credentials=other)
    assert config["backend_url"] == "https://environment.invalid/v3"
    assert provider.provider_key(config) == "environment-fixture"
    monkeypatch.setenv("REWARD_ENGINE_BACKEND_URL", "https://generic.invalid/v4")
    assert (
        provider.resolve_provider(credentials=selected, default_credentials=other)[
            "backend_url"
        ]
        == "https://generic.invalid/v4"
    )
    config = provider.resolve_provider(
        credentials=selected,
        key_env="CUSTOM_API_KEY",
        backend_url="https://cli.invalid/custom/",
        default_credentials=other,
    )
    assert config["backend_url"] == "https://cli.invalid/custom"
    assert "fixture" not in json.dumps(config)


def test_credentials_and_key_name_environment_over_dotenv(tmp_path, monkeypatch):
    selected = tmp_path / "selected.env"
    selected.write_text(
        "REWARD_ENGINE_KEY_ENV=YIBU_API_KEY\nYIBU_API_KEY=unused-fixture\n"
    )
    monkeypatch.setenv("REWARD_ENGINE_CREDENTIALS_FILE", str(selected))
    monkeypatch.setenv("REWARD_ENGINE_KEY_ENV", "CUSTOM_API_KEY")
    monkeypatch.setenv("CUSTOM_API_KEY", "environment-fixture")
    monkeypatch.setenv("CUSTOM_BASE_URL", "http://offline.invalid")
    config = provider.resolve_provider(default_credentials=tmp_path / "missing")
    assert config["credentials_file"] == str(selected)
    assert config["backend_key_env"] == "CUSTOM_API_KEY"
    assert config["backend_url"] == "http://offline.invalid/v1"
    config = provider.resolve_provider(
        default_credentials=tmp_path / "missing",
        key_env="YIBU_API_KEY",
        backend_url="https://cli.invalid/v1",
    )
    assert provider.provider_key(config) == "unused-fixture"


def test_documented_dotenv_yibu_base_url_is_used_by_both_launchers(tmp_path):
    credentials = tmp_path / ".env"
    credentials.write_text(
        "YIBU_API_KEY=offline-fixture\nYIBU_BASE_URL=https://chosen.invalid/v1/\n"
    )
    options = provider.prepare_options(
        SimpleNamespace(), default_credentials=credentials
    )
    assert options["backend_url"] == "https://chosen.invalid/v1"
    assert options["credentials"] == str(credentials)
    assert options["key_env"] == "YIBU_API_KEY"


@pytest.mark.parametrize(
    "url",
    [
        "",
        "ftp://host",
        "https://user:fixture@host/v1",
        "https://host/v1?key=fixture",
        "https://host/v1#fixture",
        "https://host:invalid/v1",
        "https://ho st/v1",
    ],
)
def test_invalid_urls_fail_without_echoing_credentials(tmp_path, url):
    with pytest.raises(ValueError) as failure:
        provider.resolve_provider(
            default_credentials=tmp_path / "missing", backend_url=url
        )
    assert "fixture" not in str(failure.value)


def test_custom_key_cannot_silently_use_default_provider(tmp_path):
    with pytest.raises(ValueError, match="explicit base URL"):
        provider.resolve_provider(
            default_credentials=tmp_path / "missing", key_env="CUSTOM_API_KEY"
        )


def test_empty_environment_key_does_not_fall_back_to_dotenv(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text("YIBU_API_KEY=dotenv-fixture\n")
    config = provider.resolve_provider(default_credentials=path)
    monkeypatch.setenv("YIBU_API_KEY", "")
    with pytest.raises(ValueError, match="Missing provider API key"):
        provider.provider_key(config)


@pytest.mark.parametrize("launcher", ["direct", "smoke", "pipeline"])
def test_real_workbuddy_prepare_environment_only(
    prepare_tree, tmp_path, monkeypatch, launcher
):
    repo = tmp_path / "repo"
    upstream = repo / "_local/native/workbuddy/runner/upstream"
    (upstream / ".env").unlink()
    assert not (repo / ".env").exists()
    monkeypatch.setenv("REWARD_ENGINE_KEY_ENV", "CUSTOM_API_KEY")
    monkeypatch.setenv("CUSTOM_API_KEY", "environment-only-fixture")
    monkeypatch.setenv("CUSTOM_BASE_URL", "https://selected.invalid/api/v2/")
    if launcher == "smoke":
        monkeypatch.setattr(smoke, "REPO", repo)
        config = smoke.prepare(
            SimpleNamespace(
                output=tmp_path / "frozen", benchmark="workbuddy", task_id="task_00"
            )
        )
    elif launcher == "pipeline":
        # No --credentials: exercise a missing default .env through BOTH resolutions.
        config = validate_pipeline.prepare(
            SimpleNamespace(
                output=tmp_path / "frozen", benchmark="workbuddy", task_id="task_00"
            )
        )
    else:
        config = prepare_tree().config
    assert config["backend_key_env"] == "CUSTOM_API_KEY"
    assert config["backend_url"] == "https://selected.invalid/api/v2"
    assert provider.provider_key(config) == "environment-only-fixture"
    experiment.verify(config)
    root = Path(config["output"])
    assert "environment-only-fixture" not in (root / "config.json").read_text()
    relative = "reward_engine/adapters/provider_config.py"
    assert read(root / "manifest.json")["code_hashes"][relative] == file_hash(
        Path(provider.__file__)
    )
    assert not Path(config["credentials_file"]).exists()


def test_real_workbuddy_prepare_custom_file_without_yibu(prepare_tree, tmp_path):
    (tmp_path / "repo/_local/native/workbuddy/runner/upstream/.env").unlink()
    credentials = tmp_path / "custom.env"
    credentials.write_text(
        "CUSTOM_API_KEY=custom-file-fixture\nCUSTOM_BASE_URL=https://custom.invalid/v2/\n"
    )
    frozen = prepare_tree(credentials=credentials, key_env="CUSTOM_API_KEY")
    assert frozen.config["backend_url"] == "https://custom.invalid/v2"
    assert frozen.config["backend_key_env"] == "CUSTOM_API_KEY"
    experiment.verify(frozen.config)
    assert "custom-file-fixture" not in (frozen.output / "config.json").read_text()


def test_missing_key_fails_real_prepare_before_creating_output(prepare_tree, tmp_path):
    (tmp_path / "repo/_local/native/workbuddy/runner/upstream/.env").unlink()
    with pytest.raises(ValueError, match="Missing provider API key"):
        prepare_tree(key_env="CUSTOM_API_KEY", backend_url="https://custom.invalid/v1")
    assert not (tmp_path / "frozen").exists()


def test_real_claw_prepare_environment_only_binds_shared_module(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    module = repo / "reward_engine/adapters/provider_config.py"
    module.parent.mkdir(parents=True)
    module.write_bytes(Path(provider.__file__).read_bytes())
    (repo / "experiments").mkdir()
    source = tmp_path / "benchmark"
    task = source / "tasks/task_0"
    task.mkdir(parents=True)
    (task / "task.yaml").write_text("{}\n")
    manifest = tmp_path / "tasks.json"
    write(
        manifest,
        {
            "task_ids": [f"task_{i}" for i in range(104)],
            "groups": {"fresh_sandbox_25": ["task_0"]},
        },
    )
    binary = tmp_path / "opencode"
    binary.write_text("offline binary fixture")
    monkeypatch.setenv("OPENCODE_BINARY", str(binary))
    monkeypatch.setenv("REWARD_ENGINE_KEY_ENV", "CUSTOM_API_KEY")
    monkeypatch.setenv("CUSTOM_API_KEY", "environment-only-fixture")
    monkeypatch.setenv("CUSTOM_BASE_URL", "https://custom.invalid/v2")
    monkeypatch.setattr(campaign, "REPO", repo)
    monkeypatch.setattr(campaign, "SOURCE", source)
    monkeypatch.setattr(campaign, "MANIFEST", manifest)
    monkeypatch.setattr(campaign, "free_port", lambda: 12345)
    monkeypatch.setattr(
        campaign.subprocess,
        "check_output",
        Mock(side_effect=["offline-image\n", "/offline/docker\n"]),
    )
    config = campaign.prepare(tmp_path / "frozen", task_ids=["task_0"])
    assert config["backend_url"] == "https://custom.invalid/v2"
    assert config["backend_key_env"] == "CUSTOM_API_KEY"
    campaign.verify_freeze(config)
    assert read(tmp_path / "frozen/manifest.json")["files"][
        "code/reward_engine/adapters/provider_config.py"
    ] == file_hash(module)
    assert not Path(config["credentials_file"]).exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("channel", ["wb-task", "roles", "grade-task"])
async def test_real_gateway_actor_role_and_grader_use_selected_env(
    tmp_path, monkeypatch, channel
):
    credentials = tmp_path / "custom.env"
    credentials.write_text("CUSTOM_API_KEY=dotenv-fixture\n")
    monkeypatch.setenv("CUSTOM_API_KEY", "environment-fixture")
    config = {
        **provider.resolve_provider(
            credentials=credentials,
            key_env="CUSTOM_API_KEY",
            backend_url="http://selected.invalid/v2",
            default_credentials=credentials,
        ),
        "output": str(tmp_path),
        "auth_file": str(tmp_path / "auth"),
        "roles": {"actor": "fixture-model"},
        "model_concurrency": 1,
    }
    (tmp_path / "auth").write_text("campaign-fixture")
    calls = []

    def upstream(request):
        assert request.headers["authorization"] == "Bearer environment-fixture"
        assert str(request.url) == "http://selected.invalid/v2/chat/completions"
        calls.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": "offline"}, "finish_reason": "stop"}
                ]
            },
        )

    app = provider.application(
        config,
        client_factory=lambda **kw: httpx.AsyncClient(
            transport=httpx.MockTransport(upstream), **kw
        ),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://offline"
    ) as client:
        response = await client.post(
            f"/{channel}/v1/chat/completions",
            headers={"Authorization": "Bearer campaign-fixture"},
            json={"model": "fixture-model", "messages": [], "stream": False},
        )
    assert response.status_code == 200
    assert len(calls) == 1
    for dirname in ("transport", "official.private"):
        for path in (tmp_path / dirname).rglob("*.json"):
            assert "environment-fixture" not in path.read_text()
            assert "dotenv-fixture" not in path.read_text()


@pytest.mark.asyncio
async def test_workbuddy_campaign_uses_shared_gateway_subprocess(
    prepare_tree, monkeypatch
):
    config = prepare_tree().config
    seen = []

    async def capture(*argv, **kwargs):
        seen.append(argv)
        raise RuntimeError("offline spawn intercepted")

    monkeypatch.setattr(experiment.asyncio, "create_subprocess_exec", capture)
    with pytest.raises(RuntimeError, match="offline spawn intercepted"):
        await experiment.campaign(config)
    assert seen == [
        (
            sys.executable,
            "-m",
            "reward_engine.adapters.provider_config",
            "--config",
            str(Path(config["output"]) / "config.json"),
        )
    ]


def test_gateway_subprocess_entrypoint_accepts_environment_only(tmp_path, monkeypatch):
    import uvicorn

    monkeypatch.setenv("CUSTOM_API_KEY", "environment-fixture")
    config = {
        **provider.resolve_provider(
            key_env="CUSTOM_API_KEY",
            backend_url="http://selected.invalid/v2",
            default_credentials=tmp_path / "missing.env",
        ),
        "output": str(tmp_path),
        "roles": {"actor": "fixture-model"},
        "model_concurrency": 1,
        "gateway_port": 12345,
    }
    write(tmp_path / "config.json", config)
    monkeypatch.setattr(
        sys, "argv", ["provider_config", "--config", str(tmp_path / "config.json")]
    )
    serve = Mock()
    monkeypatch.setattr(uvicorn, "run", serve)
    provider.main()
    assert serve.call_count == 1
    assert serve.call_args.kwargs["port"] == 12345


@pytest.mark.parametrize("module", [smoke, validate_pipeline, experiment, campaign])
def test_all_cli_entrypoints_forward_provider_flags(tmp_path, monkeypatch, module):
    credentials = tmp_path / "custom.env"
    flags = [
        "--credentials",
        str(credentials),
        "--key-env",
        "CUSTOM_API_KEY",
        "--backend-url",
        "https://cli.invalid/v2",
    ]
    output = tmp_path / "run"
    seen = []

    class Captured(Exception):
        pass

    def capture(*args, **kwargs):
        seen.append((args, kwargs))
        raise Captured

    if module in (smoke, validate_pipeline):
        argv = [
            module.__name__,
            "--benchmark",
            "workbuddy",
            "--task-id",
            "task",
            "--output",
            str(output),
            *flags,
            "--workbuddy-root",
            str(tmp_path / "native"),
        ]
    else:
        argv = [module.__name__, "prepare", "--output", str(output), *flags]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(module, "prepare", capture)
    with pytest.raises(Captured):
        module.main()
    if module in (smoke, validate_pipeline):
        args = seen[0][0][0]
        assert args.credentials == credentials
        assert args.key_env == "CUSTOM_API_KEY"
        assert args.backend_url == "https://cli.invalid/v2"
        assert args.workbuddy_root == tmp_path / "native"
    else:
        assert seen[0][1]["credentials"] == credentials
        assert seen[0][1]["key_env"] == "CUSTOM_API_KEY"
        assert seen[0][1]["backend_url"] == "https://cli.invalid/v2"


@pytest.mark.parametrize("module", [smoke, validate_pipeline, experiment, campaign])
def test_execution_cannot_override_frozen_provider(tmp_path, monkeypatch, module):
    action = ["--execute"] if module in (smoke, validate_pipeline) else ["run"]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            module.__name__,
            *action,
            "--output",
            str(tmp_path / "absent"),
            "--key-env",
            "CUSTOM_API_KEY",
        ],
    )
    with pytest.raises(SystemExit) as failure:
        module.main()
    assert failure.value.code == 2
