"""Mock HTTP only: no inference about real-model reliability or native success."""

import asyncio
import json

import httpx
import pytest

from reward_engine.examples.live import Live, LiveConfig
from reward_engine.examples.minimal import run
from reward_engine.examples.offline import Offline


@pytest.fixture
def clean_config_environment(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for name in ("OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_BASE_URL", "YIBU_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def completion(answer, *, finish="stop"):
    return {
        "choices": [
            {
                "index": 0,
                "finish_reason": finish,
                "message": {"role": "assistant", "content": json.dumps(answer)},
            }
        ]
    }


def test_credentials_literal_interpolation_environment_precedence_and_redaction(
    clean_config_environment, tmp_path
):
    patch = clean_config_environment
    file = tmp_path / "credentials.env"
    file.write_text(
        'YIBU_API_KEY="literal-${DO_NOT_EXPAND}"\nOPENAI_MODEL=file-model\nOPENAI_BASE_URL=https://example.invalid/v1\n'
    )
    patch.setenv("DO_NOT_EXPAND", "would-corrupt-key")
    config = LiveConfig.load(key_env="YIBU_API_KEY", credentials=file)
    assert config.api_key == "literal-${DO_NOT_EXPAND}"
    assert config.model == "file-model"
    assert config.api_key not in repr(config)
    patch.setenv("YIBU_API_KEY", "environment-value")
    patch.setenv("OPENAI_MODEL", "environment-model")
    config = LiveConfig.load(
        key_env="YIBU_API_KEY",
        credentials=file,
        model="explicit",
        base_url="https://other.invalid/api",
    )
    assert config.api_key == "environment-value"
    assert config.model == "explicit" and config.base_url == "https://other.invalid/api"


def test_custom_key_requires_explicit_endpoint(clean_config_environment):
    patch = clean_config_environment
    patch.setenv("YIBU_API_KEY", "third-party-secret")
    with pytest.raises(ValueError, match="Custom --key-env requires"):
        LiveConfig.load(model="single-model", key_env="YIBU_API_KEY")
    config = LiveConfig.load(
        model="single-model",
        key_env="YIBU_API_KEY",
        base_url="https://provider.invalid/v1",
    )
    assert config.base_url == "https://provider.invalid/v1"
    patch.setenv("OPENAI_BASE_URL", "https://environment.invalid/v1")
    assert (
        LiveConfig.load(model="single-model", key_env="YIBU_API_KEY").base_url
        == "https://environment.invalid/v1"
    )


def test_standard_key_default_and_missing_credentials_fail(
    clean_config_environment, tmp_path
):
    patch = clean_config_environment
    with pytest.raises(ValueError, match="model and a nonempty key"):
        LiveConfig.load(model="single-model")
    patch.setenv("OPENAI_API_KEY", "test-key")
    assert LiveConfig.load(model="single-model").base_url == "https://api.openai.com/v1"
    with pytest.raises(ValueError, match="Credentials file"):
        LiveConfig.load(model="single-model", credentials=tmp_path / "missing")


@pytest.mark.parametrize(
    "url",
    [
        "https://@example.invalid/v1",
        "https://:secret@example.invalid/v1",
        "https:///v1",
        "https://:443/v1",
        "https://example.invalid:wrong/v1",
        "https://example.invalid:65536/v1",
        "https://example.invalid:0/v1",
        "file:///tmp/endpoint",
        "https://example.invalid/v1?key=secret",
        "https://example.invalid/v1#fragment",
        "https://exam ple.invalid/v1",
        "https://example.invalid\\other/v1",
        "https://[invalid/v1",
    ],
)
def test_invalid_urls_fail_before_any_http(clean_config_environment, url):
    clean_config_environment.setenv("OPENAI_API_KEY", "secret")
    with pytest.raises(ValueError, match="Base URL"):
        LiveConfig.load(model="single-model", base_url=url)


def test_live_transport_controller_end_to_end_is_explicit_simulation(tmp_path):
    requests = []
    scripted = Offline()

    async def handler(request):
        assert str(request.url) == "https://mock.invalid/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer test-only-key"
        payload = json.loads(request.content)
        requests.append(payload)
        assert payload["model"] == "one-model" and payload["stream"] is False
        messages = payload["messages"]
        if messages[0]["content"].startswith("You are an internal"):
            answer = await scripted.role(json.loads(messages[-1]["content"]))
        else:
            # Model-authored simulation: continuation corrects the prior total.
            revised = any(m["role"] == "assistant" for m in messages)
            answer = {"final_response": json.dumps({"total": 6 if revised else 5})}
        return httpx.Response(200, json=completion(answer))

    backend = Live(
        LiveConfig("one-model", "https://mock.invalid/v1", "test-only-key"),
        transport=httpx.MockTransport(handler),
    )
    root = tmp_path / "mock-http-only"
    summary = asyncio.run(run(root, backend=backend))
    assert summary["mode"] == "live_openai_compatible"
    assert summary["completed_rounds"] == 3 and summary["actor_count"] == 9
    assert all(r["revision_count"] == 1 for r in summary["rounds"])
    assert len(requests) == summary["role_and_actor_calls"]
    assert summary["native_evaluation"] == "not_run"
    for path in root.rglob("*.json"):
        assert "test-only-key" not in path.read_text()


@pytest.mark.parametrize(
    "kind",
    ["http", "timeout", "truncated", "two_choices", "not_json", "wrong_envelope"],
)
def test_live_failure_never_falls_back_or_fills_required_fields(tmp_path, kind):
    calls = []

    def handler(request):
        calls.append(request)
        if kind == "http":
            return httpx.Response(401, text="private-provider-body")
        if kind == "timeout":
            raise httpx.ReadTimeout("private-provider-body")
        response = completion({"final_response": '{"total":6}'})
        if kind == "truncated":
            response["choices"][0]["finish_reason"] = "length"
        if kind == "two_choices":
            response["choices"] *= 2
        if kind == "not_json":
            response["choices"][0]["message"]["content"] = "not JSON"
        if kind == "wrong_envelope":
            response = completion({"text": "host must not rename this"})
        return httpx.Response(200, json=response)

    root = tmp_path / kind
    backend = Live(
        LiveConfig("one-model", "https://mock.invalid/v1", "test-only-key"),
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises((ValueError, RuntimeError)) as error:
        asyncio.run(run(root, backend=backend))
    assert "private-provider-body" not in str(error.value)
    assert len(calls) == 1  # first seed fails; no transport retries/fallback actors
    assert not (root / "summary.json").exists()
    failure = json.loads((root / "failure.json").read_text())
    assert failure["status"] == "failed"
    if kind != "wrong_envelope":
        assert list((root / "calls").glob("*/failure.json"))
    else:
        assert list(
            (root / "calls").glob("*/response.json")
        )  # malformed answer retained
