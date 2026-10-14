"""Prospective GLM options and strict terminal handling, with offline fixtures only."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import httpx
import pytest

from reward_engine.adapters import model_config as profiles
from reward_engine.adapters.workbuddy import models
from reward_engine.adapters.workbuddy.io import read, write
from tests.unit import test_validation_model_config as profile_tests
from tests.unit import test_workbuddy_role_models as role_tests

launcher = profile_tests.launcher
offline = profile_tests.offline
thinking_role = role_tests.thinking_role
PUBLIC_PROFILE = Path(profiles.__file__).parents[1] / "configs/bailian_token_plan.json"
DISABLED = {"thinking": {"type": "disabled"}, "enable_thinking": False}


def role_profile_bytes(benchmark):
    raw = PUBLIC_PROFILE.read_bytes()
    if benchmark == "claw_eval":
        # The public profile also configures WorkBuddy's independent native grader;
        # these Claw controls exercise only the shared internal-role settings.
        profile = json.loads(raw)
        profile.pop("native_grader", None)
        profile.pop("delivery_review_json_schema_models", None)
        return json.dumps(profile).encode()
    return raw


@pytest.mark.parametrize(
    "options",
    [{}, DISABLED, *[{"reasoning_effort": value} for value in ("low", "high", "max")]],
)
def test_profile_and_runtime_share_the_exact_option_allowlist(options):
    profile = {
        "schema": profiles.SCHEMA,
        "role_request_options_by_model": {"glm-5.3": options},
    }
    before = deepcopy(profile)
    parsed = profiles.parse_profile(json.dumps(profile).encode())
    config = {
        "roles": {"jury_member_3": "glm-5.3"},
        "role_request_options": DISABLED,
        "role_request_options_by_model": parsed["role_request_options_by_model"],
    }
    result = models._role_request_options(config, "glm-5.3")
    assert result == options
    assert result is not options
    assert profile == before
    assert models._validate_request_options is profiles._validate_request_options


@pytest.mark.parametrize(
    "options",
    [
        {"reasoning_effort": value}
        for value in (None, True, 0, [], {}, "", "none", "medium", "LOW", "low ")
    ]
    + [
        {"reasoning_effort": "low", "enable_thinking": True},
        {"reasoning_effort": "low", **DISABLED},
        {"reasoning_effort": "low", "max_tokens": 65536},
        {"reasoning_effort": "low", "model": "other"},
    ],
)
@pytest.mark.asyncio
async def test_invalid_effort_or_mixed_options_fail_before_http(thinking_role, options):
    role, calls, _ = thinking_role
    role.config["role_request_options_by_model"] = {"gemini-3.8-flash": options}
    with pytest.raises(ValueError, match="role_request_options"):
        await role({"mode": "offline-fixture"})
    assert calls == []
    assert not list(role.output.glob("*/request_body.json"))
    profile = {
        "schema": profiles.SCHEMA,
        "role_request_options_by_model": {"gemini-3.8-flash": options},
    }
    with pytest.raises(ValueError, match="role_request_options"):
        profiles.parse_profile(json.dumps(profile).encode())


def test_reasoning_effort_is_opt_in_per_model_only():
    with pytest.raises(ValueError, match="role_request_options"):
        models._role_request_options(
            {
                "roles": {"jury_member_3": "glm-5.3"},
                "role_request_options": {"reasoning_effort": "low"},
            },
            "glm-5.3",
        )
    assert (
        models._role_request_options({"roles": {"jury_member_3": "glm-5.3"}}, "glm-5.3")
        == {}
    )


def test_public_profile_budgets_and_options_are_frozen_exactly(launcher):
    raw = role_profile_bytes(launcher.benchmark)
    profile = profiles.parse_profile(raw)
    glm_roles = {role for role, model in profile["roles"].items() if model == "glm-5.3"}
    assert glm_roles == {"jury_member_3", "hypothesis_member_3"}
    for role, tokens in profile["role_max_tokens"].items():
        assert tokens == (
            65536 if profile["roles"][role] in {"glm-5.3", "qwen3.8-max"} else 32768
        )
    assert profile["role_request_options_by_model"] == {
        "glm-5.3": {"reasoning_effort": "low"},
        "qwen3.8-max": {"reasoning_effort": "low"},
    }
    config = launcher.prepare(raw=raw, role_thinking="disabled")
    assert (launcher.output / "model_config.json").read_bytes() == raw
    assert config["model_config"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert (
        read(launcher.output / "manifest.json")["model_config"]
        == config["model_config"]
    )
    effective = config["model_config"]["effective"]
    assert (
        effective["role_request_options_by_model"]
        == profile["role_request_options_by_model"]
    )
    assert effective["role_max_tokens"] == profile["role_max_tokens"]
    protocol = read(launcher.output / "validation_protocol.json")["role_protocol"]
    assert (
        protocol["role_request_options_by_model"]
        == effective["role_request_options_by_model"]
    )
    assert protocol["role_max_tokens"] == effective["role_max_tokens"]
    assert protocol["role_request_options"] == DISABLED
    profiles.verify_profile(config)


@pytest.mark.parametrize(
    "tamper", ["frozen_bytes", "effective_options", "effective_budget"]
)
def test_frozen_effort_or_budget_drift_is_rejected(launcher, tamper):
    config = launcher.prepare(
        raw=role_profile_bytes(launcher.benchmark), role_thinking="disabled"
    )
    if tamper == "frozen_bytes":
        path = launcher.output / "model_config.json"
        value = read(path)
        value["role_request_options_by_model"]["glm-5.3"] = {"reasoning_effort": "high"}
        write(path, value)
    elif tamper == "effective_options":
        config["role_request_options_by_model"]["glm-5.3"] = {
            "reasoning_effort": "high"
        }
    else:
        config["validation_role_limits"]["role_max_tokens"]["jury_member_3"] = 32768
    with pytest.raises(ValueError, match="changed"):
        profiles.verify_profile(config)


def _response(
    model, *, finish="stop", content='{"fixture":true}', completion_tokens=17
):
    event = {
        "model": model,
        "choices": [
            {
                "delta": {"reasoning_content": "offline fixture", "content": content},
                "finish_reason": finish,
            }
        ],
        "usage": {
            "completion_tokens": completion_tokens,
            "completion_tokens_details": {"reasoning_tokens": completion_tokens - 2},
        },
    }
    return httpx.Response(
        200, text="data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n"
    )


@pytest.mark.asyncio
async def test_actual_role_bodies_keep_model_overrides_separate_from_global_disabled(
    launcher, monkeypatch
):
    config = launcher.prepare(
        raw=role_profile_bytes(launcher.benchmark), role_thinking="disabled"
    )
    calls = []

    def reply(request):
        body = json.loads(request.content)
        calls.append(body)
        if body["model"] in {"glm-5.3", "qwen3.8-max"}:
            assert body["reasoning_effort"] == "low" and body["max_tokens"] == 65536
            assert "thinking" not in body and "enable_thinking" not in body
        else:
            assert body["max_tokens"] == 32768 and "reasoning_effort" not in body
            assert {key: body[key] for key in DISABLED} == DISABLED
        assert body["response_format"] == {"type": "json_object"}
        assert body["stream"] is True
        return _response(body["model"])

    client = httpx.AsyncClient
    monkeypatch.setattr(
        models.httpx,
        "AsyncClient",
        lambda **kw: client(transport=httpx.MockTransport(reply), **kw),
    )
    for name in (
        "jury_member_3",
        "hypothesis_member_3",
        "jury_reducer",
        "jury_member_2",
    ):
        role = models.Role(config, name, launcher.output / name)
        assert await role({"mode": "offline-fixture"}) == {"fixture": True}
        receipt = read(next(role.output.glob("*/thinking_parameters.json")))
        assert receipt["options"] == (
            {"reasoning_effort": "low"}
            if config["roles"][name] in {"glm-5.3", "qwen3.8-max"}
            else DISABLED
        )
    assert len(calls) == 4
    profiles.verify_profile(config)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content,completion_tokens", [("", 32768), ('{"fixture":true}', 65536)]
)
async def test_length_is_still_rejected_with_low_and_larger_budget(
    launcher, monkeypatch, content, completion_tokens
):
    config = launcher.prepare(
        raw=role_profile_bytes(launcher.benchmark), role_thinking="disabled"
    )
    calls = []

    def reply(request):
        body = json.loads(request.content)
        calls.append(body)
        assert body["reasoning_effort"] == "low" and body["max_tokens"] == 65536
        return _response(
            body["model"],
            finish="length",
            content=content,
            completion_tokens=completion_tokens,
        )

    client = httpx.AsyncClient
    monkeypatch.setattr(
        models.httpx,
        "AsyncClient",
        lambda **kw: client(transport=httpx.MockTransport(reply), **kw),
    )
    role = models.Role(config, "jury_member_3", launcher.output / "role")
    with pytest.raises(ValueError, match="finish=length"):
        await role({"mode": "offline-fixture"})
    assert len(calls) == 1
    assert not list(role.output.glob("*/response.json"))
    assert not list((role.output / "completed").glob("*.json"))
    failure = read(next(role.output.glob("*/failure.json")))
    assert failure["finish_reason"] == "length"
    assert failure["usage"]["completion_tokens"] == completion_tokens
    assert (
        failure["usage"]["completion_tokens_details"]["reasoning_tokens"]
        == completion_tokens - 2
    )


@pytest.mark.asyncio
async def test_effort_changes_cache_identity_and_preserves_prior_cache(thinking_role):
    role, calls, _ = thinking_role
    request = {"mode": "offline-fixture"}
    for effort in ("low", "high", "max"):
        role.config["role_request_options_by_model"] = {
            "gemini-3.8-flash": {"reasoning_effort": effort}
        }
        assert await role(request) == {"status": "pass"}
        assert await role(request) == {"status": "pass"}
    assert [body["reasoning_effort"] for body in calls] == ["low", "high", "max"]
    assert len(list((role.output / "completed").glob("*.json"))) == 3


@pytest.mark.asyncio
async def test_existing_opt_in_transport_recovery_retains_effort_receipt(thinking_role):
    role, calls, responses = thinking_role
    role.config["role_request_options_by_model"] = {
        "gemini-3.8-flash": {"reasoning_effort": "low"}
    }
    role.nonstream_recovery_models = {"gemini-3.8-flash"}
    responses.extend(
        [
            httpx.Response(
                200, text='data: {"error":{"message":"offline fixture"}}\n\n'
            ),
            httpx.Response(
                200,
                json={
                    "model": "gemini-3.8-flash",
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": '{"status":"pass"}'},
                        }
                    ],
                },
            ),
        ]
    )
    assert await role({"mode": "offline-fixture"}) == {"status": "pass"}
    assert len(calls) == 2
    assert all(body["reasoning_effort"] == "low" for body in calls)
    receipts = list(role.output.glob("*/thinking_parameters.json"))
    assert len(receipts) == 2 and read(receipts[0]) == read(receipts[1])
