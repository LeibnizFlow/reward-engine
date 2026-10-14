"""Frozen independent native grading, using temporary trees and offline HTTP only."""

import asyncio
import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest

from experiments import validate_pipeline as validation
from experiments.workbuddy import experiment, worker
from reward_engine.adapters import model_config as profiles
from reward_engine.adapters.workbuddy.gateway import application
from reward_engine.adapters.workbuddy.io import digest, file_hash, read, write
from tests.unit import test_validation_model_config as profile_tests

offline = profile_tests.offline
launcher = profile_tests.launcher
prepare_tree = profile_tests.prepare_tree
REPO = Path(__file__).resolve().parents[2]
PUBLIC_PROFILE = REPO / "reward_engine/configs/bailian_token_plan.json"
GRADER = {
    "model": "qwen3.8-max",
    "max_tokens": 32768,
    "request_options": {"reasoning_effort": "low"},
}
DISABLED = {"thinking": {"type": "disabled"}, "enable_thinking": False}
SOURCES = (
    "reward_engine/adapters/model_config.py",
    "reward_engine/adapters/workbuddy/gateway.py",
    "experiments/workbuddy/worker.py",
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
    return prepare_tree(
        with_launcher=True, model_config=source, role_thinking="disabled"
    )


def test_real_prepare_freezes_grader_bytes_effective_identity_and_protocol(frozen):
    config = frozen.config
    assert config["native_grader"] == GRADER
    assert config["roles"]["actor"] == "qwen3.8-flash"
    assert "native_grader" not in config["roles"]
    assert (
        frozen.output / "model_config.json"
    ).read_bytes() == PUBLIC_PROFILE.read_bytes()
    assert config["model_config"]["sha256"] == file_hash(PUBLIC_PROFILE)
    assert frozen.manifest["model_config"] == config["model_config"]
    assert config["identity"] == digest(
        {k: v for k, v in config.items() if k != "identity"}
    )
    assert frozen.manifest["config_identity"] == config["identity"]
    assert config["official_protocol"] == (
        "pinned WorkBuddy CompositeVerifier with qwen3.8-max; max_tokens=32768; "
        'request_options={"reasoning_effort":"low"}'
    )
    effective = config["model_config"]["effective"]
    assert effective["native_grader"] == GRADER
    assert effective["official_protocol"] == config["official_protocol"]
    protocol = read(frozen.output / "validation_protocol.json")["role_protocol"]
    assert protocol["native_grader"] == GRADER
    assert protocol["native_grader_output_tokens"] == 32768
    assert (
        protocol["native_grader_output_tokens_scope"]
        == "frozen gateway request max_tokens"
    )
    assert protocol["actor_output_tokens"] == 32768
    for relative in (*SOURCES, "experiments/validate_pipeline.py"):
        assert frozen.manifest["code_hashes"][relative] == file_hash(REPO / relative)
        assert (frozen.code / relative).read_bytes() == (REPO / relative).read_bytes()
    profiles.verify_profile(config)
    experiment.verify(config)


@pytest.mark.parametrize(
    "options",
    [
        {},
        DISABLED,
        {"reasoning_effort": "low"},
        {"reasoning_effort": "high"},
        {"reasoning_effort": "max"},
    ],
)
def test_grader_reuses_exact_option_allowlist(options):
    value = {**GRADER, "request_options": options}
    profile = {"schema": profiles.SCHEMA, "native_grader": value}
    assert (
        profiles.parse_profile(json.dumps(profile).encode())["native_grader"] == value
    )
    assert profiles.validate_native_grader(value) == value
    assert profiles.validate_native_grader(value)["request_options"] is not options


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        {},
        {"model": "m"},
        {**GRADER, "extra": 1},
        {**GRADER, "model": ""},
        {**GRADER, "model": True},
        {**GRADER, "model": "model with spaces"},
        *[{**GRADER, "max_tokens": v} for v in (True, 0, -1, 131073, 1.5, "32768")],
        *[
            {**GRADER, "request_options": v}
            for v in (
                None,
                [],
                {"reasoning_effort": "medium"},
                {"reasoning_effort": True},
                {"reasoning_effort": "low", **DISABLED},
                {"enable_thinking": False},
                {"model": "substitution"},
                {"max_tokens": 65536},
                {"response_format": {"type": "json_object"}},
                {"temperature": 0},
            )
        ],
    ],
)
def test_invalid_native_configuration_rejected(value):
    with pytest.raises(ValueError):
        profiles.parse_profile(
            json.dumps({"schema": profiles.SCHEMA, "native_grader": value}).encode()
        )
    with pytest.raises(ValueError):
        profiles.validate_native_grader(value)


@pytest.mark.parametrize("launcher", ["claw_eval"], indirect=True)
def test_claw_rejects_unused_grader_before_freeze(launcher):
    with pytest.raises(ValueError, match="only for WorkBuddy"):
        launcher.prepare(raw=PUBLIC_PROFILE.read_bytes())
    launcher.freeze.assert_not_called()
    assert not launcher.output.exists()


@pytest.mark.parametrize(
    "tamper",
    [
        "model",
        "max_tokens",
        "request_options",
        "removed",
        "official_protocol",
        "raw_bytes",
        "manifest",
        "binding",
        "consistent_effective_but_wrong_protocol",
    ],
)
def test_grader_tampering_rejected_before_gateway(frozen, monkeypatch, tamper):
    config = frozen.config
    if tamper in {"model", "max_tokens", "request_options"}:
        config["native_grader"][tamper] = {
            "model": "other",
            "max_tokens": 65536,
            "request_options": {},
        }[tamper]
    elif tamper == "removed":
        del config["native_grader"]
    elif tamper == "raw_bytes":
        path = frozen.output / "model_config.json"
        value = read(path)
        value["native_grader"]["max_tokens"] = 65536
        write(path, value)
    elif tamper == "manifest":
        manifest = read(frozen.output / "manifest.json")
        manifest["model_config"]["effective"]["native_grader"]["model"] = "other"
        write(frozen.output / "manifest.json", manifest)
    elif tamper == "binding":
        del config["model_config"]
    else:
        config["official_protocol"] = "wrong model description"
        if tamper == "consistent_effective_but_wrong_protocol":
            config["model_config"]["effective"]["official_protocol"] = config[
                "official_protocol"
            ]
            manifest = read(frozen.output / "manifest.json")
            manifest["model_config"] = config["model_config"]
            write(frozen.output / "manifest.json", manifest)
    gateway = Mock(side_effect=AssertionError("Gateway must not start"))
    monkeypatch.setattr(validation, "application", gateway)
    with pytest.raises(ValueError):
        asyncio.run(validation.run(config))
    gateway.assert_not_called()


@pytest.mark.parametrize("launcher", ["workbuddy"], indirect=True)
@pytest.mark.parametrize("with_profile", [False, True])
def test_missing_grader_retains_legacy_effective_binding(launcher, with_profile):
    config = launcher.prepare(profile_tests.profile_value() if with_profile else None)
    assert "native_grader" not in config
    assert "native_grader" not in profiles.effective_settings(config)
    assert "official_protocol" not in profiles.effective_settings(config)
    if with_profile:
        protocol = read(launcher.output / "validation_protocol.json")["role_protocol"]
        assert protocol["native_grader_output_tokens"] == 32768
        assert protocol["native_grader_output_tokens_scope"] == (
            "configured verifier environment only; no upstream cap asserted"
        )
    profiles.verify_profile(config)
    config["native_grader"] = deepcopy(GRADER)
    with pytest.raises(ValueError):
        profiles.verify_profile(config)


def test_applying_profile_without_grader_cannot_leave_an_inherited_route(frozen):
    config = deepcopy(frozen.config)
    profile = profiles.parse_profile(PUBLIC_PROFILE.read_bytes())
    del profile["native_grader"]
    profiles.apply_profile(config, profile)
    assert "native_grader" not in config
    assert (
        config["official_protocol"]
        == "pinned WorkBuddy CompositeVerifier with qwen3.8-flash"
    )
    # Applying a different profile in memory does not authorize changing a frozen run.
    with pytest.raises(ValueError):
        profiles.verify_profile(config)


def trial_spec(tmp_path):
    return {
        "candidate_id": "wb-fixture",
        "task_path": str(tmp_path / "task"),
        "output": str(tmp_path / "actor"),
        "actor_timeout_s": 60,
        "spec_path": str(tmp_path / "spec.json"),
        "grade": True,
    }


def gateway_config(tmp_path, grader=GRADER):
    token = tmp_path / "auth"
    token.write_text("offline-campaign")
    return {
        "output": str(tmp_path),
        "auth_file": str(token),
        "backend_url": "https://provider.invalid/v1",
        "gateway_container": "http://host.docker.internal:1234",
        "harness_mount": str(tmp_path / "harness"),
        "model_concurrency": 2,
        "per_model_concurrency": 1,
        "roles": {"actor": "qwen3.8-flash", "jury_reducer": "qwen3.8-max"},
        "role_request_options": DISABLED,
        "role_request_options_by_model": {"qwen3.8-max": {"reasoning_effort": "high"}},
        **({"native_grader": deepcopy(grader)} if grader is not None else {}),
    }


@pytest.mark.parametrize("grader", [None, GRADER])
def test_actual_worker_keeps_actor_and_routes_only_verifier(tmp_path, grader):
    config = gateway_config(tmp_path, grader)
    before = deepcopy(config)
    trial_config = worker.trial_config(trial_spec(tmp_path), config)
    trial = trial_config.model_dump(mode="json")
    assert config == before
    assert trial["agent"]["model_name"] == "qwen3.8-flash"
    assert trial["agent"]["kwargs"]["connection"]["model_route"] == "qwen3.8-flash"
    assert trial["agent"]["kwargs"]["model_params"] == {
        "max_output_tokens": 32768,
        "temperature": 0.95,
    }
    env = trial["verifier"]["env"]
    assert env["WORKBUDDY_VERIFIER_LLM_MODEL"] == (
        "qwen3.8-max" if grader else "qwen3.8-flash"
    )
    # Harbor redacts any TOKEN-named env field in serialized reports.
    assert (
        trial_config.verifier.env["WORKBUDDY_VERIFIER_LLM_MAX_OUTPUT_TOKENS"] == "32768"
    )
    assert env["WORKBUDDY_VERIFIER_LLM_BASE_URL"].endswith("/grade-wb-fixture/v1")


async def send_requests(
    config, cases, *, actual_model=None, content="unmodified fixture response"
):
    calls = []
    response_bodies = []

    def upstream(request):
        body = json.loads(request.content)
        calls.append(body)
        response = {
            "model": actual_model or body["model"],
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        }
        raw = json.dumps(response).encode()
        response_bodies.append(raw)
        return httpx.Response(200, content=raw)

    app = application(
        config,
        backend_key="offline-provider",
        client_factory=lambda **kw: httpx.AsyncClient(
            transport=httpx.MockTransport(upstream), **kw
        ),
    )
    replies = []
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://offline"
    ) as client:
        for channel, body in cases:
            replies.append(
                await client.post(
                    f"/{channel}/v1/chat/completions",
                    json=body,
                    headers={"Authorization": "Bearer offline-campaign"},
                )
            )
        health = (await client.get("/health")).json()
    return calls, replies, response_bodies, health


@pytest.mark.asyncio
@pytest.mark.parametrize("options", [{}, DISABLED, {"reasoning_effort": "low"}])
async def test_worker_to_gateway_wire_and_receipts_are_exact_and_isolated(
    tmp_path, options
):
    config = gateway_config(tmp_path, {**GRADER, "request_options": options})
    trial = worker.trial_config(trial_spec(tmp_path), config).model_dump(mode="json")
    env = trial["verifier"]["env"]
    native = {
        "model": env["WORKBUDDY_VERIFIER_LLM_MODEL"],
        "messages": [{"role": "user", "content": "private offline fixture"}],
    }
    actor = {"model": "qwen3.8-flash", "messages": [], "max_tokens": 123}
    internal = {
        "model": "qwen3.8-max",
        "messages": [],
        "reasoning_effort": "high",
        "max_tokens": 456,
    }
    cases = [("grade-wb-fixture", native), ("wb-fixture", actor), ("roles", internal)]
    before = deepcopy(cases)
    calls, replies, raw, health = await send_requests(config, cases)
    assert cases == before
    assert calls == [
        {**native, "max_tokens": 32768, **options},
        {**actor, "temperature": 0.95},
        internal,
    ]
    assert [r.status_code for r in replies] == [200, 200, 200]
    assert [r.content for r in replies] == raw  # No host-authored output/schema repair.
    assert health["inflight"] == 0 and all(
        v == 0 for v in health["model_inflight"].values()
    )
    path = next(tmp_path.glob("official.private/grade-wb-fixture/*"))
    assert read(path / "received_request.json") == native
    assert read(path / "request.json") == calls[0]
    assert (path / "response.json").read_bytes() == raw[0]
    receipt = read(path / "receipt.json")
    assert receipt["model"] == receipt["actual_model"] == GRADER["model"]
    assert receipt["status"] == "completed" and receipt["attempts"] == 1


@pytest.mark.asyncio
async def test_legacy_grade_wire_remains_unchanged(tmp_path):
    config = gateway_config(tmp_path, None)
    body = {"model": "qwen3.8-flash", "messages": [], "stream": False}
    calls, replies, raw, _ = await send_requests(config, [("grade-wb-fixture", body)])
    assert calls == [body] and replies[0].content == raw[0]
    path = next(tmp_path.glob("official.private/grade-wb-fixture/*"))
    assert read(path / "request.json") == body
    assert not (path / "received_request.json").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "channel,extra",
    [
        ("grade-wb-fixture", {"model": "qwen3.8-flash"}),
        ("grade-wb-fixture", {"max_tokens": 65536}),
        ("grade-wb-fixture", {"max_tokens": True}),
        ("grade-wb-fixture", {"reasoning_effort": "high"}),
        ("grade-wb-fixture", {"thinking": {"type": "disabled"}}),
        ("grade-wb-fixture", {"temperature": 0}),
        ("grade-wb-fixture", {"response_format": {"type": "json_object"}}),
        ("grade-wb-fixture", {"stream": True}),
        ("grade-wb-", {}),
        ("grade-other", {}),
    ],
)
async def test_grader_route_or_parameter_drift_never_reaches_wire(
    tmp_path, channel, extra
):
    config = gateway_config(tmp_path)
    calls, replies, _, _ = await send_requests(
        config, [(channel, {"model": GRADER["model"], "messages": [], **extra})]
    )
    assert calls == [] and replies[0].status_code == 400
    assert not (tmp_path / "official.private").exists()


@pytest.mark.asyncio
async def test_grader_only_model_is_not_admitted_on_actor_or_role_channels(tmp_path):
    config = gateway_config(tmp_path, {**GRADER, "model": "grader-only"})
    body = {"model": "grader-only", "messages": []}
    calls, replies, _, _ = await send_requests(
        config, [("wb-fixture", body), ("roles", body), ("grade-wb-fixture", body)]
    )
    assert [r.status_code for r in replies] == [400, 400, 200]
    assert calls == [{**body, "max_tokens": 32768, "reasoning_effort": "low"}]


@pytest.mark.asyncio
async def test_native_provider_model_mismatch_still_rejected(tmp_path):
    config = gateway_config(tmp_path)
    body = {"model": GRADER["model"], "messages": []}
    calls, replies, _, _ = await send_requests(
        config, [("grade-wb-fixture", body)], actual_model="silent-substitution"
    )
    assert len(calls) == 1 and replies[0].status_code == 502
    receipt = read(
        next(tmp_path.glob("official.private/grade-wb-fixture/*/receipt.json"))
    )
    assert receipt["status"] == "failed" and receipt["error_type"] == "ValueError"
