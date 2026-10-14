"""Offline generation-shape controls through the real Role and bounded ask loop."""

import asyncio
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from jsonschema import Draft202012Validator

from experiments import validate_pipeline as validation
from experiments.workbuddy import experiment
from reward_engine.adapters import delivery_review_schema as schema
from reward_engine.adapters import model_config as profiles
from reward_engine.adapters.workbuddy import models
from reward_engine.adapters.workbuddy.io import digest, file_hash, read, write
from reward_engine.core.outcome.call_budget import ACTIVE
from reward_engine.core.outcome.context import ResponseContractError, ask, build_context
from reward_engine.core.outcome.delivery_triage import validate_review
from reward_engine.core.outcome.harness import defaults
from reward_engine.core.outcome.improvement import validate as validate_improvements
from reward_engine.core.outcome.runner import DeliveryReview
from tests.support.claw_outcome_memory_runner import PACK, TASK
from tests.unit import test_validation_model_config as profile_tests

offline = profile_tests.offline
launcher = profile_tests.launcher
prepare_tree = profile_tests.prepare_tree
REPO = Path(__file__).resolve().parents[2]
PUBLIC_PROFILE = REPO / "reward_engine/configs/bailian_token_plan.json"
MODEL = "qwen3.8-max"
SHAPE = {field: "existing core field description" for field in schema.FIELDS}


def assessment():
    return {
        "outcome_goal": "Deliver a complete JSON answer for this fixture task",
        "status": "target_satisfied",
        "reason": "The fixture delivery meets the supplied goal",
        "full_score_gaps": [],
        "improvements": [],
        "claim_checks": [],
        "gaps": [],
        "verify": [],
    }


def validator(closed=False):
    value = schema.response_format(evidence_closed=closed)["json_schema"]["schema"]
    Draft202012Validator.check_schema(value)
    return Draft202012Validator(value)


@pytest.mark.parametrize(
    "wrap", ["bare", "response", "typed", "outer_read", "typed_outer_read"]
)
def test_complete_assessment_and_existing_action_envelopes_remain_legal(wrap):
    row = assessment()
    row["improvements"] = [
        {
            "id": "action-1",
            "gap_ids": ["gap-1"],
            "action": "Model-authored action",
            "kind": "outcome_gain",
            "task_quote": "fixture task",
            "factual_premises": [{"claim": "Model-authored premise", "conditions": []}],
            "future_admission_extension": {"nested": [1, {"value": True}]},
        }
    ]
    if wrap != "bare":
        row = {"response": row}
        if wrap in {"typed", "typed_outer_read"}:
            row["type"] = "json_object"
        if wrap in {"outer_read", "typed_outer_read"}:
            row["need_evidence"] = []
    validator().validate(row)
    validator(True).validate(row)


@pytest.mark.parametrize(
    "row",
    [
        {"need_evidence": ["source:1"]},
        {"response": {"need_evidence": ["source:1"]}},
        {"type": "json_object", "response": {"need_evidence": ["source:1"]}},
        {**assessment(), "need_evidence": ["source:1"]},
        {"response": assessment(), "need_evidence": ["source:1"]},
    ],
)
def test_evidence_branch_preserved_until_read_access_closes(row):
    assert validator().is_valid(row)
    assert not validator(True).is_valid(row)


@pytest.mark.parametrize(
    "row",
    [
        {k: v for k, v in assessment().items() if k != "outcome_goal"},
        {**assessment(), "outcome_goal": ""},
        {**assessment(), "outcome_goal": None},
        {**assessment(), "outcome_goal": {}},
        {**assessment(), "gaps": {}},
        {**assessment(), "improvements": ["not an action object"]},
        {"need_evidence": []},
        {"need_evidence": {"ref": "source:1"}},
        {"need_evidence": [{"ref": "source:1"}]},
        {"need_tools": [{"tool": "workspace.read"}]},
        {"response": {"response": assessment()}},
        {"response": assessment(), "type": "invented"},
    ],
)
def test_shape_rejects_missing_goal_and_illegal_action_branches(row):
    assert not validator().is_valid(row)


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    config = {
        "output": str(tmp_path),
        "auth_file": str(tmp_path / "auth"),
        "gateway_local": "http://offline.invalid",
        "validation_benchmark": "workbuddy",
        "roles": {"review": MODEL, "other": "other-model"},
        "role_bounded_stage": True,
        "role_stage_timeout_s": 60,
        "role_response_format": "json_object",
        "role_request_options_by_model": {MODEL: {"reasoning_effort": "low"}},
        schema.CONFIG_KEY: [MODEL],
    }
    (tmp_path / "auth").write_text("offline-test-token")
    write(
        tmp_path / "role_limits.json",
        {"role_stream_timeout_s": 60, "role_max_tokens": {"review": 65536}},
    )
    fixture = SimpleNamespace(
        config=config, calls=[], requests=[], budgets=[], answers=[]
    )

    def respond(request):
        body = json.loads(request.content)
        index = len(fixture.calls)
        assert index < len(fixture.answers), "Unexpected extra call or fallback"
        fixture.calls.append(body)
        fixture.requests.append(json.loads(body["messages"][1]["content"]))
        fixture.budgets.append(deepcopy(ACTIVE.get()))
        answer = fixture.answers[index]
        if isinstance(answer, httpx.Response):
            return answer
        content = answer if isinstance(answer, str) else json.dumps(answer)
        event = {
            "model": body["model"],
            "choices": [{"delta": {"content": content}, "finish_reason": "stop"}],
        }
        return httpx.Response(
            200, text="data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n"
        )

    client = httpx.AsyncClient
    monkeypatch.setattr(
        models.httpx,
        "AsyncClient",
        lambda **kw: client(transport=httpx.MockTransport(respond), **kw),
    )
    fixture.role = models.Role(config, "review", tmp_path / "role")
    return fixture


def review_context():
    context, evidence = build_context(TASK, [{"artifact": {"final_response": "{}"}}])
    for i in range(1, 4):
        evidence[f"source:{i}"] = {"content": f"Full offline source {i}"}
    return context, evidence


async def run_review(runtime, *, preview=False):
    context, evidence = review_context()
    if preview:
        context["products"][0]["sources"].append(
            {"ref": "source:1", "preview_only": True}
        )
    return await ask(
        runtime.role,
        schema.MODE,
        context,
        evidence,
        "Use the supplied review contract",
        SHAPE,
        verify_used_previews=preview,
        validate=lambda row: validate_improvements(
            validate_review(row, evidence, set()), evidence
        ),
    )


def assert_wire(runtime):
    for body, request in zip(runtime.calls, runtime.requests, strict=True):
        assert body["response_format"] == schema.for_request(request)
        assert body["max_tokens"] == 65536 and body["reasoning_effort"] == "low"
        assert body["model"] == MODEL and body["stream"] is True
    for path in runtime.role.output.glob("*/request_body.json"):
        assert read(path) in runtime.calls


@pytest.mark.asyncio
async def test_actual_delivery_controller_uses_fixed_schema(runtime):
    runtime.answers = [assessment()]
    controller = DeliveryReview(TASK, deepcopy(PACK), defaults(runtime.role))
    assert await controller({"artifact": {"final_response": "{}"}}) is None
    assert controller.status == "full_score_target_satisfied"
    assert set(runtime.requests[0]["response"]) == set(schema.FIELDS)
    assert_wire(runtime)


@pytest.mark.asyncio
async def test_schema_on_six_calls_keeps_shared_repairs_reads_and_closure(runtime):
    bad = assessment()
    del bad["outcome_goal"]
    runtime.answers = [
        '{"status":',
        bad,
        {"need_evidence": ["source:1"]},
        {"response": {"need_evidence": ["source:2"]}},
        {"need_evidence": ["source:3"]},
        assessment(),
    ]
    assert (await run_review(runtime))["outcome_goal"] == assessment()["outcome_goal"]
    assert len(runtime.calls) == 6
    assert [v["requests"] for v in runtime.budgets] == list(range(1, 7))
    assert runtime.budgets[-1] == {
        "requests": 6,
        "syntax_repairs": 1,
        "contract_repairs": 1,
    }
    assert set(runtime.requests[4]["requested_evidence"]) == {"source:1", "source:2"}
    assert runtime.requests[5]["evidence_access_closed"] is True
    assert "source:3" not in runtime.requests[5]["requested_evidence"]
    assert_wire(runtime)


@pytest.mark.asyncio
@pytest.mark.parametrize("followup_missing_goal", [False, True])
async def test_mandatory_preview_followup_does_not_reset_repair_budget(
    runtime, followup_missing_goal
):
    row = assessment()
    row["claim_checks"] = [
        {
            "claim": "Fixture claim",
            "claim_ref": "product_1:artifact:final_response",
            "support_refs": ["source:1"],
            "assessment": "supported",
            "reason": "Fixture support",
            "test_id": None,
        }
    ]
    bad = deepcopy(row)
    del bad["outcome_goal"]
    runtime.answers = [bad, row, bad if followup_missing_goal else row]
    if followup_missing_goal:
        with pytest.raises(ResponseContractError) as failure:
            await run_review(runtime, preview=True)
        assert failure.value.details["field"] == "outcome_goal"
    else:
        assert (await run_review(runtime, preview=True))["outcome_goal"] == row[
            "outcome_goal"
        ]
    assert len(runtime.calls) == 3
    assert (
        runtime.requests[1]["response_recovery"]["details"]["field"] == "outcome_goal"
    )
    assert "evidence_followup" in runtime.requests[2]
    assert "source:1" in runtime.requests[2]["requested_evidence"]
    assert runtime.budgets[-1]["contract_repairs"] == 1
    assert_wire(runtime)


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_old_malformed_review_still_fails_without_host_goal_default(
    runtime, enabled
):
    if not enabled:
        runtime.config.pop(schema.CONFIG_KEY)
    bad = assessment()
    del bad["outcome_goal"]
    runtime.answers = [bad, bad]
    with pytest.raises(ResponseContractError) as failure:
        await run_review(runtime)
    assert failure.value.details["field"] == "outcome_goal"
    assert len(runtime.calls) == 2
    assert all(
        "outcome_goal" not in read(p)["answer"]
        for p in runtime.role.output.glob("*/response.json")
    )


@pytest.mark.asyncio
async def test_provider_rejects_schema_visibly_without_fallback(runtime):
    runtime.answers = [
        httpx.Response(400, json={"error": {"message": "unsupported schema"}})
    ]
    with pytest.raises(httpx.HTTPStatusError):
        await run_review(runtime)
    assert len(runtime.calls) == 1
    assert not list((runtime.role.output / "completed").glob("*.json"))
    assert (
        read(next(runtime.role.output.glob("*/failure.json")))["error_type"]
        == "HTTPStatusError"
    )
    assert_wire(runtime)


@pytest.mark.asyncio
async def test_cache_binds_actual_schema_not_old_json_object_marker(
    runtime, monkeypatch
):
    request = {"mode": schema.MODE, "response": SHAPE}
    runtime.answers = [assessment(), assessment(), assessment()]
    runtime.config.pop(schema.CONFIG_KEY)
    await runtime.role(request)
    await runtime.role(request)
    assert len(runtime.calls) == 1
    runtime.config[schema.CONFIG_KEY] = [MODEL]
    await runtime.role(request)
    await runtime.role(request)
    assert len(runtime.calls) == 2
    original = schema.for_request

    def changed(value):
        result = original(value)
        result["json_schema"]["schema"]["title"] = "different exact frozen schema"
        return result

    monkeypatch.setattr(schema, "for_request", changed)
    await runtime.role(request)
    assert len(runtime.calls) == 3
    assert len(list((runtime.role.output / "completed").glob("*.json"))) == 3


@pytest.mark.asyncio
async def test_other_modes_and_unlisted_models_keep_same_wire_and_cache(runtime):
    runtime.answers = [assessment(), assessment()]
    request = {"mode": "outcome_measure", "response": SHAPE}
    await runtime.role(request)
    runtime.config.pop(schema.CONFIG_KEY)
    await runtime.role(request)
    assert len(runtime.calls) == 1
    runtime.config[schema.CONFIG_KEY] = [MODEL]
    other = models.Role(runtime.config, "other", runtime.role.output / "other")
    await other({"mode": schema.MODE, "response": SHAPE})
    assert len(runtime.calls) == 2
    assert all(
        body["response_format"] == {"type": "json_object"} for body in runtime.calls
    )
    assert runtime.calls[1]["model"] == "other-model"


@pytest.mark.parametrize(
    "value", [None, {}, "qwen3.8-max", [True], [""], ["model name"], [MODEL, MODEL]]
)
def test_profile_rejects_malformed_model_allowlist(value):
    with pytest.raises(ValueError):
        profiles.parse_profile(
            json.dumps({"schema": profiles.SCHEMA, schema.CONFIG_KEY: value}).encode()
        )


@pytest.mark.parametrize(
    "models_list,benchmark",
    [
        (["unknown"], "workbuddy"),
        (["alias"], "workbuddy"),
        (["grader-only"], "workbuddy"),
        ([MODEL], "claw_eval"),
    ],
)
def test_runtime_scope_is_explicit_roles_only_without_alias_substitution(
    models_list, benchmark
):
    config = {
        "validation_benchmark": benchmark,
        "roles": {"review": MODEL},
        "model_aliases": {MODEL: ["alias"]},
        "native_grader": {"model": "grader-only"},
        schema.CONFIG_KEY: models_list,
    }
    with pytest.raises(ValueError):
        profiles.delivery_review_schema_models(config)


@pytest.fixture
def frozen(prepare_tree, tmp_path):
    for relative in (
        "reward_engine/adapters/model_config.py",
        "reward_engine/adapters/delivery_review_schema.py",
        "reward_engine/adapters/workbuddy/models.py",
    ):
        target = tmp_path / "repo" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((REPO / relative).read_bytes())
    source = tmp_path / "profile.json"
    source.write_bytes(PUBLIC_PROFILE.read_bytes())
    return prepare_tree(with_launcher=True, model_config=source)


def test_real_freeze_binds_raw_effective_schema_and_helper_source(frozen):
    config = frozen.config
    assert config[schema.CONFIG_KEY] == [MODEL]
    assert (
        frozen.output / "model_config.json"
    ).read_bytes() == PUBLIC_PROFILE.read_bytes()
    assert config["model_config"]["sha256"] == file_hash(PUBLIC_PROFILE)
    effective = config["model_config"]["effective"]
    assert effective[schema.CONFIG_KEY] == [MODEL]
    assert effective["delivery_review_json_schema"] == schema.binding()
    assert frozen.manifest["model_config"] == config["model_config"]
    assert config["identity"] == digest(
        {k: v for k, v in config.items() if k != "identity"}
    )
    name = "reward_engine/adapters/delivery_review_schema.py"
    assert frozen.manifest["code_hashes"][name] == file_hash(REPO / name)
    assert (frozen.code / name).read_bytes() == (REPO / name).read_bytes()
    profiles.verify_profile(config)
    experiment.verify(config)


@pytest.mark.parametrize(
    "change", ["list", "remove", "raw", "schema_hash", "helper_version"]
)
def test_frozen_schema_or_model_drift_rejected_before_gateway(
    frozen, monkeypatch, change
):
    config = frozen.config
    if change == "list":
        config[schema.CONFIG_KEY] = []
    elif change == "remove":
        del config[schema.CONFIG_KEY]
    elif change == "raw":
        value = read(frozen.output / "model_config.json")
        value[schema.CONFIG_KEY] = []
        write(frozen.output / "model_config.json", value)
    elif change == "schema_hash":
        manifest = read(frozen.output / "manifest.json")
        manifest["model_config"]["effective"]["delivery_review_json_schema"][
            "response_format_sha256"
        ]["evidence_open"] = "0" * 64
        write(frozen.output / "manifest.json", manifest)
    else:
        monkeypatch.setattr(schema, "VERSION", "changed-version")
    gateway = Mock(side_effect=AssertionError("Must not reach a gateway"))
    monkeypatch.setattr(validation, "application", gateway)
    with pytest.raises(ValueError):
        asyncio.run(validation.run(config))
    gateway.assert_not_called()


@pytest.mark.parametrize("launcher", ["workbuddy", "claw_eval"], indirect=True)
def test_legacy_profile_effective_map_unchanged(launcher):
    config = launcher.prepare(profile_tests.profile_value())
    assert schema.CONFIG_KEY not in config
    assert "delivery_review_json_schema" not in profiles.effective_settings(config)
    profiles.verify_profile(config)
