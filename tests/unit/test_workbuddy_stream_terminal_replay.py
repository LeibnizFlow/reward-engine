"""Replay the retained Qwen Claw R3 stream with offline HTTP, plus corrupt controls."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import httpx
import pytest

from reward_engine.adapters.workbuddy import models
from reward_engine.adapters.workbuddy.io import read, write

CAPTURE = (
    Path(__file__).parents[1] / "fixtures/workbuddy/qwen_duplicate_usage_terminal.sse"
)
CAPTURE_SHA256 = "f5b7cc4ff6e31d123c829bfcc5b24a34b35650bb946dd924088612a18fb5c86c"
# Exact response-1.sse from full_claw_20261010_v3/transport/roles/
# dff4dfc18686414486bbad9fdec2d985; diagnostic: claw_round3_stream_diagnostic.md.
# Provider output is evidence, not a validated Jury vote or benchmark success.


def event(delta, finish=None):
    return {
        "id": "completion-1",
        "model": "qwen3.8-max",
        "created": 123,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish,
                "native_finish_reason": finish,
            }
        ],
    }


def trailer_stream():
    terminal = event({"content": "", "role": "assistant"}, "stop")
    terminal["usage"] = {"completion_tokens": 2, "estimated": True}
    repeated = deepcopy(terminal)
    repeated["usage"] = {"completion_tokens": 3, "estimated": True}
    repeated["service_tier"] = None
    return [
        event({"content": '{"status":"unknown","reason":"observed"}'}),
        terminal,
        repeated,
    ]


def encode(events):
    return (
        "".join("data: " + json.dumps(row) + "\n\n" for row in events)
        + "data: [DONE]\n\n"
    )


@pytest.fixture
def transport(tmp_path, monkeypatch):
    config = {
        "output": str(tmp_path),
        "roles": {"judge": "qwen3.8-max"},
        "auth_file": str(tmp_path / "auth"),
        "gateway_local": "http://offline",
    }
    (tmp_path / "auth").write_text("offline-only")
    responses, calls = [], []

    def upstream(request):
        calls.append(json.loads(request.content))
        assert responses, "Unexpected extra physical attempt"
        return responses.pop(0)

    client = httpx.AsyncClient
    monkeypatch.setattr(
        models.httpx,
        "AsyncClient",
        lambda **kw: client(transport=httpx.MockTransport(upstream), **kw),
    )
    return models.Role(config, "judge", tmp_path / "role"), responses, calls


@pytest.mark.asyncio
async def test_exact_saved_qwen_duplicate_terminal_keeps_raw_usage_and_no_new_vote(
    transport,
):
    role, responses, calls = transport
    raw = CAPTURE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == CAPTURE_SHA256
    events = [
        json.loads(line[5:])
        for line in raw.decode().splitlines()
        if line.startswith("data:") and line[5:].strip() != "[DONE]"
    ]
    content = "".join(
        c["delta"].get("content", "") for e in events for c in e["choices"]
    )
    responses.append(httpx.Response(200, content=raw))
    assert await role({"mode": "test"}) == json.loads(content)
    assert await role({"mode": "test"}) == json.loads(content)
    assert len(calls) == 1
    receipt = read(next(role.output.glob("*/response.json")))
    assert receipt["raw_output"] == content
    assert receipt["usage_events"] == [
        e["usage"] for e in events if e.get("usage") is not None
    ]
    terminals = [e for e in events if e["choices"][0].get("finish_reason")]
    assert receipt["terminal_events"] == terminals
    assert receipt["usage"] == terminals[-1]["usage"]
    assert [t["usage"]["total_tokens"] for t in terminals] == [47060, 63726]
    assert receipt["normalizations"] == [
        {"operation": "accept_repeated_empty_usage_terminal", "terminal_events": 2}
    ]
    received = read(next(role.output.glob("*/stream_lines.json")))
    assert received == raw.decode().splitlines()[: len(received)]
    assert received[-1] == "data: [DONE]"
    assert read(next(role.output.glob("*/request_body.json"))) == calls[0]


@pytest.mark.parametrize(
    "fault",
    [
        "conflicting_finish",
        "native_truncation",
        "first_truncated",
        "different_id",
        "different_created",
        "different_model",
        "missing_id",
        "missing_usage",
        "multiple_choices",
        "wrong_index",
        "bool_index",
        "content",
        "reasoning",
        "tools",
        "unknown_delta",
        "error",
        "invalid_json",
        "competing_answers",
    ],
)
@pytest.mark.asyncio
async def test_terminal_trailer_cannot_redeem_ambiguous_or_failed_stream(
    transport, fault
):
    role, responses, calls = transport
    events = trailer_stream()
    second = events[-1]
    choice = second["choices"][0]
    if fault == "conflicting_finish":
        choice["finish_reason"] = "tool_calls"
    elif fault == "native_truncation":
        choice["native_finish_reason"] = "length"
    elif fault == "first_truncated":
        events[1]["choices"][0]["finish_reason"] = "length"
    elif fault == "different_id":
        second["id"] = "other"
    elif fault == "different_created":
        second["created"] = 456
    elif fault == "different_model":
        second["model"] = "other"
    elif fault == "missing_id":
        for e in events:
            e.pop("id")
    elif fault == "missing_usage":
        second.pop("usage")
    elif fault == "multiple_choices":
        second["choices"].append(deepcopy(choice))
    elif fault == "wrong_index":
        choice["index"] = 1
    elif fault == "bool_index":
        choice["index"] = False
    elif fault == "content":
        choice["delta"]["content"] = " "
    elif fault == "reasoning":
        choice["delta"]["reasoning_content"] = "more"
    elif fault == "tools":
        choice["delta"]["tool_calls"] = [{"index": 0}]
    elif fault == "unknown_delta":
        choice["delta"]["refusal"] = "refused"
    elif fault == "error":
        events.append({"error": {"message": "late provider failure"}})
    elif fault == "competing_answers":
        events[0]["choices"][0]["delta"]["content"] = (
            '{"status":"fail"}{"status":"pass"}'
        )
    raw = encode(events)
    if fault == "invalid_json":
        raw = raw.replace("data: [DONE]", "data: {broken\n\ndata: [DONE]")
    responses.append(httpx.Response(200, text=raw))
    with pytest.raises((ValueError, models.ProviderStreamError)):
        await role({"mode": "test"})
    assert len(calls) == 1
    failure = read(next(role.output.glob("*/failure.json")))
    assert failure["usage_events"] == [e["usage"] for e in events if "usage" in e]
    assert not list((role.output / "completed").glob("*.json"))
    assert not list(role.output.glob("*/response.json"))
    received = read(next(role.output.glob("*/stream_lines.json")))
    assert received == raw.splitlines()[: len(received)]


@pytest.mark.asyncio
async def test_completed_opening_fence_uses_existing_lossless_decoder_at_role_boundary(
    transport,
):
    role, responses, calls = transport
    # Existing decoder accepts one complete value under an unclosed presentation
    # fence. The outer json_answer_text previously rejected it before decoding.
    events = [event({"content": '```json\n{"status":"unknown"}'}, "stop")]
    responses.append(httpx.Response(200, text=encode(events)))
    assert await role({"mode": "test"}) == {"status": "unknown"}
    receipt = read(next(role.output.glob("*/response.json")))
    assert receipt["raw_output"] == events[0]["choices"][0]["delta"]["content"]
    assert receipt["normalizations"] == [
        {"operation": "unwrap_complete_json_presentation"}
    ]
    assert len(calls) == 1


@pytest.mark.parametrize(
    "raw",
    [
        '```json\n{"status":',
        '{"status":"fail","status":"pass"}',
        '{"status":"unknown","claims":[',
        "not JSON",
    ],
)
@pytest.mark.asyncio
async def test_parse_failure_does_not_trigger_opt_in_transport_retry(transport, raw):
    role, responses, calls = transport
    role.nonstream_recovery_models = {"qwen3.8-max"}
    responses.append(
        httpx.Response(200, text=encode([event({"content": raw}, "stop")]))
    )
    with pytest.raises(ValueError):
        await role({"mode": "test"})
    assert len(calls) == 1
    assert read(next(role.output.glob("*/failure.json")))["partial_output"] == raw
    assert not list((role.output / "completed").glob("*.json"))


@pytest.mark.parametrize("fault", ["error", "index", "bool_index", "truncated", "json"])
@pytest.mark.asyncio
async def test_nonstream_retry_retains_failed_attempts_and_identical_model_parameters(
    transport, fault
):
    role, responses, calls = transport
    role.nonstream_recovery_models = {"qwen3.8-max"}
    role.config["role_request_options"] = {
        "thinking": {"type": "disabled"},
        "enable_thinking": False,
    }
    write(
        Path(role.config["output"]) / "role_limits.json",
        {"role_max_tokens": {"judge": 32768}},
    )
    recovered = {
        "model": "qwen3.8-max",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"content": '{"status":"unknown"}'},
            }
        ],
    }
    if fault == "error":
        recovered["error"] = {"message": "failed despite choice"}
    elif fault == "index":
        recovered["choices"][0]["index"] = 1
    elif fault == "bool_index":
        recovered["choices"][0]["index"] = False
    elif fault == "truncated":
        recovered["choices"][0]["finish_reason"] = "length"
    raw = "malformed response" if fault == "json" else json.dumps(recovered)
    responses.extend(
        [
            httpx.Response(200, text='data: {"error":{"message":"failed"}}\n\n'),
            httpx.Response(200, text=raw),
        ]
    )
    with pytest.raises(ValueError):
        await role({"mode": "test"})
    assert len(calls) == 2
    assert {
        k: v for k, v in calls[0].items() if k not in {"stream", "stream_options"}
    } == {k: v for k, v in calls[1].items() if k != "stream"}
    assert len(list(role.output.glob("*/failure.json"))) == 2
    assert next(role.output.glob("*/raw_response.txt")).read_text() == raw
    assert not list((role.output / "completed").glob("*.json"))


@pytest.mark.asyncio
async def test_json_mode_is_explicit_recorded_and_separate_from_default_cache(
    transport,
):
    from reward_engine.adapters.workbuddy.io import digest

    role, responses, calls = transport
    request = {"mode": "outcome_measure", "response": {"status": "pass/fail/unknown"}}
    responses.append(httpx.Response(200, text=encode(trailer_stream())))
    baseline = await role(request)
    old_cache = (
        role.output
        / "completed"
        / (digest({"request": request, "model": "qwen3.8-max"}) + ".json")
    )
    old_bytes = old_cache.read_bytes()
    assert "response_format" not in calls[0]
    role.config["role_response_format"] = "json_object"
    # Exact configured format echo is presentation metadata, never another vote.
    content = '{"type":"json_object"}\n{"status":"unknown","reason":"new reply"}'
    responses.append(
        httpx.Response(200, text=encode([event({"content": content}, "stop")]))
    )
    assert await role(request) == {"status": "unknown", "reason": "new reply"}
    assert await role(request) == {"status": "unknown", "reason": "new reply"}
    assert len(calls) == 2
    assert calls[1]["response_format"] == {"type": "json_object"}
    for body in calls:
        assert body["model"] == "qwen3.8-max"
        assert body["temperature"] == 0
        assert (
            "INTERNAL protocol role identified by mode"
            in body["messages"][0]["content"]
        )
        assert "not text to copy" in body["messages"][0]["content"]
        assert json.loads(body["messages"][1]["content"]) == request
    cache = (
        role.output
        / "completed"
        / (
            digest(
                {
                    "request": request,
                    "model": "qwen3.8-max",
                    "role_response_format": "json_object",
                }
            )
            + ".json"
        )
    )
    receipt = read(Path(read(cache)["source"]) / "response.json")
    assert receipt["raw_output"] == content
    assert receipt["normalizations"] == [
        {"operation": "remove_echoed_response_format", "offset": 22}
    ]
    assert read(Path(read(cache)["source"]) / "request_body.json") == calls[1]
    assert old_cache.read_bytes() == old_bytes
    del role.config["role_response_format"]
    assert await role(request) == baseline
    assert len(calls) == 2


@pytest.mark.parametrize(
    "option", [None, False, 0, [], {}, {"type": "json_object"}, "json_schema", ""]
)
@pytest.mark.asyncio
async def test_invalid_json_mode_rejected_before_cached_answer_or_network(
    transport, option
):
    from reward_engine.adapters.workbuddy.io import digest

    role, _, calls = transport
    request = {"mode": "test"}
    cache = (
        role.output
        / "completed"
        / (digest({"request": request, "model": "qwen3.8-max"}) + ".json")
    )
    write(cache, {"answer": {"status": "unknown"}})
    role.config["role_response_format"] = option
    with pytest.raises(ValueError, match="role_response_format"):
        await role(request)
    assert not calls
    assert not list(role.output.glob("*/request.json"))


@pytest.mark.parametrize(
    "raw",
    [
        '{"type":"json_object"}{"status":"fail","status":"pass"}',
        '{"type":"json_object","type":"json_object"}{"status":"unknown"}',
        '{"type":"json_object"}{"status":"unknown"}{"status":"pass"}',
        '{"type":"json_object"}{"status":"unknown",}',
        '{"type":"json_object"}{"status":NaN}',
        '{"type":"json_object"}{"type":"json_object"}',
        '{"status":"fail"}{"status":"pass"}',
        '{"type":"json_object"}[]',
    ],
)
@pytest.mark.asyncio
async def test_json_mode_does_not_choose_answers_or_stack_repairs(transport, raw):
    role, responses, calls = transport
    role.config["role_response_format"] = "json_object"
    responses.append(
        httpx.Response(200, text=encode([event({"content": raw}, "stop")]))
    )
    with pytest.raises(ValueError):
        await role({"mode": "test"})
    assert len(calls) == 1
    assert not list((role.output / "completed").glob("*.json"))
    assert read(next(role.output.glob("*/failure.json")))["partial_output"] == raw


@pytest.mark.asyncio
async def test_format_echo_is_not_removed_without_explicit_json_mode(transport):
    role, responses, calls = transport
    responses.append(
        httpx.Response(
            200,
            text=encode(
                [
                    event(
                        {"content": '{"type":"json_object"}{"status":"unknown"}'},
                        "stop",
                    )
                ]
            ),
        )
    )
    with pytest.raises(ValueError):
        await role({"mode": "test"})
    assert len(calls) == 1
    assert not list((role.output / "completed").glob("*.json"))


@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize("failure_kind", ["syntax", "contract"])
@pytest.mark.asyncio
async def test_real_role_activates_existing_bounded_ask_repairs_only_when_opted_in(
    transport, bounded, failure_kind
):
    from reward_engine.core.outcome.context import ask, build_context

    original, responses, calls = transport
    role = models.Role(
        {
            **original.config,
            "role_bounded_stage": bounded,
            "role_stage_timeout_s": 30,
            "role_response_format": "json_object",
        },
        "judge",
        original.output,
    )
    context, evidence = build_context(
        {"task_id": "t", "prompt": "Verify the authored deliverable"}
    )
    rejected = (
        '{"status":' if failure_kind == "syntax" else '{"unrelated":"must retain"}'
    )
    responses.extend(
        httpx.Response(200, text=encode([event({"content": text}, "stop")]))
        for text in [rejected, '{"status":"unknown"}']
    )

    def validate(value):
        if set(value) != {"status"} or value["status"] not in {
            "pass",
            "fail",
            "unknown",
        }:
            raise ValueError("An explicit status is required")
        return value

    operation = ask(
        role,
        "fixture_contract",
        context,
        evidence,
        "Check only supplied evidence",
        {"status": "pass/fail/unknown"},
        validate=validate,
    )
    if bounded:
        assert await operation == {"status": "unknown"}
        assert len(calls) == 2
        requests = [json.loads(c["messages"][1]["content"]) for c in calls]
        recovery = requests[1].pop("response_recovery")
        assert requests[1] == requests[0]
        assert "Do not make a new assessment" in recovery["instruction"]
        if failure_kind == "syntax":
            assert recovery["rejected_raw_response"] == rejected
        else:
            assert recovery["rejected_response"] == json.loads(rejected)
    else:
        with pytest.raises(ValueError):
            await operation
        assert len(calls) == 1
    receipts = [read(p) for p in role.output.glob("*/response.json")]
    failures = [read(p) for p in role.output.glob("*/failure.json")]
    if failure_kind == "syntax":
        assert failures[0]["partial_output"] == rejected
    else:
        # Parsed transport response is retained even though core rejects it.
        assert any(r["raw_output"] == rejected for r in receipts)
    assert len(list(role.output.glob("*/stream_lines.json"))) == len(calls)
    for body in calls:
        assert body["response_format"] == {"type": "json_object"}
        assert body["temperature"] == 0
        assert body["model"] == "qwen3.8-max"
    assert role.bounded_stage is bounded
    assert role.stage_timeout_s == 30


@pytest.mark.parametrize("failure_kind", ["syntax", "contract"])
@pytest.mark.asyncio
async def test_real_bounded_role_stops_after_one_failed_repair(transport, failure_kind):
    from reward_engine.core.outcome.context import ask, build_context

    original, responses, calls = transport
    role = models.Role(
        {**original.config, "role_bounded_stage": True}, "judge", original.output
    )
    context, evidence = build_context(
        {"task_id": "t", "prompt": "Preserve uncertainty"}
    )
    raw = '{"status":' if failure_kind == "syntax" else '{"unsupported":true}'
    responses.extend(
        httpx.Response(200, text=encode([event({"content": raw}, "stop")]))
        for _ in range(2)
    )

    def invalid(value):
        raise ValueError("No supported verdict")

    with pytest.raises(ValueError):
        await ask(
            role,
            "fixture_contract",
            context,
            evidence,
            "No defaults",
            {"status": "pass/fail/unknown"},
            validate=invalid,
        )
    assert len(calls) == 2
    assert len(list(role.output.glob("*/stream_lines.json"))) == 2
    assert all(
        read(p)["answer"] == {"unsupported": True}
        for p in role.output.glob("*/response.json")
    )


@pytest.mark.parametrize(
    "config",
    [
        {"role_bounded_stage": 1},
        {"role_bounded_stage": "true"},
        {"role_bounded_stage": None},
        {"role_stage_timeout_s": True},
        {"role_stage_timeout_s": 0},
        {"role_stage_timeout_s": 3601},
        {"role_stage_timeout_s": 30.0},
        {"role_stage_timeout_s": "600"},
    ],
)
def test_invalid_stage_settings_fail_at_construction(tmp_path, config):
    with pytest.raises(ValueError):
        models.Role(config, "judge", tmp_path)


@pytest.mark.asyncio
async def test_stage_protocol_settings_prevent_old_cache_reuse(transport):
    from reward_engine.adapters.workbuddy.io import digest

    original, responses, calls = transport
    request = {"mode": "test"}
    responses.extend(
        httpx.Response(200, text=encode(trailer_stream())) for _ in range(3)
    )
    await original(request)
    assert original.bounded_stage is False and original.stage_timeout_s == 600
    old = (
        original.output
        / "completed"
        / (digest({"request": request, "model": "qwen3.8-max"}) + ".json")
    )
    saved = old.read_bytes()
    for timeout in [600, 1200]:
        role = models.Role(
            {
                **original.config,
                "role_bounded_stage": True,
                "role_stage_timeout_s": timeout,
            },
            "judge",
            original.output,
        )
        await role(request)
        await role(request)
        cache = (
            role.output
            / "completed"
            / (
                digest(
                    {
                        "request": request,
                        "model": "qwen3.8-max",
                        "role_bounded_stage": True,
                        "role_stage_timeout_s": timeout,
                    }
                )
                + ".json"
            )
        )
        root = Path(read(cache)["source"])
        assert read(root / "protocol_settings.json") == {
            "role_bounded_stage": True,
            "role_stage_timeout_s": timeout,
        }
    assert len(calls) == 3
    assert old.read_bytes() == saved
    await models.Role(
        {**original.config, "role_bounded_stage": False}, "judge", original.output
    )(request)
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_role_requests_use_shared_stage_budget_including_transport_recovery(
    transport,
):
    from reward_engine.core.outcome.call_budget import ACTIVE

    role, responses, calls = transport
    role.nonstream_recovery_models = {"qwen3.8-max"}
    responses.append(
        httpx.Response(200, text='data: {"error":{"message":"failed"}}\n\n')
    )
    budget = {"requests": 5, "contract_repairs": 0, "syntax_repairs": 0}
    token = ACTIVE.set(budget)
    try:
        with pytest.raises(RuntimeError, match="six physical requests"):
            await role({"mode": "test"})
    finally:
        ACTIVE.reset(token)
    assert len(calls) == 1
    assert budget["requests"] == 6
    assert len(list(role.output.glob("*/failure.json"))) == 2
    assert not list((role.output / "completed").glob("*.json"))


@pytest.mark.asyncio
async def test_cached_reply_does_not_charge_exhausted_stage_budget(transport):
    from reward_engine.core.outcome.call_budget import ACTIVE

    original, responses, calls = transport
    role = models.Role(
        {**original.config, "role_bounded_stage": True}, "judge", original.output
    )
    responses.append(httpx.Response(200, text=encode(trailer_stream())))
    request = {"mode": "test"}
    expected = await role(request)
    budget = {"requests": 6, "contract_repairs": 0, "syntax_repairs": 0}
    before = deepcopy(budget)
    token = ACTIVE.set(budget)
    try:
        assert await role(request) == expected
    finally:
        ACTIVE.reset(token)
    assert budget == before
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_bounded_stage_timeout_retains_partial_stream_and_restores_budget(
    transport,
):
    import asyncio

    from reward_engine.core.outcome.call_budget import ACTIVE
    from reward_engine.core.outcome.context import ask, build_context

    class StalledStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield (
                "data: " + json.dumps(event({"content": '{"status":'})) + "\n\n"
            ).encode()
            await asyncio.Event().wait()

    original, responses, calls = transport
    role = models.Role(
        {**original.config, "role_bounded_stage": True, "role_stage_timeout_s": 1},
        "judge",
        original.output,
    )
    responses.append(httpx.Response(200, stream=StalledStream()))
    context, evidence = build_context(
        {"task_id": "t", "prompt": "Check authored output"}
    )
    before = ACTIVE.get()
    with pytest.raises(TimeoutError):
        await ask(
            role,
            "test",
            context,
            evidence,
            "No defaults",
            {"status": "pass/fail/unknown"},
        )
    assert ACTIVE.get() is before
    assert len(calls) == 1
    failure = read(next(role.output.glob("*/failure.json")))
    assert failure["error_type"] == "CancelledError"
    assert failure["partial_output"] == '{"status":'
    assert read(next(role.output.glob("*/stream_lines.json")))
    assert not list((role.output / "completed").glob("*.json"))


@pytest.mark.asyncio
async def test_empty_native_tool_terminal_replay_does_not_repeat_tool_arguments(
    transport,
):
    from reward_engine.adapters.workbuddy.measurement import WorkBuddyTools

    role, responses, calls = transport
    request = {
        "mode": "outcome_measure",
        "measurement_tools": WorkBuddyTools().catalog(),
        "response": {"need_tools": []},
    }
    call = {
        "index": 0,
        "id": "call-1",
        "type": "function",
        "function": {"name": "workspace.read", "arguments": '{"paths":["output.md"]}'},
    }
    events = trailer_stream()
    events[0]["choices"][0]["delta"] = {"tool_calls": [call]}
    for e in events[1:]:
        e["choices"][0].update(
            finish_reason="tool_calls", native_finish_reason="tool_calls"
        )
    responses.append(httpx.Response(200, text=encode(events)))
    assert await role(request) == {
        "need_tools": [
            {"tool": "workspace.read", "arguments": {"paths": ["output.md"]}}
        ]
    }
    receipt = read(next(role.output.glob("*/response.json")))
    assert receipt["native_tool_call_chunks"] == [[call]]
    assert len(calls) == 1
