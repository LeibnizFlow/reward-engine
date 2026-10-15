"""Synthetic empty tool placeholders never erase real calls."""

import json
from copy import deepcopy

import httpx
import pytest

from reward_engine.adapters.workbuddy import models
from reward_engine.adapters.workbuddy.io import read, write
from reward_engine.adapters.workbuddy.measurement import WorkBuddyTools

EMPTY_FUNCTION = {"name": "", "arguments": ""}


def event(delta, finish=None):
    return {
        "id": "offline-glm",
        "model": "glm-5.3-flash",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def encode(events):
    return (
        "".join("data: " + json.dumps(e) + "\n\n" for e in events) + "data: [DONE]\n\n"
    )


def text_events():
    return [
        event({"content": '{"status":"unknown"}', "tool_calls": []}),
        event({"tool_calls": [], "function_call": EMPTY_FUNCTION}, "stop"),
    ]


def native_call():
    return {
        "index": 0,
        "id": "call-1",
        "type": "function",
        "function": {"name": "workspace.read", "arguments": '{"paths":["output.md"]}'},
    }


def protocol():
    return {
        "mode": "outcome_measure",
        "measurement_tools": WorkBuddyTools().catalog(),
        "response": {"need_tools": []},
    }


@pytest.fixture
def transport(tmp_path, monkeypatch):
    config = {
        "output": str(tmp_path),
        "roles": {"jury_member_3": "glm-5.3-flash"},
        "auth_file": str(tmp_path / "auth"),
        "gateway_local": "http://offline",
        "role_response_format": "json_object",
        "role_bounded_stage": True,
        "role_stage_timeout_s": 1800,
        "role_request_options": {
            "thinking": {"type": "disabled"},
            "enable_thinking": False,
        },
    }
    (tmp_path / "auth").write_text("offline-only")
    write(tmp_path / "role_limits.json", {"role_max_tokens": {"jury_member_3": 65536}})
    responses, calls = [], []

    def upstream(request):
        assert request.url.host == "offline"
        calls.append(json.loads(request.content))
        assert responses, "Unexpected extra physical call"
        return responses.pop(0)

    client = httpx.AsyncClient
    monkeypatch.setattr(
        models.httpx,
        "AsyncClient",
        lambda **kw: client(transport=httpx.MockTransport(upstream), **kw),
    )
    return models.Role(config, "jury_member_3", tmp_path / "role"), responses, calls


@pytest.mark.asyncio
async def test_synthetic_empty_batches_decode_json_and_preserve_receipts(
    transport,
):
    role, responses, calls = transport
    events = text_events()
    lines = encode(events).splitlines()
    request = {"mode": "test"}
    deltas = [c["delta"] for e in events for c in e["choices"]]
    batches = [d["tool_calls"] for d in deltas if d.get("tool_calls") is not None]
    assert batches == [[], []]
    assert [
        d["function_call"] for d in deltas if d.get("function_call") is not None
    ] == [EMPTY_FUNCTION]
    content = "".join(d.get("content") or "" for d in deltas)
    expected = json.loads(content)
    assert expected == {"status": "unknown"}
    responses.append(httpx.Response(200, text="\n".join(lines) + "\n"))
    assert await role(request) == expected
    assert await role(request) == expected
    assert len(calls) == 1
    assert calls[0]["max_tokens"] == 65536
    assert calls[0]["response_format"] == {"type": "json_object"}
    assert json.loads(calls[0]["messages"][-1]["content"]) == request
    saved = read(next(role.output.glob("*/response.json")))
    assert saved["raw_output"] == content
    assert saved["native_tool_call_chunks"] == batches
    assert saved["legacy_function_call_chunks"] == [EMPTY_FUNCTION]
    assert saved["usage_events"] == [
        e["usage"] for e in events if e.get("usage") is not None
    ]
    assert saved["terminal_events"] == [events[-1]]
    assert saved["normalizations"] == []
    received = read(next(role.output.glob("*/stream_lines.json")))
    assert received == lines[: len(received)]
    assert received[-1] == "data: [DONE]"
    assert not list(role.output.glob("*/failure.json"))


@pytest.mark.parametrize(
    "field,fragment",
    [
        ("tool_calls", {}),
        ("tool_calls", ""),
        ("tool_calls", False),
        ("tool_calls", 0),
        ("tool_calls", [{}]),
        ("tool_calls", [[]]),
        ("tool_calls", [None]),
        ("tool_calls", [{"index": 0}]),
        ("tool_calls", [{"id": "call-1"}]),
        ("tool_calls", [{"function": {"name": "workspace.read"}}]),
        ("tool_calls", [{"function": {"arguments": "{}"}}]),
        ("tool_calls", [{"function": EMPTY_FUNCTION}]),
        ("function_call", {}),
        ("function_call", []),
        ("function_call", ""),
        ("function_call", False),
        ("function_call", 0),
        ("function_call", {"name": "workspace.read", "arguments": ""}),
        ("function_call", {"name": "", "arguments": "{}"}),
        ("function_call", {"name": ""}),
        ("function_call", {"arguments": ""}),
        ("function_call", {"name": None, "arguments": ""}),
        ("function_call", {"name": "", "arguments": "", "id": "call-1"}),
        ("function_call", {"name": "", "arguments": "", "index": 0}),
    ],
)
@pytest.mark.parametrize("after_stop", [False, True])
@pytest.mark.asyncio
async def test_empty_placeholders_never_hide_attempted_or_malformed_calls(
    transport, field, fragment, after_stop
):
    role, responses, calls = transport
    events = text_events()
    if after_stop:
        events.append(event({field: fragment}))
    else:
        events[0]["choices"][0]["delta"][field] = fragment
    raw = encode(events)
    responses.append(httpx.Response(200, text=raw))
    with pytest.raises(ValueError):
        await role(protocol())
    assert len(calls) == 1
    saved = read(next(role.output.glob("*/failure.json")))
    key = (
        "native_tool_call_chunks"
        if field == "tool_calls"
        else "legacy_function_call_chunks"
    )
    assert fragment in saved[key]
    assert saved["partial_output"] == '{"status":"unknown"}'
    assert saved["content_after_finish"] is after_stop
    assert not list((role.output / "completed").glob("*.json"))
    received = read(next(role.output.glob("*/stream_lines.json")))
    assert received == raw.splitlines()[: len(received)]


@pytest.mark.parametrize("finish", ["stop", "tool_calls"])
@pytest.mark.asyncio
async def test_genuine_call_with_json_text_is_never_dropped(transport, finish):
    role, responses, _ = transport
    events = text_events()
    events.insert(1, event({"tool_calls": [native_call()]}))
    events[-1]["choices"][0]["finish_reason"] = finish
    responses.append(httpx.Response(200, text=encode(events)))
    with pytest.raises(ValueError):
        await role(protocol())
    failure = read(next(role.output.glob("*/failure.json")))
    assert failure["native_tool_call_chunks"] == [[], [native_call()], []]
    assert not list((role.output / "completed").glob("*.json"))


@pytest.mark.asyncio
async def test_empty_batches_around_real_tool_fragments_preserve_need_tools(transport):
    role, responses, _ = transport
    first = native_call()
    first["function"]["arguments"] = '{"paths":'
    last = {"index": 0, "function": {"arguments": '["output.md"]}'}}
    batches = [[], [first], [], [last], []]
    events = [event({"tool_calls": batch}) for batch in batches]
    events[-1]["choices"][0].update(finish_reason="tool_calls")
    responses.append(httpx.Response(200, text=encode(events)))
    assert await role(protocol()) == {
        "need_tools": [
            {"tool": "workspace.read", "arguments": {"paths": ["output.md"]}}
        ]
    }
    saved = read(next(role.output.glob("*/response.json")))
    assert saved["native_tool_call_chunks"] == batches
    assert saved["normalizations"][0]["executed_by_adapter"] is False


@pytest.mark.parametrize(
    "fault",
    [
        "tool_stop_without_calls",
        "length",
        "missing_stop",
        "native_length",
        "multiple_choices",
        "content_after_stop",
        "error",
    ],
)
@pytest.mark.asyncio
async def test_placeholders_cannot_redeem_invalid_completion(transport, fault):
    role, responses, _ = transport
    events = text_events()
    terminal = events[-1]["choices"][0]
    if fault == "tool_stop_without_calls":
        terminal["finish_reason"] = "tool_calls"
    elif fault == "length":
        terminal["finish_reason"] = "length"
    elif fault == "missing_stop":
        terminal["finish_reason"] = None
    elif fault == "native_length":
        terminal["native_finish_reason"] = "length"
    elif fault == "multiple_choices":
        events[0]["choices"].append(deepcopy(events[0]["choices"][0]))
    elif fault == "content_after_stop":
        events.append(event({"content": " ", "tool_calls": []}))
    elif fault == "error":
        events.append({"error": {"message": "offline provider failure"}})
    responses.append(httpx.Response(200, text=encode(events)))
    with pytest.raises((ValueError, models.ProviderStreamError)):
        await role(protocol())
    assert list(role.output.glob("*/failure.json"))
    assert not list((role.output / "completed").glob("*.json"))


@pytest.mark.parametrize(
    "delta",
    [
        {},
        {"tool_calls": None, "function_call": None},
        {"tool_calls": []},
        {"function_call": EMPTY_FUNCTION},
    ],
)
@pytest.mark.asyncio
async def test_absent_null_or_exact_empty_placeholders_preserve_text(transport, delta):
    role, responses, _ = transport
    events = [event({"content": '{"status":"unknown"}', **delta}, "stop")]
    responses.append(httpx.Response(200, text=encode(events)))
    assert await role({"mode": "test"}) == {"status": "unknown"}


@pytest.mark.parametrize(
    "fragment",
    [
        {},
        {"name": "workspace.read", "arguments": ""},
        {"name": "", "arguments": "{}"},
        {"name": "", "arguments": "", "index": 0},
    ],
)
@pytest.mark.asyncio
async def test_nonstream_recovery_never_hides_legacy_function_attempt(
    transport, fragment
):
    role, responses, calls = transport
    role.nonstream_recovery_models = {"glm-5.3-flash"}
    responses.extend(
        [
            httpx.Response(
                200, text=encode([{"error": {"message": "offline failure"}}])
            ),
            httpx.Response(
                200,
                json={
                    "model": "glm-5.3-flash",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {
                                "content": '{"status":"unknown"}',
                                "function_call": fragment,
                            },
                        }
                    ],
                },
            ),
        ]
    )
    with pytest.raises(ValueError, match="Legacy function-call"):
        await role({"mode": "test"})
    assert len(calls) == 2
    raw = read(next(role.output.glob("*/raw_response.json")))
    assert raw["choices"][0]["message"]["function_call"] == fragment
    assert len(list(role.output.glob("*/failure.json"))) == 2
    assert not list((role.output / "completed").glob("*.json"))
