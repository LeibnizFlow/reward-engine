"""Replay native reads into the real bounded measurement protocol, never live HTTP."""

import json
from pathlib import Path

import httpx
import pytest

from reward_engine.adapters.workbuddy import models
from reward_engine.adapters.workbuddy.evidence import collect
from reward_engine.adapters.workbuddy.io import read
from reward_engine.adapters.workbuddy.measurement import WorkBuddyTools
from reward_engine.adapters.workbuddy.reply import decode_native_tool_reply
from reward_engine.core.outcome.context import build_context
from reward_engine.core.outcome.harness import defaults

FIXTURE = Path(__file__).parents[1] / "fixtures/workbuddy/native_workspace_read.json"


def protocol():
    return {
        "mode": "outcome_measure",
        "measurement_tools": WorkBuddyTools().catalog(),
        "tool_protocol": 'Before a verdict return ONLY {"need_tools":[]}',
        "response": {"status": "pass/fail/unknown"},
    }


def call(index=0, name="workspace.read", arguments='{"paths":["output.md"]}'):
    return {
        "index": index,
        "id": f"call-{index}",
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def event(delta, finish=None, *, index=0):
    return {
        "model": "deepseek-flash",
        "choices": [{"index": index, "delta": delta, "finish_reason": finish}],
    }


def native_events(calls=None, content=""):
    return [
        event(
            {"content": content, "tool_calls": calls if calls is not None else [call()]}
        ),
        event({}, "tool_calls"),
    ]


def text_events(answer):
    return [event({"content": json.dumps(answer)}, "stop")]


def sse(events):
    return (
        "".join("data: " + json.dumps(row) + "\n\n" for row in events)
        + "data: [DONE]\n\n"
    )


@pytest.fixture
def transport(tmp_path, monkeypatch):
    config = {
        "output": str(tmp_path),
        "roles": {"reward_calibration_judge": "deepseek-v4-flash"},
        "model_aliases": {"deepseek-v4-flash": ["deepseek-flash"]},
        "auth_file": str(tmp_path / "auth"),
        "gateway_local": "http://offline",
    }
    (tmp_path / "auth").write_text("offline-token")
    requests, responses = [], []

    def upstream(request):
        body = json.loads(request.content)
        requests.append(json.loads(body["messages"][-1]["content"]))
        assert responses, "Unexpected provider call"
        response = responses.pop(0)
        if isinstance(response, dict):
            return httpx.Response(200, json=response)
        return httpx.Response(
            200, text=sse(response), headers={"content-type": "text/event-stream"}
        )

    client = httpx.AsyncClient
    monkeypatch.setattr(
        models.httpx,
        "AsyncClient",
        lambda **kw: client(transport=httpx.MockTransport(upstream), **kw),
    )
    role = models.Role(config, "reward_calibration_judge", tmp_path / "role")
    return role, requests, responses


@pytest.mark.asyncio
async def test_actual_sanitized_stream_becomes_only_need_tools_with_lossless_provenance(
    transport,
):
    role, requests, responses = transport
    fixture = read(FIXTURE)
    events = fixture["events"]
    responses.append(events)
    raw_chunks = [
        choice["delta"]["tool_calls"]
        for row in events
        for choice in row["choices"]
        if choice["delta"].get("tool_calls")
    ]
    raw_content = "".join(
        choice["delta"].get("content", "")
        for row in events
        for choice in row["choices"]
    )
    raw_args = "".join(
        chunk["function"].get("arguments", "")
        for batch in raw_chunks
        for chunk in batch
    )
    expected = {
        "need_tools": [{"tool": "workspace.read", "arguments": json.loads(raw_args)}]
    }
    assert isinstance(expected["need_tools"][0]["arguments"]["paths"], str)
    answer = await role(protocol())
    assert answer == expected
    assert "status" not in answer
    assert await role(protocol()) == expected
    assert len(requests) == 1
    saved = read(next(role.output.glob("*/response.json")))
    assert saved["raw_output"] == raw_content
    assert saved["native_tool_call_chunks"] == raw_chunks
    assert saved["finish_reason"] == "tool_calls"
    assert saved["actual_models"] == ["deepseek-flash"]
    assert saved["usage"]["total_tokens"] == 15433
    note = saved["normalizations"][0]
    assert note["commentary"] == raw_content
    assert note["call_ids"] == ["call_sanitized_read"]
    assert note["executed_by_adapter"] is False
    assert not list(role.output.glob("**/measurement_tool_*.json"))


@pytest.mark.asyncio
@pytest.mark.parametrize("actual_stream", [False, True])
async def test_actual_string_paths_are_rejected_by_core_then_corrected_within_budget(
    transport, tmp_path, actual_stream
):
    role, requests, responses = transport
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "output.md").write_text("Delivered value 7")
    product, _ = collect(
        workspace,
        candidate_id="actual",
        final_response="Delivered value 7",
        initial_hashes={},
    )
    context, evidence = build_context(
        {"task_id": "t", "prompt": "Write value 7 to output.md"}, [product]
    )
    tool_ref = "tool:check:1" if actual_stream else "tool:check:0"
    responses.extend(
        [
            read(FIXTURE)["events"],
            text_events(
                {
                    "need_tools": [
                        {
                            "tool": "workspace.read",
                            "arguments": {"paths": ["output.md"]},
                        }
                    ]
                }
            ),
            text_events(
                {
                    "status": "pass",
                    "reason": "Observed delivered value",
                    "evidence_refs": [tool_ref],
                    "claims": [
                        {
                            "claim": "Delivered 7",
                            "claim_refs": [tool_ref],
                            "evidence_refs": [tool_ref],
                            "observation": "Read contains 7",
                            "judgment": "supported",
                        }
                    ],
                }
            ),
        ]
    )
    if not actual_stream:
        responses[:2] = [native_events(content="Reading the delivered file.")]
    result = await defaults(role, tools=WorkBuddyTools()).run(
        {"test_id": "check", "criterion_id": "value", "checker": "evidence.review"},
        context,
        evidence,
    )
    assert len(requests) == (3 if actual_stream else 2)
    assert result["tool_observations"][tool_ref]["status"] == "observed"
    assert (
        "Delivered value 7" in result["tool_observations"][tool_ref]["result"]["text"]
    )
    if actual_stream:
        first = result["tool_observations"]["tool:check:0"]
        assert first["status"] == "unknown"
        assert first["reason"] == "Supply one to 32 path patterns"
        assert isinstance(first["arguments"]["paths"], str)
        assert requests[1]["context"]["tool_observations"]["tool:check:0"] == first
        assert requests[2]["measurement_budget"]["final_verdict_required"] is True
        # A corrected read of another file cannot waive the original unknowns.
        assert result["status"] == "unknown"
        assert result["unresolved_measurement_refs"] == ["tool:check:0"]
    else:
        assert result["status"] == "pass", result
        assert result["unresolved_measurement_refs"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("count,expected_executions", [(1, 2), (4, 0)])
async def test_native_requests_remain_subject_to_core_wave_and_probe_limits(
    transport, count, expected_executions
):
    role, requests, responses = transport
    responses.extend([native_events([call(i) for i in range(count)]) for _ in range(3)])
    context, evidence = build_context(
        {"task_id": "t", "prompt": "Inspect delivery"},
        [{"artifact": {"final_response": "Output"}}],
    )
    executed = []
    tools = WorkBuddyTools(record=lambda name, value: executed.append(value))
    result = await defaults(role, tools=tools).run(
        {"test_id": "check", "criterion_id": "value", "checker": "evidence.review"},
        context,
        evidence,
    )
    assert len(requests) == 3
    assert len(executed) == expected_executions
    assert result["status"] == "unknown" and result["error"] is True
    assert "wave budget exhausted" in result["reason"]
    assert requests[-1]["measurement_budget"]["final_verdict_required"] is True
    if count == 4:
        assert requests[1]["measurement_feedback"]["status"] == "request_not_executed"


@pytest.mark.parametrize(
    "name,args",
    [
        ("workspace.read", {"paths": ["literal[1].txt"], "offset": 1}),
        ("workspace.inspect", {"paths": ["*.md"]}),
        ("document.read", {"ref": "product_1:source:0"}),
    ],
)
def test_fragmented_advertised_read_calls_preserve_arguments(name, args):
    encoded = json.dumps(args)
    chunks = [
        [
            {
                **call(name=name, arguments=encoded[:3]),
                "function": {"name": name[:4], "arguments": encoded[:3]},
            }
        ],
        [{"index": 0, "function": {"name": name[4:], "arguments": encoded[3:]}}],
    ]
    answer, _ = decode_native_tool_reply(
        protocol(),
        chunks,
        content="Inspecting the supplied evidence.",
        finish_reason="tool_calls",
    )
    assert answer == {"need_tools": [{"tool": name, "arguments": args}]}


@pytest.mark.parametrize(
    "raw",
    [
        '{"paths":',
        '["output.md"]',
        "null",
        '{"paths":[],"paths":["other"]}',
        '{"offset":NaN}',
        '{"offset":Infinity}',
        '{"paths":[]} trailing',
        '{"paths":[],"command":"rm -rf /"}',
    ],
)
@pytest.mark.asyncio
async def test_malformed_native_arguments_fail_without_cache_and_keep_raw_chunks(
    transport, raw
):
    role, _, responses = transport
    calls = [call(arguments=raw)]
    responses.append(native_events(calls))
    with pytest.raises(ValueError):
        await role(protocol())
    saved = read(next(role.output.glob("*/failure.json")))
    assert saved["native_tool_call_chunks"] == [calls]
    assert saved["finish_reason"] == "tool_calls"
    assert saved["model"] == "deepseek-v4-flash"
    assert saved["actual_models"] == ["deepseek-flash"]
    assert not list((role.output / "completed").glob("*.json"))


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown",
        "unadvertised",
        "unavailable",
        "write_tool",
        "nonmeasurement",
        "verdict_only",
    ],
)
@pytest.mark.asyncio
async def test_unadvertised_or_non_readonly_native_calls_fail_closed(
    transport, mutation
):
    role, _, responses = transport
    request = protocol()
    name = "workspace.read"
    if mutation in {"unknown", "write_tool"}:
        name = "invented.read" if mutation == "unknown" else "code.run"
        request["measurement_tools"][name] = {"available": True}
    elif mutation == "unadvertised":
        del request["measurement_tools"][name]
    elif mutation == "unavailable":
        request["measurement_tools"][name]["available"] = False
    elif mutation == "nonmeasurement":
        request["mode"] = "outcome_ranking_reduce"
    else:
        del request["tool_protocol"]
    responses.append(native_events([call(name=name)]))
    with pytest.raises(ValueError):
        await role(request)
    assert not list((role.output / "completed").glob("*.json"))


@pytest.mark.parametrize(
    "content",
    [
        '{"status":"pass"}',
        'Explanation\n{"status":"unknown"}',
        '```json\n{"need_tools":[]}\n```',
        '[{"status":"pass"}]',
        '"pass"',
        "pass",
    ],
)
@pytest.mark.asyncio
async def test_native_calls_mixed_with_json_or_verdict_are_rejected_intact(
    transport, content
):
    role, _, responses = transport
    responses.append(native_events(content=content))
    with pytest.raises(ValueError, match="mixed"):
        await role(protocol())
    failure = read(next(role.output.glob("*/failure.json")))
    assert failure["partial_output"] == content
    assert failure["native_tool_call_chunks"]


@pytest.mark.parametrize(
    "fault",
    [
        "stop",
        "length",
        "missing_finish",
        "wrong_model",
        "second_choice",
        "after_finish",
        "no_calls",
        "duplicate_id",
        "missing_index",
        "bool_index",
        "gap_index",
        "nonstring_fragment",
    ],
)
@pytest.mark.asyncio
async def test_incomplete_or_ambiguous_native_streams_are_not_cached(transport, fault):
    role, _, responses = transport
    events = native_events()
    if fault in {"stop", "length", "missing_finish"}:
        events[-1]["choices"][0]["finish_reason"] = (
            None if fault == "missing_finish" else fault
        )
    elif fault == "wrong_model":
        events[0]["model"] = "wrong-model"
    elif fault == "second_choice":
        events[0]["choices"][0]["index"] = 1
    elif fault == "after_finish":
        events.append(event({"content": "Late content"}))
    elif fault == "no_calls":
        events[0]["choices"][0]["delta"].pop("tool_calls")
    else:
        calls = events[0]["choices"][0]["delta"]["tool_calls"]
        if fault == "duplicate_id":
            calls.append({**call(1), "id": calls[0]["id"]})
        elif fault == "missing_index":
            calls[0].pop("index")
        elif fault in {"bool_index", "gap_index"}:
            calls[0]["index"] = True if fault == "bool_index" else 1
        else:
            calls[0]["function"]["arguments"] = {"paths": []}
    responses.append(events)
    with pytest.raises(ValueError):
        await role(protocol())
    assert not list((role.output / "completed").glob("*.json"))


@pytest.mark.asyncio
async def test_leading_prose_json_still_fails_without_native_tools(transport):
    role, _, responses = transport
    content = 'Here is my answer: {"status":"pass"}'
    responses.append([event({"content": content}, "stop")])
    with pytest.raises(json.JSONDecodeError):
        await role(protocol())
    assert read(next(role.output.glob("*/failure.json")))["partial_output"] == content


def test_explicit_need_tools_response_allows_read_only_native_request():
    request = protocol()
    request.pop("tool_protocol")
    request["response"] = {"need_tools": [{"tool": "catalog key", "arguments": {}}]}
    answer, _ = decode_native_tool_reply(
        request, [[call()]], content="", finish_reason="tool_calls"
    )
    assert answer == {
        "need_tools": [
            {"tool": "workspace.read", "arguments": {"paths": ["output.md"]}}
        ]
    }


def test_interleaved_native_reads_reassemble_by_index_without_argument_coercion():
    request = protocol()
    chunks = [
        [call(1, "document.read", '{"ref":'), call(0, "workspace.read", '{"paths":')],
        [{"index": 0, "function": {"arguments": '["output.md"]}'}}],
        [{"index": 1, "function": {"arguments": '"product_1:source:0"}'}}],
    ]
    answer, notes = decode_native_tool_reply(
        request, chunks, content="", finish_reason="tool_calls"
    )
    assert answer == {
        "need_tools": [
            {"tool": "workspace.read", "arguments": {"paths": ["output.md"]}},
            {"tool": "document.read", "arguments": {"ref": "product_1:source:0"}},
        ]
    }
    assert notes[0]["call_ids"] == ["call-0", "call-1"]


@pytest.mark.asyncio
async def test_nonstream_recovery_cannot_hide_native_tools_inside_a_verdict(transport):
    role, _, responses = transport
    role.nonstream_recovery_models = {"deepseek-v4-flash"}
    responses.extend(
        [
            [{"error": {"message": "offline stream failure"}}],
            {
                "model": "deepseek-flash",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": '{"status":"pass"}',
                            "tool_calls": [call()],
                        },
                    }
                ],
            },
        ]
    )
    with pytest.raises(ValueError, match="Native nonstream tool replies"):
        await role(protocol())
    assert not list((role.output / "completed").glob("*.json"))
    raw = read(next(role.output.glob("*/raw_response.json")))
    assert raw["choices"][0]["message"]["tool_calls"] == [call()]


@pytest.mark.parametrize(
    "fault",
    [
        "empty_stop_then_content",
        "late_content",
        "late_reasoning",
        "length_then_stop",
        "tool_finish_then_stop",
        "duplicate_stop",
        "choice_one",
        "bool_choice",
        "string_choice",
        "split_choices",
        "duplicate_choices_in_event",
    ],
)
@pytest.mark.asyncio
async def test_text_stream_requires_one_coherent_terminal_choice(transport, fault):
    role, _, responses = transport
    events = text_events({"status": "pass"})
    if fault == "empty_stop_then_content":
        events = [event({}, "stop"), event({"content": '{"status":"pass"}'})]
    elif fault in {"late_content", "late_reasoning"}:
        field = "content" if fault == "late_content" else "reasoning_content"
        events.append(event({field: " "}))
    elif fault in {"length_then_stop", "tool_finish_then_stop", "duplicate_stop"}:
        terminal = {
            "length_then_stop": "length",
            "tool_finish_then_stop": "tool_calls",
            "duplicate_stop": "stop",
        }[fault]
        events.insert(0, event({}, terminal))
    elif fault in {"choice_one", "bool_choice", "string_choice"}:
        events[0]["choices"][0]["index"] = {
            "choice_one": 1,
            "bool_choice": False,
            "string_choice": "0",
        }[fault]
    elif fault == "split_choices":
        events = [
            event({"content": '{"status":'}),
            event({"content": '"pass"}'}, "stop", index=1),
        ]
    else:
        events[0]["choices"].insert(0, {"index": 0, "delta": {}, "finish_reason": None})
    responses.append(events)
    with pytest.raises(ValueError, match="completed choice"):
        await role(protocol())
    saved = read(next(role.output.glob("*/failure.json")))
    assert saved["partial_output"] == "".join(
        choice["delta"].get("content", "")
        for row in events
        for choice in row["choices"]
    )
    assert saved["choice_indices"] == [
        choice["index"] for row in events for choice in row["choices"]
    ]
    assert not list((role.output / "completed").glob("*.json"))


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.asyncio
async def test_final_usage_only_chunk_is_accepted_after_terminal(transport, native):
    role, requests, responses = transport
    events = (
        native_events()
        if native
        else [
            event({"role": "assistant", "content": ""}),
            event({"content": '{"status":'}),
            event({"content": '"pass"}'}, "stop"),
        ]
    )
    usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    events.append({"choices": [], "usage": usage})
    responses.append(events)
    answer = await role(protocol())
    assert answer == (
        {
            "need_tools": [
                {"tool": "workspace.read", "arguments": {"paths": ["output.md"]}}
            ]
        }
        if native
        else {"status": "pass"}
    )
    assert await role(protocol()) == answer
    assert len(requests) == 1
    assert read(next(role.output.glob("*/response.json")))["usage"] == usage


@pytest.mark.parametrize("recovery", [False, True])
@pytest.mark.asyncio
async def test_duplicate_verdict_keys_never_cache_and_keep_original(
    transport, recovery
):
    role, _, responses = transport
    raw = '{"status":"fail","status":"pass"}'
    if recovery:
        role.nonstream_recovery_models = {"deepseek-v4-flash"}
        responses.extend(
            [
                [{"error": {"message": "offline stream failure"}}],
                {
                    "model": "deepseek-flash",
                    "choices": [{"finish_reason": "stop", "message": {"content": raw}}],
                },
            ]
        )
    else:
        responses.append([event({"content": raw}, "stop")])
    with pytest.raises(json.JSONDecodeError, match="Duplicate"):
        await role(protocol())
    assert not list((role.output / "completed").glob("*.json"))
    if recovery:
        saved = read(next(role.output.glob("*/raw_response.json")))
        assert saved["choices"][0]["message"]["content"] == raw
    else:
        assert read(next(role.output.glob("*/failure.json")))["partial_output"] == raw


@pytest.mark.asyncio
async def test_fragmented_native_arguments_reject_duplicates_at_nested_depth(transport):
    role, _, responses = transport
    chunks = [call(arguments='{"paths":[{"path":"first",')]
    tail = [{"index": 0, "function": {"arguments": '"path":"second"}]}'}}]
    responses.append(
        [event({"tool_calls": chunks}), event({"tool_calls": tail}, "tool_calls")]
    )
    with pytest.raises(ValueError, match="Duplicate"):
        await role(protocol())
    saved = read(next(role.output.glob("*/failure.json")))
    assert saved["native_tool_call_chunks"] == [chunks, tail]
    assert not list((role.output / "completed").glob("*.json"))
