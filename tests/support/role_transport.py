"""Synthetic SSE and offline Role transport helpers; no captured data or network."""

import json
from copy import deepcopy

import httpx
import pytest

from reward_engine.adapters.workbuddy import models
from reward_engine.adapters.workbuddy.models import Role
from reward_engine.core.outcome.harness import defaults


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


async def replay(transport, replies, context, evidence):
    original, responses, calls = transport
    role = Role(
        {**original.config, "role_bounded_stage": True}, "judge", original.output
    )
    before = deepcopy(replies)
    responses.extend(
        httpx.Response(
            200,
            text=encode(
                [
                    event(
                        {
                            "content": r
                            if isinstance(r, str)
                            else json.dumps(r, ensure_ascii=False)
                        },
                        "stop",
                    )
                ]
            ),
        )
        for r in replies
    )
    result = await defaults(role).run(
        {
            "test_id": "contract",
            "criterion_id": "contract",
            "checker": "evidence.review",
        },
        context,
        evidence,
    )
    assert replies == before
    assert len(calls) == len(replies) and not responses
    requests = [json.loads(c["messages"][1]["content"]) for c in calls]
    assert result["execution_usage"] == {
        "model_requests": len(replies),
        "tool_calls": 0,
    }
    receipts = [json.loads(p.read_text()) for p in role.output.glob("*/response.json")]
    failures = [json.loads(p.read_text()) for p in role.output.glob("*/failure.json")]
    assert len(receipts) + len(failures) == len(replies)
    assert all(f["error_type"] == "JSONDecodeError" for f in failures)
    assert sorted(
        [r["raw_output"] for r in receipts] + [f["partial_output"] for f in failures]
    ) == sorted(
        r if isinstance(r, str) else json.dumps(r, ensure_ascii=False) for r in replies
    )
    return result, requests
