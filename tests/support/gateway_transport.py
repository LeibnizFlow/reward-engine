"""Explicit transient SSE errors retry locally before any model output escapes."""

import json

import httpx

from reward_engine.adapters.workbuddy.gateway import application


def event(value):
    return "data: " + json.dumps(value, ensure_ascii=False) + "\n\n"


def tool_chunk(arguments, *, content="", finish=None):
    return event(
        {
            "model": "model",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": content,
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_same",
                                "type": "function",
                                "function": {"name": "shell", "arguments": arguments},
                            }
                        ],
                    },
                    "finish_reason": finish,
                }
            ],
        }
    )


def failure(
    code="LITELLM_UNAVAILABLE",
    *,
    error_type="new_api_error",
    message="Model service temporarily unavailable",
):
    return event({"error": {"type": error_type, "code": code, "message": message}})


PARTIAL = tool_chunk('{"command":"discard me"}', content="discarded answer")
SUCCESS = (
    tool_chunk('{"command":"keep me"}', content="final answer", finish="tool_calls")
    + "data: [DONE]\n\n"
)


async def exercise(tmp_path, replies):
    token = tmp_path / "auth"
    token.write_text("local-test-only")
    requests, responses, waits = [], [], []

    async def upstream(request):
        index = len(requests)
        requests.append(json.loads(request.content))
        assert index < len(replies), "Retry exceeded the scripted transport budget"
        reply = replies[index]
        status, body = reply if isinstance(reply, tuple) else (200, reply)
        response = httpx.Response(status, content=body.encode())
        responses.append(response)
        return response

    async def no_wait(seconds):
        waits.append(seconds)

    config = {
        "output": str(tmp_path),
        "auth_file": str(token),
        "backend_url": "https://unused.invalid/v1",
        "model_concurrency": 1,
        "roles": {"actor": "model"},
    }
    app = application(
        config,
        backend_key="local-test-only",
        retry_wait=no_wait,
        client_factory=lambda **kwargs: httpx.AsyncClient(
            transport=httpx.MockTransport(upstream),
            **kwargs,
        ),
    )
    body = {
        "model": "model",
        "messages": [{"role": "user", "content": "task"}],
        "stream": True,
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://local"
    ) as client:
        response = await client.post(
            "/roles/v1/chat/completions",
            headers={"Authorization": "Bearer local-test-only"},
            json=body,
        )
        health = (await client.get("/health")).json()
    assert response.status_code == 200  # Stream errors are SSE, not HTTP retries.
    assert health["inflight"] == 0 and health["model_inflight"] == {"model": 0}
    assert all(response.is_closed for response in responses)
    assert requests and all(request == body for request in requests)
    root = next(tmp_path.glob("transport/roles/*"))
    receipt = json.loads((root / "receipt.json").read_text())
    return response, receipt, root, requests, waits


def forwarded_data(response):
    return [line for line in response.text.splitlines() if line.startswith("data:")]
