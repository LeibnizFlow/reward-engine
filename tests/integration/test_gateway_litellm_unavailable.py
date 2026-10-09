"""Explicit transient SSE errors retry locally before any model output escapes."""

import json

import httpx
import pytest

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
    message="模型服务暂时不可用",
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


@pytest.mark.asyncio
async def test_unavailable_discards_partial_answer_and_forwards_tool_action_once(
    tmp_path,
):
    response, receipt, root, requests, waits = await exercise(
        tmp_path, [PARTIAL + failure(), SUCCESS]
    )
    assert len(requests) == 2 and waits == [15]
    assert receipt["status"] == "completed"
    assert forwarded_data(response) == [
        line for line in SUCCESS.splitlines() if line.startswith("data:")
    ]
    assert response.text.count('"id": "call_same"') == 1
    assert "discard" not in response.text
    assert (root / "response.sse").read_text() == SUCCESS
    # The error data line is logged; parsing stops before its trailing blank line.
    assert (root / "response-1.sse").read_text() == (PARTIAL + failure())[:-1]
    original = receipt["provider_errors"][0]
    assert original == {
        "attempt": 1,
        "stream_attempt": 1,
        "code": "LITELLM_UNAVAILABLE",
        "type": "new_api_error",
        "message": "模型服务暂时不可用",
        "reason": "provider_litellm_unavailable",
    }
    assert receipt["retries"][0]["error_code"] == "LITELLM_UNAVAILABLE"
    assert receipt["retries"][0]["attempt"] == 1
    assert (
        receipt["stream_attempts"][0]["retry_reason"] == "provider_litellm_unavailable"
    )


@pytest.mark.asyncio
async def test_unavailable_exhausts_existing_six_attempt_budget_without_forwarding_tools(
    tmp_path,
):
    response, receipt, root, requests, waits = await exercise(
        tmp_path, [PARTIAL + failure()] * 6
    )
    assert len(requests) == 6 and waits == [15, 30, 60, 60, 60]
    assert receipt["status"] == "failed"
    assert receipt["error_reason"] == "Provider transport retries exhausted"
    assert len(receipt["retries"]) == 5
    assert [row["attempt"] for row in receipt["provider_errors"]] == list(range(1, 7))
    assert len(receipt["stream_attempts"]) == 6
    assert all(
        row["error_code"] == "LITELLM_UNAVAILABLE" for row in receipt["stream_attempts"]
    )
    assert forwarded_data(response) == [
        'data: {"error":{"message":"Campaign upstream stream failed"}}'
    ]
    assert "discard" not in response.text and "tool_calls" not in response.text
    assert not (root / "response.sse").exists()
    assert len(list(root.glob("response-*.sse"))) == 6


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code",
    ["LITELLM_AUTH_FAILED", "content_policy_violation", "litellm_unavailable", "", 503],
)
@pytest.mark.parametrize("error_type", ["new_api_error", "rate_limit_error"])
async def test_other_explicit_codes_never_retry_even_with_transient_sounding_message(
    tmp_path, code, error_type
):
    message = "The model service is temporarily unavailable; stream ended before the response was complete."
    raw = PARTIAL + failure(code, error_type=error_type, message=message)
    response, receipt, root, requests, waits = await exercise(tmp_path, [raw])
    assert len(requests) == 1 and waits == []
    assert receipt["status"] == "failed" and not receipt.get("retries")
    assert receipt["provider_errors"][0]["code"] == code
    assert receipt["provider_errors"][0]["message"] == message
    assert receipt["provider_errors"][0]["attempt"] == 1
    assert receipt["provider_errors"][0]["reason"] == "provider_stream_error"
    assert "tool_calls" not in response.text and "discard" not in response.text
    assert not (root / "response.sse").exists()
    assert (root / "response-1.sse").read_text() == raw[:-1]


@pytest.mark.asyncio
async def test_error_code_does_not_reset_incomplete_stream_budget(tmp_path):
    response, receipt, _, requests, waits = await exercise(
        tmp_path, [PARTIAL] + [failure()] * 5
    )
    assert len(requests) == 6 and waits == [15, 30, 60, 60, 60]
    assert receipt["stream_attempts"][0]["retry_reason"] == "incomplete_stream"
    assert receipt["error_reason"] == "Provider transport retries exhausted"
    assert "tool_calls" not in response.text


@pytest.mark.asyncio
async def test_receipt_preserves_physical_attempt_after_header_retry(tmp_path):
    _, receipt, _, requests, waits = await exercise(
        tmp_path, [(502, "upstream failed"), failure(), SUCCESS]
    )
    assert len(requests) == 3 and waits == [5, 15]
    assert receipt["status"] == "completed"
    assert receipt["provider_errors"][0]["attempt"] == 2
    assert receipt["provider_errors"][0]["stream_attempt"] == 1
    assert receipt["retries"][1]["attempt"] == 2
    assert receipt["retries"][1]["error_code"] == "LITELLM_UNAVAILABLE"


@pytest.mark.asyncio
async def test_completed_semantic_output_never_triggers_transport_retry(tmp_path):
    raw = (
        event(
            {
                "model": "model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": "Wrong answer. LITELLM_UNAVAILABLE 模型服务暂时不可用"
                        },
                        "finish_reason": "stop",
                    }
                ],
            }
        )
        + "data: [DONE]\n\n"
    )
    response, receipt, _, requests, waits = await exercise(tmp_path, [raw])
    assert len(requests) == 1 and waits == []
    assert receipt["status"] == "completed" and not receipt.get("provider_errors")
    assert response.text == raw
