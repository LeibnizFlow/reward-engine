"""Replay the exact error data lines captured in v3; no live service access."""

import json

import pytest

from tests.integration.test_gateway_litellm_unavailable import (
    PARTIAL,
    SUCCESS,
    event,
    exercise,
    failure,
    forwarded_data,
)

# Source frames (including param and null/code representation) are preserved in
# .aris/traces/experiment-audit/2026-10-10_run01/005-gateway.observed-frames.json.
OBSERVED = [
    (
        'data: {"error":{"message":"Service temporarily unavailable","type":"api_error","param":"","code":null}}\n',
        "provider_api_service_unavailable",
    ),
    (
        'data: {"error":{"message":"Stream error occurred","type":"upstream_error","param":"","code":"error"}}\n',
        "provider_upstream_stream_error",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("frame,reason", OBSERVED)
async def test_captured_error_retries_and_discards_all_partial_tool_output(
    tmp_path, frame, reason
):
    response, receipt, root, requests, waits = await exercise(
        tmp_path,
        [(502, "header retry"), PARTIAL + frame, SUCCESS],
    )
    error = json.loads(frame[5:])["error"]
    assert len(requests) == 3 and waits == [5, 15]
    assert receipt["status"] == "completed"
    assert forwarded_data(response) == [
        line for line in SUCCESS.splitlines() if line.startswith("data:")
    ]
    assert response.text.count('"id": "call_same"') == 1
    assert "discard" not in response.text
    assert (root / "response.sse").read_text() == SUCCESS
    assert (root / "response-1.sse").read_text() == PARTIAL + frame
    assert receipt["provider_errors"] == [
        {
            "attempt": 2,
            "stream_attempt": 1,
            "code": error["code"],
            "type": error["type"],
            "message": error["message"],
            "reason": reason,
        }
    ]
    assert receipt["retries"][1]["attempt"] == 2
    assert receipt["retries"][1]["error_code"] == error["code"]
    assert receipt["retries"][1]["reason"] == reason


@pytest.mark.asyncio
@pytest.mark.parametrize("frame,reason", OBSERVED)
async def test_captured_error_exhaustion_preserves_six_frames_without_tool_forwarding(
    tmp_path, frame, reason
):
    response, receipt, root, requests, waits = await exercise(
        tmp_path, [PARTIAL + frame] * 6
    )
    assert len(requests) == 6 and waits == [15, 30, 60, 60, 60]
    assert receipt["status"] == "failed"
    assert receipt["error_reason"] == "Provider transport retries exhausted"
    assert len(receipt["retries"]) == 5
    assert len(receipt["stream_attempts"]) == len(receipt["provider_errors"]) == 6
    for i, original in enumerate(receipt["provider_errors"], 1):
        assert original["attempt"] == original["stream_attempt"] == i
        assert original["reason"] == reason
        assert (root / f"response-{i}.sse").read_text() == PARTIAL + frame
    assert not (root / "response.sse").exists()
    assert forwarded_data(response) == [
        'data: {"error":{"message":"Campaign upstream stream failed"}}'
    ]
    assert "tool_calls" not in response.text and "discard" not in response.text


@pytest.mark.asyncio
@pytest.mark.parametrize("frame,reason", OBSERVED)
@pytest.mark.parametrize(
    "change",
    [
        {"code": "authentication_error"},
        {"code": "content_policy_violation"},
        {"type": "invalid_request_error"},
        {
            "message": "Invalid request: Service temporarily unavailable; Stream error occurred"
        },
    ],
)
async def test_nearby_errors_are_not_transient_matches(tmp_path, frame, reason, change):
    error = json.loads(frame[5:])["error"]
    error.update(change)
    response, receipt, _, requests, waits = await exercise(
        tmp_path, [PARTIAL + event({"error": error})]
    )
    assert len(requests) == 1 and waits == []
    assert receipt["status"] == "failed" and not receipt.get("retries")
    assert receipt["provider_errors"][0]["reason"] == "provider_stream_error"
    assert "tool_calls" not in response.text


@pytest.mark.asyncio
async def test_generic_error_code_alone_is_not_retryable(tmp_path):
    _, receipt, _, requests, waits = await exercise(
        tmp_path,
        [
            failure("error", error_type="upstream_error", message="Permission denied"),
        ],
    )
    assert len(requests) == 1 and waits == []
    assert receipt["status"] == "failed" and not receipt.get("retries")


@pytest.mark.asyncio
async def test_captured_errors_share_existing_budget_with_other_transport_errors(
    tmp_path,
):
    frames = [
        PARTIAL,
        failure(),
        OBSERVED[0][0],
        OBSERVED[1][0],
        OBSERVED[0][0],
        OBSERVED[1][0],
    ]
    response, receipt, _, requests, waits = await exercise(tmp_path, frames)
    assert len(requests) == 6 and waits == [15, 30, 60, 60, 60]
    assert receipt["error_reason"] == "Provider transport retries exhausted"
    assert [r["retry_reason"] for r in receipt["stream_attempts"]] == [
        "incomplete_stream",
        "provider_litellm_unavailable",
        OBSERVED[0][1],
        OBSERVED[1][1],
        OBSERVED[0][1],
        OBSERVED[1][1],
    ]
    assert "tool_calls" not in response.text


@pytest.mark.asyncio
@pytest.mark.parametrize("frame,reason", OBSERVED)
async def test_error_shaped_semantic_output_is_forwarded_without_retry(
    tmp_path, frame, reason
):
    raw = (
        event(
            {
                "model": "model",
                "choices": [
                    {"index": 0, "delta": {"content": frame}, "finish_reason": "stop"}
                ],
            }
        )
        + "data: [DONE]\n\n"
    )
    response, receipt, _, requests, waits = await exercise(tmp_path, [raw])
    assert response.text == raw
    assert len(requests) == 1 and waits == []
    assert receipt["status"] == "completed" and not receipt.get("provider_errors")
