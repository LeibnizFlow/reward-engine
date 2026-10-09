"""Replay captured service errors through real Role/gateway/checker boundaries."""

import json
from pathlib import Path

import httpx
import pytest

from reward_engine.adapters.workbuddy import models
from reward_engine.adapters.workbuddy.gateway import application
from reward_engine.adapters.workbuddy.io import read, write
from reward_engine.core.outcome.context import build_context
from reward_engine.core.outcome.harness import Registry, defaults

CAPTURES = read(
    Path(__file__).parents[1] / "fixtures/workbuddy/provider_stream_errors.json"
)["captures"]


@pytest.fixture
def make_role(tmp_path, monkeypatch):
    client = httpx.AsyncClient

    def make(responses, *, gateway=False):
        calls, waits = [], []
        config = {
            "output": str(tmp_path),
            "roles": {"reward_calibration_judge": "deepseek-v4-flash"},
            "auth_file": str(tmp_path / "auth"),
            "gateway_local": "http://offline",
            "backend_url": "http://provider.invalid/v1",
            "model_concurrency": 1,
        }
        (tmp_path / "auth").write_text("offline-token")

        def upstream(request):
            calls.append(json.loads(request.content))
            assert responses, "Unexpected retry beyond the fixture budget"
            return responses.pop(0)

        def backend(**kwargs):
            return client(transport=httpx.MockTransport(upstream), **kwargs)

        if gateway:

            async def retry_wait(delay):
                waits.append(delay)

            app = application(
                config,
                backend_key="offline",
                client_factory=backend,
                retry_wait=retry_wait,
            )

            def factory(**kwargs):
                return client(transport=httpx.ASGITransport(app), **kwargs)
        else:
            factory = backend
        monkeypatch.setattr(models.httpx, "AsyncClient", factory)
        return (
            models.Role(config, "reward_calibration_judge", tmp_path / "role"),
            calls,
            waits,
        )

    return make


def error_response(capture):
    return httpx.Response(
        200,
        text=capture["sse_line"] + "\n\n",
        headers={"content-type": "text/event-stream"},
    )


@pytest.mark.parametrize("capture", CAPTURES, ids=lambda row: row["name"])
@pytest.mark.parametrize("gateway", [False, True], ids=["direct", "real_gateway"])
@pytest.mark.asyncio
async def test_captured_service_error_is_infrastructure_unknown_and_keeps_receipts(
    make_role, tmp_path, capture, gateway
):
    attempts = 6 if gateway else 1
    role, calls, waits = make_role(
        [error_response(capture) for _ in range(attempts)], gateway=gateway
    )
    context, evidence = build_context(
        {"task_id": "t", "prompt": "Check delivered answer"},
        [{"artifact": {"final_response": "Delivered answer"}}],
    )
    result = await defaults(role).run(
        {"test_id": "check", "criterion_id": "c", "checker": "evidence.review"},
        context,
        evidence,
    )
    assert result["status"] == "unknown" and result["error"] is True
    assert result["failure_layer"] == "infrastructure"
    assert result["reason"] == "Checker failed: ProviderStreamError"
    # Only the existing gateway retry budget; no Role retry or contract repair.
    assert len(calls) == attempts
    assert waits == ([15, 30, 60, 60, 60] if gateway else [])
    failure = read(next(role.output.glob("*/failure.json")))
    assert failure["error_type"] == "ProviderStreamError"
    assert "http_status" not in failure
    assert not list((role.output / "completed").glob("*.json"))
    assert not list(role.output.glob("*/response.json"))
    if gateway:
        assert failure["provider_error_event"] == {
            "error": {"message": "Campaign upstream stream failed"}
        }
        transport = next((tmp_path / "transport/roles").glob("*/receipt.json"))
        receipt = read(transport)
        assert receipt["status"] == "failed"
        assert len(receipt["provider_errors"]) == attempts
        assert "upstream_http_status" not in receipt
        assert {
            key: receipt["provider_errors"][0][key]
            for key in ("code", "type", "message")
        } == {
            key: capture["event"]["error"][key] for key in ("code", "type", "message")
        }
        for attempt in range(1, attempts + 1):
            assert (
                transport.parent / f"response-{attempt}.sse"
            ).read_text().splitlines()[0] == capture["sse_line"]
    else:
        assert failure["provider_error_event"] == capture["event"]


@pytest.mark.asyncio
async def test_checker_runtime_error_remains_evaluation_execution():
    async def broken_checker(test, context, evidence):
        # Even the old provider message cannot turn a checker bug into transport.
        raise RuntimeError("Provider returned a stream error")

    registry = Registry()
    registry.register("broken", broken_checker, "Deliberately failing local checker")
    context, evidence = build_context({"task_id": "t", "prompt": "Check delivery"}, [])
    result = await registry.run(
        {"test_id": "check", "criterion_id": "c", "checker": "broken"},
        context,
        evidence,
    )
    assert result["status"] == "unknown" and result["error"] is True
    assert result["failure_layer"] == "evaluation_execution"
    assert result["reason"] == "Checker failed: RuntimeError"


@pytest.mark.parametrize("recovers", [False, True])
@pytest.mark.asyncio
async def test_typed_error_preserves_one_opt_in_recovery_and_original_receipt(
    make_role, recovers
):
    capture = CAPTURES[0]
    recovered = {
        "model": "deepseek-v4-flash",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": '{"status":"unknown"}'},
            }
        ],
    }
    responses = [
        error_response(capture),
        httpx.Response(200, json=recovered if recovers else capture["event"]),
    ]
    role, calls, _ = make_role(responses)
    role.nonstream_recovery_models = {"deepseek-v4-flash"}
    if recovers:
        assert await role({"mode": "test"}) == {"status": "unknown"}
        assert await role({"mode": "test"}) == {"status": "unknown"}
    else:
        with pytest.raises(ValueError, match="Incomplete or mismatched"):
            await role({"mode": "test"})
        assert not list((role.output / "completed").glob("*.json"))
    assert len(calls) == 2
    assert [body["stream"] for body in calls] == [True, False]
    transport = read(next(role.output.glob("*/transport.json")))
    original = read(Path(transport["original_failure"]))
    assert original["error_type"] == "ProviderStreamError"
    assert original["provider_error_event"] == capture["event"]
    assert read(next(role.output.glob("*/raw_response.json"))) == (
        recovered if recovers else capture["event"]
    )


@pytest.mark.parametrize("slot", ["seeds/1", "round_1/actors/A"])
@pytest.mark.parametrize("error_type", ["ProviderStreamError", "RuntimeError"])
def test_claw_recovery_retains_receipts_and_does_not_admit_generic_runtime_errors(
    tmp_path, slot, error_type
):
    from experiments.claw_eval.opencode.campaign import recover_failed_canary

    task = tmp_path / "tasks/task"
    failure = {"error_type": error_type, "reason": "Provider returned a stream error"}
    write(task / "state.json", {"status": "failed", **failure})
    path = task / slot / "FAILURE.json"
    write(path, failure)
    original = path.read_bytes()
    cache = task / "round_1/models/jury/completed/cache.json"
    write(cache, {"answer": {"status": "unknown"}})
    cached = cache.read_bytes()
    if error_type == "RuntimeError":
        with pytest.raises(RuntimeError, match="Refusing to recover"):
            recover_failed_canary(tmp_path, "task")
        assert path.read_bytes() == original
    else:
        assert recover_failed_canary(tmp_path, "task") is True
        archive = next(
            (tmp_path / "implementation_revisions").glob("*/task_recovery/task")
        )
        assert (archive / slot / "FAILURE.json").read_bytes() == original
        assert not path.exists()
    assert cache.read_bytes() == cached


def test_claw_capacity_recognizes_only_typed_stream_failure_before_completed_rounds():
    from experiments.claw_eval.opencode.capacity import (
        retryable_task_error,
        retryable_task_result,
    )

    error = models.ProviderStreamError(CAPTURES[0]["event"])
    assert isinstance(error, RuntimeError)
    assert not hasattr(error, "status_code")
    assert retryable_task_error(error)
    assert not retryable_task_error(RuntimeError(str(error)))
    result = {
        "status": "failed",
        "completed_rounds": 0,
        "construction_failure": {"error_type": "ProviderStreamError"},
    }
    assert retryable_task_result(result)
    assert not retryable_task_result({**result, "completed_rounds": 1})
    assert not retryable_task_result(
        {**result, "construction_failure": {"error_type": "RuntimeError"}}
    )


@pytest.mark.parametrize("attempts", [0, 2])
@pytest.mark.asyncio
async def test_claw_task_recovery_keeps_existing_two_retry_limit(
    tmp_path, monkeypatch, attempts
):
    from experiments.claw_eval.opencode import campaign

    task = tmp_path / "tasks/task"
    result = {
        "status": "failed",
        "completed_rounds": 0,
        "construction_failure": {"error_type": "ProviderStreamError"},
    }
    write(task / "result.private.json", result)
    for index in range(attempts):
        archive = tmp_path / f"archive-{index}"
        write(archive / "state.json", {"error_type": "ProviderStreamError"})
        write(task / f"recovery_events/{index}.json", {"archived_to": str(archive)})

    class ReachedFreshTask(Exception):
        pass

    def stop_before_execution(*args):
        raise ReachedFreshTask

    monkeypatch.setattr(campaign, "public_task", stop_before_execution)
    if attempts == 0:
        with pytest.raises(ReachedFreshTask):
            await campaign.task_run({"output": str(tmp_path)}, None, "task")
        archive = next(
            (tmp_path / "implementation_revisions").glob("*/task_recovery/task")
        )
        assert read(archive / "result.private.json") == result
    else:
        assert (
            await campaign.task_run({"output": str(tmp_path)}, None, "task") == result
        )
        assert read(task / "result.private.json") == result
        assert not (tmp_path / "implementation_revisions").exists()
