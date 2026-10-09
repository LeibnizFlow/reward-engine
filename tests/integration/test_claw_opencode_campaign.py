"""Contracts for the local OpenCode/native Claw bridge (no paid requests)."""

import json
from types import SimpleNamespace

import httpx
import pytest

from experiments.claw_eval.opencode.campaign import (
    corrected_failure_retry_limit,
    recover_failed_canary,
    substantive_recovery_count,
)
from experiments.claw_eval.opencode.capacity import (
    retryable_task_error,
    retryable_task_result,
    transport_window,
)
from experiments.claw_eval.opencode.monitor import campaign_process_state
from experiments.claw_eval.opencode.runtime import Actor, Runtime, opencode_config
from reward_engine.adapters.workbuddy.gateway import application
from reward_engine.adapters.workbuddy.io import digest, write


@pytest.mark.asyncio
async def test_gateway_opt_in_capacity_keeps_legacy_default(tmp_path):
    token = tmp_path / "auth"
    token.write_text("test")
    cfg = {
        "output": str(tmp_path),
        "auth_file": str(token),
        "roles": {"actor": "model"},
        "model_concurrency": 8,
        "per_model_concurrency": 8,
        "backend_url": "https://unused",
    }
    app = application(cfg, backend_key="unused")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://local"
    ) as client:
        assert (
            await client.post(
                "/control/concurrency",
                headers={"Authorization": "Bearer test"},
                json={"global_limit": 12},
            )
        ).status_code == 400
    cfg["max_model_concurrency"] = 128
    app = application(cfg, backend_key="unused")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://local"
    ) as client:
        assert (
            await client.post("/control/concurrency", json={"global_limit": 12})
        ).status_code == 401
        result = await client.post(
            "/control/concurrency",
            headers={"Authorization": "Bearer test"},
            json={"global_limit": 12},
        )
        assert result.json()["limit"] == 12


@pytest.mark.asyncio
async def test_gateway_retries_retryable_http_twice_with_recorded_backoff(tmp_path):
    token = tmp_path / "auth"
    token.write_text("test")
    attempts, waits = [], []

    async def upstream(request):
        attempts.append(request)
        if len(attempts) < 3:
            return httpx.Response(502, text="temporary")
        return httpx.Response(200, json={"model": "model", "choices": []})

    async def no_wait(seconds):
        waits.append(seconds)

    config = {
        "output": str(tmp_path),
        "auth_file": str(token),
        "backend_url": "https://fake/v1",
        "model_concurrency": 1,
        "roles": {"actor": "model"},
    }
    app = application(
        config,
        backend_key="unused",
        retry_wait=no_wait,
        client_factory=lambda **kw: httpx.AsyncClient(
            transport=httpx.MockTransport(upstream), **kw
        ),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://local"
    ) as client:
        response = await client.post(
            "/roles/v1/chat/completions",
            headers={"Authorization": "Bearer test"},
            json={"model": "model", "messages": []},
        )
    receipt = next(tmp_path.glob("transport/*/*/receipt.json"))
    recorded = json.loads(receipt.read_text())
    assert response.status_code == 200
    assert len(attempts) == 3 and waits == [5, 15]
    assert recorded["attempts"] == 3
    assert [row["status"] for row in recorded["http_errors"]] == [502, 502]


@pytest.mark.asyncio
async def test_gateway_retries_read_timeout_before_response_is_exposed(tmp_path):
    token = tmp_path / "auth"
    token.write_text("test")
    attempts, waits = [], []

    async def upstream(request):
        attempts.append(request)
        if len(attempts) == 1:
            raise httpx.ReadTimeout("headers delayed", request=request)
        return httpx.Response(200, json={"model": "model", "choices": []})

    async def no_wait(seconds):
        waits.append(seconds)

    config = {
        "output": str(tmp_path),
        "auth_file": str(token),
        "backend_url": "https://fake/v1",
        "model_concurrency": 1,
        "roles": {"actor": "model"},
    }
    app = application(
        config,
        backend_key="unused",
        retry_wait=no_wait,
        client_factory=lambda **kw: httpx.AsyncClient(
            transport=httpx.MockTransport(upstream), **kw
        ),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://local"
    ) as client:
        response = await client.post(
            "/roles/v1/chat/completions",
            headers={"Authorization": "Bearer test"},
            json={"model": "model", "messages": []},
        )
    recorded = json.loads(next(tmp_path.glob("transport/*/*/receipt.json")).read_text())
    assert response.status_code == 200 and len(attempts) == 2 and waits == [5]
    assert recorded["attempts"] == 2
    assert recorded["retries"][0]["reason"] == "ReadTimeout"


@pytest.mark.asyncio
async def test_gateway_retries_temporary_provider_stream_error(tmp_path):
    token = tmp_path / "auth"
    token.write_text("test")
    attempts, waits = [], []

    async def upstream(request):
        attempts.append(request)
        if len(attempts) == 1:
            body = (
                'data: {"error":{"type":"new_api_error",'
                '"message":"The model service is temporarily unavailable."}}\n\n'
            )
        else:
            body = (
                'data: {"model":"model","choices":[{"index":0,'
                '"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n'
                "data: [DONE]\n\n"
            )
        return httpx.Response(200, content=body.encode())

    async def no_wait(seconds):
        waits.append(seconds)

    config = {
        "output": str(tmp_path),
        "auth_file": str(token),
        "backend_url": "https://fake/v1",
        "model_concurrency": 1,
        "roles": {"actor": "model"},
    }
    app = application(
        config,
        backend_key="unused",
        retry_wait=no_wait,
        client_factory=lambda **kw: httpx.AsyncClient(
            transport=httpx.MockTransport(upstream), **kw
        ),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://local"
    ) as client:
        response = await client.post(
            "/roles/v1/chat/completions",
            headers={"Authorization": "Bearer test"},
            json={"model": "model", "messages": [], "stream": True},
        )

    recorded = json.loads(next(tmp_path.glob("transport/*/*/receipt.json")).read_text())
    assert response.status_code == 200 and "ok" in response.text
    assert len(attempts) == 2 and waits == [15]
    assert recorded["status"] == "completed"
    assert recorded["retries"][0]["reason"] == "provider_temporarily_unavailable"


@pytest.mark.asyncio
async def test_gateway_retries_provider_incomplete_stream_error(tmp_path):
    token = tmp_path / "auth"
    token.write_text("test")
    attempts, waits = [], []

    async def upstream(request):
        attempts.append(request)
        if len(attempts) == 1:
            body = (
                'data: {"error":{"type":"new_api_error",'
                '"message":"The upstream stream ended before the response was complete."}}\n\n'
            )
        else:
            body = (
                'data: {"model":"model","choices":[{"index":0,'
                '"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n'
                "data: [DONE]\n\n"
            )
        return httpx.Response(200, content=body.encode())

    async def no_wait(seconds):
        waits.append(seconds)

    config = {
        "output": str(tmp_path),
        "auth_file": str(token),
        "backend_url": "https://fake/v1",
        "model_concurrency": 1,
        "roles": {"actor": "model"},
    }
    app = application(
        config,
        backend_key="unused",
        retry_wait=no_wait,
        client_factory=lambda **kw: httpx.AsyncClient(
            transport=httpx.MockTransport(upstream), **kw
        ),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://local"
    ) as client:
        response = await client.post(
            "/roles/v1/chat/completions",
            headers={"Authorization": "Bearer test"},
            json={"model": "model", "messages": [], "stream": True},
        )

    recorded = json.loads(next(tmp_path.glob("transport/*/*/receipt.json")).read_text())
    assert response.status_code == 200 and "ok" in response.text
    assert len(attempts) == 2 and waits == [15]
    assert recorded["status"] == "completed"
    assert recorded["retries"][0]["reason"] == "provider_incomplete_stream"


@pytest.mark.asyncio
async def test_gateway_retries_malformed_provider_stream_event(tmp_path):
    token = tmp_path / "auth"
    token.write_text("test")
    attempts, waits = [], []

    async def upstream(request):
        attempts.append(request)
        if len(attempts) == 1:
            body = 'data: {"choices":[{"index":0,data: {"error":"timeout"}}]}\n\n'
        else:
            body = (
                'data: {"model":"model","choices":[{"index":0,'
                '"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n'
                "data: [DONE]\n\n"
            )
        return httpx.Response(200, content=body.encode())

    async def no_wait(seconds):
        waits.append(seconds)

    config = {
        "output": str(tmp_path),
        "auth_file": str(token),
        "backend_url": "https://fake/v1",
        "model_concurrency": 1,
        "roles": {"actor": "model"},
    }
    app = application(
        config,
        backend_key="unused",
        retry_wait=no_wait,
        client_factory=lambda **kw: httpx.AsyncClient(
            transport=httpx.MockTransport(upstream), **kw
        ),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://local"
    ) as client:
        response = await client.post(
            "/roles/v1/chat/completions",
            headers={"Authorization": "Bearer test"},
            json={"model": "model", "messages": [], "stream": True},
        )

    recorded = json.loads(next(tmp_path.glob("transport/*/*/receipt.json")).read_text())
    assert response.status_code == 200 and "ok" in response.text
    assert len(attempts) == 2 and waits == [15]
    assert recorded["status"] == "completed"
    assert recorded["retries"][0]["reason"] == "malformed_stream"


def test_capacity_requeues_only_recoverable_task_outcomes():
    assert retryable_task_result(
        {"status": "partial", "completed_rounds": 0, "stop_reason": "incomplete_arms"}
    )
    assert retryable_task_result(
        {
            "status": "failed",
            "completed_rounds": 0,
            "construction_failure": {"error_type": "ReadTimeout"},
        }
    )
    assert not retryable_task_result({"status": "completed", "completed_rounds": 1})
    assert retryable_task_error(TimeoutError("actor timeout"))
    assert retryable_task_error(RuntimeError("docker exited 137"))
    assert retryable_task_error(
        ValueError("Cannot reuse a changed or failed Actor slot")
    )
    assert not retryable_task_error(ValueError("bad task data"))


def test_failed_zero_round_canary_is_archived_without_touching_caches(tmp_path):
    task = tmp_path / "tasks/task"
    write(task / "result.private.json", {"status": "failed", "completed_rounds": 0})
    write(task / "state.json", {"status": "failed"})
    write(task / "round_1/construction_failure.json", {"stage": "jury"})
    write(task / "seeds/1/RESULT.json", {"status": "completed"})
    write(task / "round_1/models/jury_member/completed/cache.json", {"answer": {}})
    assert recover_failed_canary(tmp_path, "task")
    assert not (task / "result.private.json").exists()
    assert (task / "seeds/1/RESULT.json").exists()
    assert (task / "round_1/models/jury_member/completed/cache.json").exists()
    archive = next((tmp_path / "implementation_revisions").glob("*/task_recovery/task"))
    assert (archive / "result.private.json").exists()
    assert (archive / "round_1/construction_failure.json").exists()
    assert not recover_failed_canary(tmp_path, "task")


def test_pre_session_dependency_failure_is_recoverable_without_legacy_runner(tmp_path):
    task = tmp_path / "tasks/task"
    write(task / "state.json", {"status": "running"})
    for arm in "ABC":
        slot = task / "round_1/actors" / arm
        write(slot / "STARTED.json", {"identity": arm})
        write(
            slot / "FAILURE.json",
            {
                "error_type": "ModuleNotFoundError",
                "reason": "No module named 'swarm_reward'",
                "session": None,
            },
        )
    assert recover_failed_canary(tmp_path, "task")
    assert not (task / "round_1/actors/A").exists()
    archive = next((tmp_path / "implementation_revisions").glob("*/task_recovery/task"))
    assert (archive / "round_1/actors/A/FAILURE.json").exists()

    config = opencode_config("model", "http://gateway/channel", "secret", 200000, 32768)
    assert config["model"] == "campaign/model"
    assert (
        config["provider"]["campaign"]["options"]["baseURL"]
        == "http://gateway/channel/v1"
    )
    assert config["provider"]["campaign"]["models"]["model"]["limit"]["output"] == 32768


def test_failed_seed_timeout_is_archived_for_infrastructure_retry(tmp_path):
    task = tmp_path / "tasks/task"
    write(task / "state.json", {"status": "failed", "error_type": "TimeoutError"})
    write(task / "seeds/1/STARTED.json", {"identity": "seed"})
    write(
        task / "seeds/1/FAILURE.json",
        {"error_type": "TimeoutError", "reason": "", "session": None},
    )
    write(task / "seeds/2/RESULT.json", {"status": "completed"})

    assert recover_failed_canary(tmp_path, "task")
    assert not (task / "seeds/1").exists()
    assert (task / "seeds/2/RESULT.json").exists()
    archive = next((tmp_path / "implementation_revisions").glob("*/task_recovery/task"))
    assert (archive / "seeds/1/FAILURE.json").exists()


def test_failed_seed_docker_exit_is_archived_for_infrastructure_retry(tmp_path):
    task = tmp_path / "tasks/task"
    write(
        task / "state.json",
        {
            "status": "failed",
            "error_type": "RuntimeError",
            "reason": "docker exited 1: ",
        },
    )
    write(task / "seeds/1/STARTED.json", {"identity": "seed"})
    write(
        task / "seeds/1/FAILURE.json",
        {"error_type": "RuntimeError", "reason": "docker exited 1: ", "session": None},
    )

    assert recover_failed_canary(tmp_path, "task")
    assert not (task / "seeds/1").exists()
    archive = next((tmp_path / "implementation_revisions").glob("*/task_recovery/task"))
    assert (archive / "seeds/1/FAILURE.json").exists()


def test_failed_actor_timeout_preserves_completed_sibling(tmp_path):
    task = tmp_path / "tasks/task"
    write(task / "state.json", {"status": "partial", "completed_rounds": 0})
    write(
        task / "result.private.json",
        {
            "status": "partial",
            "completed_rounds": 0,
            "stop_reason": "incomplete_arms",
        },
    )
    write(task / "round_1/actors/A/RESULT.json", {"status": "completed"})
    write(task / "round_1/actors/C/STARTED.json", {"identity": "arm-c"})
    write(
        task / "round_1/actors/C/FAILURE.json",
        {"error_type": "TimeoutError", "reason": "", "session": "session-c"},
    )

    assert recover_failed_canary(tmp_path, "task")
    assert (task / "round_1/actors/A/RESULT.json").exists()
    assert not (task / "round_1/actors/C").exists()
    archive = next((tmp_path / "implementation_revisions").glob("*/task_recovery/task"))
    assert (archive / "round_1/actors/C/FAILURE.json").exists()


def test_explicitly_interrupted_actor_slot_is_recoverable(tmp_path):
    task = tmp_path / "tasks/task"
    write(task / "state.json", {"status": "partial", "completed_rounds": 0})
    write(
        task / "result.private.json",
        {
            "status": "partial",
            "completed_rounds": 0,
            "stop_reason": "incomplete_arms",
            "rounds": [
                {
                    "arms": {
                        "B": {
                            "status": "failed",
                            "error_type": "RuntimeError",
                            "reason": "Interrupted Actor slot needs inspection; not resampling",
                        }
                    }
                }
            ],
        },
    )
    write(task / "round_1/actors/A/RESULT.json", {"status": "completed"})
    write(task / "round_1/actors/B/STARTED.json", {"identity": "arm-b"})
    write(task / "round_1/actors/B/actor_state.json", {"turn": 2})

    assert recover_failed_canary(tmp_path, "task")
    assert (task / "round_1/actors/A/RESULT.json").exists()
    assert not (task / "round_1/actors/B").exists()
    archive = next((tmp_path / "implementation_revisions").glob("*/task_recovery/task"))
    assert (archive / "round_1/actors/B/actor_state.json").exists()


def test_user_criterion_provenance_repair_gets_one_follow_up_retry(tmp_path):
    task = tmp_path / "task"
    write(
        task / "result.private.json",
        {
            "construction_failure": {
                "error": "User criterion must cite an actual user statement"
            }
        },
    )
    assert corrected_failure_retry_limit(task) == 5


def test_cancelled_slots_are_recoverable_after_graceful_campaign_stop(tmp_path):
    task = tmp_path / "tasks/task"
    write(
        task / "state.json",
        {"status": "failed", "error_type": "CancelledError", "reason": ""},
    )
    write(task / "seeds/1/RESULT.json", {"status": "completed"})
    write(
        task / "seeds/2/FAILURE.json",
        {"error_type": "CancelledError", "reason": "", "session": "seed-2"},
    )
    write(task / "round_1/actors/A/RESULT.json", {"status": "completed"})
    write(
        task / "round_1/actors/B/FAILURE.json",
        {"error_type": "CancelledError", "reason": "", "session": "arm-b"},
    )

    assert recover_failed_canary(tmp_path, "task")
    assert (task / "seeds/1/RESULT.json").exists()
    assert (task / "round_1/actors/A/RESULT.json").exists()
    assert not (task / "seeds/2").exists()
    assert not (task / "round_1/actors/B").exists()
    archive = next((tmp_path / "implementation_revisions").glob("*/task_recovery/task"))
    assert (archive / "seeds/2/FAILURE.json").exists()
    assert (archive / "round_1/actors/B/FAILURE.json").exists()


def test_administratively_interrupted_slots_preserve_completed_work(tmp_path):
    task = tmp_path / "tasks/task"
    write(
        task / "state.json",
        {
            "status": "failed",
            "error_type": "CampaignInterrupted",
            "reason": "controlled scheduler restart",
        },
    )
    write(task / "seeds/1/RESULT.json", {"status": "completed"})
    write(task / "seeds/2/STARTED.json", {"identity": "seed-2"})
    write(task / "round_1/actors/A/RESULT.json", {"status": "completed"})
    write(task / "round_1/actors/B/STARTED.json", {"identity": "arm-b"})

    assert recover_failed_canary(tmp_path, "task")
    assert (task / "seeds/1/RESULT.json").exists()
    assert (task / "round_1/actors/A/RESULT.json").exists()
    assert not (task / "seeds/2").exists()
    assert not (task / "round_1/actors/B").exists()
    archive = next((tmp_path / "implementation_revisions").glob("*/task_recovery/task"))
    assert (archive / "seeds/2/STARTED.json").exists()
    assert (archive / "round_1/actors/B/STARTED.json").exists()


def test_operational_restarts_do_not_consume_model_retry_budget(tmp_path):
    task = tmp_path / "tasks/task"
    states = [
        {"status": "failed", "completed_rounds": 0},
        {
            "status": "failed",
            "error_type": "CampaignInterrupted",
            "reason": "controlled restart",
        },
        {
            "status": "failed",
            "error_type": "ValueError",
            "reason": "Cannot reuse a changed or failed Actor slot",
        },
    ]
    for index, state in enumerate(states):
        archive = tmp_path / "archives" / str(index)
        write(archive / "state.json", state)
        write(
            task / "recovery_events" / f"{index}.json",
            {"archived_to": str(archive)},
        )

    assert substantive_recovery_count(task) == 1


@pytest.mark.asyncio
async def test_task_run_retries_bounded_length_construction_failure(
    tmp_path, monkeypatch
):
    from experiments.claw_eval.opencode import campaign

    task = tmp_path / "tasks/task"
    write(
        task / "result.private.json",
        {
            "status": "failed",
            "completed_rounds": 0,
            "construction_failure": {
                "error_type": "ValueError",
                "error": "Incomplete or mismatched role reply: finish=length, models={'model'}",
            },
        },
    )
    write(task / "state.json", {"status": "failed"})

    class StopAfterRecovery(Runtime):
        async def sample(self, task_id, output, **kwargs):
            raise RuntimeError("reached resumed execution")

    monkeypatch.setattr(campaign, "public_task", lambda *_: (None, {"task_id": "task"}))
    with pytest.raises(RuntimeError, match="reached resumed execution"):
        await campaign.task_run(
            {"output": str(tmp_path)}, StopAfterRecovery({}), "task"
        )
    assert not (task / "result.private.json").exists()
    assert len(list((task / "recovery_events").glob("*.json"))) == 1


def test_monitor_distinguishes_running_exited_and_completed(tmp_path):
    proc = tmp_path / "proc"
    output = tmp_path / "run"
    output.mkdir()
    write(output / "process.json", {"pid": 42})
    assert campaign_process_state(output, proc)["status"] == "exited"
    (proc / "42").mkdir(parents=True)
    (proc / "42/cmdline").write_bytes(
        (
            "python -m experiments.claw_eval.opencode.campaign run --output "
            + str(output)
        )
        .replace(" ", "\0")
        .encode()
    )
    assert campaign_process_state(output, proc) == {"status": "running", "pid": 42}
    write(output / "acceptance.json", {"accepted": True})
    assert campaign_process_state(output, proc) == {"status": "completed", "pid": 42}


@pytest.mark.asyncio
async def test_slots_never_redraw_interrupted_or_failed_actor(tmp_path):
    runtime = Runtime({"identity": "frozen"})
    slot = tmp_path / "actor"
    slot.mkdir()
    with pytest.raises(RuntimeError, match="Interrupted"):
        await runtime.sample("task", slot)
    write(slot / "RESULT.json", {"identity": "other", "status": "completed"})
    with pytest.raises(ValueError, match="Cannot reuse"):
        await runtime.sample("task", slot)


@pytest.mark.asyncio
async def test_actor_cache_uses_protocol_identity_across_operational_changes(tmp_path):
    runtime = Runtime(
        {"identity": "new-operational-config", "actor_protocol_identity": "frozen"}
    )
    slot = tmp_path / "actor"
    identity = digest(
        {"task": "task", "guidance": "", "review": False, "protocol": "frozen"}
    )
    expected = {"identity": identity, "status": "completed", "candidate": "kept"}
    write(slot / "RESULT.json", expected)

    assert await runtime.sample("task", slot) == expected


@pytest.mark.asyncio
async def test_changed_actor_guidance_is_archived_before_resampling(
    tmp_path, monkeypatch
):
    from experiments.claw_eval.opencode import runtime as runtime_module

    runtime = Runtime(
        {
            "identity": "operational",
            "actor_protocol_identity": "protocol",
            "recover_changed_actor_identity": True,
            "output": str(tmp_path),
            "docker_data_root": str(tmp_path),
        }
    )
    slot = tmp_path / "tasks/task/round_1/actors/C"
    write(slot / "RESULT.json", {"identity": "old-guidance", "status": "completed"})

    class FreshActor:
        session = None

        def __init__(self, *_args):
            pass

        async def setup(self):
            raise RuntimeError("fresh Actor started")

        async def close(self):
            pass

    monkeypatch.setattr(runtime_module, "Actor", FreshActor)
    with pytest.raises(RuntimeError, match="fresh Actor started"):
        await runtime.sample("task", slot, guidance="new")

    archives = list(
        (tmp_path / "implementation_revisions").glob(
            "*_actor_identity_mismatch/tasks/task/round_1/actors/C/RESULT.json"
        )
    )
    assert len(archives) == 1
    assert json.loads(archives[0].read_text())["identity"] == "old-guidance"


@pytest.mark.asyncio
async def test_failed_actor_result_is_archived_before_resampling(tmp_path, monkeypatch):
    from experiments.claw_eval.opencode import runtime as runtime_module

    runtime = Runtime(
        {
            "identity": "operational",
            "actor_protocol_identity": "protocol",
            "recover_changed_actor_identity": True,
            "output": str(tmp_path),
            "docker_data_root": str(tmp_path),
        }
    )
    slot = tmp_path / "tasks/task/round_1/actors/B"
    identity = digest(
        {"task": "task", "guidance": "", "review": False, "protocol": "protocol"}
    )
    write(slot / "RESULT.json", {"identity": identity, "status": "failed"})

    class FreshActor:
        session = None

        def __init__(self, *_args):
            pass

        async def setup(self):
            raise RuntimeError("fresh Actor started")

        async def close(self):
            pass

    monkeypatch.setattr(runtime_module, "Actor", FreshActor)
    with pytest.raises(RuntimeError, match="fresh Actor started"):
        await runtime.sample("task", slot)

    archives = list(
        (tmp_path / "implementation_revisions").glob(
            "*_actor_identity_mismatch/tasks/task/round_1/actors/B/RESULT.json"
        )
    )
    assert len(archives) == 1
    assert json.loads(archives[0].read_text())["status"] == "failed"


def test_private_grader_files_not_injected_as_public_inputs(tmp_path):
    actor = object.__new__(Actor)
    actor.workspace = tmp_path / "workspace"
    actor.workspace.mkdir()
    task = tmp_path / "task"
    task.mkdir()
    (task / "input.csv").write_text("value\n1\n")
    (task / "grader.py").write_text("private")
    actor.task = SimpleNamespace(task_file=str(task / "task.yaml"))
    actor.config = {"benchmark_root": str(tmp_path)}
    actor.inject(["input.csv"])
    assert (actor.workspace / "input.csv").read_text() == "value\n1\n"
    assert not (actor.workspace / "grader.py").exists()
    with pytest.raises(ValueError, match="Unsafe"):
        actor.inject(["../task/grader.py"])


def test_capacity_counts_recovered_rate_limits(tmp_path):
    write(
        tmp_path / "transport/x/one/receipt.json",
        {"queued_at": 100, "status": "completed", "http_errors": [{"status": 429}]},
    )
    write(
        tmp_path / "official.private/grade/two/receipt.json",
        {"queued_at": 101, "status": "failed"},
    )
    write(
        tmp_path / "transport/x/cancelled/receipt.json",
        {
            "queued_at": 101,
            "status": "failed",
            "error_type": "CancelledError",
        },
    )
    assert transport_window(tmp_path, 99) == {
        "requests": 2,
        "failed": 1,
        "rate_limited": True,
        "completed": 1,
    }
    assert transport_window(tmp_path, 102)["requests"] == 0


def test_grader_injection_cannot_follow_actor_symlink_outside_workspace(tmp_path):
    actor = object.__new__(Actor)
    actor.workspace = tmp_path / "workspace"
    actor.workspace.mkdir()
    task = tmp_path / "task"
    (task / "fixtures").mkdir(parents=True)
    (task / "fixtures/check.py").write_text("private grader")
    outside = tmp_path / "outside"
    outside.mkdir()
    (actor.workspace / "fixtures").symlink_to(outside, target_is_directory=True)
    actor.task = SimpleNamespace(task_file=str(task / "task.yaml"))
    actor.config = {"benchmark_root": str(tmp_path)}
    with pytest.raises(ValueError, match="symlink"):
        actor.inject(["fixtures/check.py"])
    assert not (outside / "check.py").exists()


def test_search_backend_rejects_non_results_and_preserves_source_text():
    from experiments.claw_eval.opencode.search import parse_rss

    payload = b"<rss><channel><item><title>Original title</title><link>https://example.org</link><description>Actual snippet</description></item></channel></rss>"
    assert parse_rss(payload, 1) == [
        {
            "title": "Original title",
            "url": "https://example.org",
            "description": "Actual snippet",
            "date": "",
        }
    ]
    with pytest.raises(ValueError):
        parse_rss(b"<html>captcha</html>", 10)


@pytest.mark.asyncio
async def test_real_opencode_mcp_and_same_session_revision_offline(
    tmp_path, monkeypatch
):
    """Real container/binary/tools; deterministic local model, no paid API calls."""
    import asyncio
    import json
    import os
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from pathlib import Path

    import uvicorn

    from experiments.claw_eval.opencode.runtime import free_port
    from reward_engine.core.outcome.memory import Memory

    benchmark = os.environ.get("CLAW_EVAL_ROOT")
    binary = os.environ.get("CLAW_OPENCODE_BINARY")
    image = os.environ.get("CLAW_ACTOR_IMAGE")
    if not all((benchmark, binary, image)):
        pytest.skip(
            "Set CLAW_EVAL_ROOT, CLAW_OPENCODE_BINARY and CLAW_ACTOR_IMAGE for real offline integration"
        )
    monkeypatch.syspath_prepend(str(Path(benchmark) / "src"))
    folder = tmp_path / "benchmark/tasks/fixture"
    folder.mkdir(parents=True)
    (folder / "task.yaml").write_text(
        'task_id: fixture\ntask_name: fixture\nversion: "1"\nprompt:\n  text: Write result.txt with initial.\n'
    )
    token = tmp_path / "auth"
    token.write_text("offline")
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            text = json.dumps(body["messages"])
            stage = "revised" if "DIAGNOSTIC_REVISION" in text else "initial"
            command = f"printf {stage} > /workspace/result.txt"
            used = any(
                command in str(m.get("tool_calls", "")) for m in body["messages"]
            )
            delta = (
                {"content": "Completed " + stage}
                if used
                else {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call-" + stage,
                            "type": "function",
                            "function": {
                                "name": "claw_Bash",
                                "arguments": json.dumps({"command": command}),
                            },
                        }
                    ]
                }
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            chunks = [
                {
                    "id": "offline",
                    "model": "deepseek-v4-flash",
                    "choices": [{"index": 0, "delta": delta}],
                },
                {
                    "id": "offline",
                    "model": "deepseek-v4-flash",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop" if used else "tool_calls",
                        }
                    ],
                },
            ]
            for chunk in chunks:
                self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
            self.wfile.write(b"data: [DONE]\n\n")

    server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = free_port()
    config = {
        "output": str(tmp_path),
        "identity": "offline-only",
        "campaign_id": "offline-" + tmp_path.name,
        "benchmark_root": str(tmp_path / "benchmark"),
        "auth_file": str(token),
        "actor_timeout_s": 90,
        "roles": {"actor": "deepseek-v4-flash"},
        "gateway_container": f"http://host.docker.internal:{server.server_port}",
        "tools_port": port,
        "tools_container": f"http://host.docker.internal:{port}",
        "opencode_binary": binary,
        "image": image,
        "docker_data_root": "/var/lib/docker",
    }
    runtime = Runtime(config)
    web = uvicorn.Server(
        uvicorn.Config(runtime.app, host="0.0.0.0", port=port, log_level="warning")
    )
    serving = asyncio.create_task(web.serve())
    await asyncio.sleep(0.5)
    observed = []

    class Review:
        memory = Memory()
        history = []
        status = "not_checked"
        progress = SimpleNamespace(snapshot=lambda: {})

        async def __call__(self, candidate):
            observed.append(candidate)
            self.status = "checked"
            return (
                "DIAGNOSTIC_REVISION: replace result.txt with revised."
                if len(observed) == 1
                else None
            )

    try:
        row = await runtime.sample("fixture", tmp_path / "actor", review=Review())
        assert row["status"] == "completed"
        assert len(observed) == 2
        assert (
            tmp_path / "actor/snapshots/1/workspace/result.txt"
        ).read_text() == "initial"
        assert (
            tmp_path / "actor/snapshots/2/workspace/result.txt"
        ).read_text() == "revised"
        from claw_eval.trace.reader import load_trace

        start, messages, dispatches, media, end, audits = load_trace(
            tmp_path / "actor/trace.jsonl"
        )
        assert len(dispatches) == 2 and end.trace_id == start.trace_id
        assert all(d.tool_name == "Bash" for d in dispatches)
        sessions = set()
        for path in (tmp_path / "actor").glob("actor-*.jsonl"):
            sessions.update(
                json.loads(line).get("sessionID")
                for line in path.read_text().splitlines()
                if line.startswith("{")
            )
        assert sessions == {row["session"]}
        tool_requests = [r for r in requests if r.get("tools")]
        assert tool_requests and all(
            t["function"]["name"].startswith("claw_")
            for r in tool_requests
            for t in r["tools"]
        )
    finally:
        web.should_exit = True
        await serving
        server.shutdown()
        server.server_close()
