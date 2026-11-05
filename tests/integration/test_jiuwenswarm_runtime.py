"""Real Docker/CLI tests. Missing explicit dependencies fail, never silently skip."""

import asyncio
import json
import os
import shlex
from pathlib import Path
from uuid import uuid4

import pytest

from reward_engine.adapters.jiuwenswarm.bundle import verify
from reward_engine.adapters.jiuwenswarm.container import docker, prepare
from reward_engine.adapters.jiuwenswarm.gateway import register
from reward_engine.adapters.jiuwenswarm.office import PROBE
from reward_engine.adapters.jiuwenswarm.policy import policy
from reward_engine.adapters.workbuddy.gateway import application
from tests.support.jiuwenswarm_local import local_model, serving, tool


@pytest.fixture(scope="session")
def runtime_paths():
    bundle = os.environ.get("JIUWENSWARM_TEST_BUNDLE")
    image = os.environ.get("JIUWENSWARM_TEST_IMAGE")
    assert bundle and image, (
        "BLOCKED: set JIUWENSWARM_TEST_BUNDLE and JIUWENSWARM_TEST_IMAGE for real Docker validation"
    )
    verify(bundle)
    return bundle, image


async def container(tmp_path, image, bundle):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "input.txt").write_text("SYNTHETIC_INPUT")
    name = "reward-engine-test-" + uuid4().hex
    await docker(
        "run",
        "-d",
        "--name",
        name,
        "--user",
        "root",
        "--add-host",
        "host.docker.internal:host-gateway",
        "-v",
        str(workspace) + ":/workspace",
        "-v",
        bundle + ":/opt/jiuwenswarm:ro",
        "--entrypoint",
        "sleep",
        image,
        "infinity",
    )
    await docker(
        "exec",
        name,
        "sh",
        "-c",
        "id dev >/dev/null 2>&1 || useradd -m dev; chown -R dev /workspace",
    )
    return name


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["agent.code.normal", "agent.work.normal"])
async def test_real_tools_delivery_resume_and_office_environment(
    tmp_path, runtime_paths, mode
):
    bundle, image = runtime_paths
    config = {
        "output": str(tmp_path),
        "identity": "local",
        "actor_backend": "jiuwenswarm",
        "jiuwenswarm_bundle": bundle,
        "jiuwenswarm_mode": mode,
        "roles": {"actor": "fixture"},
        "model_concurrency": 1,
    }
    auth = tmp_path / "auth"
    auth.write_text("controller")
    config["auth_file"] = str(auth)
    token = register(config, "actor-fixture", policy())
    command = (
        "python -c "
        + shlex.quote(PROBE)
        + "; cat /workspace/input.txt; printf 'DELIVERED' > /workspace/output.txt"
    )

    async def respond(body, index):
        if index == 1:
            return tool("bash", {"command": command})
        if index == 3:
            return tool(
                "bash",
                {
                    "command": "printf 'REVISED' >> /workspace/output.txt; cat /workspace/output.txt"
                },
                "revision",
            )
        return {"role": "assistant", "content": "Complete synthetic delivery."}

    name = await container(tmp_path, image, bundle)
    try:
        async with local_model(respond) as (port, calls):
            config["backend_url"] = f"http://127.0.0.1:{port}/v1"
            async with serving(
                application(config, backend_key="local-only")
            ) as gateway_port:
                process = await prepare(
                    name,
                    tmp_path / "actor",
                    config,
                    {
                        "base_url": f"http://host.docker.internal:{gateway_port}/actor-fixture/v1",
                        "model": "fixture",
                        "api_key": token,
                    },
                )
                first = (
                    await process.run(
                        "Create the synthetic output.",
                        workspace=Path("/workspace"),
                        timeout_s=90,
                    )
                ).require_completed()
                second = (
                    await process.run(
                        "Revise it in this same session.",
                        workspace=Path("/workspace"),
                        session_id=first.session_id,
                        timeout_s=90,
                    )
                ).require_completed()
                assert first.session_id == second.session_id and len(calls) == 4
                assert (
                    tmp_path / "workspace/output.txt"
                ).read_text() == "DELIVEREDREVISED"
                assert (
                    tmp_path / "workspace/input.txt"
                ).read_text() == "SYNTHETIC_INPUT"
                outputs = str(
                    [m for c in calls for m in c["messages"] if m["role"] == "tool"]
                )
                assert (
                    "offline_roundtrip" in outputs
                    and "ModuleNotFoundError" not in outputs
                )
                assert all(
                    {t["function"]["name"] for t in c["tools"]}
                    <= set(policy()["allowed"])
                    for c in calls
                )
                with pytest.raises(ValueError, match="Cannot change mode"):
                    await process.run(
                        "invalid",
                        workspace=Path("/workspace"),
                        session_id=first.session_id,
                        mode="agent.work.normal"
                        if mode == "agent.code.normal"
                        else "agent.code.normal",
                    )
    finally:
        await docker("rm", "-f", name)


@pytest.mark.asyncio
async def test_real_native_502_continues_live_session_and_partial_workspace(tmp_path, runtime_paths):
    from unittest.mock import AsyncMock, Mock

    from fastapi import HTTPException

    from experiments.claw_eval.opencode.runtime import Actor
    from reward_engine.adapters.jiuwenswarm.recovery import CONTINUE
    from reward_engine.adapters.jiuwenswarm.recovery import verify as verify_recovery
    from reward_engine.adapters.workbuddy.io import read

    bundle, image = runtime_paths
    config = {'output': str(tmp_path), 'identity': 'synthetic-live-recovery',
              'actor_backend': 'jiuwenswarm', 'jiuwenswarm_bundle': bundle,
              'jiuwenswarm_mode': 'agent.code.normal', 'roles': {'actor': 'fixture'},
              'generation_protocol': 'three_b_fixed_baselines_v1', 'actor_timeout_s': 180,
              'model_concurrency': 1, 'auth_file': str(tmp_path / 'auth')}
    (tmp_path / 'auth').write_text('synthetic-controller')
    token = register(config, 'actor-fixture', policy())

    async def respond(body, index):
        messages = body['messages']
        continued = any(CONTINUE in str(m.get('content')) for m in messages if m['role'] == 'user')
        outputs = str([m for m in messages if m['role'] == 'tool'])
        if not continued:
            if 'PARTIAL' in outputs:
                raise HTTPException(status_code=502, detail='Synthetic native recovery failure')
            return tool('bash', {'command': "printf 'PARTIAL' > /workspace/output.txt; cat /workspace/output.txt"})
        assert 'PARTIAL' in outputs, 'The continued session lost its previous tool result'
        if 'PARTIAL-RECOVERED' not in outputs:
            return tool('bash', {'command': "printf '%s' '-RECOVERED' >> /workspace/output.txt; cat /workspace/output.txt"}, 'recovered')
        return {'role': 'assistant', 'content': 'Completed the same synthetic task.'}

    name = await container(tmp_path, image, bundle)
    try:
        async with local_model(respond) as (port, calls):
            config['backend_url'] = f'http://127.0.0.1:{port}/v1'
            async with serving(application(config, backend_key='synthetic-local-only')) as gateway_port:
                actor = object.__new__(Actor)
                actor.config, actor.output, actor.container = config, tmp_path / 'actor', name
                actor.execution_s, actor.turn, actor.session = 0, 0, None
                actor.usage, actor.raw_events = [], []
                actor.trace_message, actor._snapshot = Mock(), AsyncMock(return_value={'synthetic': 'delivery'})
                actor.process = await prepare(name, actor.output, config, {
                    'base_url': f'http://host.docker.internal:{gateway_port}/actor-fixture/v1',
                    'model': 'fixture', 'api_key': token})
                assert await actor._jiuwenswarm_turn('Complete the synthetic file task.') == {'synthetic': 'delivery'}
                assert actor.turn == 1 and actor.transport_recoveries == 1
                assert (tmp_path / 'workspace/output.txt').read_text() == 'PARTIAL-RECOVERED'
                process = actor.output / 'jiuwenswarm/process'
                first = read(process / 'turn-1.result.json')
                second = read(process / 'turn-1-recovery-1.result.json')
                assert first['status'] == 'failed' and second['status'] == 'completed'
                assert first['session_id'] == second['session_id'] == actor.session
                assert read(process / 'turn-1-recovery-1.request.json')['timeout_seconds'] < 180
                verify_recovery(actor.output, {'resources': {'turns': 1, 'transport_recoveries': 1,
                    'process_executions': 2, 'actor_timeout_s': 180}})
                assert len(calls) >= 4
    finally:
        await docker('rm', '-f', name)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_real_container_cleanup(tmp_path, runtime_paths, cancel):
    bundle, image = runtime_paths

    async def respond(body, index):
        return tool("bash", {"command": "echo started > /workspace/running; sleep 300"})

    name = await container(tmp_path, image, bundle)
    try:
        async with local_model(respond) as (port, calls):
            process = await prepare(
                name,
                tmp_path / "actor",
                {"jiuwenswarm_bundle": bundle},
                {
                    "base_url": f"http://host.docker.internal:{port}/v1",
                    "model": "fixture",
                    "api_key": "local",
                },
            )
            task = asyncio.create_task(
                process.run(
                    "Run the command",
                    workspace=Path("/workspace"),
                    timeout_s=120,
                    host_timeout_s=15 if not cancel else 60,
                )
            )
            # Cancellation is exercised after the tool starts. A cold runtime
            # can need over 30 seconds, while its unchanged host deadline is 60.
            # Leave five seconds to cancel before that deadline; the timeout
            # variant still raises from task.done() at its own 15-second bound.
            for _ in range(550):
                if (tmp_path / "workspace/running").exists():
                    break
                if task.done():
                    await task
                await asyncio.sleep(0.1)
            assert (tmp_path / "workspace/running").exists()
            if cancel:
                task.cancel()
            with pytest.raises(asyncio.CancelledError if cancel else TimeoutError):
                await task
            assert (
                await docker("inspect", "--format", "{{.State.Running}}", name)
            ).strip() == "false"
            receipt = json.loads(
                next(process.raw_dir.glob("*.execution.json")).read_text()
            )
            assert receipt["status"] == ("cancelled" if cancel else "timed_out")
    finally:
        await docker("rm", "-f", name)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["agent.code.normal", "agent.work.normal"])
async def test_real_cli_rejects_unregistered_tool(tmp_path, runtime_paths, mode):
    bundle, image = runtime_paths
    config = {
        "output": str(tmp_path),
        "identity": "local",
        "jiuwenswarm_bundle": bundle,
        "jiuwenswarm_mode": mode,
        "roles": {"actor": "fixture"},
        "model_concurrency": 1,
    }
    auth = tmp_path / "auth"
    auth.write_text("controller")
    config["auth_file"] = str(auth)
    token = register(config, "actor-denial", policy([]))

    attempted = False

    async def respond(body, index):
        nonlocal attempted
        if any(isinstance(m.get("content"), list) for m in body["messages"]):
            return {"role": "assistant", "content": "black"}
        assert body.get("tools") == []
        if not attempted:
            attempted = True
            return tool(
                "write_file",
                {"path": "/workspace/forbidden.txt", "content": "must not execute"},
            )
        return {"role": "assistant", "content": "The requested tool was denied."}

    name = await container(tmp_path, image, bundle)
    try:
        async with local_model(respond) as (port, calls):
            config["backend_url"] = f"http://127.0.0.1:{port}/v1"
            async with serving(
                application(config, backend_key="local-only")
            ) as gateway_port:
                process = await prepare(
                    name,
                    tmp_path / "actor",
                    config,
                    {
                        "base_url": f"http://host.docker.internal:{gateway_port}/actor-denial/v1",
                        "model": "fixture",
                        "api_key": token,
                    },
                    allowed_tools=[],
                )
                result = await process.run(
                    "Write forbidden.txt", workspace=Path("/workspace"), timeout_s=90
                )
                assert not (tmp_path / "workspace/forbidden.txt").exists()
                outputs = str(
                    [r for r in result.events if r["event_type"] == "chat.tool_result"]
                ).lower()
                assert (
                    "denied" in outputs
                    or "not allowed" in outputs
                    or "not found" in outputs
                )
                assert len(calls) >= 2
    finally:
        await docker("rm", "-f", name)
