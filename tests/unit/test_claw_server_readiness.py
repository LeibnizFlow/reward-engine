"""Claw must wait for both local servers before probing models or actors."""

import asyncio
from types import SimpleNamespace

import httpx
import pytest

from experiments.claw_eval.opencode import campaign
from reward_engine.adapters import provider_config


@pytest.mark.asyncio
@pytest.mark.parametrize("slow_server", [0, 1])
async def test_probe_waits_for_delayed_local_server(tmp_path, monkeypatch, slow_server):
    instances, probes = [], []

    class Server:
        def __init__(self, config):
            self.index = len(instances)
            self.started = self.should_exit = False
            instances.append(self)

        async def serve(self):
            await asyncio.sleep(1.2 if self.index == slow_server else 0)
            self.started = True
            while not self.should_exit:
                await asyncio.sleep(0.01)

    async def probe(config):
        if not all(server.started for server in instances):
            raise httpx.ConnectError("All connection attempts failed")
        probes.append(True)

    monkeypatch.setattr(campaign, "Runtime", lambda _: SimpleNamespace(app=object()))
    monkeypatch.setattr(provider_config, "application", lambda _: object())
    monkeypatch.setattr(campaign.uvicorn, "Config", lambda *a, **kw: object())
    monkeypatch.setattr(campaign.uvicorn, "Server", Server)
    monkeypatch.setattr(campaign, "probe", probe)
    await campaign.serve({"output": str(tmp_path), "benchmark_root": str(tmp_path),
                          "gateway_port": 1, "tools_port": 2,
                          "generation_protocol": "three_b_fixed_baselines_v1"}, probe_only=True)
    assert probes == [True]
    assert all(server.should_exit for server in instances)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, ValueError("synthetic startup failure")])
async def test_server_startup_exit_is_reported(failure):
    async def startup():
        if failure:
            raise failure

    task = asyncio.create_task(startup())
    with pytest.raises(type(failure) if failure else RuntimeError,
                       match="synthetic startup failure" if failure else "gateway server exited"):
        await campaign.wait_for_servers({"gateway": SimpleNamespace(started=False)}, [task])


@pytest.mark.asyncio
async def test_server_readiness_has_a_deadline():
    task = asyncio.create_task(asyncio.Event().wait())
    try:
        with pytest.raises(TimeoutError):
            await campaign.wait_for_servers(
                {"gateway": SimpleNamespace(started=False)}, [task], timeout=0.01
            )
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [TimeoutError, asyncio.CancelledError])
async def test_failed_readiness_cleans_up_servers(tmp_path, monkeypatch, failure):
    instances, stopped = [], []

    class Server:
        def __init__(self, config):
            self.should_exit = False
            instances.append(self)

        async def serve(self):
            try:
                await asyncio.Event().wait()
            finally:
                stopped.append(self)

    async def wait(servers, tasks):
        await asyncio.sleep(0)
        raise failure()

    monkeypatch.setattr(campaign, "Runtime", lambda _: SimpleNamespace(app=object()))
    monkeypatch.setattr(provider_config, "application", lambda _: object())
    monkeypatch.setattr(campaign.uvicorn, "Config", lambda *a, **kw: object())
    monkeypatch.setattr(campaign.uvicorn, "Server", Server)
    monkeypatch.setattr(campaign, "wait_for_servers", wait)
    with pytest.raises(failure):
        await campaign.serve({"output": str(tmp_path), "benchmark_root": str(tmp_path),
                              "gateway_port": 1, "tools_port": 2}, probe_only=True)
    assert stopped == instances
    assert all(server.should_exit for server in instances)
