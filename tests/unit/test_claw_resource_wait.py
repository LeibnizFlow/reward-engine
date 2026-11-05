import asyncio

import pytest

from experiments.claw_eval.opencode import runtime as module
from reward_engine.adapters.workbuddy.io import read


@pytest.mark.parametrize("floor,allowed", [(None, False), (16, True), (20, False)])
def test_explicit_storage_reserve_preserves_default(tmp_path, monkeypatch, floor, allowed):
    from types import SimpleNamespace
    runtime = object.__new__(module.Runtime)
    runtime.config = {"output": str(tmp_path), "docker_data_root": str(tmp_path)}
    if floor is not None:
        runtime.config["minimum_free_gib"] = floor
    monkeypatch.setattr(module.shutil, "disk_usage", lambda path: SimpleNamespace(free=19 * 1024**3))
    monkeypatch.setattr(module.Path, "read_text", lambda path: "MemAvailable: 104857600 kB\n")
    if allowed:
        runtime.check_resources()
    else:
        with pytest.raises(RuntimeError, match="Storage admission"):
            runtime.check_resources()


@pytest.mark.parametrize("value", [0, 4, True, 16.5, 1025])
def test_storage_reserve_rejects_invalid_values(value):
    runtime = object.__new__(module.Runtime)
    runtime.config = {"minimum_free_gib": value}
    with pytest.raises(ValueError, match="minimum_free_gib"):
        runtime.check_resources()


def test_storage_reserve_cannot_change_an_existing_run(monkeypatch, tmp_path, capsys):
    from experiments.claw_eval.opencode.campaign import main
    monkeypatch.setattr("sys.argv", ["campaign", "run", "--output", str(tmp_path), "--minimum-free-gib", "16"])
    with pytest.raises(SystemExit) as caught:
        main()
    assert caught.value.code == 2
    assert "frozen during fresh preparation" in capsys.readouterr().err


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_sample_waits_before_actor_creation_and_reuses_completed_slot(tmp_path, monkeypatch, cancel):
    runtime = object.__new__(module.Runtime)
    runtime.config = {"identity": "bound", "failure_policy": "isolate_tasks_v1"}
    slot = tmp_path / "actors/B1"
    blocked, release = asyncio.Event(), asyncio.Event()
    calls = []

    def resources():
        if not release.is_set():
            raise RuntimeError("Storage admission floor reached")

    async def wait(_):
        blocked.set()
        await release.wait()

    class Actor:
        session, execution_s, turn, events, usage = "first-session", 7, 1, [], {}

        def __init__(self, *args):
            calls.append("actor")

        async def setup(self):
            pass

        async def finalize(self):
            pass

        async def close(self):
            pass

    async def deliver(*args, **kwargs):
        return {"artifact": {"final_response": "synthetic"}}

    runtime.check_resources = resources
    monkeypatch.setattr(module.asyncio, "sleep", wait)
    monkeypatch.setattr(module, "Actor", Actor)
    monkeypatch.setattr(module, "run_delivery", deliver)
    task = asyncio.create_task(runtime.sample("synthetic", slot))
    await blocked.wait()
    assert not slot.exists() and calls == []
    assert read(slot.parent / "B1.resource_wait.json")["status"] == "waiting"
    if cancel:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not slot.exists() and calls == []
    else:
        release.set()
        result = await task
        assert result["actor_execution_s"] == 7
        assert await runtime.sample("synthetic", slot) == result
        assert calls == ["actor"]
        assert read(slot.parent / "B1.resource_wait.json")["status"] == "admitted"


@pytest.mark.asyncio
@pytest.mark.parametrize("policy,reason", [(None, "Storage admission floor reached"), ("isolate_tasks_v1", "unexpected")])
async def test_other_failures_and_legacy_policy_still_raise(tmp_path, policy, reason):
    runtime = object.__new__(module.Runtime)
    runtime.config = {"identity": "bound", "failure_policy": policy}

    def resources():
        raise RuntimeError(reason)

    runtime.check_resources = resources
    with pytest.raises(RuntimeError, match=reason):
        await runtime.sample("synthetic", tmp_path / "B1")
    assert not (tmp_path / "B1").exists()
