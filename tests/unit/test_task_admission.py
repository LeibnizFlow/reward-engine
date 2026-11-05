"""Resource exhaustion drains active work and leaves queued tasks resumable."""

import asyncio
from types import SimpleNamespace

import pytest

from reward_engine.adapters import task_admission


def test_space_checks_both_filesystems_and_custom_docker_root(monkeypatch):
    checked = []

    def usage(path):
        checked.append(path)
        return SimpleNamespace(free=100 if path == "output" else 9)

    monkeypatch.setattr(task_admission.shutil, "disk_usage", usage)
    with pytest.raises(OSError, match="available=9, required=10"):
        task_admission.check_space({
            "output": "output", "docker_data_root": "docker", "minimum_free_bytes": 10,
        })
    assert checked == ["output", "docker"]


def test_isolated_admission_rechecks_after_a_resource_failure():
    checks = []

    def check(_):
        checks.append(True)
        if len(checks) == 1:
            raise OSError("synthetic disk pressure")

    admission = task_admission.TaskAdmission({"failure_policy": "isolate_tasks_v1"}, check)
    with pytest.raises(OSError, match="synthetic disk pressure"):
        admission.admit()
    admission.admit()
    assert len(checks) == 2 and admission.failure is None


@pytest.mark.asyncio
@pytest.mark.parametrize("benchmark", ["claw", "office"])
async def test_low_space_drains_active_task_and_resume_only_runs_pending(monkeypatch, benchmark):
    from experiments.claw_eval.opencode import campaign
    from experiments.workbuddy.schedule import run_tasks

    config = {"generation_protocol": "three_b_fixed_baselines_v1",
              "rounds": 1, "task_concurrency": 2}
    state = {"tasks": {}}
    calls, completed, checks = [], [], []
    denied = asyncio.Event()

    def check(_):
        checks.append(True)
        if len(checks) == 2:
            denied.set()
            raise OSError("synthetic disk full")

    async def one_task(_, name):
        calls.append(name)
        if name == "a":
            await asyncio.wait_for(denied.wait(), timeout=1)
        completed.append(name)
        return {"status": "completed", "scored": 6}

    async def claw_task(cfg, runtime, name):
        # The real task runner reuses completed checkpoints.
        if name not in state["tasks"]:
            state["tasks"][name] = await one_task(cfg, name)

    async def run():
        if benchmark == "office":
            await run_tasks(config, state, ["a", "b", "c"], one_task=one_task,
                            check_space=check, record=lambda _: None)
        else:
            await campaign.run_three_b_tasks(config, None, ["a", "b", "c"])

    monkeypatch.setattr(task_admission, "check_space", check)
    monkeypatch.setattr(campaign, "task_run", claw_task)
    monkeypatch.setattr(campaign, "report", lambda _: None)
    if benchmark == "claw":
        with pytest.raises(OSError, match="synthetic disk full"):
            await run()
    else:
        await run()
        assert state["dispatch_blocked"]["reason"] == "synthetic disk full"
    assert calls == completed == ["a"]
    assert len(checks) == 2  # Admission stays closed even if space recovers.
    assert set(state["tasks"]) == {"a"}
    await run()
    assert calls == completed == ["a", "b", "c"]
    assert "dispatch_blocked" not in state
