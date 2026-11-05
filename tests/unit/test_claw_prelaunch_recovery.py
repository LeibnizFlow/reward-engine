import pytest

from reward_engine.adapters.claw_eval.recovery import recover_prelaunch
from reward_engine.adapters.workbuddy.io import read, write


def setup_failure(root):
    stage = {"status": "started", "identity": "bound", "attempt": 1}
    write(root / "seed_0/stage.json", stage)
    write(root / "seed_0/failure.json", {**stage, "error_type": "ServiceStartError"})
    write(root / "seeds/1/STARTED.json", {"identity": "actor"})
    write(root / "seeds/1/FAILURE.json", {"identity": "actor", "session": None,
                                        "error_type": "ServiceStartError"})
    return stage


def test_proven_prelaunch_failure_is_archived(tmp_path):
    stage = setup_failure(tmp_path)
    failure = (tmp_path / "seeds/1/FAILURE.json").read_bytes()
    assert recover_prelaunch(tmp_path, "seed_0", stage)
    archive, = (tmp_path / "prelaunch_recoveries").iterdir()
    assert (archive / "slot/FAILURE.json").read_bytes() == failure
    assert read(archive / "stage.json") == stage
    assert not (tmp_path / "seeds/1").exists()
    assert (tmp_path / "seed_0/failure.json").exists()


@pytest.mark.parametrize("evidence", ["request", "session", "identity", "attempt"])
def test_ambiguous_actor_or_changed_binding_cannot_be_retried(tmp_path, evidence):
    stage = setup_failure(tmp_path)
    if evidence == "request":
        write(tmp_path / "seeds/1/jiuwenswarm/process/turn-1.request.json", {})
    elif evidence == "session":
        path = tmp_path / "seeds/1/FAILURE.json"
        write(path, {**read(path), "session": "actual-session"})
    else:
        stage[evidence] = "changed"
    assert not recover_prelaunch(tmp_path, "seed_0", stage)
    assert (tmp_path / "seeds/1/FAILURE.json").exists()


@pytest.mark.asyncio
async def test_stage_recovery_preserves_attempt_and_failure_history(tmp_path):
    from reward_engine.core.outcome.construction import fingerprint
    from reward_engine.core.outcome.three_b import Stages

    stage = setup_failure(tmp_path)
    stage["identity"] = fingerprint({"run": {}, "inputs": {}})
    write(tmp_path / "seed_0/stage.json", stage)
    failure = {**stage, "error_type": "ServiceStartError"}
    write(tmp_path / "seed_0/failure.json", failure)
    write(tmp_path / "seed_0/failures/1.json", failure)
    def restore(key):
        path = tmp_path / (key + ".json")
        return read(path) if path.exists() else None
    stages = Stages({}, lambda k, v: write(tmp_path / (k + ".json"), v), restore,
                    recover_actor=lambda name, state: recover_prelaunch(tmp_path, name, state))
    async def execute():
        return {"session": "first-actual-session"}
    assert await stages.run("seed_0", {}, execute, actor=True) == {
        "session": "first-actual-session"}
    assert read(tmp_path / "seed_0/stage.json")["attempt"] == 2
    assert read(tmp_path / "seed_0/failures/1.json") == failure


def resource_failure(root):
    stage = {"status": "started", "identity": "bound", "attempt": 1, "implementation": "revision"}
    failure = {**stage, "error_type": "RuntimeError", "reason": "Storage admission floor reached"}
    write(root / "round_1/B1/stage.json", stage)
    write(root / "round_1/B1/failure.json", failure)
    write(root / "round_1/B1/failures/1.json", failure)
    return stage


def test_unstarted_resource_denial_preserves_original_receipts(tmp_path):
    stage = resource_failure(tmp_path)
    original = (tmp_path / "round_1/B1/failure.json").read_bytes()
    assert recover_prelaunch(tmp_path, "round_1/B1", stage)
    archive, = (tmp_path / "prelaunch_recoveries").iterdir()
    assert (archive / "failure.json").read_bytes() == original
    assert (tmp_path / "round_1/B1/failures/1.json").read_bytes() == original
    assert not (tmp_path / "round_1/actors/B1").exists()
    assert read(archive / "recovery.json")["actor_started"] is False


@pytest.mark.parametrize("change", ["slot", "history", "identity", "implementation", "reason", "attempt", "symlink"])
def test_resource_denial_recovery_rejects_ambiguous_evidence(tmp_path, change):
    stage = resource_failure(tmp_path)
    if change == "slot":
        write(tmp_path / "round_1/actors/B1/STARTED.json", {})
    elif change == "symlink":
        (tmp_path / "round_1/actors").symlink_to(tmp_path / "absent", target_is_directory=True)
    elif change in {"identity", "implementation", "attempt"}:
        stage[change] = "changed"
    else:
        path = tmp_path / ("round_1/B1/failures/1.json" if change == "history" else "round_1/B1/failure.json")
        write(path, {**read(path), "reason": "unrelated error"})
    assert not recover_prelaunch(tmp_path, "round_1/B1", stage)


@pytest.mark.asyncio
async def test_resource_recovery_resumes_stage_once_and_reuses_its_completion(tmp_path):
    from reward_engine.core.outcome.construction import fingerprint
    from reward_engine.core.outcome.three_b import Stages

    stage = resource_failure(tmp_path)
    stage["identity"] = fingerprint({"run": {}, "inputs": {}})
    write(tmp_path / "round_1/B1/stage.json", stage)
    failure = {**stage, "error_type": "RuntimeError", "reason": "Storage admission floor reached"}
    for name in ("failure.json", "failures/1.json"):
        write(tmp_path / "round_1/B1" / name, failure)

    def restore(key):
        path = tmp_path / (key + ".json")
        return read(path) if path.exists() else None

    calls = []

    async def execute():
        calls.append("execute")
        return {"session": "first-actual-session"}

    stages = Stages({}, lambda k, v: write(tmp_path / (k + ".json"), v), restore,
                    implementation="new-revision",
                    recover_actor=lambda name, state: recover_prelaunch(tmp_path, name, state))
    first = await stages.run("round_1/B1", {}, execute, actor=True)
    assert await stages.run("round_1/B1", {}, execute, actor=True) == first
    assert calls == ["execute"]
    assert read(tmp_path / "round_1/B1/stage.json")["attempt"] == 2
    assert read(tmp_path / "round_1/B1/failures/1.json") == failure
