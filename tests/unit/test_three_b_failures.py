"""Failed phases remain failures in CLI/reporting and respect configured deadlines."""

import sys

import pytest

from experiments.workbuddy import experiment
from reward_engine.adapters.jiuwenswarm import report as reporting
from reward_engine.adapters.workbuddy.io import write
from reward_engine.core.batch import BatchPolicy
from reward_engine.core.outcome import validated_construction
from reward_engine.core.outcome.construction import fingerprint
from reward_engine.core.outcome.three_b import Stages, require_measurement


@pytest.mark.asyncio
async def test_incomplete_measurement_retains_diagnostic_in_failed_stage():
    records = {}
    measured = {"pack_id": "synthetic", "results": [
        {"test_id": "check", "status": "unknown", "error": True,
         "diagnostic": "claims[0].evidence_refs is missing"}]}
    async def check():
        require_measurement({"pack_id": "synthetic", "tests": [{"test_id": "check"}]}, measured)
    stages = Stages({}, lambda k, v: records.__setitem__(k, v), records.get)
    with pytest.raises(ValueError, match="Full measurement"):
        await stages.run("construction_checks/0", {}, check)
    assert records["construction_checks/0/failures/1"]["measurement"] == measured
    assert records["construction_checks/0/stage"]["status"] == "started"


@pytest.mark.parametrize("status,code", [("partial", 1), ("completed", None)])
def test_office_cli_exit_follows_campaign_status(tmp_path, monkeypatch, status, code):
    write(tmp_path / "config.json", {"output": str(tmp_path), "code": str(experiment.REPO)})
    async def campaign(config, **kwargs):
        write(tmp_path / "status.json", {"status": status})
    monkeypatch.setattr(experiment, "campaign", campaign)
    monkeypatch.setattr(sys, "argv", ["experiment", "run", "--output", str(tmp_path)])
    if code:
        with pytest.raises(SystemExit) as error:
            experiment.main()
        assert error.value.code == code
    else:
        experiment.main()


@pytest.mark.asyncio
async def test_construction_uses_configured_batch_deadline(monkeypatch):
    policy = BatchPolicy(deadline_s=1200)
    class ReachedBatch(Exception):
        pass
    async def propose(*args, **kwargs):
        assert kwargs["policy"] is policy
        raise ReachedBatch
    monkeypatch.setattr(validated_construction.jury, "_propose_once", propose)
    with pytest.raises(ReachedBatch):
        await validated_construction.construct_reward({}, {}, None, lambda _: None,
            policy=policy, adversarial_review_enabled=True)


def test_report_counts_completed_baselines_before_failed_round(tmp_path, monkeypatch):
    task = tmp_path / "tasks/synthetic"
    write(task / "seeds/1/snapshots/1/workspace/stage.json", {"untrusted": "task input"})
    value = {"session": "actual-retained-session", "candidate": {"artifact": "synthetic"}}
    write(task / "seed_0/stage.json", {"status": "completed", "value": value,
        "value_sha256": fingerprint(value)})
    write(task / "round_1/construction/stage.json", {"status": "started", "attempt": 1})
    write(task / "round_1/construction/failure.json", {"attempt": 1, "error_type": "ValueError", "reason": "synthetic failure"})
    verified = []
    monkeypatch.setattr(reporting, "verify_actor", lambda row: verified.append(row))
    result = reporting.report({"output": str(tmp_path), "task_ids": ["synthetic"], "rounds": 1}, tmp_path / "report")
    assert verified == [value]
    assert result["completed_actors"] == 1 and result["valid_scores"] == 0
    assert result["completed_tasks"] == 0
    assert result["tasks"][0]["status"] == "failed"
    assert result["tasks"][0]["rounds"][0]["status"] == "failed"
