"""Exercise production task wrappers with offline actors and a captured core call."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from experiments.claw_eval.opencode import campaign
from experiments.workbuddy import experiment, resume
from reward_engine.adapters.workbuddy.io import digest, read, write


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "settings, rounds, audit",
    [
        ({}, 1, False),
        ({"rounds": 3, "full_audit": True}, 3, True),
        ({"rounds": 2, "full_audit": False}, 2, False),
    ],
)
async def test_claw_forwards_rounds_and_audit_without_changing_identity(
    tmp_path, monkeypatch, settings, rounds, audit
):
    config = {"output": str(tmp_path), "batch": {}, **settings}
    config["identity"] = digest(config)
    before = deepcopy(config)
    task = {"task_id": "task", "prompt": "Create a report"}
    monkeypatch.setattr(campaign, "public_task", lambda *_: (None, task))
    monkeypatch.setattr(campaign, "registry", Mock(return_value={}))
    review = Mock(return_value=object())
    monkeypatch.setattr(campaign, "DeliveryReview", review)

    async def actor(task_id, slot, **kwargs):
        return {
            "status": "completed",
            "candidate": {"candidate_id": str(slot.relative_to(tmp_path))},
            "runtime_directory": str(slot),
        }

    runtime = SimpleNamespace(sample=AsyncMock(side_effect=actor))
    grader = AsyncMock(return_value={"status": "scored", "score": 1})
    monkeypatch.setattr(campaign, "grade", grader)

    async def core(actual_task, seeds, **kwargs):
        assert actual_task == task
        assert seeds == [{"candidate_id": f"tasks/task/seeds/{i}"} for i in range(1, 4)]
        assert kwargs["rounds"] == rounds
        assert kwargs["full_audit"] is audit
        assert kwargs["continuation_policy"] == "fixed_rounds"
        assert kwargs["enabled_arms"] == ("A", "B", "C")
        assert kwargs["construction_rounds"] == 1
        for number in range(1, rounds + 1):
            for arm in "ABC":
                row = await kwargs["sample"](arm, number, "guidance", {})
                assert row["runtime_directory"] == str(
                    tmp_path / f"tasks/task/round_{number}/actors/{arm}"
                )
                assert await kwargs["evaluate"](task, row) == {
                    "status": "scored",
                    "score": 1,
                }
        return {"status": "completed", "completed_rounds": rounds}

    run = AsyncMock(side_effect=core)
    monkeypatch.setattr(campaign, "run_task", run)
    result = await campaign.task_run(config, runtime, "task")
    run.assert_awaited_once()
    assert runtime.sample.await_count == 3 + 3 * rounds
    assert grader.await_count == 3 * rounds
    assert review.call_count == rounds
    assert read(tmp_path / "tasks/task/result.private.json") == result
    assert config == before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "settings, audit",
    [({}, False), ({"full_audit": False}, False), ({"full_audit": True}, True)],
)
@pytest.mark.parametrize("completed", [0, 1, 2])
async def test_workbuddy_forwards_audit_and_remaining_rounds_preserving_checkpoint(
    tmp_path, monkeypatch, settings, audit, completed
):
    instruction = tmp_path / "dataset/task/instruction.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_text("Create a report")
    config = {
        "output": str(tmp_path),
        "tasks_root": str(instruction.parent.parent),
        "rounds": 3,
        "batch": {},
        **settings,
    }
    config["identity"] = digest(config)
    before = deepcopy(config)
    history = [
        {"official": {arm: {"status": "scored"} for arm in "ABC"}}
        for _ in range(completed)
    ]
    checkpoint = (
        {
            "pool": [{"candidate_id": "frozen-pool-candidate"}],
            "memory": [{"memory_id": "frozen-memory-id"}],
            "checks": [{"test_id": "frozen-test-id"}],
            "previous_pack": {"pack_id": "frozen-pack-id"},
            "expectation_cache": {"frozen-expectation": "retained"},
            "known_gaps": [{"gap": "retained"}],
            "decisions": [{"decision": "retained"}],
        }
        if completed
        else None
    )
    retained = deepcopy((history, checkpoint))
    monkeypatch.setattr(
        resume, "completed_rounds", Mock(return_value=(history, checkpoint))
    )
    monkeypatch.setattr(experiment, "registry", Mock(return_value={}))

    async def launch(actual_config, task, slot, **kwargs):
        candidate = {"candidate_id": str(slot.relative_to(tmp_path))}
        if slot.parent.name == "seeds":
            assert kwargs == {"grade": False}
        else:
            assert kwargs == {"prompt": "guidance", "pack": {"pack_id": "test"}}
            write(slot / "spec.json", candidate)
            write(slot / "OFFICIAL.private.json", {"status": "scored", "score": 1})
        return {"status": "completed", "candidate": candidate}

    actor = AsyncMock(side_effect=launch)
    monkeypatch.setattr(experiment, "launch", actor)

    async def core(task, pool, **kwargs):
        assert kwargs["rounds"] == 3 - completed
        assert kwargs["full_audit"] is audit
        assert kwargs["continuation_policy"] == "fixed_rounds"
        if completed:
            assert kwargs["start_round"] == completed + 1
            assert pool == checkpoint["pool"]
            assert kwargs["memory"].rows == checkpoint["memory"]
            for argument, key in (
                ("initial_checks", "checks"),
                ("initial_pack", "previous_pack"),
                ("initial_expectation_cache", "expectation_cache"),
                ("initial_known_gaps", "known_gaps"),
                ("initial_decisions", "decisions"),
            ):
                assert kwargs[argument] == checkpoint[key]
        else:
            assert "start_round" not in kwargs
            assert pool == [
                {"candidate_id": f"tasks/task/seeds/{i}"} for i in range(1, 4)
            ]
        rounds = []
        for number in range(completed + 1, 4):
            official = {}
            for arm in "ABC":
                row = await kwargs["sample"](
                    arm, number, "guidance", {"pack_id": "test"}
                )
                assert (
                    row["candidate"]["candidate_id"]
                    == f"tasks/task/round_{number}/actors/{arm}"
                )
                official[arm] = await kwargs["evaluate"](task, row)
                assert official[arm] == {"status": "scored", "score": 1}
            rounds.append({"official": official})
        return {
            "status": "completed",
            "completed_rounds": 3 - completed,
            "rounds": rounds,
        }

    run = AsyncMock(side_effect=core)
    monkeypatch.setattr(experiment, "run_task", run)
    assert await experiment.one_task(config, "task") == {
        "status": "completed",
        "completed_rounds": 3,
        "scored": 9,
    }
    run.assert_awaited_once()
    assert actor.await_count == 3 + 3 * (3 - completed)
    result = read(tmp_path / "tasks/task/result.private.json")
    assert len(result["rounds"]) == result["completed_rounds"] == 3
    assert result["rounds"][:completed] == history
    assert (history, checkpoint) == retained
    assert config == before
