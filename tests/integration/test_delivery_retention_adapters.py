"""Offline native result serialization; no transport or official grader execution."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from experiments.claw_eval.opencode import runtime as claw_runtime
from reward_engine.adapters.workbuddy.io import read, write
from reward_engine.core.outcome.harness import defaults
from reward_engine.core.outcome.runner import DeliveryReview
from reward_engine.core.outcome.version_selection import validate_review_session
from tests.support.delivery_retention import (
    TASK,
    ScriptedActor,
    published_pack,
    scripted_review,
)


def assert_native_and_retained(row, pack):
    assert row["candidate"]["artifact"]["final_response"] == "{"
    assert row["final_measurement"]["results"][0]["status"] == "fail"
    assert row["retained_candidate"]["artifact"]["final_response"] == "{}"
    assert row["retained_measurement"]["results"][0]["status"] == "pass"
    assert row["retained_delivery"]["native_state_restored"] is False
    assert row["retained_delivery"]["official_evaluation"] == "not_run"
    assert "feedback_execution" not in row
    validate_review_session(row, TASK, pack)
    for version in row["review_session"]["versions"]:
        recorded = read(version["measurement_source"])
        assert recorded["candidate"] == version["candidate"]
        assert recorded["measurement"] == version["measurement"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["regression", "generic", "disabled"])
async def test_claw_serializes_terminal_state_and_explicit_retention(
    tmp_path, monkeypatch, mode
):
    actors = []

    class Actor(ScriptedActor):
        def __init__(self, runtime, task_id, output):
            super().__init__()
            self.session, self.execution_s = "synthetic-session", 0
            self.finalized = None
            self.closed = False
            actors.append(self)

        async def setup(self):
            pass

        async def finalize(self):
            self.finalized = deepcopy(self.current)

        async def close(self):
            self.closed = True

    class GenericReview:
        status, memory, history = "generic-transport", None, []
        progress = SimpleNamespace(snapshot=lambda: {})

        def __init__(self):
            self.calls = 0

        async def __call__(self, candidate):
            self.calls += 1
            return "try again" if self.calls == 1 else None

    output = tmp_path / "trial"
    checks = output / "checks"
    pack = published_pack()
    review = (
        DeliveryReview(
            TASK,
            pack,
            defaults(scripted_review),
            source_prefix=str(checks),
            record=lambda name, value: write(checks / (name + ".json"), value),
        )
        if mode == "regression"
        else GenericReview()
        if mode == "generic"
        else None
    )
    monkeypatch.setattr(claw_runtime, "Actor", Actor)
    runtime = claw_runtime.Runtime({"identity": "offline", "output": str(tmp_path)})
    monkeypatch.setattr(runtime, "check_resources", lambda: None)
    row = await runtime.sample("offline", output, review=review)
    assert read(output / "RESULT.json") == row
    assert actors[0].closed and actors[0].finalized == row["candidate"]
    assert row["native_grader_trial"] == str(output)
    if mode == "regression":
        assert_native_and_retained(row, pack)
    else:
        assert "review_session" not in row and "retained_candidate" not in row
        assert row["final_measurement"] is None
        assert actors[0].calls == (
            ["start", "resume"] if mode == "generic" else ["start"]
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("review_enabled", [True, False])
async def test_workbuddy_serializes_terminal_state_and_explicit_retention(
    tmp_path, monkeypatch, review_enabled
):
    from reward_engine.adapters.workbuddy import agent as workbuddy_agent

    pack = published_pack()
    agent = object.__new__(workbuddy_agent.OutcomeOpenCodeAgent)
    agent.spec = {"task": TASK, "actor_timeout_s": 10, "prompt": ""}
    if review_enabled:
        agent.spec["pack"] = pack
    agent.config, agent.result_root = {"batch": {}}, tmp_path
    agent.actor_remaining, agent.session, agent.turn = 10, "synthetic-session", 0
    agent.render_instruction = lambda value: value
    agent.snapshot = AsyncMock(return_value=({}, {}))
    agent._stop_opencode = AsyncMock()
    actor = ScriptedActor()
    agent.start, agent.resume = actor.start, actor.resume
    monkeypatch.setattr(
        workbuddy_agent, "registry", lambda *args: defaults(scripted_review)
    )
    await agent.run("instruction", object(), None)
    row = read(tmp_path / "ACTOR_RESULT.json")
    assert row["candidate"] == actor.current
    assert row["official_verifier_observed"] is False
    agent._stop_opencode.assert_not_awaited()
    agent.snapshot.assert_awaited_once_with("initial")
    if review_enabled:
        assert_native_and_retained(row, pack)
    else:
        assert "review_session" not in row and "retained_candidate" not in row
        assert row["review_status"] == "disabled"
        assert row["final_measurement"] is None
