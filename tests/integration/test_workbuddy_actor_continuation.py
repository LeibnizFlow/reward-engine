"""Output truncation resumes the same actor without accepting a partial delivery."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from reward_engine.adapters.workbuddy.agent import OutcomeOpenCodeAgent


def actor(tmp_path, reasons, *, sessions=None, exhaust=False):
    agent = object.__new__(OutcomeOpenCodeAgent)
    agent.turn, agent.session, agent.actor_remaining = 0, None, 60
    agent.result_root = tmp_path
    agent.logs_dir = tmp_path / "logs"
    agent.logs_dir.mkdir()
    agent.environment = SimpleNamespace()
    agent._error_messages = lambda: []
    agent.snapshot = AsyncMock(return_value=({"candidate_id": "retained"}, {}))
    calls = []

    async def run(environment, prompt, *, raw_name, session, timeout_sec):
        i = len(calls)
        calls.append({"session": session, "timeout": timeout_sec, "prompt": prompt})
        sid = sessions[i] if sessions else "same-session"
        events = [
            {
                "type": "text",
                "sessionID": sid,
                "part": {
                    "messageID": str(i),
                    "text": "partial" if reasons[i] != "stop" else "done",
                },
            }
        ]
        if reasons[i] is not None:
            events.append(
                {
                    "type": "step_finish",
                    "sessionID": sid,
                    "part": {"reason": reasons[i]},
                }
            )
        (agent.logs_dir / raw_name).write_text("\n".join(json.dumps(e) for e in events))
        if exhaust:
            agent.actor_remaining = 0
        return True

    agent._opencode_run_bounded = run
    return agent, calls


@pytest.mark.asyncio
async def test_length_continues_same_session_with_shared_budget(tmp_path):
    agent, calls = actor(tmp_path, ["length", "stop"])
    assert await agent._turn("original task") == {"candidate_id": "retained"}
    assert [c["session"] for c in calls] == [None, "same-session"]
    assert 0 < calls[1]["timeout"] < calls[0]["timeout"]
    assert agent.turn == 1
    agent.snapshot.assert_awaited_once_with("delivery-1", "done")
    assert len(list((tmp_path / "logs").glob("*.jsonl"))) == 2
    receipt = json.loads((tmp_path / "continuations/turn-1-0.json").read_text())
    assert receipt["will_continue"] and receipt["session"] == "same-session"


@pytest.mark.asyncio
async def test_repeated_length_stops_after_two_continuations(tmp_path):
    agent, calls = actor(tmp_path, ["length"] * 3)
    with pytest.raises(RuntimeError, match="after two same-session continuations"):
        await agent._turn("original task")
    assert len(calls) == 3
    agent.snapshot.assert_not_awaited()
    assert (
        json.loads((tmp_path / "continuations/turn-1-2.json").read_text())[
            "will_continue"
        ]
        is False
    )


@pytest.mark.asyncio
async def test_length_does_not_reset_execution_budget(tmp_path):
    agent, calls = actor(tmp_path, ["length"], exhaust=True)
    with pytest.raises(TimeoutError, match="budget exhausted"):
        await agent._turn("original task")
    assert len(calls) == 1
    agent.snapshot.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", [None, "tool-calls"])
async def test_missing_terminal_is_not_silently_retried(tmp_path, reason):
    agent, calls = actor(tmp_path, [reason])
    with pytest.raises(ValueError, match="terminal session receipt"):
        await agent._turn("original task")
    assert len(calls) == 1
    agent.snapshot.assert_not_awaited()


@pytest.mark.asyncio
async def test_continuation_rejects_changed_session(tmp_path):
    agent, _ = actor(tmp_path, ["length", "stop"], sessions=["original", "different"])
    with pytest.raises(ValueError, match="changed session"):
        await agent._turn("original task")
    agent.snapshot.assert_not_awaited()
