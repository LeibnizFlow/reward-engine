from copy import deepcopy

import pytest

from reward_engine.core.outcome.task_sequence import (
    ScopedFeedback,
    TaskSequence,
    require_version_scope,
)


def sequence():
    return TaskSequence(
        {
            "task_id": "task",
            "prompt": "Prepare a report in USD.",
            "documents": [{"path": "schema.json", "content": "USD"}],
        }
    )


def test_only_revealed_user_turns_change_scope_and_old_deliveries_cannot_be_restored():
    state = sequence()
    first = state.observe({"artifact": {"final_response": "FIRST"}})
    assert state.advance(
        "Use CNY instead.",
        event_id="native-2",
        documents=[{"path": "schema.json", "content": "CNY"}],
    )
    second = state.observe({"artifact": {"final_response": "SECOND"}})
    assert first["task_scope_sha256"] != second["task_scope_sha256"]
    assert state.task["documents"][0]["content"] == "CNY"
    assert (
        "FIRST" not in state.task["prompt"]
    )  # Actor claims are not user requirements.
    assert not state.advance(
        "Use CNY instead.",
        event_id="native-2",
        documents=[{"path": "schema.json", "content": "CNY"}],
    )
    with pytest.raises(ValueError, match="changed"):
        state.advance("Use EUR.", event_id="native-2")
    with pytest.raises(ValueError, match="earlier"):
        require_version_scope([{"candidate": first}], second)
    require_version_scope([{"candidate": second}], second)


@pytest.mark.asyncio
@pytest.mark.parametrize("defect", ["hash", "different_candidate"])
async def test_single_version_shortcut_still_checks_immutable_current_delivery(defect):
    from reward_engine.core.outcome.construction import fingerprint
    from reward_engine.core.outcome.version_selection import select_versions
    from tests.support.claw_outcome_multiturn import TASK, pack

    state = TaskSequence(TASK)
    candidate = state.observe({"artifact": {"final_response": "Delivered"}})
    version = {
        "candidate_id": "delivery_1",
        "candidate": deepcopy(candidate),
        "artifact_sha256": fingerprint(candidate["artifact"]),
    }
    if defect == "hash":
        version["artifact_sha256"] = "changed"
    else:
        candidate["artifact"]["final_response"] = "Not the recorded version"
    rows = {
        "C": {
            "candidate": candidate,
            "effective_task": TASK,
            "effective_pack": pack(),
            "review_session": {"versions": [version]},
        }
    }
    with pytest.raises(ValueError, match="mismatch|differs"):
        await select_versions(TASK, pack(), rows, None, lambda _: None)


@pytest.mark.asyncio
async def test_scope_refresh_never_reuses_old_authority_or_resets_native_deadline():
    state = sequence()
    now, calls, records = [0], [], {}

    async def refresh(task, deliveries):
        calls.append(deepcopy(task))
        assert flow.current is None
        if len(calls) == 2:
            raise ValueError("Rebuild unavailable")
        return {"pack": {"task_id": task["task_id"], "task_prompt": task["prompt"]}}

    flow = ScopedFeedback(
        state, refresh, deadline=100, clock=lambda: now[0], record=records.__setitem__
    )
    assert await flow.advance("Add a table.", event_id="2")
    assert await flow.advance("Add a table.", event_id="2")
    assert len(calls) == 1
    assert await flow.advance("Remove the table.", event_id="3") is None
    assert flow.current is None and flow.deadline == 100
    now[0] = 101
    assert await flow.advance("Use a chart.", event_id="4") is None
    assert len(calls) == 2
    assert records["task_scope/" + state.scope]["reason"] == "native_deadline"
