"""Native artifacts stay bound even when two consecutive deliveries are equal."""

import pytest

from reward_engine.adapters.jiuwenswarm.checkpoints import (
    artifact_bindings,
    verify_actor,
)
from reward_engine.adapters.workbuddy.io import file_hash, write


@pytest.mark.parametrize("changed", ["session", "delivery", "result", "turn"])
def test_native_actor_replay_rejects_mutation(tmp_path, changed):
    candidate = {"artifact": {"final_response": "same delivery"}}
    for turn in (1, 2):
        root = tmp_path / "snapshots" / str(turn)
        (root / "workspace").mkdir(parents=True)
        (root / "workspace/output.txt").write_text("same")
        write(root / "candidate.json", candidate)
        write(root / "inventory.json", [{"path": "output.txt", "sha256": file_hash(root / "workspace/output.txt")}])
    session = tmp_path / "sessions/2/history.jsonl"
    session.parent.mkdir(parents=True)
    session.write_text('{"session":"real"}\n')
    row = {"candidate": candidate, "session": "real", "runtime_directory": str(tmp_path),
           "resources": {"turns": 2}, "artifact_bindings": artifact_bindings(tmp_path)}
    write(tmp_path / "RESULT.json", row)
    verify_actor(row)
    if changed == "session":
        session.write_text("changed")
    elif changed == "delivery":
        (tmp_path / "snapshots/2/workspace/output.txt").write_text("changed")
    elif changed == "result":
        write(tmp_path / "RESULT.json", {**row, "session": "different"})
    else:
        row["resources"]["turns"] = 3
    with pytest.raises(ValueError):
        verify_actor(row)
