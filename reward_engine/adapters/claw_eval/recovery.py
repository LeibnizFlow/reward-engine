"""Retry a proven service setup failure without redrawing an actor session."""

import os
import re
import shutil
from pathlib import Path
from uuid import uuid4

from reward_engine.adapters.workbuddy.io import file_hash, read, write

RESOURCE_DENIALS = frozenset({
    "Storage admission floor reached", "Memory admission floor reached",
})


def _resource_denial(root, name, stage, slot, failure_path):
    """Recover only a bound denial that happened before an Actor slot existed."""
    history = root / name / "failures" / f"{stage.get('attempt')}.json"
    stage_path = root / name / "stage.json"
    if (type(stage.get("attempt")) is not int or not 1 <= stage["attempt"] < 3
            or stage.get("status") != "started" or not stage.get("identity")
            or slot.exists() or slot.is_symlink()):
        return False
    for path in (slot, history, failure_path, stage_path):
        if any(p.is_symlink() for p in (path, *path.parents)):
            return False
    if not all(p.is_file() for p in (history, failure_path, stage_path)):
        return False
    failure = read(failure_path)
    if (read(stage_path) != stage or file_hash(history) != file_hash(failure_path)
            or failure.get("error_type") != "RuntimeError"
            or failure.get("reason") not in RESOURCE_DENIALS
            or any(failure.get(key) != stage.get(key)
                   for key in ("identity", "attempt", "implementation"))):
        return False
    archive = root / "prelaunch_recoveries" / uuid4().hex
    archive.mkdir(parents=True)
    for path in (stage_path, failure_path):
        shutil.copy2(path, archive / path.name)
    write(archive / "recovery.json", {
        "stage": name, "identity": stage["identity"], "attempt": stage["attempt"],
        "reason": "Resource admission denied before Actor slot creation",
        "slot": str(slot), "actor_started": False,
        "file_hashes": {str(p.relative_to(root)): file_hash(p)
                        for p in (stage_path, failure_path, history)},
    })
    return True


def recover_prelaunch(root, name, stage):
    root = Path(root)
    if re.fullmatch(r"seed_[0-2]", name):
        slot = root / "seeds" / str(int(name[-1]) + 1)
    elif re.fullmatch(r"round_[1-3]/B[1-3]", name):
        round_name, arm = name.split("/")
        slot = root / round_name / "actors" / arm
    else:
        return False
    failure_path = root / name / "failure.json"
    if _resource_denial(root, name, stage, slot, failure_path):
        return True
    if not failure_path.is_file() or not (slot / "FAILURE.json").is_file():
        return False
    failure, actor_failure = read(failure_path), read(slot / "FAILURE.json")
    if (
        type(stage.get("attempt")) is not int
        or not 1 <= stage["attempt"] < 3
        or failure.get("attempt") != stage.get("attempt")
        or failure.get("identity") != stage.get("identity")
        or failure.get("error_type") != "ServiceStartError"
        or actor_failure.get("error_type") != "ServiceStartError"
        or actor_failure.get("session") is not None
        or not (slot / "STARTED.json").is_file()
        or read(slot / "STARTED.json").get("identity") != actor_failure.get("identity")
        # Process writes its request before spawning the CLI. Even a failed
        # request is execution evidence, so absence of a result is insufficient.
        or (slot / "jiuwenswarm").exists()
        or any((slot / filename).exists() for filename in ("RESULT.json", "trace.jsonl", "tool_events.jsonl"))
        or slot.is_symlink()
        or any(path.is_symlink() for path in slot.rglob("*"))
    ):
        return False
    archive = root / "prelaunch_recoveries" / uuid4().hex
    archive.mkdir(parents=True)
    hashes = {str(path.relative_to(slot)): file_hash(path)
              for path in slot.rglob("*") if path.is_file()}
    shutil.copy2(root / name / "stage.json", archive / "stage.json")
    shutil.copy2(failure_path, archive / "failure.json")
    write(archive / "recovery.json", {
        "stage": name, "identity": stage["identity"], "attempt": stage["attempt"],
        "reason": "Service setup failed before any Process request or actor session",
        "slot": str(slot), "file_hashes": hashes,
    })
    os.replace(slot, archive / "slot")
    return True
