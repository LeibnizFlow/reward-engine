"""Verify retained native delivery bytes before reusing a completed stage."""

from pathlib import Path

from reward_engine.adapters.workbuddy.io import file_hash, read


def verify_actor(row):
    from reward_engine.adapters.claw_eval.actor_retry import verify_actor_retry

    from .recovery import verify as verify_recoveries

    root = Path(row["runtime_directory"])
    result = read(root / "RESULT.json")
    if "actor_retry" in row or "actor_retry" in result:
        if any(row.get(key) != result.get(key) for key in
               ("actor_retry", "actor_execution_s", "identity")):
            raise ValueError("Retained Actor retry lineage or execution cost changed")
    verify_actor_retry(row)
    verify_recoveries(root, row)
    if (
        "artifact_bindings" in row
        and artifact_bindings(root) != row["artifact_bindings"]
    ):
        raise ValueError("Retained native/session artifact bytes changed")
    if (
        result.get("candidate") != row["candidate"]
        or result.get("session") != row["session"]
    ):
        raise ValueError("Retained native actor result changed")
    turn = row.get("resources", {}).get("turns")
    if type(turn) is not int or turn < 1:
        raise ValueError("Retained actor turn binding missing")
    snapshots = [root / "snapshots" / name / "candidate.json"
                 for name in (str(turn), f"delivery-{turn}")
                 if (root / "snapshots" / name / "candidate.json").is_file()]
    matches = [p for p in snapshots if read(p) == row["candidate"]]
    if len(matches) != 1:
        raise ValueError("Delivery snapshot binding missing or ambiguous")
    parent = matches[0].parent
    for item in read(parent / "inventory.json"):
        path = parent / "workspace" / item["path"]
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to((parent / "workspace").resolve())
            or file_hash(path) != item["sha256"]
        ):
            raise ValueError("Retained delivery file changed")


def artifact_bindings(root):
    root = Path(root)
    files = [
        p
        for directory in ("snapshots", "sessions", "jiuwenswarm/process")
        for p in (root / directory).rglob("*")
        if p.is_file()
    ]
    files.extend(
        root / name
        for name in ("trace.jsonl", "environment-snapshot.json")
        if (root / name).exists()
    )
    if any(
        p.is_symlink() or not p.resolve().is_relative_to(root.resolve()) for p in files
    ):
        raise ValueError("Native artifact escapes retained trial")
    return {str(p.relative_to(root)): file_hash(p) for p in sorted(files)}
