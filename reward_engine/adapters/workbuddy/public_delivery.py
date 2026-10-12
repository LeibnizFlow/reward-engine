"""Public contract freezing and honest pre-verifier workspace inventory collection."""

import hashlib
import json
import os
import stat
from pathlib import Path

from reward_engine.core.outcome.public_delivery import (
    INVENTORY,
    fingerprint,
    safe_path,
    validate,
)


def _open_regular(workspace, relative):
    """Open beneath the trusted root without following any path component link."""
    parts = safe_path(relative).split("/")
    directory = os.open(workspace, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in parts[:-1]:
            child = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory
            )
            os.close(directory)
            directory = child
        return os.open(
            parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory
        )
    finally:
        os.close(directory)


def scan_workspace(workspace, *, excluded=(), initial_hashes=None):
    workspace = Path(workspace)
    files, blocked = [], []
    complete = True
    if workspace.is_symlink() or not workspace.is_dir():
        raise ValueError("Workspace must be a real directory, not a link")

    def failure(error):
        nonlocal complete
        complete = False
        path = Path(error.filename) if error.filename else workspace
        blocked.append(
            {"path": path.relative_to(workspace).as_posix(), "reason": "scan_error"}
        )

    for folder, dirs, names in os.walk(workspace, followlinks=False, onerror=failure):
        for name in sorted(dirs + names):
            path = Path(folder) / name
            relative = path.relative_to(workspace).as_posix()
            if name in excluded:
                if name in dirs:
                    dirs.remove(name)
                continue
            try:
                info = path.lstat()
                if stat.S_ISLNK(info.st_mode):
                    blocked.append({"path": relative, "reason": "symlink"})
                    if name in dirs:
                        dirs.remove(name)
                elif stat.S_ISDIR(info.st_mode):
                    blocked.append({"path": relative, "reason": "directory"})
                elif not stat.S_ISREG(info.st_mode):
                    blocked.append({"path": relative, "reason": "special"})
                else:
                    descriptor = _open_regular(workspace, relative)
                    with os.fdopen(descriptor, "rb") as stream:
                        before = os.fstat(stream.fileno())
                        if not stat.S_ISREG(before.st_mode):
                            raise OSError("Path changed to a nonregular file")
                        sha = hashlib.file_digest(stream, "sha256").hexdigest()
                        after = os.fstat(stream.fileno())

                    def identity(s):
                        return (
                            s.st_dev,
                            s.st_ino,
                            s.st_size,
                            s.st_mtime_ns,
                            s.st_ctime_ns,
                        )

                    if (
                        identity(info) != identity(before)
                        or identity(before) != identity(after)
                        or identity(after) != identity(path.lstat())
                    ):
                        complete = False
                        blocked.append(
                            {"path": relative, "reason": "changed_during_collection"}
                        )
                        continue
                    files.append(
                        {
                            "path": relative,
                            "sha256": sha,
                            "bytes": after.st_size,
                            "changed": initial_hashes is not None
                            and initial_hashes.get(relative) != sha,
                        }
                    )
            except OSError:
                complete = False
                blocked.append({"path": relative, "reason": "read_error"})
    files.sort(key=lambda row: row["path"])
    blocked.sort(key=lambda row: (row["path"], row["reason"]))
    return {
        "schema": INVENTORY,
        "phase": "before_official_verifier",
        "complete": complete,
        "files": files,
        "blocked": blocked,
        "excluded_components": sorted(excluded),
        "inventory_sha256": fingerprint(files),
    }


def read_bound_contract(binding, task):
    if (
        set(binding) != {"path", "sha256", "contract_sha256", "task_id", "schema"}
        or binding["schema"] != "public_delivery.frozen_binding.v1"
    ):
        raise ValueError("Unknown frozen public contract binding")
    path = Path(binding["path"])
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != binding["sha256"]:
        raise ValueError("Frozen public contract bytes changed")
    contract = validate(json.loads(data), task)
    if (
        binding["contract_sha256"] != fingerprint(contract)
        or binding["task_id"] != task["task_id"]
    ):
        raise ValueError("Frozen public contract binding mismatch")
    return contract


def freeze_contract(source, output, tasks_root, task_ids):
    data = Path(source).read_bytes()
    contract = json.loads(data)
    task_id = safe_path(contract["task_id"])
    if "/" in task_id or task_id not in task_ids:
        raise ValueError("Public contract must bind a selected task")
    source_path = safe_path(contract["source"]["path"])
    task_root = Path(tasks_root) / task_id
    source_file = task_root / source_path
    current = task_root
    for part in source_path.split("/"):
        current /= part
        if current.is_symlink():
            raise ValueError("Public instruction source cannot contain links")
    if (
        not source_file.resolve().is_relative_to(task_root.resolve())
        or source_file.is_symlink()
    ):
        raise ValueError("Public instruction source escapes task")
    # The actual adapter task prompt, not an arbitrary file chosen by a contract.
    prompt = (task_root / "instruction.md").read_bytes().decode("utf-8")
    if source_file.read_bytes() != prompt.encode():
        raise ValueError("Contract source is not the public task prompt")
    validate(contract, {"task_id": task_id, "prompt": prompt})
    target = Path(output) / "public_delivery_contract.json"
    with target.open("xb") as stream:
        stream.write(data)
    return {
        "path": str(target.resolve()),
        "sha256": hashlib.sha256(data).hexdigest(),
        "contract_sha256": fingerprint(contract),
        "task_id": task_id,
        "schema": "public_delivery.frozen_binding.v1",
    }
