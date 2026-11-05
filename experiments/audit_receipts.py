"""Audit receipts for the existing public entry point."""

import hashlib
import json
import math
from pathlib import Path


class IncompleteReceipt(Exception):
    """A producer has not published a readable receipt yet."""


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def text(value):
    return isinstance(value, str) and bool(value.strip())


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def receipt_digest(value):
    """SHA-256 of UTF-8 canonical JSON: sorted keys, compact, no ASCII escaping."""
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def strings(value):
    return isinstance(value, list) and bool(value) and all(map(text, value))


def ids(rows, key):
    require(isinstance(rows, list) and rows, f"Nonempty {key} rows required")
    values = [row[key] for row in rows]
    require(all(map(text, values)), f"Nonempty {key} identities required")
    require(len(set(values)) == len(values), f"Duplicate {key} identities")
    return values


def candidate(row):
    require(text(row.get("candidate_id")), "Candidate identity missing")
    artifact = row["artifact"]
    require(text(artifact.get("final_response")), "No nonempty native final response")
    return row["candidate_id"]


def retained_workspace(r, snapshot, observed):
    """Verify retained native bytes, not merely the extracted candidate JSON."""
    from reward_engine.adapters.workbuddy.evidence import EXCLUDED

    directory = snapshot.parent
    for parent in (directory, *directory.parents):
        require(not (r.root / parent).is_symlink(), "Linked native snapshot directory")
    inventory = r.read(directory / "inventory.json")
    require(isinstance(inventory, list), "Native inventory must be a list")
    workspace = r.path(directory / "workspace")
    require(
        workspace.is_dir() and not workspace.is_symlink(),
        "Native retained workspace missing or linked",
    )

    def relative_path(name):
        require(text(name), "Empty inventory path")
        path = Path(name)
        require(
            not path.is_absolute()
            and name == path.as_posix()
            and path.parts
            and not {".", ".."}.intersection(path.parts),
            "Unsafe native inventory path",
        )
        return path

    names = []
    for entry in inventory:
        path = relative_path(entry["path"])
        require(not set(path.parts) & EXCLUDED, "Excluded file in native inventory")
        require(entry["path"] not in names, "Duplicate native inventory path")
        names.append(entry["path"])
        require(
            type(entry.get("bytes")) is int
            and entry["bytes"] >= 0
            and type(entry.get("changed")) is bool,
            "Invalid native inventory metadata",
        )
        for parent in (path, *path.parents):
            require(
                not (workspace / parent).is_symlink(), "Linked native inventory path"
            )
        raw = r.raw(directory / "workspace" / path)
        require(
            len(raw) == entry["bytes"]
            and hashlib.sha256(raw).hexdigest() == entry.get("sha256"),
            "Retained native workspace bytes differ from inventory: " + entry["path"],
        )
    actual = {
        path.relative_to(workspace).as_posix()
        for path in workspace.rglob("*")
        if not set(path.relative_to(workspace).parts) & EXCLUDED
        and path.is_file()
        and not path.is_symlink()
    }
    require(
        actual == set(names), "Retained workspace files differ from native inventory"
    )
    require(
        observed["artifact"].get("changed_files")
        == [row for row in inventory if row["changed"]],
        "Candidate changed-file hashes differ from native inventory",
    )
    environment = observed["environment"]
    require(
        environment.get("file_count") == len(inventory),
        "Candidate file count differs from native inventory",
    )
    deleted = environment.get("deleted_files")
    require(
        isinstance(deleted, list) and len(set(deleted)) == len(deleted),
        "Invalid native deleted-file inventory",
    )
    for name in deleted:
        path = relative_path(name)
        require(
            name not in actual
            and not (workspace / path).exists()
            and not (workspace / path).is_symlink(),
            "Recorded deleted file is still present",
        )


class Receipts:
    def __init__(self, root, closed=False):
        self.root, self.closed = root, closed
        self.paths, self.facts = [], {}

    def path(self, relative):
        path = self.root / relative
        require(
            path.resolve().is_relative_to(self.root), "Evidence escapes run directory"
        )
        name = path.relative_to(self.root).as_posix()
        if name not in self.paths:
            self.paths.append(name)
        return path

    def raw(self, relative):
        try:
            return self.path(relative).read_bytes()
        except FileNotFoundError as error:
            raise IncompleteReceipt(f"Missing receipt: {relative}") from error

    def read(self, relative, *, lines=False):
        try:
            raw = self.raw(relative).decode("utf-8")
            return (
                [json.loads(line) for line in raw.splitlines() if line.strip()]
                if lines
                else json.loads(raw)
            )
        except (UnicodeError, json.JSONDecodeError) as error:
            raise IncompleteReceipt(
                f"Incomplete or malformed JSON: {relative}"
            ) from error

    def optional(self, relative):
        path = self.path(relative)
        return self.read(relative) if path.exists() else None

    def bound_path(self, recorded, expected):
        require(text(recorded), "Missing native evidence path")
        observed = Path(recorded)
        if not observed.is_absolute():
            observed = self.root / observed
        require(
            observed.resolve() == self.path(expected).resolve(),
            "Native path belongs to another slot",
        )
