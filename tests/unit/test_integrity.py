"""Integrity remains byte-based while traversal and reporting are bounded."""

import json
import os
import threading
import time

import pytest

from reward_engine.adapters.integrity import tree_hashes, verification_progress
from reward_engine.adapters.workbuddy.io import file_hash


def test_full_inventory_and_bytes_match_reference(tmp_path):
    for relative in ("a", "nested/b", "nested/deeper/c"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative)
    expected = {str(p.relative_to(tmp_path)): file_hash(p)
                for p in tmp_path.rglob("*") if p.is_file()}
    assert tree_hashes(tmp_path, file_hash) == expected


def test_restored_timestamp_does_not_hide_modified_bytes(tmp_path):
    path = tmp_path / "receipt"
    path.write_text("before")
    stat = path.stat()
    first = tree_hashes(tmp_path, file_hash)
    path.write_text("after!")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert tree_hashes(tmp_path, file_hash) != first


def test_new_deleted_files_and_symlinks_cannot_hide(tmp_path):
    path = tmp_path / "receipt"
    path.write_text("original")
    first = tree_hashes(tmp_path, file_hash)
    (tmp_path / "new").write_text("new")
    assert tree_hashes(tmp_path, file_hash) != first
    path.unlink()
    assert "receipt" not in tree_hashes(tmp_path, file_hash)
    path.symlink_to(tmp_path / "new")
    with pytest.raises(ValueError, match="symlinks"):
        tree_hashes(tmp_path, file_hash)


def test_readers_are_bounded_and_no_file_is_read_twice(tmp_path):
    for i in range(30):
        (tmp_path / str(i)).write_text(str(i))
    lock = threading.Lock()
    active = peak = 0
    calls = []

    def counted(path):
        nonlocal active, peak
        with lock:
            calls.append(path)
            active += 1
            peak = max(peak, active)
        try:
            time.sleep(.001)
            return file_hash(path)
        finally:
            with lock:
                active -= 1

    assert len(tree_hashes(tmp_path, counted)) == 30
    assert len(calls) == len(set(calls)) == 30
    assert peak <= 2


@pytest.mark.parametrize("fail", [False, True])
def test_progress_reports_completion_or_failure_without_model_calls(tmp_path, fail):
    root = tmp_path / "evidence"
    root.mkdir()
    (root / "receipt").write_text("synthetic")
    destination = tmp_path / "progress.json"

    def run():
        with verification_progress(destination, "startup_verification"):
            assert json.loads(destination.read_text())["status"] == "running"
            tree_hashes(root, file_hash, expected_files=1)
            if fail:
                raise ValueError("synthetic mismatch")

    if fail:
        with pytest.raises(ValueError):
            run()
    else:
        run()
    report = json.loads(destination.read_text())
    assert report["status"] == ("failed" if fail else "completed")
    assert report["steps"][0]["files_hashed"] == 1
    assert report["steps"][0]["expected_files"] == 1
    assert report["phase"] == "startup_verification"


def test_runtime_scans_once_and_rejects_changed_bytes(tmp_path, monkeypatch):
    from reward_engine.adapters.jiuwenswarm import bundle

    (tmp_path / "bin").mkdir()
    (tmp_path / "bin/jiuwenswarm-process").write_text("#!/bin/sh\nexit 0\n")
    code = tmp_path / "code.py"
    code.write_text("pass\n")
    manifest = {"format": "jiuwenswarm-runtime-bundle-v1", "schema": "0.1",
                "packages": {"workswarm": "0.2.7"},
                "files": tree_hashes(tmp_path, file_hash)}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(type(tmp_path), "rglob", lambda *_: pytest.fail("Repeated recursive scan"))
    assert bundle.verify(tmp_path)["jiuwenswarm_version"] == "0.2.7"
    stamp = code.stat()
    code.write_text("oops\n")
    os.utime(code, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    with pytest.raises(ValueError, match="Runtime bundle changed"):
        bundle.verify(tmp_path)


def test_legacy_runtime_still_rejects_symlinks_in_ignored_bytecode(tmp_path):
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "receipt").write_text("synthetic")
    (tmp_path / "__pycache__/alias").symlink_to(tmp_path / "receipt")
    with pytest.raises(ValueError, match="symlinks"):
        tree_hashes(tmp_path, file_hash, ignore_bytecode=True, inspect_ignored=True)
