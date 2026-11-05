"""Lineage reuse avoids inventory rereads while observing changed inputs."""

import os
import time

import pytest

from reward_engine.adapters.workbuddy import models
from reward_engine.adapters.workbuddy.io import read, write


@pytest.fixture
def lineage(tmp_path):
    parent = {
        "identity": "parent", "checkpoint_config_identity": "original",
        "output": str(tmp_path), "roles": {"one": "model-a", "two": "model-b"},
    }
    parent_path = tmp_path / "parent.json"
    manifest = tmp_path / "revision" / "manifest.json"
    write(parent_path, parent)
    write(manifest, {"parent_config": str(parent_path), "files": {"fixture": "hash"}})
    current = {**parent, "identity": "current",
               "implementation_revision": str(manifest.parent)}
    # Frozen revisions are old enough for metadata reuse; new files are not.
    time.sleep(1.02)
    return current, manifest, parent_path


def test_roles_reuse_manifest_pointer_but_observe_parent_config(lineage, monkeypatch):
    config, manifest, parent_path = lineage
    reads = []

    def counted(path):
        reads.append(path)
        return read(path)

    monkeypatch.setattr(models, "read", counted)
    assert models.compatible_revisions(config, "one") == ["parent"]
    assert models.compatible_revisions(config, "two") == ["parent"]
    assert reads.count(manifest) == 1
    parent = read(parent_path)
    parent["roles"]["one"] = "model-b"
    write(parent_path, parent)
    assert models.compatible_revisions(config, "one") == []
    assert models.compatible_revisions(config, "two") == ["parent"]
    assert reads.count(manifest) == 1


def test_manifest_edit_with_restored_mtime_invalidates_pointer(lineage):
    config, manifest, parent_path = lineage
    assert models.compatible_revisions(config, "one") == ["parent"]
    alternate = parent_path.with_name("second.json")
    write(alternate, {**read(parent_path), "identity": "second"})
    stamp = manifest.stat()
    # Same-length in-place replacement: mtime and size alone cannot detect it.
    manifest.write_text(manifest.read_text().replace("parent.json", "second.json"))
    os.utime(manifest, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    assert manifest.stat().st_size == stamp.st_size
    assert models.compatible_revisions(config, "one") == ["second"]


def test_dynamic_role_amendments_are_rechecked(lineage, tmp_path):
    config, _, parent_path = lineage
    parent_output = tmp_path / "old-output"
    parent_output.mkdir()
    write(parent_path, {**read(parent_path), "output": str(parent_output)})
    assert models.compatible_revisions(config, "one") == ["parent"]
    write(tmp_path / "role_models.json", {"roles": {"one": "model-b"}})
    assert models.compatible_revisions(config, "one") == []
    assert models.compatible_revisions(config, "two") == ["parent"]


def test_changed_or_invalid_manifest_is_not_cached(lineage, monkeypatch):
    config, manifest, parent_path = lineage

    def changing(path):
        value = read(path)
        if path == manifest:
            write(manifest, {"parent_config": str(parent_path), "changed": True})
        return value

    monkeypatch.setattr(models, "read", changing)
    with pytest.raises(ValueError, match="changed during lineage"):
        models.compatible_revisions(config, "one")
    monkeypatch.setattr(models, "read", read)
    manifest.write_text("{")
    with pytest.raises(ValueError):
        models.compatible_revisions(config, "one")
    write(manifest, {"parent_config": str(parent_path)})
    assert models.compatible_revisions(config, "one") == ["parent"]


def test_warmed_manifest_cannot_be_replaced_by_symlink(lineage):
    config, manifest, _ = lineage
    assert models.compatible_revisions(config, "one") == ["parent"]
    archived = manifest.with_suffix(".original")
    manifest.rename(archived)
    manifest.symlink_to(archived)
    with pytest.raises(ValueError, match="regular file"):
        models.compatible_revisions(config, "one")


def test_recent_manifests_are_read_even_when_metadata_collides(lineage, monkeypatch):
    config, manifest, parent_path = lineage
    stamp = models.file_identity(manifest)
    monkeypatch.setattr(models, "file_identity", lambda _: stamp)
    monkeypatch.setattr(models.time, "time_ns", lambda: stamp[5] + 1)
    assert models.compatible_revisions(config, "one") == ["parent"]
    alternate = parent_path.with_name("second.json")
    write(alternate, {**read(parent_path), "identity": "second"})
    write(manifest, {"parent_config": str(alternate)})
    assert models.compatible_revisions(config, "one") == ["second"]
