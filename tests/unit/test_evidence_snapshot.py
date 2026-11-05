"""Snapshot storage limits must not discard or weaken retained evidence."""

import errno
import os
from pathlib import Path

import pytest

from reward_engine.adapters.evidence_snapshot import EvidenceSnapshots
from reward_engine.adapters.integrity import verification_cache
from reward_engine.adapters.workbuddy.io import file_hash


def refuse_archive_link(monkeypatch, archive, code=errno.EMLINK, before=None):
    original = os.link

    def link(source, destination, **kwargs):
        if Path(destination).parent == archive:
            if before:
                before(Path(source))
            raise OSError(code, os.strerror(code))
        return original(source, destination, **kwargs)

    monkeypatch.setattr(os, 'link', link)


def test_saturated_object_keeps_old_snapshots_and_copies_new_immutable_bytes(tmp_path, monkeypatch):
    source = tmp_path / 'live'
    source.write_bytes(b'synthetic retained receipt')
    archive = tmp_path / 'archive'
    archive.mkdir()
    with verification_cache(tmp_path / 'cache.sqlite3') as cache:
        snapshots = EvidenceSnapshots(tmp_path / 'objects', cache)
        digest = snapshots.copy(source, archive / 'first')
        original_inode = (archive / 'first').stat().st_ino
        refuse_archive_link(monkeypatch, archive)
        for name in ('second', 'third'):
            target = archive / name
            assert snapshots.copy(source, target) == digest
            assert file_hash(target) == digest
            assert target.stat().st_ino not in {original_inode, source.stat().st_ino}
            assert not target.stat().st_mode & 0o222
        source.write_bytes(b'changed live task')
        assert all(file_hash(archive / name) == digest for name in ('first', 'second', 'third'))
        assert (archive / 'first').stat().st_ino == original_inode


@pytest.mark.parametrize('code', [errno.ENOSPC, errno.EIO])
def test_other_storage_errors_propagate(tmp_path, monkeypatch, code):
    source = tmp_path / 'live'
    source.write_text('synthetic evidence')
    archive = tmp_path / 'archive'
    archive.mkdir()
    refuse_archive_link(monkeypatch, archive, code)
    with verification_cache(tmp_path / 'cache.sqlite3') as cache:
        with pytest.raises(OSError) as error:
            EvidenceSnapshots(tmp_path / 'objects', cache).copy(source, archive / 'receipt')
    assert error.value.errno == code
    assert not (archive / 'receipt').exists()


def test_link_fallback_never_overwrites_existing_destination(tmp_path, monkeypatch):
    source = tmp_path / 'live'
    source.write_text('synthetic evidence')
    archive = tmp_path / 'archive'
    archive.mkdir()
    target = archive / 'receipt'
    target.write_bytes(b'existing historical evidence')
    refuse_archive_link(monkeypatch, archive)
    with verification_cache(tmp_path / 'cache.sqlite3') as cache:
        with pytest.raises(FileExistsError):
            EvidenceSnapshots(tmp_path / 'objects', cache).copy(source, target)
    assert target.read_bytes() == b'existing historical evidence'


def test_link_fallback_rejects_corrupt_object_and_removes_partial_copy(tmp_path, monkeypatch):
    source = tmp_path / 'live'
    source.write_bytes(b'synthetic evidence')
    archive = tmp_path / 'archive'
    archive.mkdir()

    def corrupt(path):
        path.chmod(0o644)
        path.write_bytes(b'corrupted evidence')
        path.chmod(0o444)

    refuse_archive_link(monkeypatch, archive, before=corrupt)
    with verification_cache(tmp_path / 'cache.sqlite3') as cache:
        with pytest.raises(ValueError, match='snapshot'):
            EvidenceSnapshots(tmp_path / 'objects', cache).copy(source, archive / 'receipt')
    assert not (archive / 'receipt').exists()
