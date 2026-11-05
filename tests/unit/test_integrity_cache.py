"""Persistent reuse must not hide edits, changed inventories, or corrupt caches."""

import json
import os
import sqlite3
import subprocess
import sys

import pytest

from reward_engine.adapters.integrity import tree_hashes, verification_cache
from reward_engine.adapters.integrity_cache import DigestCache
from reward_engine.adapters.workbuddy.io import file_hash


def test_warm_verification_reads_no_unchanged_bytes_across_instances(tmp_path):
    tree = tmp_path / 'tree'
    tree.mkdir()
    (tree / 'a').write_text('evidence')
    cache_path = tmp_path / 'cache.sqlite3'
    with verification_cache(cache_path) as cache:
        expected = tree_hashes(tree, file_hash)
        assert (cache.reads, cache.hits) == (1, 0)
    with verification_cache(cache_path) as cache:
        assert tree_hashes(tree, file_hash) == expected
        assert (cache.reads, cache.hits) == (0, 1)


def test_another_process_can_reuse_verified_hashes(tmp_path):
    path = tmp_path / 'receipt'
    path.write_text('evidence')
    database = tmp_path / 'cache.sqlite3'
    cache = DigestCache(database)
    expected = cache.digest(path)
    cache.close()
    result = subprocess.check_output([
        sys.executable, '-c',
        'import json,sys; from reward_engine.adapters.integrity_cache import DigestCache; '
        'c=DigestCache(sys.argv[1]); d=c.digest(sys.argv[2]); '
        'print(json.dumps([d,c.reads,c.hits])); c.close()',
        str(database), str(path)], text=True)
    assert json.loads(result) == [expected, 0, 1]


def test_force_full_rechecks_bytes_even_with_warm_cache(tmp_path, monkeypatch):
    tree = tmp_path / 'tree'
    tree.mkdir()
    (tree / 'a').write_text('evidence')
    database = tmp_path / 'cache.sqlite3'
    with verification_cache(database):
        expected = tree_hashes(tree, file_hash)
    monkeypatch.setenv('REWARD_ENGINE_VERIFY_FULL', '1')
    with verification_cache(database) as cache:
        assert tree_hashes(tree, file_hash) == expected
        assert (cache.reads, cache.hits) == (1, 0)


@pytest.mark.parametrize('replace', [False, True])
def test_same_size_edit_with_restored_mtime_invalidates_cache(tmp_path, replace):
    path = tmp_path / 'receipt'
    path.write_text('before')
    stamp = path.stat()
    cache = DigestCache(tmp_path / 'cache.sqlite3')
    old = cache.digest(path)
    if replace:
        other = tmp_path / 'replacement'
        other.write_text('after!')
        other.replace(path)
    else:
        path.write_text('after!')
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    assert cache.digest(path) != old
    assert cache.reads == 2
    cache.close()


def test_warm_cache_still_enumerates_additions_deletions_and_symlinks(tmp_path):
    tree = tmp_path / 'tree'
    tree.mkdir()
    receipt = tree / 'receipt'
    receipt.write_text('evidence')
    with verification_cache(tmp_path / 'cache.sqlite3'):
        initial = tree_hashes(tree, file_hash)
        (tree / 'new').write_text('new')
        assert set(tree_hashes(tree, file_hash)) == {'new', 'receipt'}
        receipt.unlink()
        assert tree_hashes(tree, file_hash) != initial
        receipt.symlink_to(tree / 'new')
        with pytest.raises(ValueError, match='symlinks'):
            tree_hashes(tree, file_hash)


def test_tampered_cache_value_cannot_replace_actual_digest(tmp_path):
    path = tmp_path / 'receipt'
    path.write_text('evidence')
    database = tmp_path / 'cache.sqlite3'
    cache = DigestCache(database)
    expected = cache.digest(path)
    cache.close()
    with sqlite3.connect(database) as connection:
        connection.execute('UPDATE digests SET digest=?', ('0' * 64,))
    cache = DigestCache(database)
    assert cache.digest(path) == expected
    assert cache.reads == 1
    cache.close()


def test_corrupt_database_falls_back_to_reading_bytes(tmp_path):
    path = tmp_path / 'receipt'
    path.write_text('evidence')
    database = tmp_path / 'cache.sqlite3'
    database.write_bytes(b'not a sqlite database')
    cache = DigestCache(database)
    assert cache.digest(path) == file_hash(path)
    assert cache.reads == 1
    cache.close()


def test_mutation_during_read_never_publishes_cache_entry(tmp_path, monkeypatch):
    from reward_engine.adapters.workbuddy import io

    path = tmp_path / 'receipt'
    path.write_text('before')
    cache = DigestCache(tmp_path / 'cache.sqlite3')
    original = io.file_hash

    def changed(value):
        result = original(value)
        value.write_text('after!')
        return result

    monkeypatch.setattr(io, 'file_hash', changed)
    with pytest.raises(ValueError, match='changed during'):
        cache.digest(path)
    assert cache.pending == {}
    cache.close()


@pytest.mark.parametrize('mutation', ['content', 'addition', 'deletion'])
def test_jiuwenswarm_bundle_reuses_cache_without_hiding_changes(tmp_path, mutation):
    from reward_engine.adapters.jiuwenswarm.bundle import verify

    root = tmp_path / 'runtime'
    (root / 'bin').mkdir(parents=True)
    launcher = root / 'bin/jiuwenswarm-process'
    launcher.write_text('#!/bin/sh\nexit 0\n')
    payload = root / 'payload'
    payload.write_text('before')
    (root / 'manifest.json').write_text(json.dumps({
        'format': 'jiuwenswarm-runtime-bundle-v1', 'schema': '0.1',
        'packages': {'workswarm': '0.2.7'},
        'files': {'bin/jiuwenswarm-process': file_hash(launcher),
                  'payload': file_hash(payload)},
    }))
    database = tmp_path / 'cache.sqlite3'
    with verification_cache(database):
        expected = verify(root)
    with verification_cache(database) as cache:
        assert verify(root) == expected
        assert (cache.reads, cache.hits) == (0, 2)
        if mutation == 'content':
            stamp = payload.stat()
            payload.write_text('after!')
            os.utime(payload, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        elif mutation == 'addition':
            (root / 'unexpected').write_text('new')
        else:
            payload.unlink()
        with pytest.raises(ValueError, match='changed'):
            verify(root)
