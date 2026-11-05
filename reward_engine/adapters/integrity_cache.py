"""Local SHA-256 reuse guarded by kernel metadata and authenticated cache rows.

This is a local optimization, not a replacement for expected digest comparison.
The cache and its key belong to the same trusted account as the verifier. Every
tree inventory is still enumerated. A cache miss or damaged cache reads bytes.
"""

import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import stat
import threading
from pathlib import Path


def identity(path):
    value = Path(path).lstat()
    if not stat.S_ISREG(value.st_mode):
        raise ValueError("Integrity input must be a regular file, not a symlink")
    return (value.st_dev, value.st_ino, value.st_mode, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns, value.st_uid, value.st_gid)


class DigestCache:
    """Only store digests whose file identity was stable across the read."""

    def __init__(self, path, *, reuse=True):
        self.path = Path(path)
        self.reuse = reuse
        try:
            self.boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        except OSError:
            # Without a kernel boot identity, do not trust a prior process's
            # inode timestamps across remounts or reboots.
            self.boot = secrets.token_hex(16)
        self.lock = threading.RLock()
        self.pending = {}
        self.connection = None
        self.key = None
        self.hits = self.reads = 0
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            key_path = self.path.with_suffix(self.path.suffix + '.key')
            try:
                descriptor = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                pass
            else:
                with os.fdopen(descriptor, 'wb') as stream:
                    stream.write(secrets.token_bytes(32))
            self.key = key_path.read_bytes()
            if len(self.key) != 32:
                return
            self.connection = sqlite3.connect(self.path, timeout=0.2, check_same_thread=False)
            self.connection.execute('PRAGMA journal_mode=WAL')
            self.connection.execute('CREATE TABLE IF NOT EXISTS digests '
                                    '(identity TEXT PRIMARY KEY, digest TEXT, signature TEXT)')
            self.connection.commit()
        except (OSError, sqlite3.Error):
            self._disable()

    def _disable(self):
        if self.connection is not None:
            self.connection.close()
        self.connection = None
        self.pending.clear()

    def _signature(self, key, digest):
        return hmac.new(self.key, (key + ':' + digest).encode(), hashlib.sha256).hexdigest()

    def digest(self, path):
        from reward_engine.adapters.workbuddy.io import file_hash

        before = identity(path)
        # Nanosecond ctime prevents restoring mtime from hiding an in-place edit;
        # inode/device distinguish replacement and copies. Schema binds SHA-256.
        key = json.dumps(('sha256-stat-v1', self.boot, *before), separators=(',', ':'))
        with self.lock:
            row = self.pending.get(key) if self.reuse else None
            if self.reuse and row is None and self.connection is not None:
                try:
                    row = self.connection.execute(
                        'SELECT digest, signature FROM digests WHERE identity=?', (key,)).fetchone()
                except sqlite3.Error:
                    self._disable()
            if (row and all(isinstance(value, str) for value in row)
                    and hmac.compare_digest(row[1], self._signature(key, row[0]))):
                if before != identity(path):
                    raise ValueError('File changed during integrity verification')
                self.hits += 1
                return row[0]
        digest = file_hash(path)
        if before != identity(path):
            raise ValueError('File changed during integrity verification')
        with self.lock:
            self.reads += 1
            if self.connection is not None:
                self.pending[key] = (digest, self._signature(key, digest))
                if len(self.pending) >= 1024:
                    self._flush()
        return digest

    def _flush(self):
        if self.connection is None or not self.pending:
            return
        try:
            self.connection.executemany('INSERT OR REPLACE INTO digests VALUES (?, ?, ?)',
                                        [(key, *row) for key, row in self.pending.items()])
            self.connection.commit()
            self.pending.clear()
        except sqlite3.Error:
            self._disable()

    def close(self):
        with self.lock:
            self._flush()
            self._disable()
