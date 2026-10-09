"""Validate public bytes while retaining original scientific identities.

Sanitization changes serialized paths and credentials. A source hash is never
presented as the checksum of the public file: both values are recorded explicitly.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


class Evidence:
    def __init__(self, data_root: Path):
        self.data_root = Path(data_root).resolve()
        self.root = self.data_root.parent
        self.records = {}
        self.canonical_identities = {}
        with (self.data_root / 'provenance/files.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                if row['path'] in self.records:
                    raise ValueError('Duplicate evidence path: ' + row['path'])
                name = Path(row['path'])
                if name.is_absolute() or '..' in name.parts or not name.parts or name.parts[0] != 'data':
                    raise ValueError('Evidence path escapes data root: ' + row['path'])
                self.records[row['path']] = row
                if 'public_canonical_sha256' in row:
                    key = row['public_canonical_sha256']
                    old = self.canonical_identities.get(key)
                    source = row['source_canonical_sha256']
                    if key in self.canonical_identities and old != source:
                        # Equivalent public values can originate from different run
                        # metadata. Such values cannot recover a unique source ID.
                        self.canonical_identities[key] = None
                    else:
                        self.canonical_identities[key] = source

    def resolve(self, name: str) -> Path:
        path = (self.root / name).resolve()
        if not path.is_relative_to(self.data_root):
            raise ValueError('Evidence path escapes data root: ' + name)
        return path

    def original_hash(self, path: Path) -> str | None:
        path = Path(path).resolve()
        if not path.is_file():
            return None
        try:
            relative = str(path.relative_to(self.root))
        except ValueError:
            return sha256(path)
        row = self.records.get(relative)
        actual = sha256(path)
        if row is None:
            if path.is_relative_to(self.data_root):
                raise ValueError('Unregistered evidence: ' + relative)
            return actual
        if actual != row['sha256']:
            raise ValueError('Public evidence checksum mismatch: ' + relative)
        return row['source_sha256']

    def original_canonical_hash(self, value) -> str:
        current = canonical_hash(value)
        original = self.canonical_identities.get(current, current)
        if original is None:
            raise ValueError('Ambiguous original canonical identity')
        return original

    def verify(self, names=None) -> int:
        selected = self.records if names is None else names
        count = 0
        for name in selected:
            row = self.records[name]
            path = self.resolve(name)
            if not path.is_file() or path.stat().st_size != row['bytes'] or sha256(path) != row['sha256']:
                raise ValueError('Public evidence missing or changed: ' + name)
            count += 1
        return count
