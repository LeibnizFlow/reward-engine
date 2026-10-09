"""Portable data access with explicit original/public hash separation."""
from pathlib import Path
import json

from release.evidence import Evidence, canonical_hash, sha256

REPO = Path(__file__).resolve().parents[3]
OUTPUT = REPO / 'data/analysis'
_evidence = None


def configure(data_root):
    global REPO, OUTPUT, _evidence
    data_root = Path(data_root).resolve()
    REPO = data_root.parent
    OUTPUT = data_root / 'analysis'
    _evidence = Evidence(data_root)


def read(path):
    path = Path(path)
    if _evidence is not None and path.resolve().is_relative_to(_evidence.data_root):
        _evidence.original_hash(path)
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    path = Path(path)
    if not path.is_file():
        return None
    return _evidence.original_hash(path) if _evidence is not None else sha256(path)


def canonical(value):
    return _evidence.original_canonical_hash(value) if _evidence is not None else canonical_hash(value)
