"""Publication must not turn sanitized or duplicate evidence into valid results."""

import hashlib
import json

import pytest

from release.evidence import Evidence
from release.report import validate_rows


def test_public_bytes_checked_before_returning_original_identity(tmp_path):
    data = tmp_path / "data"
    (data / "provenance").mkdir(parents=True)
    path = data / "receipt.json"
    path.write_text('{"score": 0, "path": "tasks/task"}\n')
    current = hashlib.sha256(path.read_bytes()).hexdigest()
    original = "a" * 64
    (data / "provenance/files.jsonl").write_text(
        json.dumps(
            {
                "path": "data/receipt.json",
                "sha256": current,
                "source_sha256": original,
                "bytes": path.stat().st_size,
            }
        )
        + "\n"
    )
    evidence = Evidence(data)
    assert evidence.original_hash(path) == original
    path.write_text('{"score": 100}\n')
    with pytest.raises(ValueError, match="checksum mismatch"):
        evidence.original_hash(path)


def test_manifest_rejects_escape(tmp_path):
    (tmp_path / "data/provenance").mkdir(parents=True)
    (tmp_path / "data/provenance/files.jsonl").write_text(
        json.dumps({"path": "../secret"}) + "\n"
    )
    with pytest.raises(ValueError, match="escapes"):
        Evidence(tmp_path / "data")


def test_duplicate_round_cannot_replace_missing_round():
    rows = [
        {
            "task": str(i),
            "round": n,
            "A": 0,
            "B": 10,
            "C": 20,
            "selected_arm": "A",
            "selected_score": 0,
        }
        for i in range(50)
        for n in (1, 2, 3)
    ]
    data = {"experiment": "workbuddy_office", "rounds": rows}
    assert validate_rows(data)["valid_scores"] == 450
    rows[-1] = rows[-2].copy()
    with pytest.raises(ValueError, match="Duplicate"):
        validate_rows(data)


def test_report_rejects_missing_score_instead_of_filling_zero():
    rows = [
        {
            "task": str(i),
            "round": n,
            "A": 0,
            "B": 10,
            "C": 20,
            "selected_arm": "B",
            "selected_score": 10,
        }
        for i in range(50)
        for n in (1, 2, 3)
    ]
    rows[0]["A"] = None
    with pytest.raises(ValueError, match="Missing"):
        validate_rows({"experiment": "workbuddy_office", "rounds": rows})


def test_ambiguous_identity_stays_ambiguous_after_duplicate(tmp_path):
    from release.evidence import canonical_hash

    (tmp_path / "data/provenance").mkdir(parents=True)
    value = {"score": 1}
    rows = [
        {
            "path": f"data/{i}.json",
            "public_canonical_sha256": canonical_hash(value),
            "source_canonical_sha256": source,
        }
        for i, source in enumerate(["a", "b", "b"])
    ]
    (tmp_path / "data/provenance/files.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    with pytest.raises(ValueError, match="Ambiguous"):
        Evidence(tmp_path / "data").original_canonical_hash(value)
