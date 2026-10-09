"""Incomplete evidence must never be advertised as a complete release."""

import hashlib
import json
import tarfile

import pytest

from release.package import export
from release.restore import restore


def evidence_tree(root):
    records = []
    for name in (
        "workbuddy_office/experiment.json",
        "claw_eval/experiment.json",
        "analysis/input.json",
    ):
        path = root / "data" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
        records.append(
            {
                "path": "data/" + name,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    manifest = root / "data/provenance/files.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("".join(json.dumps(row) + "\n" for row in records))


def test_source_only_checkout_fails_without_creating_export(tmp_path):
    output = tmp_path / "export"
    with pytest.raises(ValueError, match="Restore the complete evidence"):
        export(tmp_path, output)
    assert not output.exists()


def test_partial_restore_fails_before_creating_export(tmp_path):
    evidence_tree(tmp_path)
    (tmp_path / "data/analysis/input.json").unlink()
    output = tmp_path / "export"
    with pytest.raises(ValueError, match="data/analysis"):
        export(tmp_path, output)
    assert not output.exists()


def test_modified_evidence_fails_before_creating_export(tmp_path):
    evidence_tree(tmp_path)
    (tmp_path / "data/analysis/input.json").write_text('{"changed":true}\n')
    output = tmp_path / "export"
    with pytest.raises(ValueError, match="missing or changed"):
        export(tmp_path, output)
    assert not output.exists()


def test_complete_export_preserves_all_dataset_members(tmp_path):
    evidence_tree(tmp_path)
    (tmp_path / "README.md").write_text("Fixture publication\n")
    output = tmp_path / "export"
    result = export(tmp_path, output)
    assert {row["experiment"] for row in result["archives"]} == {
        "workbuddy_office",
        "claw_eval",
        "analysis",
        "provenance",
    }
    members = []
    for row in result["archives"]:
        with tarfile.open(output / "zenodo" / row["file"]) as archive:
            members.extend(archive.getnames())
    assert sorted(members) == sorted(
        str(p.relative_to(tmp_path))
        for p in (tmp_path / "data").rglob("*")
        if p.is_file()
    )


def test_restore_preserves_current_source_and_checks_data_members(tmp_path):
    source = tmp_path / "source"
    evidence_tree(source)
    (source / "README.md").write_text("Historical source\n")
    output = tmp_path / "export"
    export(source, output)
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "README.md").write_text("Repaired current source\n")
    result = restore(output / "zenodo", checkout)
    assert result["files_restored"] == 4
    assert (checkout / "README.md").read_text() == "Repaired current source\n"
    assert (checkout / "data/analysis/input.json").read_text() == "{}\n"
    with pytest.raises(FileExistsError, match="already exists"):
        restore(output / "zenodo", checkout)


def test_restore_checks_every_archive_before_extracting(tmp_path):
    source = tmp_path / "source"
    evidence_tree(source)
    output = tmp_path / "export"
    export(source, output)
    (output / "zenodo/workbuddy_office-001.tar.gz").write_bytes(b"corrupt")
    checkout = tmp_path / "checkout"
    with pytest.raises(ValueError, match="checksum mismatch"):
        restore(output / "zenodo", checkout)
    assert not (checkout / "data").exists()
