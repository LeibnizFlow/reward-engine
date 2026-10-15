"""Native input restoration preserves current source and verifies every archive."""

import hashlib
import io
import json
import tarfile

import pytest

from experiments.restore_native import restore


def archives(root):
    root.mkdir()
    entries = []
    checksums = []
    for dataset in ("workbuddy_office", "claw_eval", "analysis", "provenance"):
        name = dataset + ".tar.gz"
        path = root / name
        with tarfile.open(path, "w:gz") as bundle:
            member = tarfile.TarInfo(f"data/{dataset}/input.json")
            data = b"{}\n"
            member.size = len(data)
            bundle.addfile(member, io.BytesIO(data))
        entries.append({"file": name, "experiment": dataset, "files": 1})
        checksums.append(hashlib.sha256(path.read_bytes()).hexdigest() + "  " + name)
    (root / "package_manifest.json").write_text(json.dumps({"archives": entries}))
    (root / "SHA256SUMS").write_text("\n".join(checksums) + "\n")
    return root


def test_restore_preserves_source_and_refuses_existing_data(tmp_path):
    source = archives(tmp_path / "archives")
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "README.md").write_text("Current source\n")
    result = restore(source, checkout)
    assert result["files_restored"] == 4
    assert (checkout / "README.md").read_text() == "Current source\n"
    assert (checkout / "data/workbuddy_office/input.json").read_text() == "{}\n"
    with pytest.raises(FileExistsError, match="already exists"):
        restore(source, checkout)


def test_restore_checks_every_archive_before_extracting(tmp_path):
    source = archives(tmp_path / "archives")
    (source / "provenance.tar.gz").write_bytes(b"corrupt")
    checkout = tmp_path / "checkout"
    with pytest.raises(ValueError, match="checksum mismatch"):
        restore(source, checkout)
    assert not (checkout / "data").exists()
