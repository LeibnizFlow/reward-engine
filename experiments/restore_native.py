"""Verify publication archives and restore evidence without replacing source code."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def restore(archives: Path, root: Path):
    root = root.resolve()
    if (root / "data").exists():
        raise FileExistsError(
            "data/ already exists; use a fresh checkout or output directory"
        )
    manifest = json.loads((archives / "package_manifest.json").read_text())
    entries = manifest["archives"]
    expected = {}
    for line in (archives / "SHA256SUMS").read_text().splitlines():
        checksum, name = line.split(maxsplit=1)
        expected[name.lstrip("*")] = checksum
    datasets = {entry["experiment"] for entry in entries}
    if datasets != {"workbuddy_office", "claw_eval", "analysis", "provenance"}:
        raise ValueError("The archive manifest does not contain all four datasets")
    names = [entry["file"] for entry in entries]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate archive name")
    for entry in entries:
        name = entry["file"]
        if Path(name).name != name or name not in expected:
            raise ValueError("Invalid or unchecked archive name")
        if sha256(archives / name) != expected[name]:
            raise ValueError("Archive checksum mismatch: " + name)
    seen = set()
    for entry in entries:
        print("Restoring " + entry["file"], flush=True)
        count = 0
        with tarfile.open(archives / entry["file"], mode="r|gz") as bundle:
            for member in bundle:
                path = PurePosixPath(member.name)
                if (
                    not member.isfile()
                    or path.is_absolute()
                    or ".." in path.parts
                    or path.parts[:2] != ("data", entry["experiment"])
                ):
                    raise ValueError("Unexpected evidence member: " + member.name)
                if str(path) in seen:
                    raise ValueError("Duplicate evidence member: " + member.name)
                seen.add(str(path))
                bundle.extract(member, root, filter="data")
                count += 1
        if count != entry["files"]:
            raise ValueError("Archive member count mismatch: " + entry["file"])
    return {"files_restored": len(seen), "data_root": str(root / "data")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, default=Path("_local/zenodo"))
    parser.add_argument("--root", type=Path, default=Path("_local"))
    args = parser.parse_args()
    print(json.dumps(restore(args.archives, args.root)))


if __name__ == "__main__":
    main()
