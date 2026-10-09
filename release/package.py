"""Export a clean GitHub tree and size-bounded Zenodo archives, without uploads."""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import tarfile
from pathlib import Path

from .evidence import Evidence, sha256

SOURCE_TREES = ("reward_engine", "experiments", "release", "tests", "docs", "reports")
SOURCE_FILES = (
    "README.md",
    "PUBLICATION.md",
    "LICENSE",
    "pyproject.toml",
    ".gitignore",
    ".gitattributes",
    ".env.example",
)
IGNORED = {
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".git",
    "node_modules",
}


def source_files(root):
    for name in SOURCE_FILES:
        p = root / name
        if p.is_file():
            yield p
    for name in SOURCE_TREES:
        for p in sorted((root / name).rglob("*")):
            if (
                p.is_file()
                and not p.is_symlink()
                and not any(part in IGNORED for part in p.parts)
                and p.suffix != ".pyc"
            ):
                yield p


def write_archive(path, files, root):
    if shutil.disk_usage(path.parent).free < 10 * 1024**3:
        raise OSError("Insufficient space for the 10 GiB reserve")
    tmp = path.with_suffix(path.suffix + ".tmp")
    # Reproducible timestamps, owner IDs and gzip header; original evidence
    # timestamps remain in the recorded receipts.
    with (
        tmp.open("wb") as stream,
        gzip.GzipFile(
            filename="", fileobj=stream, mode="wb", mtime=0, compresslevel=6
        ) as zipped,
    ):
        with tarfile.open(
            fileobj=zipped, mode="w|", format=tarfile.PAX_FORMAT
        ) as archive:
            for p in files:
                info = archive.gettarinfo(str(p), str(p.relative_to(root)))
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mtime = 0
                if info.isfile():
                    with p.open("rb") as data:
                        archive.addfile(info, data)
                else:
                    archive.addfile(info)
                if shutil.disk_usage(path.parent).free < 10 * 1024**3:
                    raise OSError("Packaging stopped to preserve disk reserve")
    tmp.replace(path)


def export(root: Path, output: Path, part_bytes=4_000_000_000):
    if output.exists():
        raise FileExistsError("Use a new export directory")
    # A source checkout alone is not a complete evidence release. Validate before
    # creating any output, so missing or partial restores cannot look successful.
    required = [
        root / "data/workbuddy_office/experiment.json",
        root / "data/claw_eval/experiment.json",
        root / "data/provenance/files.jsonl",
    ]
    missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
    for dataset in ("workbuddy_office", "claw_eval", "analysis", "provenance"):
        directory = root / "data" / dataset
        if not directory.is_dir() or not any(directory.iterdir()):
            missing.append(str(directory.relative_to(root)))
    if missing:
        raise ValueError(
            "Restore the complete evidence archives before packaging; missing: "
            + ", ".join(missing)
        )
    evidence = Evidence(root / "data")
    if not evidence.records:
        raise ValueError("The evidence manifest is empty")
    evidence.verify()
    output.mkdir(parents=True)
    github = output / "github"
    github.mkdir()
    sources = list(source_files(root))
    for p in sources:
        q = github / p.relative_to(root)
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, q)
    zenodo = output / "zenodo"
    zenodo.mkdir()
    write_archive(zenodo / "source-and-reports.tar.gz", sources, root)
    packages = []
    for dataset in ("workbuddy_office", "claw_eval", "analysis", "provenance"):
        files = sorted(
            p
            for p in (root / "data" / dataset).rglob("*")
            if p.is_file()
            and not any(part in IGNORED for part in p.parts)
            and p.suffix != ".pyc"
        )
        group = []
        size = 0
        number = 1
        for p in files:
            if p.is_symlink():
                raise ValueError("Data packages must not contain symlinks")
            current = p.stat().st_size
            if group and size + current > part_bytes:
                name = f"{dataset}-{number:03d}.tar.gz"
                write_archive(zenodo / name, group, root)
                packages.append(
                    {"file": name, "experiment": dataset, "files": len(group)}
                )
                print(name, flush=True)
                group = []
                size = 0
                number += 1
            group.append(p)
            size += current
        if group:
            name = f"{dataset}-{number:03d}.tar.gz"
            write_archive(zenodo / name, group, root)
            packages.append({"file": name, "experiment": dataset, "files": len(group)})
            print(name, flush=True)
    archives = sorted(zenodo.glob("*.tar.gz"))
    checksum_lines = [f"{sha256(p)}  {p.name}" for p in archives]
    (zenodo / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n")
    total = sum(p.stat().st_size for p in archives)
    manifest = {
        "schema_version": 1,
        "archives": packages,
        "compressed_bytes": total,
        "file_count": len(archives),
        "default_zenodo_quota_ok": total <= 50_000_000_000 and len(archives) + 4 <= 100,
        "metadata_pending": [
            "creators",
            "github_repository",
            "license_for_original_data_and_reports",
        ],
        "uploaded": False,
        "original_git_history_included": False,
    }
    (zenodo / "package_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (zenodo / "README.md").write_text(
        "# Experimental evidence release\n\n"
        "Verify SHA256SUMS, then extract every archive into the same empty directory. "
        "Part numbers are packaging volumes, not experiment batches.\n\n"
        "Run `python -m release.verify --data-root data` and see `release/README.md` for offline report reproduction. "
        "The GitHub source export excludes data; these archives restore it.\n\n"
        "Publication metadata is pending: creators and the license for original data/reports. "
        "Third-party licenses remain attached to their respective materials.\n"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--part-bytes", type=int, default=4_000_000_000)
    args = parser.parse_args()
    if args.part_bytes <= 0:
        parser.error("--part-bytes must be positive")
    print(
        json.dumps(export(args.root.resolve(), args.output.resolve(), args.part_bytes))
    )


if __name__ == "__main__":
    main()
