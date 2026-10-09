"""Scan public files and nested containers without printing sensitive values."""

import argparse
import gzip
import hashlib
import io
import json
import re
import tarfile
import zipfile
from pathlib import Path

RULES = {
    "provider_key": re.compile(rb"(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_-]{20,}"),
    "bearer_credential": re.compile(rb"(?i)bearer\s+[A-Za-z0-9_.=-]{24,}"),
    "host_absolute_path": re.compile(rb'/(?:data[0-9]+|Users)/[^\s"\'<>`\\]+'),
}


def inspect(raw, label, findings, deny=(), depth=0):
    for rule, expression in RULES.items():
        if rule == "provider_key" and b"sk-" not in raw:
            continue
        if (
            rule == "host_absolute_path"
            and b"/data" not in raw
            and b"/Users/" not in raw
        ):
            continue
        if expression.search(raw):
            findings.append({"file": label, "rule": rule})
    if deny and deny.search(raw):
        findings.append({"file": label, "rule": "retired_collection_identifier"})
    if depth >= 4:
        return
    if raw.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                for name in archive.namelist():
                    if not name.endswith("/"):
                        inspect(
                            archive.read(name),
                            label + "!" + name,
                            findings,
                            deny,
                            depth + 1,
                        )
        except (zipfile.BadZipFile, RuntimeError):
            findings.append({"file": label, "rule": "unreadable_zip"})
    elif label.endswith((".tar.gz", ".tgz", ".tar")):
        try:
            with tarfile.open(fileobj=io.BytesIO(raw), mode="r:*") as archive:
                for member in archive:
                    if member.isfile():
                        inspect(
                            archive.extractfile(member).read(),
                            label + "!" + member.name,
                            findings,
                            deny,
                            depth + 1,
                        )
        except tarfile.TarError:
            findings.append({"file": label, "rule": "unreadable_tar"})
    elif label.endswith(".gz"):
        try:
            inspect(gzip.decompress(raw), label[:-3], findings, deny, depth + 1)
        except (OSError, EOFError):
            findings.append({"file": label, "rule": "unreadable_gzip"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", type=Path, nargs="*")
    parser.add_argument(
        "--file-list", type=Path, help="Newline-delimited file paths to scan"
    )
    parser.add_argument(
        "--deny-file",
        type=Path,
        help="JSON array of retired identifiers; only rule names are reported",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--exceptions",
        type=Path,
        default=Path(__file__).with_name("audit_exceptions.json"),
    )
    args = parser.parse_args()
    if args.file_list:
        args.roots.extend(Path(s) for s in args.file_list.read_text().splitlines() if s)
    if not args.roots:
        parser.error("Provide roots or --file-list")
    names = json.loads(args.deny_file.read_text()) if args.deny_file else []
    deny = (
        re.compile(b"|".join(re.escape(x.encode()) for x in names)) if names else None
    )
    exceptions = (
        json.loads(args.exceptions.read_text()) if args.exceptions.exists() else []
    )
    findings = []
    reviewed = []
    count = 0
    for root in args.roots:
        paths = [root] if root.is_file() else sorted(root.rglob("*"))
        for path in paths:
            if not path.is_file() or any(
                x in path.parts for x in ("__pycache__", ".git")
            ):
                continue
            raw = path.read_bytes()
            current = []
            inspect(raw, str(path), current, deny)
            for finding in current:
                match = next(
                    (
                        entry
                        for entry in exceptions
                        if entry["rule"] == finding["rule"]
                        and finding["file"] == str(path)
                        and entry["sha256"] == hashlib.sha256(raw).hexdigest()
                    ),
                    None,
                )
                if match:
                    reviewed.append(
                        {
                            **finding,
                            "sha256": match["sha256"],
                            "reason": match["reason"],
                        }
                    )
                else:
                    findings.append(finding)
            count += 1
    args.output.write_text(
        json.dumps(
            {
                "files_scanned": count,
                "findings": findings,
                "reviewed_exceptions": reviewed,
                "status": "passed" if not findings else "needs_review",
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"files_scanned": count, "findings": len(findings)}))
    if findings:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
