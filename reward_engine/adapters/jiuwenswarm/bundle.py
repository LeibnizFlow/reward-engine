"""Export a self-contained, content-addressed Linux runtime from a pinned interpreter.

Run with the interpreter that has workswarm installed. Only distribution files,
Python's standard library and its shared libraries are copied; no user config.
"""

import argparse
import importlib.metadata as metadata
import json
import platform
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

from reward_engine.adapters.integrity import tree_hashes
from reward_engine.adapters.workbuddy.io import file_hash


def build(target):
    from jiuwenswarm.channels.process_cli.protocol.version import CURRENT_SCHEMA_VERSION
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    if CURRENT_SCHEMA_VERSION != "0.1":
        raise RuntimeError("Review the adapter before changing Process schema")
    target = Path(target).resolve()
    target.mkdir(parents=True, exist_ok=False)
    prefix = Path(sys.base_prefix)
    if sys.version_info[:2] != (3, 12) or platform.system() != "Linux":
        raise RuntimeError("The benchmark bundle requires Linux Python 3.12")
    pending, packages = [("workswarm", set())], {}
    extras_seen = {}
    while pending:
        name, extras = pending.pop()
        key = canonicalize_name(name)
        if key in packages and extras <= extras_seen[key]:
            continue
        dist = metadata.distribution(name)
        packages[key] = dist
        extras_seen.setdefault(key, set()).update(extras)
        for spec in dist.requires or []:
            requirement = Requirement(spec)
            if requirement.marker and not any(
                requirement.marker.evaluate({"extra": extra})
                for extra in (extras_seen[key] | {""})
            ):
                continue
            dependency = metadata.distribution(requirement.name)
            if (
                requirement.specifier
                and dependency.version not in requirement.specifier
            ):
                raise RuntimeError(
                    f"Installed dependency violates {spec}: {dependency.version}"
                )
            pending.append((requirement.name, set(requirement.extras)))
    if packages["workswarm"].version != "0.2.7":
        raise RuntimeError("Expected workswarm 0.2.7; review protocol before updating")

    def copy(source, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination, follow_symlinks=True)

    copy(Path(sys.executable), target / "bin/python")
    dynamic = subprocess.check_output(
        ["readelf", "-d", str(target / "bin/python")], text=True
    )
    if "$ORIGIN/../lib" not in dynamic:
        raise RuntimeError("Bundle interpreter requires $ORIGIN/../lib RPATH")
    stdlib = Path(sysconfig.get_path("stdlib"))
    shutil.copytree(
        stdlib,
        target / "lib/python3.12",
        ignore=shutil.ignore_patterns("site-packages", "__pycache__", "*.pyc"),
    )
    for dist in packages.values():
        for entry in dist.files or []:
            source = Path(dist.locate_file(entry))
            if not source.is_file() or "__pycache__" in source.parts:
                continue
            # Installed scripts contain absolute shebangs. Runtime launchers below
            # use the bundled interpreter instead; distribution RECORD remains provenance.
            try:
                relative = source.relative_to(stdlib / "site-packages")
            except ValueError:
                continue
            if ".." not in relative.parts:
                copy(source, target / "lib/python3.12/site-packages" / relative)
    # Conda's interpreter/extensions link shared libraries relative to lib/.
    # Copy these runtime libraries, not package caches or the whole environment.
    for path in (prefix / "lib").glob("*.so*"):
        if path.is_file():
            copy(path, target / "lib" / path.name)
    shutil.copytree("/usr/share/zoneinfo", target / "share/zoneinfo", symlinks=False)
    for name in ("supervisor.py", "entry.py"):
        copy(Path(__file__).with_name(name), target / "bin" / name)
    launcher = target / "bin/jiuwenswarm-process"
    launcher.write_text(
        '#!/bin/sh\nbase=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)\nunset LD_LIBRARY_PATH\nexport PYTHONTZPATH="$base/share/zoneinfo"\nexec "$base/bin/python" "$base/bin/supervisor.py" "$@"\n'
    )
    launcher.chmod(0o755)
    records = {
        str(path.relative_to(target)): file_hash(path)
        for path in sorted(target.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "format": "jiuwenswarm-runtime-bundle-v1",
        "python": platform.python_version(),
        "platform": platform.machine(),
        "schema": "0.1",
        "packages": {name: dist.version for name, dist in sorted(packages.items())},
        "files": records,
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    )
    (target / "requirements.lock").write_text(
        "".join(f"{name}=={dist.version}\n" for name, dist in sorted(packages.items()))
    )
    print(
        json.dumps(
            {
                "bundle": str(target),
                "packages": len(packages),
                "manifest_sha256": file_hash(target / "manifest.json"),
            }
        )
    )


def repackage(source, target):
    """Derive an immutable launcher repair without changing pinned package bytes."""
    source, target = Path(source).resolve(), Path(target).resolve()
    identity = _verify(source, allow_legacy_launcher=True)
    shutil.copytree(
        source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    dynamic = subprocess.check_output(
        ["readelf", "-d", str(target / "bin/python")], text=True
    )
    if "$ORIGIN/../lib" not in dynamic:
        raise RuntimeError("Bundle interpreter requires $ORIGIN/../lib RPATH")
    launcher = target / "bin/jiuwenswarm-process"
    launcher.write_text(
        launcher.read_text().replace(
            'export LD_LIBRARY_PATH="$base/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"\n',
            "unset LD_LIBRARY_PATH\n",
        )
    )
    manifest = json.loads((target / "manifest.json").read_text())
    manifest["parent"] = identity
    manifest["loader_policy"] = "rpath-only"
    manifest["files"]["bin/jiuwenswarm-process"] = file_hash(launcher)
    (target / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    )
    return verify(target)


def verify(target):
    return _verify(target)


def _manifest(target):
    manifest = json.loads((target / "manifest.json").read_text())
    if (
        manifest.get("format") != "jiuwenswarm-runtime-bundle-v1"
        or manifest.get("schema") != "0.1"
        or manifest.get("packages", {}).get("workswarm") != "0.2.7"
    ):
        raise ValueError("Unsupported runtime bundle protocol")
    return manifest


def _identity(target, manifest):
    return {
        "jiuwenswarm_bundle": file_hash(target / "manifest.json"),
        "jiuwenswarm_schema": manifest["schema"],
        "jiuwenswarm_version": manifest["packages"]["workswarm"],
    }


def identity(target):
    """Read the declared bundle identity without scanning installed files."""
    target = Path(target).resolve()
    return _identity(target, _manifest(target))


def _verify(target, *, allow_legacy_launcher=False):
    target = Path(target).resolve()
    manifest = _manifest(target)
    launcher = (target / "bin/jiuwenswarm-process").read_text()
    if "export LD_LIBRARY_PATH=" in launcher and not allow_legacy_launcher:
        raise ValueError("Bundle launcher leaks LD_LIBRARY_PATH; repackage it first")
    actual = tree_hashes(
        target, file_hash, ignore_bytecode=allow_legacy_launcher,
        exclude=("manifest.json", "requirements.lock"),
        expected_files=len(manifest["files"]),
        inspect_ignored=True,
    )
    if set(actual) != set(manifest["files"]):
        raise ValueError("Runtime bundle file inventory changed")
    lock = target / "requirements.lock"
    if lock.exists() and lock.read_text() != "".join(
        f"{name}=={version}\n" for name, version in sorted(manifest["packages"].items())
    ):
        raise ValueError("Runtime bundle requirements lock changed")
    for relative, expected in manifest["files"].items():
        if actual[relative] != expected:
            raise ValueError(f"Runtime bundle changed: {relative}")
    return _identity(target, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "verify", "repackage"))
    parser.add_argument("target", type=Path)
    parser.add_argument(
        "--source", type=Path, help="Verified source bundle for repackage"
    )
    args = parser.parse_args()
    if args.action == "build":
        build(args.target)
    elif args.action == "repackage":
        if args.source is None:
            parser.error("repackage requires --source")
        print(json.dumps(repackage(args.source, args.target)))
    else:
        print(json.dumps(verify(args.target)))


if __name__ == "__main__":
    main()
