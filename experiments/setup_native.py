"""Prepare local WorkBuddy paths and native imports from verified publication data."""

import argparse
import json
import shutil
import sys
import sysconfig
from pathlib import Path


def setup(root: Path, credentials: Path):
    if sys.version_info < (3, 12):
        raise RuntimeError("The native WorkBuddy runtime requires Python 3.12+")
    if sys.prefix == sys.base_prefix:
        raise RuntimeError(
            "Activate a virtual environment before installing native paths"
        )
    root = root.resolve()
    environment = root / "data/workbuddy_office/environment"
    local = root / "local/workbuddy/runner"
    upstream = local / "upstream"
    sources = [
        (environment / "workbuddy_swarm", local / "workbuddy_swarm"),
        (environment / "benchmark_source", upstream / "src"),
        (environment / "configs", upstream / "configs"),
        (
            root / "data/workbuddy_office/benchmark/tasks",
            upstream / "datasets/wb-bench-office-v1.0",
        ),
    ]
    if any(not source.is_dir() for source, _ in sources):
        raise FileNotFoundError("Restore the evidence data before native setup")
    if not credentials.is_file():
        raise FileNotFoundError("Provide a private credentials file; see .env.example")
    upstream.mkdir(parents=True, exist_ok=True)
    for source, destination in sources:
        if not destination.exists():
            shutil.copytree(source, destination)
    target = upstream / ".env"
    if target.exists() and target.read_bytes() != credentials.read_bytes():
        raise FileExistsError("WorkBuddy credentials already exist and differ")
    if not target.exists():
        target.touch(mode=0o600)
        shutil.copyfile(credentials, target)
    target.chmod(0o600)
    paths = [root, local, upstream / "src", root / "data/claw_eval/benchmark/src"]
    config = Path(sysconfig.get_paths()["purelib"]) / "reward_engine_native.pth"
    config.write_text("\n".join(map(str, paths)) + "\n")
    return {
        "workbuddy_root": str(local),
        "python": sys.executable,
        "paths_file": str(config),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--credentials", type=Path, default=Path(".env"))
    args = parser.parse_args()
    print(json.dumps(setup(args.root, args.credentials.resolve())))


if __name__ == "__main__":
    main()
