"""Local execution settings and JSON IO, shared by adapters and experiments."""

import json
import os
import shlex
from pathlib import Path

from reward_engine.adapters.paths import PYTHON as PYTHON
from reward_engine.adapters.paths import REPO as REPO


def load_named_key(name, env_file):
    """Read only the named credential, without importing an Agent SDK."""
    if os.environ.get(name):
        return os.environ[name]
    for line in Path(env_file).expanduser().read_text().splitlines():
        key, separator, value = line.strip().removeprefix("export ").partition("=")
        if separator and key.strip() == name:
            parts = shlex.split(value, comments=True)
            if len(parts) == 1 and parts[0] and "$" not in parts[0]:
                return parts[0]
    raise ValueError(f"Required credential {name} was not found")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
