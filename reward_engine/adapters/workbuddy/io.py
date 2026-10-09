"""Small atomic experiment records and explicit local configuration."""

import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

STREAM_TIMEOUT_S = 1800


def role_model(config, role):
    """Resolve an explicit runtime role amendment without rebinding old samples."""
    path = Path(config["output"]) / "role_models.json"
    overrides = read(path)["roles"] if path.exists() else {}
    if (
        not isinstance(overrides, dict)
        or not set(overrides) <= set(config["roles"])
        or any(
            not isinstance(model, str) or model not in config["roles"].values()
            for model in overrides.values()
        )
    ):
        raise ValueError(
            "Role model amendments must use configured roles and gateway models"
        )
    return overrides.get(role, config["roles"][role])


def role_timeout_s(config, default=STREAM_TIMEOUT_S):
    path = Path(config["output"]) / "role_limits.json"
    limits = read(path) if path.exists() else config
    value = limits.get("role_stream_timeout_s", default)
    if type(value) not in (int, float) or not 0 < value <= 3600:
        raise ValueError(
            "Role stream timeout must be positive and at most 3600 seconds"
        )
    return value


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str)
        + "\n"
    )
    os.replace(temporary, path)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()
    ).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def auth(config):
    return Path(config["auth_file"]).read_text().strip()


def model_matches(config, requested, actual):
    return actual == requested or actual in config.get("model_aliases", {}).get(
        requested, []
    )
