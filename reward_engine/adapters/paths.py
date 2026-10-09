"""Configurable checkout, interpreter and output locations for local tools."""

import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PYTHON = os.environ.get("REWARD_ENGINE_PYTHON", sys.executable)
LOCAL = Path(os.environ.get("REWARD_ENGINE_LOCAL", REPO / "local")).expanduser().resolve()
REPORTS = Path(os.environ.get("REWARD_ENGINE_REPORTS", REPO / "reports")).expanduser().resolve()
CLAW_ROOT = Path(os.environ.get("CLAW_EVAL_ROOT", LOCAL / "claw-eval")).expanduser().resolve()
WORKBUDDY_ROOT = Path(os.environ.get("WORKBUDDY_ROOT", LOCAL / "workbuddy/runner")).expanduser().resolve()


def experiment_id(value):
    """Canonical run label; original task/model/version identifiers stay untouched."""
    name = Path(value).name
    name = re.sub(r"(?<!\d)(20\d{2})[-.](\d{2})[-.](\d{2})(?!\d)", r"\1\2\3", name)
    name = re.sub(r"[-.\s]+", "_", name).lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_]*", name):
        raise ValueError(f"Invalid experiment name: {value}")
    return name


def report_directory(direction, run, output=None):
    """Use a stable report location; explicit CLI output takes precedence."""
    target = Path(output) if output is not None else REPORTS / experiment_id(direction) / experiment_id(run)
    target.mkdir(parents=True, exist_ok=True)
    return target.resolve()
