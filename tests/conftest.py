"""Keep generated test reports out of the repository's experiment reports."""

from pathlib import Path

import pytest

from reward_engine.adapters import paths


def pytest_configure(config):
    """Pytest creates basetemp itself, but requires its parent to exist."""
    if config.option.basetemp:
        Path(config.option.basetemp).parent.mkdir(parents=True, exist_ok=True)


@pytest.fixture(scope="session", autouse=True)
def isolated_report_root(tmp_path_factory):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(paths, "REPORTS", tmp_path_factory.mktemp("reports"))
        yield
