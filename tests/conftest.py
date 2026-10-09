"""Keep generated test reports out of the repository's experiment reports."""

import pytest

from reward_engine.adapters import paths


@pytest.fixture(scope="session", autouse=True)
def isolated_report_root(tmp_path_factory):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(paths, "REPORTS", tmp_path_factory.mktemp("reports"))
        yield
