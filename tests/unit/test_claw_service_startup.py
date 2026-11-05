from types import SimpleNamespace

import pytest

from reward_engine.adapters.claw_eval import services
from reward_engine.adapters.workbuddy.io import read


@pytest.mark.parametrize("fail", [False, True])
def test_slow_startup_has_logs_and_owned_process_cleanup(tmp_path, monkeypatch, fail):
    clock, killed, clients = [0], [], []

    class Process:
        pid, returncode = 123456789, None

        def __init__(self, command, **kwargs):
            kwargs["stdout"].write(b"synthetic startup diagnostic\n")
            assert kwargs["start_new_session"]

        def poll(self):
            return 1 if fail else None

        def wait(self, **kwargs):
            return 0

    class Client:
        def __init__(self, **kwargs):
            clients.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def request(self, *args, **kwargs):
            return SimpleNamespace(status_code=200 if clock[0] >= 12 else 503)

    monkeypatch.setattr(services.subprocess, "Popen", Process)
    monkeypatch.setattr(services.httpx, "Client", Client)
    monkeypatch.setattr(services.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(services.time, "sleep", lambda _: clock.__setitem__(0, clock[0] + 3))
    monkeypatch.setattr(services.os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    service = SimpleNamespace(name="synthetic", command="python fixture.py", env={},
                              ready_timeout=10, health_check="http://localhost:1/health",
                              health_check_method="GET")
    manager = services.ServiceManager([service], tmp_path, output=tmp_path / "logs")
    if fail:
        with pytest.raises(services.ServiceStartError, match="log:"):
            manager.__enter__()
    else:
        with manager:
            assert not killed
    assert len(clients) == 1 and killed
    assert (tmp_path / "logs/0.log").read_text() == "synthetic startup diagnostic\n"
    receipt = read(tmp_path / "logs/0.json")
    assert receipt["status"] == ("failed" if fail else "ready")
    assert receipt["timeout_s"] == 60
