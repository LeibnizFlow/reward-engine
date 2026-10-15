"""Load the real actor with unused native installation boundaries isolated.

Continuation and serialization tests execute the actual actor/transport methods. They supply their
own session log and mock execution; benchmark installation and endpoints must not
be consulted. Harbor remains the real declared Python dependency.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import reward_engine.adapters.workbuddy


def native_actor_module(monkeypatch):
    root = Path(reward_engine.adapters.workbuddy.__file__).parent

    def unused(*args, **kwargs):
        raise AssertionError("Adapter regression must not use native installation")

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    boundaries = {
        "workbuddy_bench.agents._agent_user": {"ensure_agent_user": unused},
        "workbuddy_bench.runner.model_endpoints": {"host_reachable_url": unused},
        "workbuddy_swarm.endpoint": {"ProxyJsonEndpoint": unused, "api_base": unused},
    }
    with monkeypatch.context() as patch:
        for name, exports in boundaries.items():
            module = ModuleType(name)
            module.__dict__.update(exports)
            patch.setitem(sys.modules, name, module)
        name = "reward_engine.adapters.workbuddy.opencode"
        patch.setitem(sys.modules, name, load(name, root / "opencode.py"))
        actor = load("tests.support._isolated_actor", root / "agent.py")
    return actor
