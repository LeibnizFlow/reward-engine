"""Restore a verified delivery for native grading without invoking an actor."""

from pathlib import Path

from harbor.agents.base import BaseAgent

from reward_engine.adapters.jiuwenswarm.checkpoints import verify_actor
from reward_engine.adapters.workbuddy.io import read


class FrozenDeliveryAgent(BaseAgent):
    def __init__(self, logs_dir, *args, **kwargs):
        self.state = read(kwargs.pop("actor_result"))
        super().__init__(logs_dir, *args, **kwargs)

    @staticmethod
    def name():
        return "frozen-office-delivery"

    def version(self):
        return "1"

    async def setup(self, environment):
        verify_actor(self.state)
        root = Path(self.state["runtime_directory"])
        snapshot = root / "snapshots" / f"delivery-{self.state['resources']['turns']}"
        cleared = await environment.exec(
            user="root",
            command='python -c "import pathlib,shutil; '
            "[(shutil.rmtree(p) if p.is_dir() and not p.is_symlink() else p.unlink()) "
            "for p in pathlib.Path('/workspace').iterdir() if p.name != '.git']\"",
        )
        if cleared.return_code:
            raise RuntimeError("Cannot restore native grading workspace")
        await environment.upload_dir(snapshot / "workspace", "/workspace")

    async def run(self, instruction, environment, context):
        # The verifier consumes native files; no model or feedback runs here.
        return None
