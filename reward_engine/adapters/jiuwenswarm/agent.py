"""Harbor Office actor using Process CLI without the OpenCode dependency."""

import json
import shlex
import time
from pathlib import Path

from harbor.agents.base import BaseAgent

from reward_engine.adapters.workbuddy.delivery import OfficeDelivery
from reward_engine.adapters.workbuddy.io import read, write

from .container import docker, prepare
from .gateway import register
from .office import PROBE
from .policy import policy
from .receipts import tool_outcomes


class OutcomeJiuwenSwarmAgent(OfficeDelivery, BaseAgent):
    def __init__(self, logs_dir, *args, **kwargs):
        self.spec = read(kwargs.pop("OUTCOME_SPEC"))
        self.config = read(self.spec["campaign_config"])
        self.result_root = Path(self.spec["output"])
        self.actor_remaining = self.spec["actor_timeout_s"]
        self.session, self.turn, self.usage, self.tool_errors = None, 0, [], 0
        self.connection = kwargs.pop("connection")
        super().__init__(logs_dir, *args, **kwargs)

    @staticmethod
    def name():
        return "workbuddy-outcome-jiuwenswarm"

    def version(self):
        return "0.2.7"

    @staticmethod
    def render_instruction(instruction):
        return instruction

    async def exec_as_agent(self, environment, **kwargs):
        return await environment.exec(
            user=getattr(environment, "default_user", None) or "dev", **kwargs
        )

    async def setup(self, environment):
        user = str(getattr(environment, "default_user", None) or "dev")
        if user == "root":
            raise ValueError("Office actor requires a non-root user")
        created = await environment.exec(
            user="root",
            command=f"id {shlex.quote(user)} >/dev/null 2>&1 || useradd -m -s /bin/bash {shlex.quote(user)}; "
            f"mkdir -p /workspace; chown -R {shlex.quote(user)} /workspace",
        )
        if created.return_code:
            raise RuntimeError("Could not prepare actor user")
        found = await environment._run_docker_compose_command(["ps", "-q", "main"])
        self.container = found.stdout.strip()
        if not self.container or "\n" in self.container:
            raise RuntimeError("Expected one Harbor main container")
        proof = await environment.exec(
            user=user, command="python -c " + shlex.quote(PROBE)
        )
        if proof.return_code:
            raise RuntimeError(
                "Office dependency/Python/read-write preflight failed: " + proof.stderr
            )
        proof = json.loads(proof.stdout)
        if self.config.get("task_images_manifest"):
            row = next(
                r
                for r in read(self.config["task_images_manifest"])["records"]
                if r["task_id"] == self.spec["task"]["task_id"]
            )
            if proof["inputs"] != row["input_sha256"]:
                raise ValueError("Office image task input hashes changed")
        write(self.result_root / "preflight.json", proof)
        channel = "actor-" + self.spec["candidate_id"]
        token = register(self.config, channel, policy())
        self.process = await prepare(
            self.container,
            self.result_root,
            self.config,
            {
                "base_url": self.config["gateway_container"] + "/" + channel + "/v1",
                "api_key": token,
                "model": self.config["roles"]["actor"],
            },
            user=user,
        )

    async def _turn(self, prompt):
        if self.actor_remaining <= 0:
            raise TimeoutError("JiuwenSwarm execution budget exhausted")
        self.turn += 1
        started = time.monotonic()
        try:
            result = (
                await self.process.run(
                    prompt,
                    workspace=Path("/workspace"),
                    session_id=self.session,
                    timeout_s=self.actor_remaining,
                    label=f"turn-{self.turn}",
                )
            ).require_completed()
        finally:
            self.actor_remaining -= time.monotonic() - started
        self.session = result.session_id
        self.usage.append(result.result.get("usage") or None)
        outcomes = tool_outcomes(result.events)
        write(self.result_root / "tool_outcomes" / f"{self.turn}.json", outcomes)
        self.tool_errors = (
            None
            if self.tool_errors is None or outcomes["unknown"]
            else self.tool_errors + outcomes["reported_errors"]
        )
        archive = self.result_root / "sessions" / str(self.turn)
        archive.parent.mkdir(parents=True, exist_ok=True)
        await docker("cp", self.container + ":" + self.process.child_root, str(archive))
        if not result.output.strip():
            raise ValueError("No public final response")
        candidate, _ = await self.snapshot(f"delivery-{self.turn}", result.output)
        write(
            self.result_root / "actor_state.json",
            {
                "session": self.session,
                "turn": self.turn,
                "actor_backend": "jiuwenswarm",
                "remaining_actor_s": self.actor_remaining,
            },
        )
        return candidate

    async def _stop_opencode(self, environment):
        if hasattr(self, "process"):
            await self.process.close()
