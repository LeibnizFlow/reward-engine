"""Shared native Office snapshot and completion/review lifecycle."""

import shlex
import time

from reward_engine.adapters.workbuddy.evidence import collect, extract_archive
from reward_engine.adapters.workbuddy.io import write
from reward_engine.adapters.workbuddy.models import registry
from reward_engine.core.batch import BatchPolicy
from reward_engine.core.outcome.memory import Memory
from reward_engine.core.outcome.runner import (
    DeliveryReview,
    review_result_fields,
    run_delivery,
)


class OfficeDelivery:
    async def snapshot(self, name, final_response=""):
        target = self.result_root / "snapshots" / name
        target.mkdir(parents=True, exist_ok=False)
        archive = "/tmp/workbuddy-outcome-snapshot.tar.gz"
        command = (
            "tar --exclude='./.git' --exclude='./node_modules' --exclude='./.opencode' "
            "--exclude='./.venv' --exclude='*/__pycache__' --exclude='./.cache' "
            "--exclude='./.pytest_cache' --hard-dereference -czf "
            + shlex.quote(archive)
            + " -C /workspace ."
        )
        result = await self.exec_as_agent(self.environment, command=command)
        if result.return_code != 0:
            raise RuntimeError("Workspace snapshot failed")
        await self.environment.download_file(archive, target / "workspace.tar.gz")
        extract_archive(target / "workspace.tar.gz", target / "workspace")
        return collect(
            target / "workspace",
            candidate_id=self.spec["candidate_id"],
            final_response=final_response,
            initial_hashes=getattr(self, "initial_hashes", None),
            output=target,
            public_delivery=bool(self.spec["task"].get("public_delivery_contract")),
        )

    async def start(self, prompt):
        return await self._turn(self.instruction + ("\n\n" + prompt if prompt else ""))

    async def resume(self, feedback):
        if not self.session:
            raise ValueError("Cannot revise without the actual actor session")
        return await self._turn(feedback)

    async def run(self, instruction, environment, context):
        del context
        self.environment, self.instruction = (
            environment,
            self.render_instruction(instruction),
        )
        if self.spec["task"].get("public_delivery_contract"):
            from reward_engine.core.outcome.public_delivery import (
                actor_instruction,
                validate,
            )

            contract = validate(
                self.spec["task"]["public_delivery_contract"], self.spec["task"]
            )
            public = actor_instruction(contract)
            if public not in self.spec.get("prompt", ""):
                self.instruction += "\n\n" + public
        self._instruction = self.instruction
        _, self.initial_hashes = await self.snapshot("initial")
        memory, review = Memory(), None
        if self.spec.get("pack"):
            checks = self.result_root / "checks"
            review = DeliveryReview(
                self.spec["task"],
                self.spec["pack"],
                registry(self.config, checks),
                memory=memory,
                max_revisions=2,
                max_tests=3,
                audit_mode="full"
                if self.config.get("generation_protocol")
                == "three_b_fixed_baselines_v1"
                else "targeted",
                policy=BatchPolicy(**self.config["batch"]),
                source_prefix=str(checks),
                record=lambda n, v: write(checks / (n + ".json"), v),
            )
        started = time.monotonic()
        try:
            candidate = await run_delivery(
                self, review, prompt=self.spec.get("prompt", "")
            )
            write(
                self.result_root / "ACTOR_RESULT.json",
                {
                    "status": "completed",
                    "candidate": candidate,
                    "session": self.session,
                    "runtime_directory": str(self.result_root),
                    "usage": getattr(self, "usage", None),
                    "tool_errors": getattr(self, "tool_errors", None),
                    "review_status": review.status if review else "disabled",
                    "memory": memory.rows,
                    "task_progress": review.progress.snapshot() if review else None,
                    "final_measurement": review.history[-1]
                    if review and review.history
                    else None,
                    "resources": {
                        "actor_execution_s": self.spec["actor_timeout_s"]
                        - self.actor_remaining,
                        "actor_and_review_s": time.monotonic() - started,
                        "turns": self.turn,
                        "revision_turns": max(0, self.turn - 1),
                    },
                    "official_verifier_observed": False,
                    **review_result_fields(review, candidate),
                },
            )
        except BaseException as error:
            await self._stop_opencode(environment)
            write(
                self.result_root / "ACTOR_FAILURE.json",
                {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "reason": str(error),
                    "session": self.session,
                    "runtime_directory": str(self.result_root),
                    "usage": getattr(self, "usage", None),
                    "tool_errors": getattr(self, "tool_errors", None),
                },
            )
            try:
                await self.snapshot("failure")
            except Exception:
                pass
            raise
