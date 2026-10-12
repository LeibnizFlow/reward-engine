"""Harbor OpenCode agent with the current Outcome completion callback."""

import shlex
import time
from pathlib import Path

from reward_engine.adapters.workbuddy.evidence import collect, extract_archive
from reward_engine.adapters.workbuddy.io import auth, read, write
from reward_engine.adapters.workbuddy.models import registry
from reward_engine.adapters.workbuddy.opencode import (
    OpenCodeTransport,
    read_jsonl,
    session_id,
)
from reward_engine.core.batch import BatchPolicy
from reward_engine.core.outcome.memory import Memory
from reward_engine.core.outcome.runner import (
    DeliveryReview,
    review_result_fields,
    run_delivery,
)


def final_reply(events):
    replies = [event.get("part", {}) for event in events if event.get("type") == "text"]
    final_message = replies[-1].get("messageID") if replies else None
    return "\n".join(
        part.get("text", "")
        for part in replies
        if part.get("messageID") == final_message
    )


class OutcomeOpenCodeAgent(OpenCodeTransport):
    """Only the base OpenCode transport is reused; no historical reward calls."""

    def __init__(self, logs_dir, *args, **kwargs):
        self.spec = read(kwargs.pop("OUTCOME_SPEC"))
        self.config = read(self.spec["campaign_config"])
        self.result_root = Path(self.spec["output"])
        self.actor_remaining = self.spec["actor_timeout_s"]
        self.session = None
        self.turn = 0
        super().__init__(logs_dir, *args, **kwargs)

    @staticmethod
    def name():
        return "workbuddy-outcome-opencode"

    def _proxy_token(self):
        return auth(self.config)

    async def install(self, environment):
        await super().install(environment)
        # The root version probe creates this directory before the non-root rollout.
        user = shlex.quote(str(getattr(environment, "default_user", None) or "root"))
        await self.exec_as_root(
            environment,
            command=f"mkdir -p /tmp/opencode && chown -R {user} /tmp/opencode",
        )

    async def _stop_opencode(self, environment):
        await self.exec_as_root(
            environment, command="pkill -TERM -x opencode 2>/dev/null || true"
        )

    async def _opencode_run(self, environment, prompt, *, raw_name, session=None):
        raw = "/logs/agent/" + raw_name
        combined = "/logs/agent/opencode.txt"
        session_flag = " --session " + shlex.quote(session) if session else ""
        command = (
            "export PATH=/usr/local/bin:$PATH; opencode run --dir /workspace --pure --auto "
            "--format json --thinking --model "
            + shlex.quote(f"{self.provider_id}/{self._route}")
            + session_flag
            + " -- "
            + shlex.quote(prompt)
            + " > "
            + shlex.quote(raw)
            + " 2>&1; "
            "run_status=$?; "
            + (
                "cat " + shlex.quote(raw) + " >> " + combined
                if session
                else "cp " + shlex.quote(raw) + " " + combined
            )
            + "; exit $run_status"
        )
        result = await self.exec_as_agent(
            environment, command=command, env=await self._write_config(environment)
        )
        if result.return_code:
            raise RuntimeError(
                f"OpenCode exited {result.return_code}; raw events retained"
            )

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
            raise ValueError("Cannot revise without the actual OpenCode session")
        return await self._turn(feedback)

    async def _turn(self, prompt):
        self.turn += 1
        for continuation in range(3):
            if self.actor_remaining <= 0:
                raise TimeoutError("OpenCode execution budget exhausted")
            suffix = f"-continuation-{continuation}" if continuation else ""
            raw_name = f"outcome-turn-{self.turn}{suffix}.jsonl"
            started = time.monotonic()
            try:
                finished = await self._opencode_run_bounded(
                    self.environment,
                    prompt,
                    raw_name=raw_name,
                    session=self.session,
                    timeout_sec=self.actor_remaining,
                )
            finally:
                self.actor_remaining -= time.monotonic() - started
            if not finished:
                raise TimeoutError("OpenCode execution budget exhausted")
            if self._error_messages():
                raise RuntimeError("OpenCode emitted an error event")
            raw = self.logs_dir / raw_name
            if not raw.exists():
                await self.environment.download_file("/logs/agent/" + raw_name, raw)
            events = read_jsonl(raw)
            observed_session = session_id(events)
            if self.session is not None and observed_session != self.session:
                raise ValueError("OpenCode revision changed session")
            terminal = [e for e in events if e.get("type") == "step_finish"]
            if not observed_session or not terminal:
                raise ValueError("OpenCode did not produce a terminal session receipt")
            self.session = observed_session
            reason = terminal[-1].get("part", {}).get("reason")
            if reason == "stop":
                break
            if reason != "length":
                raise ValueError("OpenCode did not produce a terminal session receipt")
            write(
                self.result_root
                / "continuations"
                / f"turn-{self.turn}-{continuation}.json",
                {
                    "reason": "length",
                    "session": self.session,
                    "raw_log": str(raw),
                    "remaining_actor_s": self.actor_remaining,
                    "will_continue": continuation < 2 and self.actor_remaining > 0,
                },
            )
            if continuation == 2:
                raise RuntimeError(
                    "OpenCode output limit persisted after two same-session continuations"
                )
            prompt = (
                "The previous response reached its output token limit before completion. "
                "Continue the original task in this same session using the existing workspace and context. "
                "Complete the pending work and provide the final response."
            )
        candidate, _ = await self.snapshot(f"delivery-{self.turn}", final_reply(events))
        write(
            self.result_root / "actor_state.json",
            {
                "session": self.session,
                "turn": self.turn,
                "remaining_actor_s": self.actor_remaining,
            },
        )
        return candidate

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
                audit_mode="targeted",
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
                },
            )
            try:
                await self.snapshot("failure")
            except Exception:
                pass
            raise
