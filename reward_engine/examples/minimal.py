"""One task, three real core rounds; scripted offline by default.

Run: python -m reward_engine.examples.minimal --output runs/minimal
Live: add --live --model NAME --base-url URL --key-env NAME --credentials PATH.
Live needs pip install '.[demo]' (the runtime extra also includes these dependencies).
Output must be new. No Docker, archive, benchmark grader, or adversarial QC.
"""

import argparse
import asyncio
import json
from copy import deepcopy
from pathlib import Path

from reward_engine.core.batch import BatchPolicy
from reward_engine.core.outcome.harness import defaults
from reward_engine.core.outcome.loop import run_task
from reward_engine.core.outcome.memory import Memory
from reward_engine.core.outcome.runner import DeliveryReview, run_delivery
from reward_engine.examples.offline import TASK, Offline, candidate, seeds, total_check


class NoTools:
    """No external execution capability is supplied by this example."""

    def catalog(self):
        return {}

    async def run(self, probe, evidence):
        raise ValueError("This example does not offer measurement tools")


class Recorder:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.records = {}
        self.calls = 0
        self.errors = []

    def save(self, name, value):
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Receipt name must stay inside the example output")
        value = deepcopy(value)
        self.records[name] = value
        destination = self.root / (name + ".json")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )

    async def call(self, backend, request, name):
        self.calls += 1
        prefix = f"calls/{self.calls:04d}-{name}"
        self.save(prefix + "/request", request)
        try:
            answer = await backend(request)
        except (Exception, asyncio.CancelledError) as error:
            reason = getattr(error, "safe_reason", None)
            if reason is None:
                if isinstance(error, asyncio.CancelledError):
                    reason = "Call cancelled by enclosing stage/batch deadline or caller; exact origin is not determined here"
                elif isinstance(error, TimeoutError):
                    reason = "Call timed out; no unsafe exception details retained"
                else:
                    reason = (
                        "Call failed; exception text is not marked safe for retention"
                    )
            self.errors.append(
                {"path": prefix, "error_type": type(error).__name__, "reason": reason}
            )
            self.save(prefix + "/failure", self.errors[-1])
            raise
        self.save(prefix + "/response", answer)
        return answer


class Actor:
    """A single in-memory conversation continued by core.run_delivery."""

    def __init__(self, backend, record, identity, arm):
        self.backend, self.record, self.identity, self.arm = (
            backend,
            record,
            identity,
            arm,
        )
        self.messages = [
            {
                "role": "system",
                "content": 'Perform the user task. Respond with a JSON envelope containing only "final_response", a string holding your actual deliverable. This envelope is transport only, not part of the deliverable. Revise that deliverable when review feedback is supplied. Do not invent tool execution.',
            },
            {
                "role": "user",
                "content": (
                    "The task below specifies the deliverable text INSIDE the "
                    "final_response string. Its output-format requirements (including "
                    "ONLY JSON) apply to that inner deliverable, not the transport envelope.\n"
                    "<deliverable_task>\n" + TASK["prompt"] + "\n</deliverable_task>\n"
                    "Return exactly one transport JSON object with only the key "
                    '"final_response". Its value must be a string containing your completed '
                    "deliverable; escape any inner JSON quotes as required for a JSON string. "
                    "Do not return the inner deliverable as the outer object or add Markdown.\n"
                    "Transport shape only; replace the placeholder with your actual work:\n"
                    '{"final_response": "<your completed deliverable text>"}'
                ),
            },
        ]
        self.revisions = 0

    async def _answer(self):
        async def invoke(messages):
            return await self.backend.actor(
                messages, arm=self.arm, revision=self.revisions
            )

        answer = await self.record.call(invoke, deepcopy(self.messages), "actor")
        if (
            set(answer) != {"final_response"}
            or not isinstance(answer["final_response"], str)
            or not answer["final_response"].strip()
        ):
            raise ValueError(
                "Actor must author a nonempty final_response string; no missing fields are synthesized"
            )
        self.messages.append(
            {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)}
        )
        self.record.save(f"actors/{self.identity}/conversation", self.messages)
        return candidate(
            self.identity + f"-v{self.revisions}", answer["final_response"]
        )

    async def start(self, prompt):
        if prompt:
            self.messages.append({"role": "user", "content": prompt})
        return await self._answer()

    async def resume(self, feedback):
        self.revisions += 1
        self.messages.append({"role": "user", "content": feedback})
        return await self._answer()


def verify_execution(result, recorder):
    """Execution requirements only; never replace model verdicts with success."""
    if recorder.errors:
        raise RuntimeError(
            "One or more role/provider calls failed; see calls/*/failure.json"
        )
    if result.get("status") != "completed" or result.get("completed_rounds") != 3:
        raise RuntimeError(
            "Core did not complete three rounds; see result.private.json and construction_failure receipts"
        )
    for row in result["rounds"]:
        if row["ranking"].get("status") != "compared" or not row["selection"].get(
            "arm"
        ):
            raise RuntimeError(
                "Ranking/selection unavailable; no fallback is reported as successful execution"
            )
        n = row["round"]
        for stage in ("jury_map", "hypothesis_map", "ranking_map"):
            batch = recorder.records.get(f"round_{n}/{stage}", {})
            if batch.get("succeeded") != 3 or not batch.get("complete"):
                raise RuntimeError(
                    f"Round {n} {stage} did not complete all three roles"
                )
        hypothesis = recorder.records.get(f"round_{n}/hypothesis", {})
        if (
            hypothesis.get("status") not in {"propose", "keep"}
            or hypothesis.get("availability") == "unavailable"
        ):
            raise RuntimeError("Hypothesis stage unavailable")
        smoke = recorder.records.get(f"round_{n}/construction_smoke", {})
        if (
            smoke.get("status") != "receipt_path_checked"
            or smoke.get("receipt", {}).get("error")
            or not smoke.get("bindings")
            or any(v != "callable" for v in smoke["bindings"].values())
        ):
            raise RuntimeError("Builder/check harness was not callable")
        construction = recorder.records.get(f"round_{n}/construction_checks", [])
        if not construction or any(
            not measurement.get("results")
            or any(r.get("error") for r in measurement["results"])
            for measurement in construction
        ):
            raise RuntimeError("Construction measurement execution failed")
        for arm in "ABC":
            delivery = row["arms"].get(arm, {})
            if delivery.get("status") != "completed":
                raise RuntimeError(f"Round {n} arm {arm} failed")
            measurement = recorder.records[f"round_{n}/{arm}/outcome_estimate"]
            if not measurement.get("results") or any(
                r.get("error") for r in measurement["results"]
            ):
                raise RuntimeError(
                    "A full delivered measurement failed; see retained receipts"
                )
            if delivery.get("review_status") in {
                "check_error_unresolved",
                "check_unavailable",
            }:
                raise RuntimeError("Delivery review execution unavailable")


async def run(output, *, backend=None):
    backend = backend or Offline()
    config = getattr(backend, "config", None)
    concurrency = config.role_concurrency if config else 1
    stage_timeout = config.stage_timeout if config else 180
    # Three panel seats may run sequentially. Do not leave the core's 240s
    # default batch cutoff ahead of explicitly configured long role stages.
    policy = BatchPolicy(
        concurrency=concurrency, deadline_s=max(240, 3 * stage_timeout)
    )
    record = Recorder(output)
    scope = {
        "mode": backend.mode,
        "controller": "reward_engine.core.outcome.loop.run_task",
        "rounds": 3,
        "construction_strategy": "legacy",
        "adversarial_review_enabled": False,
        "task": TASK,
        "native_evaluation": "not_run",
        "runtime_settings": {
            "provider": config.public_settings() if config else None,
            "stage_timeout_s": stage_timeout,
            "batch_policy": {
                "concurrency": policy.concurrency,
                "deadline_s": policy.deadline_s,
                "group_concurrency": policy.group_concurrency,
                "deadline_rule": "max(240, 3 * stage_timeout_s), allowing three panel seats",
            },
        },
        "limits": [
            "Scripted offline roles are not LLM/native validation.",
            "Basic path demonstration, not all branches or adversarial QC.",
            "Live uses one model for all seats, not independent model evidence.",
            "No native scores or RL training. Live revision is observed, never forced.",
            "Behavior is deterministic offline; core UUIDs and timings are not byte-identical.",
        ],
    }
    record.save("scope", scope)
    try:

        def roles(role, prefix):
            async def call(request):
                return await record.call(backend.role, request, role)

            call.bounded_stage = True
            call.stage_timeout_s = stage_timeout
            return call

        registry = defaults(roles("measurement", "measurement"), tools=NoTools())
        registry.register(
            "example.total",
            total_check,
            'For minimal-total ONLY: parse final_response, require exactly {"total": integer sum of task quantities}. arguments={"ref":"product_1:artifact:final_response"}.',
        )
        # Keep live compilation small: allow only already registered checks, no
        # generated executable code, tools, Docker, or workspace mutation.

        async def sample(arm, number, prompt, pack):
            identity = f"round-{number}-{arm}"
            actor = Actor(backend, record, identity, arm)
            local_memory = Memory()
            review = (
                DeliveryReview(
                    TASK,
                    pack,
                    registry,
                    max_revisions=2,
                    audit_mode="full",
                    policy=policy,
                    memory=local_memory,
                    source_prefix=f"round_{number}/actors/{arm}/checks",
                    record=lambda name, value: record.save(
                        f"round_{number}/actors/{arm}/checks/{name}", value
                    ),
                )
                if pack is not None
                else None
            )
            current = await run_delivery(actor, review, prompt=prompt)
            result = {
                "status": "completed",
                "candidate": current,
                "session_id": identity,
                "revision_count": actor.revisions,
                "review_status": review.status if review else "disabled",
                "memory": deepcopy(local_memory.rows),
            }
            if review:
                if any(
                    r.get("error") for m in review.history for r in m["results"]
                ) or any(
                    m.get("completeness_review", {}).get("error")
                    for m in review.history
                ):
                    raise RuntimeError(
                        "Review measurement failed; retain attempts, do not treat as valid critique"
                    )
                result.update(
                    review.result_fields(current),
                    final_measurement=deepcopy(review.history[-1]),
                )
            record.save(f"round_{number}/{arm}/delivery", result)
            return result

        initial = seeds()
        if backend.mode != "offline_deterministic":
            initial = [
                await Actor(backend, record, f"seed-{i}", "seed").start("")
                for i in range(3)
            ]
        record.save("seeds", initial)
        result = await run_task(
            deepcopy(TASK),
            initial,
            rounds=3,
            continuation_policy="fixed_rounds",
            full_audit=True,
            roles=roles,
            registry=registry,
            sample=sample,
            memory=Memory(),
            policy=policy,
            record=record.save,
        )
        verify_execution(result, record)
        summary = {
            "status": "execution_completed",
            "mode": backend.mode,
            "native_evaluation": "not_run",
            "completed_rounds": result["completed_rounds"],
            "seed_count": len(initial),
            "actor_count": sum(len(r["arms"]) for r in result["rounds"]),
            "rounds": [
                {
                    "round": r["round"],
                    "selected_arm": r["selection"]["arm"],
                    "revision_count": r["arms"]["B"]["revision_count"],
                    "review_status": r["arms"]["B"]["review_status"],
                    "memory_entries": len(record.records[f"round_{r['round']}/memory"]),
                }
                for r in result["rounds"]
            ],
            "memory_entries": len(result["memory"]),
            "role_and_actor_calls": record.calls,
            "limits": scope["limits"],
        }
        record.save("summary", summary)
        return summary
    except Exception as error:
        record.save(
            "failure",
            {
                "status": "failed",
                "mode": backend.mode,
                "error_type": type(error).__name__,
                "reason": str(error),
            },
        )
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New directory; existing outputs are never overwritten",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Real paid provider calls; default is scripted offline",
    )
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--key-env", default="OPENAI_API_KEY")
    parser.add_argument(
        "--request-timeout",
        type=float,
        help="Live HTTP per-operation timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--stage-timeout",
        type=float,
        help="Live core role stage timeout in seconds, including queueing (default: 180)",
    )
    parser.add_argument(
        "--role-concurrency",
        type=int,
        help="Shared live model HTTP concurrency, including actors (default: 1)",
    )
    parser.add_argument(
        "--thinking",
        choices=("provider_default", "disabled"),
        help="Default: omit overrides; disabled sends thinking.type=disabled and enable_thinking=false",
    )
    parser.add_argument(
        "--credentials",
        type=Path,
        help="Optional dotenv file; defaults to .env if present; environment takes precedence",
    )
    args = parser.parse_args(argv)
    if not args.live and (
        args.model
        or args.base_url
        or args.credentials
        or args.key_env != "OPENAI_API_KEY"
        or args.request_timeout is not None
        or args.stage_timeout is not None
        or args.role_concurrency is not None
        or args.thinking is not None
    ):
        parser.error("Provider options require --live; no silent offline fallback")
    try:
        backend = None
        if args.live:
            from reward_engine.examples.live import Live, LiveConfig

            backend = Live(
                LiveConfig.load(
                    model=args.model,
                    base_url=args.base_url,
                    key_env=args.key_env,
                    credentials=args.credentials,
                    timeout=args.request_timeout
                    if args.request_timeout is not None
                    else 120,
                    stage_timeout=args.stage_timeout
                    if args.stage_timeout is not None
                    else 180,
                    role_concurrency=args.role_concurrency
                    if args.role_concurrency is not None
                    else 1,
                    thinking=args.thinking or "provider_default",
                )
            )
        summary = asyncio.run(run(args.output, backend=backend))
    except Exception as error:
        parser.exit(1, f"Example failed ({type(error).__name__}): {error}\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
