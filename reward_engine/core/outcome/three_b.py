"""Fixed baselines, independent B sessions and current-round six-way assessment.

Persistence is supplied by the benchmark through record/restore. Native scores
are only passed to evaluate and private summaries, never to learning or selection.
"""

from copy import deepcopy

from reward_engine.core.outcome import jury, pool_selector
from reward_engine.core.outcome.construction import fingerprint
from reward_engine.core.outcome.context import build_context, check_bundle
from reward_engine.core.outcome.harness import measure, smoke
from reward_engine.core.outcome.loop import actor_reward_prompt
from reward_engine.core.outcome.memory import Memory
from reward_engine.core.outcome.validated_construction import construct_reward

PROTOCOL = "three_b_fixed_baselines_v1"
ARMS = ("B1", "B2", "B3")
BASELINES = ("seed_0", "seed_1", "seed_2")
CANDIDATES = BASELINES + ARMS


def validate_config(config):
    protocol = config.get("generation_protocol")
    if protocol not in (None, "legacy_abcd", PROTOCOL):
        raise ValueError("Unknown generation_protocol")
    if protocol != PROTOCOL:
        return
    if (
        type(config.get("rounds", 3)) is not int
        or not 1 <= config.get("rounds", 3) <= 3
    ):
        raise ValueError("Three B requires one to three rounds")
    if (
        config.get("selection_policy", "legacy") != "legacy"
        or config.get("public_delivery_contract")
        or config.get("hypothesis_enabled", False) is not False
        or config.get("max_revisions", 2) != 2
        or config.get("reuse_first_round_from")
        or config.get("recover_from")
        or config.get("canary_from")
    ):
        raise ValueError(
            "Three B is incompatible with selection/delivery/replay overrides"
        )


def baseline_hashes(states):
    if len(states) != 3 or any(not row.get("candidate") for row in states):
        raise ValueError("Three fixed baseline states are required")
    return {
        key: fingerprint(row["candidate"])
        for key, row in zip(BASELINES, states, strict=True)
    }


def prepare_selection(task, candidates, checkpoint=None):
    if set(candidates) != set(CANDIDATES):
        raise ValueError(
            "Selection requires three fixed baselines and three B deliveries"
        )
    return pool_selector.prepare(
        task,
        candidates,
        runtime={"generation_protocol": PROTOCOL},
        previous=checkpoint.get("prepared") if checkpoint else None,
    )


async def select(
    task, candidates, roles, *, record, policy=None, max_bytes=393216, checkpoint=None
):
    """Three Map seats and one Reduce, preserving target evidence validation."""
    prepared = prepare_selection(task, candidates, checkpoint)
    record("selection/prepared", prepared)
    saved = checkpoint.get("ranking") if checkpoint else None
    if isinstance(saved, dict) and saved.get("status") == "unavailable":
        if checkpoint.get("decision") is not None:
            raise ValueError("Unavailable ranking cannot have a completed decision")
        record("selection/unavailable/" + fingerprint(saved), deepcopy(saved))
        saved = None
    if saved is not None:
        # A malformed claimed completion is not permission to redraw a verdict.
        jury.validate_ranking(saved, prepared["context"], prepared["evidence"])
        ranking = deepcopy(saved)
    else:
        ranking = await jury.rank(
            prepared["context"],
            prepared["evidence"],
            {str(i): roles(f"jury_member_{i}") for i in range(1, 4)},
            roles("jury_reducer"),
            policy=policy,
            reduce_max_bytes=max_bytes,
            record=lambda name, value: record("selection/" + name, value),
        )
    if ranking.get("status") == "unavailable":
        record("selection/unavailable/" + fingerprint(ranking), deepcopy(ranking))
        raise RuntimeError("Six-candidate ranking unavailable: " + ranking.get("reason", "Unknown failure"))
    decision = pool_selector.decide(ranking, prepared)
    decision.update(arm=decision.pop("candidate_key"), policy=PROTOCOL)
    if (
        checkpoint
        and checkpoint.get("decision") is not None
        and decision != checkpoint["decision"]
    ):
        raise ValueError("Frozen six-candidate selection changed")
    record("selection/ranking", ranking)
    record("selection/decision", decision)
    return ranking, decision


class IncompleteMeasurement(ValueError):
    """A failed full check with its evidence retained for diagnosis and recovery."""

    def __init__(self, measured):
        super().__init__("Full measurement did not complete; selection is blocked")
        self.measurement = deepcopy(measured)


def require_measurement(pack, measured):
    rows = measured.get("results", [])
    expected = {t["test_id"] for t in pack.get("tests", [])}
    if (
        measured.get("pack_id") != pack["pack_id"]
        or measured.get("error")
        or measured.get("unresolved")
        or not isinstance(rows, list)
        or len(rows) != len(expected)
        or {r.get("test_id") for r in rows} != expected
        or any(
            r.get("status") not in {"pass", "fail", "unknown"}
            or r.get("error")
            or r.get("unresolved_measurement_refs")
            for r in rows
        )
    ):
        raise IncompleteMeasurement(measured)


class Stages:
    """Content-bound completed stages; ambiguous actor starts are never retried."""

    def __init__(self, binding, record, restore, implementation=None, recover_actor=None,
                 admit_stage=None):
        self.binding, self.record, self.restore = binding, record, restore
        self.implementation = implementation
        self.recover_actor = recover_actor
        self.admit_stage = admit_stage

    async def run(self, name, inputs, execute, *, actor=False):
        identity = fingerprint({"run": self.binding, "inputs": inputs})
        key = name + "/stage"
        saved = self.restore(key)
        if saved is not None:
            if saved.get("identity") != identity:
                raise ValueError("Checkpoint inputs changed: " + name)
            if saved.get("status") == "completed":
                if fingerprint(saved["value"]) != saved.get("value_sha256"):
                    raise ValueError("Checkpoint content changed: " + name)
                return deepcopy(saved["value"])
        if self.admit_stage:
            self.admit_stage(name)
        if saved is not None and actor and not (self.recover_actor and self.recover_actor(name, saved)):
            raise RuntimeError(
                "Uncertain actor stage; inspect retained failure, no resampling: "
                + name
            )
        attempt = (saved or {}).get("attempt", 0) + 1
        self.record(
            key, {"identity": identity, "status": "started", "attempt": attempt,
                  "implementation": self.implementation}
        )
        try:
            value = await execute()
        except BaseException as error:
            if not hasattr(error, "stage"):
                error.stage = name
            failure = {
                "identity": identity,
                "attempt": attempt,
                "error_type": type(error).__name__,
                "reason": str(error),
                "implementation": self.implementation,
            }
            if isinstance(error, IncompleteMeasurement):
                failure["measurement"] = error.measurement
            self.record(name + f"/failures/{attempt}", failure)
            self.record(name + "/failure", failure)
            raise
        self.record(
            key,
            {
                "identity": identity,
                "status": "completed",
                "attempt": attempt,
                "value": value,
                "value_sha256": fingerprint(value),
                "implementation": self.implementation,
            },
        )
        return deepcopy(value)


def public_state(row):
    if (
        row.get("status") != "completed"
        or not row.get("candidate")
        or not row.get("session")
    ):
        raise ValueError("Actor needs a completed delivery and actual session")
    if row.get("feedback_execution", {}).get("status") == "failed" or row.get(
        "review_status"
    ) in {"failed", "unavailable", "check_unavailable", "check_error_unresolved"}:
        raise ValueError("Actor critic did not complete")
    allowed = (
        "status",
        "candidate",
        "session",
        "runtime_directory",
        "native_grader_trial",
        "review_status",
        "review_session",
        "memory",
        "resources",
        "usage",
        "tool_errors",
        "actor_execution_s",
        "actor_retry",
        "spec_identity",
        "identity",
        "artifact_bindings",
    )
    return {k: deepcopy(row[k]) for k in allowed if k in row}


async def run_task(
    task,
    *,
    rounds=3,
    runtime,
    baseline,
    sample,
    roles,
    registry,
    evaluate=None,
    policy=None,
    jury_reduce_max_bytes=393216,
    record=lambda *_: None,
    restore=lambda _: None,
    verify_actor=lambda _: None,
    implementation=None,
    recover_actor=None,
    admit_stage=None,
):
    """Run resumable stages using native adapters, with no hypothesis construction."""
    validate_config({"generation_protocol": PROTOCOL, "rounds": rounds})
    if task.get("public_delivery_contract"):
        raise ValueError("Three B cannot bypass public delivery contract selection")
    binding = {
        "task": task,
        "runtime": runtime,
        "generation_protocol": PROTOCOL,
        "rounds": rounds,
        "hypothesis_enabled": False,
    }
    old = restore("protocol")
    if old is not None and old != binding:
        raise ValueError("Task, protocol or runtime changed; incompatible checkpoint")
    record("protocol", binding)
    stages = Stages(binding, record, restore, implementation, recover_actor, admit_stage)
    states, sessions = {}, set()

    def accept(row):
        verify_actor(row)
        if row["session"] in sessions:
            raise ValueError("Baseline/B actors must use independent real sessions")
        sessions.add(row["session"])
        return row

    for index, key in enumerate(BASELINES):

        async def generate(index=index):
            return public_state(await baseline(index))

        states[key] = accept(await stages.run(key, task, generate, actor=True))
        record(key, states[key])
    fixed = baseline_hashes(list(states.values()))
    pool, checks, memory, results = (
        [s["candidate"] for s in states.values()],
        [],
        Memory(),
        [],
    )
    for number in range(1, rounds + 1):
        prefix = f"round_{number}"

        def save(name, value, prefix=prefix):
            record(prefix + "/" + name, deepcopy(value))

        context, evidence = build_context(task, pool, checks=checks)
        context["memories"] = await memory.retrieve(context, top_k=3)
        save("context", context)

        async def construct(
            context=context, evidence=evidence, prefix=prefix, save=save
        ):
            from reward_engine.core.outcome.authored_json import enabled
            from reward_engine.core.outcome.resume_inputs import reduction_checkpoint

            checkpoint = None
            if enabled(roles("jury_reducer", prefix)):
                checkpoint = reduction_checkpoint(restore(prefix + "/proposal/jury_reduce_checkpoint"), context, evidence)
                if checkpoint is not None:
                    save("proposal_reuse", {"source_sha256": fingerprint(checkpoint["jury_reduce"]),
                                            "public_input_validated": True, "member_contracts_validated": 3})
            return await construct_reward(
                context,
                evidence,
                registry,
                lambda role: roles(role, prefix),
                record=save,
                reduce_max_bytes=jury_reduce_max_bytes,
                adversarial_review_enabled=True,
                policy=policy,
                **({"checkpoint": checkpoint} if checkpoint is not None else {}),
            )

        pack = await stages.run(
            prefix + "/construction",
            {"context": context, "evidence": evidence},
            construct,
        )
        save("pack", pack)
        save("hypothesis", {"status": "disabled", "applied": False})
        save(
            "arm_protocol",
            {"id": PROTOCOL, "arms": list(ARMS), "hypothesis_applied": False},
        )

        async def smoke_pack(pack=pack):
            receipt = await smoke(pack, registry)
            if (
                receipt.get("status") != "receipt_path_checked"
                or receipt.get("receipt", {}).get("error")
                or any(v != "callable" for v in receipt.get("bindings", {}).values())
            ):
                raise ValueError("RewardPack execution preflight did not complete")
            return receipt

        await stages.run(prefix + "/construction_smoke", pack, smoke_pack)

        async def inspect(candidate, stage_name, pack=pack):
            c, e = build_context(task, [candidate])
            from reward_engine.core.outcome.authored_json import enabled

            prior = restore(stage_name + "/failure") if enabled(getattr(registry, "review_call", None)) else None
            completed = prior.get("measurement") if prior else None
            measured = await measure(pack, registry, c, e, policy=policy, test_ids=None,
                                     **({"completed": completed} if completed is not None else {}))
            require_measurement(pack, measured)
            return measured

        measured_pool = []
        for index, candidate in enumerate(pool):

            async def inspect_existing(candidate=candidate, inspect=inspect, index=index, prefix=prefix):
                return await inspect(candidate, prefix + f"/construction_checks/{index}")

            measured = await stages.run(
                prefix + f"/construction_checks/{index}",
                {"pack": pack, "candidate": candidate},
                inspect_existing,
            )
            require_measurement(pack, measured)
            measured_pool.append(
                check_bundle(
                    f"product_{index + 1}",
                    measured,
                    f"{prefix}/construction_checks/{index}/stage.json",
                )
            )
        checks = measured_pool
        arms = {}
        for arm in ARMS:

            async def generate(arm=arm, number=number, pack=pack):
                return public_state(
                    await sample(arm, number, actor_reward_prompt(pack), deepcopy(pack))
                )

            row = accept(
                await stages.run(
                    prefix + "/" + arm,
                    {"pack": pack, "baselines": fixed},
                    generate,
                    actor=True,
                )
            )
            row["hypothesis_applied"] = False
            arms[arm] = row
            save(arm + "/delivery", row)
        measurements = {}
        for arm, row in arms.items():

            async def inspect_final(row=row, inspect=inspect, arm=arm, prefix=prefix):
                return await inspect(row["candidate"], prefix + "/" + arm + "/measurement")

            measured = await stages.run(
                prefix + "/" + arm + "/measurement",
                {"pack": pack, "candidate": row["candidate"]},
                inspect_final,
            )
            require_measurement(pack, measured)
            measurements[arm] = measured
            save(arm + "/outcome_estimate", measured)
        candidates = {
            **{k: v["candidate"] for k, v in states.items()},
            **{k: v["candidate"] for k, v in arms.items()},
        }

        async def choose(candidates=candidates, prefix=prefix, save=save):
            checkpoint = {
                k: restore(prefix + "/selection/" + k)
                for k in ("prepared", "ranking", "decision")
            }
            return await select(
                task,
                candidates,
                lambda role: roles(role, prefix + "/selection"),
                record=save,
                policy=policy,
                max_bytes=jury_reduce_max_bytes,
                checkpoint=checkpoint if checkpoint["prepared"] else None,
            )

        ranking, selection = await stages.run(prefix + "/selector", candidates, choose)
        selected = selection["arm"]
        selection.update(
            round=number,
            source=(
                f"{selected}.json"
                if selected in BASELINES
                else f"{prefix}/{selected}/delivery.json"
            )
            if selected
            else None,
        )
        save("selected_delivery", selection)
        # Native grading receives original native provenance. It never enters a public checkpoint.
        official = {}
        if evaluate is not None:
            for key, row in {**states, **arms}.items():

                async def grade(row=row):
                    scored = await evaluate(deepcopy(task), deepcopy(row))
                    if scored.get("status") not in {"scored", "graded"}:
                        raise RuntimeError(
                            "Native grading incomplete; retained private receipts require retry"
                        )
                    return scored

                grade_key = fingerprint(
                    {
                        "task": task,
                        "candidate": row["candidate"],
                        "session": row["session"],
                        "native": row.get("runtime_directory"),
                        "runtime": runtime,
                    }
                )
                official[key] = await stages.run(
                    "grades.private/" + grade_key, row, grade
                )
        for arm, row in arms.items():
            pool.append(deepcopy(row["candidate"]))
            checks.append(
                check_bundle(
                    f"product_{len(pool)}",
                    measurements[arm],
                    f"{prefix}/{arm}/outcome_estimate.json",
                )
            )
            for entry in row.get("memory", []):
                if entry["memory_id"] not in {r["memory_id"] for r in memory.rows}:
                    memory.rows.append(deepcopy(entry))
        complete = bool(selected) and all(
            s.get("status") in {"scored", "graded"} for s in official.values()
        )
        summary = {
            "status": "completed" if complete else "partial",
            "round": number,
            "arms": arms,
            "ranking": ranking,
            "selection": selection,
            "official": official,
            "selected_official": official.get(selected),
            "hypothesis_applied": False,
        }
        checkpoint = {
            "binding": fingerprint(binding),
            "generation_protocol": PROTOCOL,
            "baselines": fixed,
            "completed_round": number,
            "pack_sha256": fingerprint(pack),
            "candidates": {k: fingerprint(v) for k, v in candidates.items()},
            "pool": pool,
            "checks": checks,
            "memory": memory.rows,
        }
        saved = restore(prefix + "/checkpoint")
        if saved is not None and saved != checkpoint:
            raise ValueError("Frozen round checkpoint changed")
        save("checkpoint", checkpoint)
        save("summary.private", summary)
        results.append(summary)
        if not complete:
            break
    result = {
        "status": "completed"
        if len(results) == rounds and all(r["status"] == "completed" for r in results)
        else "partial",
        "completed_rounds": sum(r["status"] == "completed" for r in results),
        "rounds": results,
        "selected_delivery": results[-1]["selection"],
        "generation_protocol": PROTOCOL,
        "hypothesis_enabled": False,
        "official_evaluation": "separate_from_outcome_estimator",
    }
    record("result.private", result)
    return result
