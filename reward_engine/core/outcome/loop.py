"""Outcome-first rounds and task-list runner; official scores never feed learning."""

import asyncio
from copy import deepcopy

from reward_engine.core.outcome import builder, hypotheses, jury
from reward_engine.core.outcome.arms import FEEDBACK, LEGACY, arm_spec, has_hypothesis
from reward_engine.core.outcome.construction import validate_rounds
from reward_engine.core.outcome.context import build_context, check_bundle
from reward_engine.core.outcome.harness import measure, smoke
from reward_engine.core.outcome.memory import Memory
from reward_engine.core.outcome.review_policy import validate_review_flag


def actor_reward_prompt(pack):
    """Same public guidance for an ordinary B run and an explicit infrastructure retry."""
    text = (
        "Outcome standards for your fresh Task. Prior user answers are not provided; "
        "clarify applicable unknown requirements yourself. "
        "These standards are fallible guidance, not proof that their factual assertions are true. "
        "For consequential claims that depend on external facts or actual runtime behavior, use your "
        "available search/read/execution tools selectively when a bounded check can resolve uncertainty. "
        "Existing adequate evidence or a sound derivation can suffice; there is no tool-call quota. "
        "Do not replace feasible verification of a central claim with a blanket 'verify later' disclaimer. "
        "Distinguish what was actually checked from proposed tests and remaining uncertainty. "
        "A request for a review/design does not authorize implementation, deployment, or unavailable "
        "hardware experiments. Respect the original Task and current user's authority.\n"
        + "\n".join(
            "- Honor the current user's actual additional requirements; clarify unknowns before committing."
            if c.get("origin") == "user"
            else f"- {c['standard']} Full satisfaction: {c['anchors']['full']}"
            for c in pack["criteria"]
        )
    )
    # Task-scoped research leads only: never attach historical product/user data,
    # private grader output, or entire research documents to an Actor prompt.
    sources = []
    for source in pack.get('measurement_sources', {}).values():
        url = source.get('url') if isinstance(source, dict) else None
        if isinstance(url, str) and url.startswith(('https://', 'http://')) and url not in sources:
            sources.append(url)
    if sources:
        text += ('\nOptional professional source leads from Task research (not verified conclusions '
                 'or extra user requirements; inspect relevance before relying on them):\n'
                 + '\n'.join('- ' + url for url in sources[:6]))
    return text


def arm_prompts(pack, method, protocol=LEGACY, *, adversarial_review_enabled=False):
    validate_review_flag(adversarial_review_enabled)
    if pack.get('feedback_policy') == 'evidence_admitted':
        from reward_engine.core.outcome.admission import require_publication
        require_publication(pack)
        hypotheses.require_validated_projection(method, allow_unverified=not adversarial_review_enabled)
    if protocol == FEEDBACK and not has_hypothesis(method):
        method = {'status': 'keep', 'method': ''}
    exploration = hypotheses.actor_prompt(method, allow_unverified=not adversarial_review_enabled)
    reward = actor_reward_prompt(pack) if protocol == LEGACY else ""
    prompts = {}
    for arm in "ABCD":
        spec = arm_spec(protocol, arm)
        prompts[arm] = "\n\n".join(p for p in (
            reward if spec["static_reward"] else "",
            exploration if spec["hypothesis"] else "",
        ) if p)
    if protocol == LEGACY:
        # Preserve the historical full-loop projection byte for byte.
        prompts['D'] = prompts['B'] + '\n\nOptional exploration (not a new requirement):\n' + prompts['C']
    return prompts


def select_delivery(ranking, arm_products, measurements):
    """Original is the fallback. Relative preference is fallible, not a GT guarantee."""
    fallback = {
        "arm": "A" if "A" in arm_products else None,
        "reason": "Keep Original unless a supported improvement clears known check regressions",
    }
    if "A" not in arm_products or ranking.get("status") != "compared":
        return fallback
    positions = {p: i for i, group in enumerate(ranking["groups"]) for p in group}
    original = positions.get(arm_products["A"])
    if original is None:
        return fallback
    baseline_checks = {r["test_id"] for r in measurements["A"]["results"] if r["status"] == "pass"}
    eligible = []
    for arm in (a for a in arm_products if a != "A"):
        pos = positions.get(arm_products.get(arm))
        if pos is None or pos >= original:
            continue
        checks = {r["test_id"]: r["status"] for r in measurements[arm]["results"] if not r.get("error")}
        if not any(checks.get(test) == "fail" for test in baseline_checks):
            eligible.append((pos, arm))
    if not eligible:
        return fallback
    return {
        "arm": min(eligible)[1],
        "reason": "Strictly preferred to Original; no known checked regression",
        "boundary": "Jury/measurement-based fallback rule, not guaranteed official-score monotonicity",
    }


async def run_task(
    task,
    seeds,
    *,
    rounds=3,
    construction_rounds=1,
    construction_strategy="legacy",
    adversarial_review_enabled=False,
    roles,
    registry,
    sample,
    evaluate=None,
    memory=None,
    top_k=3,
    skills=(),
    task_review=None,
    policy=None,
    start_round=1,
    initial_checks=(),
    initial_pack=None,
    initial_expectation_cache=None,
    initial_known_gaps=(),
    initial_decisions=(),
    full_audit=False,
    jury_reduce_max_bytes=160000,
    enabled_arms=("A", "B", "C"),
    arm_protocol=LEGACY,
    continuation_policy="jury_new_gap",
    delivery_selector=None,
    defer_final_selection=False,
    record=lambda *_: None,
):
    """One round by default; rounds caps Jury-approved important-new-gap extensions.

    sample(arm, round, prompt, pack) returns a REAL product, review log and cost.

    One independent fresh Actor per arm. arm_protocol fixes the treatment meanings;
    legacy configs keep their historical meaning. Official evaluation is separate.
    """
    validate_review_flag(adversarial_review_enabled)
    if type(rounds) is not int or not 1 <= rounds <= 3 or len(seeds) < 3:
        raise ValueError("Use at least three real seeds and one to three rounds")
    validate_rounds(construction_rounds)
    if construction_strategy not in {"legacy", "counterexample_validated"}:
        raise ValueError("Unknown construction strategy")
    hardened = construction_strategy == "counterexample_validated"
    if hardened and (construction_rounds != 1 or rounds != 1):
        raise ValueError("Counterexample construction runs once, without fixed-loop layering")
    for arm in enabled_arms:
        arm_spec(arm_protocol, arm)
    if type(start_round) is not int or start_round < 1 or start_round + rounds > 4:
        raise ValueError("Continuation must stay within the original three rounds")
    if continuation_policy not in {"jury_new_gap", "fixed_rounds"}:
        raise ValueError("Unknown continuation_policy")
    if delivery_selector is not None and (continuation_policy != "fixed_rounds" or not set(enabled_arms) <= set("ABC")):
        raise ValueError("Verified current selection requires fixed-round A/B/C execution")
    if type(full_audit) is not bool:
        raise ValueError("full_audit must be an explicit boolean")
    if type(jury_reduce_max_bytes) is not int or jury_reduce_max_bytes < 1:
        raise ValueError("jury_reduce_max_bytes must be a positive integer")
    if (not enabled_arms or len(set(enabled_arms)) != len(enabled_arms)
            or not set(enabled_arms) <= {"A", "B", "C", "D"}):
        raise ValueError("Choose distinct A/B/C/D arms")
    if type(defer_final_selection) is not bool or (defer_final_selection and (
            continuation_policy != 'fixed_rounds' or hardened or delivery_selector is not None)):
        raise ValueError('Deferred final selection requires fixed legacy construction without another selector')
    pool, memory = deepcopy(list(seeds)), memory if memory is not None else Memory()
    results, prior_checks = [], deepcopy(list(initial_checks))
    expectation_cache = deepcopy(initial_expectation_cache or {})
    previous_pack = deepcopy(initial_pack)
    known_gaps, decisions = deepcopy(list(initial_known_gaps)), deepcopy(list(initial_decisions))
    construction_failure = None
    record("loop_policy", {"default_rounds": 1, "round_limit": rounds, "start_round": start_round,
                           "construction_rounds": construction_rounds,
                           "adversarial_review_enabled": adversarial_review_enabled,
                           "continuation": continuation_policy})
    for number in range(start_round, start_round + rounds):
        prefix = f"round_{number}"

        def save(name, data, prefix=prefix):
            record(prefix + "/" + name, data)

        context, evidence = build_context(task, pool, skills=skills, checks=prior_checks)
        if task_review is not None:
            from reward_engine.core.outcome.task_review import attach

            knowledge = await task_review(context)
            save("task_review", knowledge)
            context, evidence = attach(context, evidence, knowledge)
        context["memories"] = await memory.retrieve(context, top_k=top_k)
        if decisions:
            context["next_round_focus"] = deepcopy(decisions[-1])
        save("context", context)
        if hardened:
            from reward_engine.core.outcome.validated_construction import construct
            try:
                pack, method = await construct(context, evidence, registry,
                    lambda role, prefix=prefix: roles(role, prefix), record=save, reduce_max_bytes=jury_reduce_max_bytes, adversarial_review_enabled=adversarial_review_enabled)
                definition = {"criteria": deepcopy(pack['criteria']), "gaps": []}
            except Exception as error:
                construction_failure = {"stage": "counterexample_validation", "round": number,
                    "error_type": type(error).__name__, "error": str(error)}
                save("construction_failure", construction_failure)
                decisions.append({"decision": "stop", "reason": "construction_unavailable", "round": number})
                break
        else:
            try:
                definition = await jury.propose(
                    context,
                    evidence,
                    {str(i): roles("jury_member_" + str(i), prefix) for i in range(1, 4)},
                    roles("jury_reducer", prefix),
                    policy=policy,
                    expectation_cache=expectation_cache,
                    revision_protocol=True,
                    reduce_max_bytes=jury_reduce_max_bytes,
                    record=save,
                    **({"construction_rounds": construction_rounds} if construction_rounds > 1 else {}),
                )
            except Exception as error:
                # No new Pack/Actor work after an unresolved definition. Keep previous
                # completed rounds and exact public inputs for explicit recovery.
                construction_failure = {"stage": "jury", "round": number,
                    "error_type": type(error).__name__, "error": str(error)}
                save("construction_failure", construction_failure)
                save("recovery_state", {"pool": deepcopy(pool), "checks": deepcopy(prior_checks),
                    "previous_pack": deepcopy(previous_pack), "memory": deepcopy(memory.rows),
                    "expectation_cache": deepcopy(expectation_cache), "known_gaps": deepcopy(known_gaps),
                    "completed_rounds": deepcopy(results)})
                decisions.append({"decision": "stop", "reason": "jury_unavailable", "round": number})
                save("loop_decision", decisions[-1])
                break
            memory.observe_jury(task, definition, source=prefix + "/jury.json")
            known_gaps.extend(deepcopy(definition.get("gaps", [])))
            context["gaps"] = deepcopy(definition.get("gaps", []))
            context["memories"] = await memory.retrieve(context, top_k=top_k)
            save("learning_context", context)
            # Shared Outcome context + same Jury result. Independent work runs concurrently.
            built = await asyncio.gather(
                builder.build(
                    context, evidence, definition, registry, roles("harness_designer", prefix),
                    previous=previous_pack, record=save,
                    **({"construction_rounds": construction_rounds, "policy": policy,
                        "reviewers": {str(i): roles("harness_reviewer_" + str(i), prefix)
                                      for i in range(1, 4)}} if construction_rounds > 1 else {}),
                ),
                hypotheses.propose(
                    context,
                    evidence,
                    definition,
                    {str(i): roles("hypothesis_member_" + str(i), prefix) for i in range(1, 4)},
                    roles("hypothesis_chooser", prefix),
                    policy=policy,
                    record=save,
                    **({"construction_rounds": construction_rounds} if construction_rounds > 1 else {}),
                ),
                return_exceptions=True,
            )
            failed_builds = [(stage, value) for stage, value in zip(("builder", "hypotheses"), built, strict=True)
                             if isinstance(value, BaseException)]
            if failed_builds:
                stage, build_error = failed_builds[0]
                construction_failure = {"stage": stage, "round": number,
                    "error_type": type(build_error).__name__, "error": str(build_error),
                    "failed_stages": [s for s, _ in failed_builds]}
                save("construction_failure", construction_failure)
                save("recovery_state", {"pool": deepcopy(pool), "checks": deepcopy(prior_checks),
                    "previous_pack": deepcopy(previous_pack), "memory": deepcopy(memory.rows),
                    "expectation_cache": deepcopy(expectation_cache), "known_gaps": deepcopy(known_gaps),
                    "completed_rounds": deepcopy(results), "construction_rounds": construction_rounds})
                decisions.append({"decision": "stop", "reason": stage + "_unavailable", "round": number})
                save("loop_decision", decisions[-1])
                break
            pack, method = built
        previous_pack = deepcopy(pack)

        if (not hardened and arm_protocol == FEEDBACK and method.get("availability") == "unavailable"
                and any(arm_spec(arm_protocol, arm)["hypothesis"] for arm in enabled_arms)):
            construction_failure = {"stage": "hypotheses", "round": number,
                "error_type": "HypothesesUnavailable", "error": method.get("reason", "Unavailable hypotheses")}
            save("construction_failure", construction_failure)
            save("recovery_state", {"pool": deepcopy(pool), "checks": deepcopy(prior_checks),
                "previous_pack": deepcopy(previous_pack), "memory": deepcopy(memory.rows),
                "expectation_cache": deepcopy(expectation_cache), "known_gaps": deepcopy(known_gaps),
                "completed_rounds": deepcopy(results)})
            decisions.append({"decision": "stop", "reason": "hypotheses_unavailable", "round": number})
            save("loop_decision", decisions[-1])
            break  # A transport/recovery failure is not a semantic keep treatment.

        # Validate on each real product with per-product requirements; no ranking reproduction.
        async def inspect(candidate, pack=pack, current_task=task):
            c, e = build_context(current_task, [candidate])
            return await measure(pack, registry, c, e, policy=policy,
                                 test_ids=None if full_audit else [])

        save("construction_smoke", await smoke(pack, registry))
        if full_audit:
            seed_checks = await asyncio.gather(*(inspect(c) for c in pool))
            save("construction_checks", seed_checks)
            prior_checks = [check_bundle(f"product_{i}", measured, f"{prefix}/construction_checks.json#{i}")
                            for i, measured in enumerate(seed_checks, 1)]
        prompts = arm_prompts(pack, method, arm_protocol, adversarial_review_enabled=adversarial_review_enabled)
        save("arm_protocol", {"id": arm_protocol, "arms": {
            a: arm_spec(arm_protocol, a) for a in enabled_arms},
            "hypothesis_applied": has_hypothesis(method)})

        async def one(arm, number=number, prompts=prompts, pack=pack, save=save, method=method):
            spec = arm_spec(arm_protocol, arm)
            if arm_protocol == FEEDBACK and arm == 'B' and not has_hypothesis(method):
                row = {"status": "skipped", "reason": "No actionable Hypothesis; B would duplicate A"}
                save(f"{arm}/skipped", row)
                return row
            try:
                row = await sample(arm, number, prompts[arm], pack if (
                    spec["delivery_review"] or spec["policy_critic"]) else None)
                row['hypothesis_applied'] = spec['hypothesis'] and has_hypothesis(method)
                save(f"{arm}/delivery", row)
                if not row.get("candidate") or (row.get("status") == "failed" and not row.get("review_session")):
                    raise ValueError("Sampler returned no observed product")
                return row
            except Exception as error:
                row = {"status": "failed", "error_type": type(error).__name__, "reason": str(error)}
                save(f"{arm}/failure", row)
                return row

        arms = dict(zip(enabled_arms, await asyncio.gather(*(one(a) for a in enabled_arms)), strict=True))
        valid = {a: row for a, row in arms.items() if row.get("candidate")}
        arm_products, measurements = {}, {}
        for arm, row in valid.items():
            current_task = row.get('effective_task') or task
            current_pack = row.get('effective_pack') or pack
            if current_task != task and current_pack is pack:
                from reward_engine.core.outcome.task_sequence import TaskSequence
                current_pack = TaskSequence(current_task).rebase(pack)
            measured = row.get("final_measurement")
            if measured is None or measured["pack_id"] != current_pack["pack_id"]:
                measured = await inspect(row["candidate"], pack=current_pack, current_task=current_task)
            save(f"{arm}/outcome_estimate", measured)
            measurements[arm] = measured
            pool.append(deepcopy(row["candidate"]))
            arm_products[arm] = f"product_{len(pool)}"
            prior_checks.append(
                check_bundle(f"product_{len(pool)}", measured, f"{prefix}/{arm}/outcome_estimate.json")
            )
            # Only actual delivery-review revisions supply before/after evidence.
            for entry in row.get("memory", []):
                if entry["memory_id"] not in {r["memory_id"] for r in memory.rows}:
                    memory.rows.append(deepcopy(entry))
        if hardened and delivery_selector is None:
            from reward_engine.core.outcome.version_selection import select_versions
            version_selection = await select_versions(task, pack, arms, registry,
                lambda role, prefix=prefix: roles(role, prefix + '/version_selection'), record=save,
                reduce_max_bytes=jury_reduce_max_bytes)
            for arm, row in arms.items():
                save(f'{arm}/delivery', row)
        if defer_final_selection and number == start_round + rounds - 1:
            save('check_sources', prior_checks)
            ranking = {'status': 'deferred', 'reason': 'independent_final_pool_selection'}
            selection = {'status': 'pending', 'arm': None, 'product_id': None, 'source': None, 'round': number,
                         'reason': 'Final pool is assessed separately after completed rounds'}
            save('ranking', ranking)
            save('selected_delivery', selection)
        elif hardened and delivery_selector is None:
            ranking, selection = version_selection['ranking'], version_selection['selection']
        else:
            terminal_context, terminal_evidence = build_context(task, pool, skills=skills, checks=prior_checks)
            save("check_sources", prior_checks)
            try:
                if delivery_selector is not None:
                    selected = await delivery_selector(task, {a: row["candidate"] for a, row in valid.items()}, pack, prefix)
                    ranking = {"status": "compared" if len(selected["comparisons"]) >= 2 else "unavailable",
                               "policy": selected["policy"], "audit": f"{prefix}/selection/decision.json"}
                    save("ranking", ranking)
                else:
                    selected = None
                    ranking = await jury.rank(
                        terminal_context,
                        terminal_evidence,
                        {str(i): roles("jury_member_" + str(i), prefix + "/terminal") for i in range(1, 4)},
                        roles("jury_reducer", prefix + "/terminal"),
                        policy=policy,
                        reduce_max_bytes=jury_reduce_max_bytes,
                        record=save,
                        loop_review={"known_gaps": known_gaps, "current_criteria": definition["criteria"],
                                     "prior_decisions": decisions, "current_product_ids": list(arm_products.values())}
                                    if continuation_policy == "jury_new_gap" and number < start_round + rounds - 1
                                    and len(valid) == len(enabled_arms) else None,
                    )
            except Exception as error:
                ranking = {"status": "unavailable", "error_type": type(error).__name__}
                save("ranking", ranking)
                if delivery_selector is not None:
                    # Retain Actor results and selection checkpoints for explicit resume.
                    # A mismatched checkpoint must never silently become a fresh selection.
                    raise
            if delivery_selector is None:
                selection = select_delivery(ranking, arm_products, measurements)
            else:
                selection = {k: selected.get(k) for k in ("arm", "reason", "status", "policy", "identity", "tie_break", "boundary")}
                selection["evidence_source"] = f"{prefix}/selection/decision.json"
            selection.update(
                round=number,
                product_id=arm_products.get(selection["arm"]),
                source=f"{prefix}/{selection['arm']}/delivery.json" if selection["arm"] else None,
            )
            save("selected_delivery", selection)
        decision = deepcopy(ranking.get("continuation", {"decision": "stop", "reason": "No supported important new gap"}))
        if len(valid) != len(enabled_arms):
            decision = {"decision": "stop", "reason": "incomplete_arms"}
        elif number >= start_round + rounds - 1:
            decision = {"decision": "stop", "reason": "round_limit"}
        elif continuation_policy == "fixed_rounds":
            decision = {"decision": "continue", "reason": "configured_fixed_rounds"}
        elif ranking.get("status") != "compared":
            decision = {"decision": "stop", "reason": "ranking_unavailable"}
        elif decision["decision"] == "continue" and selection["product_id"] not in decision["product_ids"]:
            decision = {"decision": "stop", "reason": "New gap does not affect selected deliverable"}
        decision["round"] = number
        decisions.append(decision)
        save("loop_decision", decision)
        if decision["decision"] == "continue" and continuation_policy == "jury_new_gap":
            known_gaps.append(decision)
        # Independent private evaluations have no data dependency. Keep their
        # existing per-call deadlines; a failed grader is not an Actor failure.
        grade_slots = asyncio.Semaphore(policy.concurrency if policy else 3)

        async def grade_one(arm, row, grade_slots=grade_slots):
            try:
                async with grade_slots:
                    result = await evaluate(task, row) if evaluate else {"status": "not_run"}
            except Exception as error:
                result = {"status": "failed", "error_type": type(error).__name__}
            return arm, result

        official = dict(await asyncio.gather(*(grade_one(a, r) for a, r in valid.items())))
        summary = {
            "round": number,
            "arm_protocol": arm_protocol,
            "arms": arms,
            "feedback_execution": {a: deepcopy(row.get("feedback_execution", {
                "status": "unknown", "reason": "execution_receipt_missing"})) for a, row in arms.items()},
            "official": official,
            "ranking": ranking,
            "selection": selection,
            "loop_decision": decision,
            "status": "completed" if len(valid) == len(enabled_arms) and all(
                not hardened or row.get('status') == 'completed' and row.get('feedback_execution', {}).get(
                    'status') in {'completed', 'disabled'} for row in valid.values()) else "partial",
            "pool_size": len(pool),
        }
        save("summary.private", summary)
        save("memory", memory.rows)
        save("checkpoint", deepcopy({"task": task, "completed_round": number, "pool": pool,
             "memory": memory.rows, "checks": prior_checks, "previous_pack": previous_pack,
             "expectation_cache": expectation_cache, "known_gaps": known_gaps, "decisions": decisions}))
        results.append(summary)
        if decision["decision"] != "continue":
            break  # Default one round. Grader scores cannot authorize another round.
    final = {
        "arm_protocol": arm_protocol,
        "status": "completed"
        if not construction_failure and results and results[-1]["status"] == "completed"
        else "partial" if results else "failed",
        "completed_rounds": sum(r["status"] == "completed" for r in results),
        "rounds": results,
        "feedback_execution": deepcopy(results[-1]["feedback_execution"]) if results else {},
        "selected_delivery": results[-1]["selection"] if results else None,
        "official_evaluation": "separate_from_outcome_estimator",
        "memory": memory.rows,
        "round_policy": "one_round_unless_jury_new_gap" if continuation_policy == "jury_new_gap" else "fixed_rounds",
        "round_limit": rounds,
        "construction_rounds": construction_rounds,
        "stop_reason": decisions[-1]["reason"],
    }
    if construction_failure:
        final["construction_failure"] = construction_failure
    record("result.private", final)
    return final


async def run_tasks(tasks, run_one, *, concurrency=2):
    """Failures isolated per Task; no score selection or mixing task pools."""
    if type(concurrency) is not int or concurrency < 1:
        raise ValueError("Positive task concurrency required")
    semaphore = asyncio.Semaphore(concurrency)

    async def one(task):
        async with semaphore:
            try:
                return {"task_id": task["task_id"], "result": await run_one(task)}
            except Exception as error:
                return {"task_id": task["task_id"], "error_type": type(error).__name__, "status": "failed"}

    return await asyncio.gather(*(one(t) for t in tasks))
