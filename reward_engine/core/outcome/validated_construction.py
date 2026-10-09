"""One propose/attack/repair/recheck pipeline, with immutable public evidence."""

from copy import deepcopy

from reward_engine.core.batch import BatchPolicy
from reward_engine.core.outcome import builder, hypotheses, jury
from reward_engine.core.outcome.construction import fingerprint
from reward_engine.core.outcome.context import ask, compact_learning_context
from reward_engine.core.outcome.counterexamples import (
    VERSION,
    attack,
    bind_case_dialogue,
    dialogue_contract,
    verify_issue,
)
from reward_engine.core.outcome.review_policy import (
    basic_review,
    require_adversarial,
    validate_review_flag,
)


def current_definition(definition):
    """Keep original draft lineage in recorded inputs, not live obligations after repair."""
    return {
        "criteria": deepcopy(definition["criteria"]),
        "gaps": [],
        "uncertainties": deepcopy(definition.get("uncertainties", [])),
        "independent_expectations": [],
        "proposal_expectations": [],
        "coverage": [],
    }


async def repair_definition(
    definition,
    findings,
    context,
    evidence,
    call,
    *,
    pack=None,
    adversarial_review_enabled=False,
):
    require_adversarial(adversarial_review_enabled)
    from reward_engine.core.outcome.counterexamples import targets_for

    targets = targets_for(pack, "reward") if pack else {}
    affected = {
        targets.get(r["target_id"], {}).get("criterion_id", r["target_id"])
        for r in findings
        if r["status"] in {"disputed", "refuted"}
    }
    candidates = {c["id"]: c for c in definition["criteria"] if c["id"] in affected}
    if not candidates:
        return definition

    def validate(row):
        if (
            row.get("decision") not in {"keep", "revise"}
            or not isinstance(row.get("reason"), str)
            or not row["reason"].strip()
        ):
            raise ValueError("Decide keep or propose revise with a concrete reason")
        updates = row.get("updates")
        if not isinstance(updates, list) or len({c.get("id") for c in updates}) != len(
            updates
        ):
            raise ValueError("Repair updates must be unique criteria")
        if row["decision"] == "keep" and updates:
            raise ValueError("A keep cannot contain updates")
        if not {c.get("id") for c in updates} <= set(candidates):
            raise ValueError("Repair only disputed criteria; retain their stable IDs")
        if updates:
            row["updates"] = jury.validate(
                {"criteria": updates, "gaps": []},
                evidence,
                context,
                clause_contract=any("clauses" in c for c in candidates.values()),
            )["criteria"]
            for update in row["updates"]:
                related = [
                    targets.get(r["target_id"], {})
                    for r in findings
                    if targets.get(r["target_id"], {}).get(
                        "criterion_id", r["target_id"]
                    )
                    == update["id"]
                ]
                if related and all(t.get("target_kind") == "clause" for t in related):
                    allowed = {t["clause_id"] for t in related}
                    original = {c["id"]: c for c in candidates[update["id"]]["clauses"]}
                    changed = {c["id"]: c for c in update["clauses"]}
                    if set(original) != set(changed) or any(
                        changed[k] != v for k, v in original.items() if k not in allowed
                    ):
                        raise ValueError(
                            "A clause-scoped repair preserves all other clauses and stable IDs"
                        )
        return row

    schema, instruction = jury.clause_schema(
        any("clauses" in c for c in candidates.values())
    )
    row = await ask(
        call,
        "outcome_attack_repair_definition",
        compact_learning_context(context),
        evidence,
        "Decide whether to KEEP this Pack or PROPOSE a bounded candidate repair against the ORIGINAL Task. "
        "A reference/checker disagreement is DISPUTED, not proof that either side is right. Inspect "
        "the material, applicability, source derivation and both rationales. Reject mistaken attacks "
        "and mistaken reference verdicts. You may keep even when a reference disagrees. "
        "Keep stable IDs. Remove false prescriptions and replace them with scoped, correct outcome "
        "requirements. Do not remove actual Task obligations or require one example/method. "
        "A failed checker is not proof every part of a criterion is false. Preserve unrelated valid "
        "requirements and inspect the independent reference derivation, not votes. Updates may be "
        "empty when only a checker needs repair. All changes are candidates, not certified corrections. "
        + instruction,
        {
            "decision": "keep/revise",
            "reason": "specific Task outcome rationale",
            "updates": schema["criteria"],
        },
        extra={
            "current_criteria": list(candidates.values()),
            "disputed_findings": findings,
        },
        validate=validate,
        max_contract_retries=1,
        max_evidence_rounds=2,
        close_evidence=True,
    )
    updated = {c["id"]: c for c in row["updates"]}
    return {
        **current_definition(definition),
        "criteria": [deepcopy(updated.get(c["id"], c)) for c in definition["criteria"]],
        "repair_decision": {"decision": row["decision"], "reason": row["reason"]},
    }


async def recheck(
    old,
    artifact,
    module,
    context,
    evidence,
    registry,
    references,
    members,
    *,
    record,
    only_targets=None,
    checkpoint_store=None,
    adversarial_review_enabled=False,
):
    """Preserve the original failing cases and add one genuinely new boundary."""
    require_adversarial(adversarial_review_enabled)
    from reward_engine.core.outcome.counterexamples import public_reference_context

    context, evidence = public_reference_context(context, evidence)
    findings = []
    for index, finding in enumerate(old["records"]):
        if only_targets is not None and finding["target_id"] not in only_targets:
            continue
        if "issue" not in finding:
            continue
        issue = deepcopy(finding["issue"])
        target = issue["target_id"]
        from reward_engine.core.outcome.counterexamples import targets_for

        if module == "reward" and target not in targets_for(artifact, module):
            # Check IDs may change during compilation; remap by the recorded rubric.
            old_tests = [r for c in finding["cases"] for r in c["current_pack_results"]]
            criteria = {r["criterion_id"] for r in old_tests}
            if len(criteria) != 1:
                raise ValueError(
                    "Cannot recheck a retired checker without its criterion identity"
                )
            issue["target_id"] = next(iter(criteria))
        ref_seat = str(int(issue["proposer_seat"]) % 3 + 1)
        binding = fingerprint(
            {
                "version": VERSION,
                "artifact": artifact,
                "module": module,
                "original_finding": finding,
                "context": context,
                "evidence": evidence,
                "tools": registry.tools.catalog(),
            }
        )

        async def stage(name, execute, *args, binding=binding, index=index, **kwargs):
            cached = (
                checkpoint_store.verification_checkpoint(binding, name)
                if checkpoint_store
                else None
            )
            if cached is not None:
                record(
                    f"recheck_{index}/{name}_recovery",
                    {
                        "binding": binding,
                        "payload_sha256": fingerprint(cached),
                        "reused": True,
                    },
                )
                return deepcopy(cached)
            value = await execute(*args, **kwargs)
            if checkpoint_store is not None:
                checkpoint_store.save_verification_checkpoint(binding, name, value)
            record(
                f"recheck_{index}/{name}_checkpoint",
                {
                    "binding": binding,
                    "payload_sha256": fingerprint(value),
                    "reused": False,
                },
            )
            return deepcopy(value)

        result = await stage(
            "original",
            verify_issue,
            issue,
            artifact,
            module,
            context,
            evidence,
            registry,
            references[ref_seat],
            record=lambda n, d, i=index: record(f"original_{i}/{n}", d),
            adversarial_review_enabled=adversarial_review_enabled,
        )
        result["origin_target_id"] = finding["target_id"]
        result["origin_validation_id"] = finding["validation_id"]

        # A fresh case is generated without exposing the repair rationale or old cases.
        def validate_boundary(row, issue=issue):
            dialogues = {
                p["product_id"] for p in context["products"] if p["user_updates"]
            }
            case = bind_case_dialogue(row.get("case"), dialogues)
            row = {**row, "case": case}
            if (
                not all(
                    isinstance(case.get(k), str) and case[k].strip()
                    for k in ("id", "artifact", "question", "why")
                )
                or case.get("expected") not in {"pass", "fail"}
                or not isinstance(case.get("probes", []), list)
                or len(case.get("probes", [])) > 3
            ):
                raise ValueError(
                    "Fresh boundary needs self-contained material and a falsifiable prediction"
                )
            if any(case["artifact"] == c["artifact"] for c in issue["cases"]):
                raise ValueError("Fresh boundary cannot replay a previous case")
            case["kind"] = "boundary"
            return row

        case_schema, dialogue_instruction = dialogue_contract(context)
        fresh = await stage(
            "fresh_case",
            ask,
            members[ref_seat],
            "outcome_attack_fresh_boundary",
            compact_learning_context(context),
            evidence,
            "Produce one new boundary case to challenge this current prescription or checker. "
            + dialogue_instruction
            + "Vary a material applicability condition. Return a self-contained scoped claim, numerical "
            "probes when helpful, and a prediction to be independently checked; no invented executions.",
            {"case": case_schema},
            extra={"target": targets_for(artifact, module)[issue["target_id"]]},
            validate=validate_boundary,
            max_contract_retries=1,
            max_evidence_rounds=1,
        )
        fresh_issue = {
            **issue,
            "cases": [c for c in issue["cases"] if c["kind"] != "boundary"]
            + [fresh["case"]],
            "proposer_seat": ref_seat,
        }
        other = str(int(ref_seat) % 3 + 1)
        boundary = await stage(
            "boundary",
            verify_issue,
            fresh_issue,
            artifact,
            module,
            context,
            evidence,
            registry,
            references[other],
            record=lambda n, d, i=index: record(f"boundary_{i}/{n}", d),
            adversarial_review_enabled=adversarial_review_enabled,
        )
        result["fresh_boundary"] = boundary
        if boundary["status"] in {"refuted", "disputed"}:
            result["status"] = "disputed"
        elif boundary["status"] != "supported_on_cases" and result["status"] not in {
            "refuted",
            "disputed",
        }:
            result["status"] = "inconclusive"
        findings.append(result)
    from reward_engine.core.outcome.counterexamples import validation_record

    tested = {r["target_id"] for r in findings}
    # Unmeasured bindings in the new revision never inherit whole-Pack authority.
    findings.extend(
        validation_record(
            key,
            "unverified",
            context=context,
            evidence=evidence,
            artifact=artifact,
            reason="Not rechecked on this candidate revision",
        )
        for key in targets_for(artifact, module)
        if key not in tested
    )
    return {
        **old,
        "records": findings,
        "artifact_sha256": fingerprint(artifact),
        "rechecked": True,
    }


def select_repair(before, after):
    """Bounded empirical preference, never a certification of the reference's truth."""
    previous = {r["target_id"]: r for r in before["records"] if r.get("cases")}
    improvements, regressions, unresolved = [], [], []
    expected_targets = {
        k for k, r in previous.items() if r["status"] in {"refuted", "disputed"}
    }
    checked_targets = {
        r.get("origin_target_id", r["target_id"])
        for r in after["records"]
        if r.get("cases")
    }
    unresolved.extend(sorted(expected_targets - checked_targets))
    for finding in after["records"]:
        if not finding.get("cases"):
            continue
        key = finding.get("origin_target_id", finding["target_id"])
        original = previous.get(key)
        if not original or not finding.get("contrast_qualified"):
            unresolved.append(key)
            continue
        old_cases = {c["case_id"]: c for c in original["cases"]}
        if set(old_cases) != {c["case_id"] for c in finding["cases"]}:
            unresolved.append(key)
            continue
        for case in finding["cases"]:
            old = old_cases[case["case_id"]]
            expected = case["reference"]["status"]
            if (
                expected not in {"pass", "fail"}
                or expected != old["reference"]["status"]
            ):
                unresolved.append(case["case_id"])
                continue
            verdict, prior = case["current_verdict"], old["current_verdict"]
            if verdict not in {"pass", "fail"}:
                unresolved.append(case["case_id"])
            elif prior == expected and verdict != expected:
                regressions.append(case["case_id"])
            elif (
                prior in {"pass", "fail"} and prior != expected and verdict == expected
            ):
                improvements.append(case["case_id"])
        if finding.get("fresh_boundary", {}).get("status") != "supported_on_cases":
            unresolved.append(key + ":fresh_boundary")
    return {
        "accepted": bool(improvements) and not regressions and not unresolved,
        "improved_cases": improvements,
        "regressed_cases": regressions,
        "unresolved": unresolved,
        "boundary": "Preference on fixed cases and a fresh boundary; references remain fallible",
    }


def publish_validation(pack, audit):
    pack = deepcopy(pack)
    if audit.get("artifact_sha256") != fingerprint(pack):
        raise ValueError("Validation belongs to a different Pack revision")
    records = audit["records"]

    def state_of(rows):
        checked = [r for r in rows if r["status"] != "unverified"]
        return (
            "disputed"
            if any(r["status"] in {"refuted", "disputed"} for r in checked)
            else "supported_on_cases"
            if checked and all(r["status"] == "supported_on_cases" for r in checked)
            else "inconclusive"
            if checked
            else "unverified"
        )

    for criterion in pack["criteria"]:
        related = [r for r in records if r["target_id"] == criterion["id"]]
        state = state_of(related)
        criterion["validation_status"] = state
        criterion["revision_authority"] = "trial_only"
        criterion["validation_ids"] = [
            r["validation_id"] for r in related if r["status"] != "unverified"
        ]
    current = {c["id"]: c for c in pack["criteria"]}
    for test in pack["tests"]:
        # A criterion-level case executes all its tests; a test-level case covers only itself.
        # Unselected aliases must not erase that support or certify sibling tests.
        related = [
            r
            for r in records
            if r["target_id"] in {test["test_id"], test["criterion_id"]}
        ]
        test["validation_status"] = state_of(related)
        test["rubric"] = deepcopy(current[test["criterion_id"]])
        rubric = current[test["criterion_id"]]
        constraints = [
            c for c in rubric.get("clauses", []) if c["use"] == "task_constraint"
        ]
        test["critical_preservation"] = (
            test["validation_status"] == "supported_on_cases"
            and bool(constraints)
            and len(constraints) == len(rubric["clauses"])
            and test.get(
                "measurement_kind",
                "deterministic" if test["checker"] == "json.parse" else "unknown",
            )
            == "deterministic"
        )
    pack["validation"] = {
        "version": VERSION,
        "status": audit["status"],
        "records": records,
        "task_sha256": fingerprint(
            {"task_id": pack["task_id"], "prompt": pack["task_prompt"]}
        ),
        "validated_improvement": False,
        "decision_policy": "bounded_candidate",
        **(
            {"adversarial_review": deepcopy(audit["adversarial_review"])}
            if "adversarial_review" in audit
            else {}
        ),
    }
    pack["feedback_policy"] = "evidence_admitted"
    from reward_engine.core.outcome.admission import (
        publication_digest,
        validation_payload,
    )

    pack["validation"]["payload_sha256"] = fingerprint(validation_payload(pack))
    pack["validation"]["publication_sha256"] = publication_digest(pack)
    return pack


async def construct_reward(
    context,
    evidence,
    registry,
    roles,
    *,
    record=lambda *_: None,
    reduce_max_bytes=393216,
    checkpoint=None,
    adversarial_review_enabled=False,
):
    """Build a Reward Pack; counterexample review requires operator opt-in."""
    validate_review_flag(adversarial_review_enabled)
    members = {str(i): roles(f"jury_member_{i}") for i in range(1, 4)}
    previous = None
    if checkpoint is not None:
        if checkpoint.get("context_sha256") != fingerprint(context) or checkpoint.get(
            "evidence_sha256"
        ) != fingerprint(evidence):
            raise ValueError(
                "Construction checkpoint belongs to different public inputs"
            )
        if checkpoint.get("jury_reduce"):
            saved = checkpoint["jury_reduce"]
            if fingerprint(saved["context"]) != fingerprint(context) or fingerprint(
                saved["evidence"]
            ) != fingerprint(evidence):
                raise ValueError("Jury checkpoint public input mismatch")
            definition = await jury.reduce_proposals(
                context,
                evidence,
                deepcopy(saved["inputs"]),
                roles("jury_reducer"),
                reduce_max_bytes=reduce_max_bytes,
                strict=True,
                clause_contract=True,
                record=lambda n, d: record("proposal/" + n, d),
            )
        else:
            definition = jury.validate(
                deepcopy(checkpoint["definition"]),
                evidence,
                context,
                clause_contract=True,
            )
            previous = deepcopy(checkpoint["pack"])
        if previous and (
            previous["task_id"] != context["task"]["task_id"]
            or previous["task_prompt"] != context["task"]["prompt"]
        ):
            raise ValueError("Checkpoint Pack Task mismatch")
        record(
            "resumed_draft",
            {
                "context_sha256": fingerprint(context),
                "evidence_sha256": fingerprint(evidence),
                "pack_sha256": fingerprint(previous),
                "new_jury_calls": 0,
            },
        )
    else:
        definition = await jury._propose_once(
            context,
            evidence,
            members,
            roles("jury_reducer"),
            policy=BatchPolicy(deadline_s=900),
            strict=True,
            task_only=True,
            reduce_max_bytes=reduce_max_bytes,
            record=lambda n, d: record("proposal/" + n, d),
            clause_contract=True,
            revision_protocol=True,
        )
    record("draft_definition", definition)
    definition = current_definition(definition)
    pack = await builder.build(
        context,
        evidence,
        definition,
        registry,
        roles("harness_designer"),
        record=lambda n, d: record("builder/" + n, d),
        **({"previous": previous} if previous is not None else {}),
    )
    record("draft_pack", pack)
    if not adversarial_review_enabled:
        audit = basic_review(pack, context, evidence, registry)
        audit["adversarial_review"] = {
            "status": "skipped",
            "reason": "adversarial_review_disabled",
        }
        record("basic_review", audit)
        if audit["basic_integrity"]["errors"]:
            raise ValueError(
                "Reward integrity errors: " + str(audit["basic_integrity"]["errors"])
            )
        pack = publish_validation(pack, audit)
        record("pack", pack)
        return pack
    attackers = {str(i): roles(f"attack_member_{i}") for i in range(1, 4)}
    references = {str(i): roles(f"attack_reference_{i}") for i in range(1, 4)}
    audit = await attack(
        pack,
        "reward",
        context,
        evidence,
        registry,
        attackers,
        references,
        record=lambda n, d: record("reward_attack/" + n, d),
        adversarial_review_enabled=adversarial_review_enabled,
    )
    disputed = [r for r in audit["records"] if r["status"] in {"disputed", "refuted"}]
    if disputed:
        before = deepcopy(pack)
        original_audit = deepcopy(audit)
        try:
            definition = await repair_definition(
                definition,
                disputed,
                context,
                evidence,
                roles("attack_reducer"),
                pack=pack,
                adversarial_review_enabled=adversarial_review_enabled,
            )
        except Exception as error:
            record(
                "repair_selection",
                {
                    "accepted": False,
                    "reason": "Repair proposal unavailable; retain original",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            pack = publish_validation(before, original_audit)
            record("pack", pack)
            return pack
        record(
            "repair_proposal", definition.get("repair_decision", {"decision": "keep"})
        )
        if definition.get("repair_decision", {}).get("decision") != "revise":
            pack = publish_validation(pack, audit)
            record("pack", pack)
            return pack
        # Rebuild only affected bindings, including a checker defect without changed criterion text.
        from reward_engine.core.outcome.counterexamples import targets_for

        targets = targets_for(pack, "reward")
        affected = {
            targets[r["target_id"]].get("criterion_id", r["target_id"])
            for r in disputed
        }
        affected |= {
            t["criterion_id"] for t in pack["tests"] if t["test_id"] in affected
        }
        previous = deepcopy(pack)
        previous["criteria"] = [
            c for c in previous["criteria"] if c["id"] not in affected
        ]
        try:
            pack = await builder.build(
                context,
                evidence,
                definition,
                registry,
                roles("harness_designer"),
                previous=previous,
                repair_findings=disputed,
                record=lambda n, d: record("builder_repair/" + n, d),
            )
            record("candidate_pack", pack)
            record(
                "repair_lineage",
                {
                    "before_sha256": fingerprint(before),
                    "after_sha256": fingerprint(pack),
                    "affected": sorted(affected),
                    "original_task_preserved": True,
                },
            )
            audit = await recheck(
                audit,
                pack,
                "reward",
                context,
                evidence,
                registry,
                references,
                attackers,
                record=lambda n, d: record("reward_recheck/" + n, d),
                only_targets={r["target_id"] for r in disputed},
                adversarial_review_enabled=adversarial_review_enabled,
            )
            decision = select_repair(original_audit, audit)
        except Exception as error:
            decision = {
                "accepted": False,
                "reason": "Candidate construction/recheck unavailable; retain original",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        record("repair_selection", decision)
        if not decision["accepted"]:
            pack, audit = before, original_audit
    pack = publish_validation(pack, audit)
    record("pack", pack)
    return pack


async def construct(
    context,
    evidence,
    registry,
    roles,
    *,
    record=lambda *_: None,
    reduce_max_bytes=393216,
    checkpoint=None,
    adversarial_review_enabled=False,
):
    validate_review_flag(adversarial_review_enabled)
    resumed = checkpoint is not None and "completed_reward" in checkpoint
    if resumed:
        from reward_engine.core.outcome.admission import require_publication

        if checkpoint["context_sha256"] != fingerprint(context) or checkpoint[
            "evidence_sha256"
        ] != fingerprint(evidence):
            raise ValueError("Completed Reward checkpoint public inputs mismatch")
        pack = deepcopy(checkpoint["completed_reward"])
        require_publication(pack)
        record("pack", pack)
    else:
        pack = await construct_reward(
            context,
            evidence,
            registry,
            roles,
            record=record,
            reduce_max_bytes=reduce_max_bytes,
            **({"checkpoint": checkpoint} if checkpoint is not None else {}),
            adversarial_review_enabled=adversarial_review_enabled,
        )
    try:
        method = await construct_hypothesis(
            pack,
            context,
            evidence,
            registry,
            roles,
            record=record,
            adversarial_review_enabled=adversarial_review_enabled,
        )
    except Exception as error:
        # A/C/D depend on the completed Reward, not B's optional prescription.
        # Preserve failure provenance; never turn failed validation into advice.
        method = {
            "status": "keep",
            "availability": "unavailable",
            "method": "",
            "reason": "Hypothesis construction or validation unavailable; B must be skipped",
            "error_type": type(error).__name__,
            "error": str(error),
        }
        record("hypothesis_unavailable", method)
    record("hypothesis", method)
    record(
        "construction_manifest",
        {
            "status": "partial"
            if method.get("availability") == "unavailable"
            else "completed",
            "adversarial_review_enabled": adversarial_review_enabled,
            "reward_available": True,
            "hypothesis_available": method.get("availability") != "unavailable",
            "version": VERSION,
            "context_sha256": fingerprint(context),
            "evidence_sha256": fingerprint(evidence),
            "pack_sha256": fingerprint(pack),
            "hypothesis_sha256": fingerprint(method),
            "boundary": (
                "Scoped case validation; actual outcome improvement has not been measured"
                if adversarial_review_enabled
                else "Basic integrity only; adversarial review skipped, Hypothesis advice is unverified"
            ),
        },
    )
    return pack, method


async def construct_hypothesis(
    pack,
    context,
    evidence,
    registry,
    roles,
    *,
    record=lambda *_: None,
    adversarial_review_enabled=False,
):
    validate_review_flag(adversarial_review_enabled)
    learning = deepcopy(context)
    learning["gaps"] = []
    learning["memories"] = []
    # Suspected/refuted rules remain audit data, never authoritative methods.
    guidance_definition = {
        "criteria": [
            deepcopy(c)
            for c in pack["criteria"]
            if c["validation_status"] not in {"refuted", "disputed"}
        ],
        "gaps": [],
        "coverage": [],
        "independent_expectations": [],
        "proposal_expectations": [],
    }
    method = await hypotheses.propose(
        learning,
        evidence,
        guidance_definition,
        {str(i): roles(f"hypothesis_member_{i}") for i in range(1, 4)},
        roles("hypothesis_chooser"),
        policy=BatchPolicy(deadline_s=480),
        record=lambda n, d: record("hypothesis_draft/" + n, d),
    )
    if method.get("availability") == "unavailable":
        return method
    if method.get("status") == "propose":
        # Fresh projection happens BEFORE attacks so the exact Actor-facing method is checked.
        method = await project_method(
            method, learning, evidence, guidance_definition, roles("hypothesis_chooser")
        )
    if method.get("status") == "propose" and not adversarial_review_enabled:
        if "validation" in method or "actor_projection" in method:
            raise ValueError(
                "A new unverified projection must not replace earlier validation records"
            )
        method["validation"] = {
            "status": "skipped",
            "reason": "adversarial_review_disabled",
            "artifact_sha256": fingerprint(method),
            "records": [],
        }
        method["actor_projection"] = {
            "method": method["method"],
            "source_sha256": fingerprint(method["method"]),
            "validated": False,
        }
        record("hypothesis_review_skipped", method["validation"])
        return method
    if method.get("status") == "propose":
        attackers = {str(i): roles(f"attack_member_{i}") for i in range(1, 4)}
        references = {str(i): roles(f"attack_reference_{i}") for i in range(1, 4)}
        method_audit = await attack(
            method,
            "hypothesis",
            learning,
            evidence,
            registry,
            attackers,
            references,
            record=lambda n, d: record("hypothesis_attack/" + n, d),
            adversarial_review_enabled=adversarial_review_enabled,
        )
        if any(r["status"] in {"refuted", "disputed"} for r in method_audit["records"]):
            method = await project_method(
                method,
                learning,
                evidence,
                guidance_definition,
                roles("hypothesis_chooser"),
                findings=method_audit["records"],
            )
            if method.get("status") == "propose":
                method_audit = await recheck(
                    method_audit,
                    method,
                    "hypothesis",
                    learning,
                    evidence,
                    registry,
                    references,
                    attackers,
                    record=lambda n, d: record("hypothesis_recheck/" + n, d),
                    adversarial_review_enabled=adversarial_review_enabled,
                )
        verified = (
            method.get("status") == "propose"
            and bool(method_audit["records"])
            and all(
                r["status"] == "supported_on_cases" for r in method_audit["records"]
            )
        )
        method["validation"] = method_audit
        if not verified:
            record("quarantined_hypothesis", method)
            unavailable = any(
                r.get("verification_error") for r in method_audit["records"]
            )
            method = {
                "status": "keep",
                "method": "",
                "reason": "No independently supported Actor prescription",
                **(
                    {
                        "availability": "unavailable",
                        "reason": "Hypothesis case verification did not complete",
                    }
                    if unavailable
                    else {}
                ),
                "validation": method_audit,
            }
        else:
            method["actor_projection"] = {
                "method": method["method"],
                "source_sha256": fingerprint(method["method"]),
                "validated": True,
            }
    return method


async def project_method(method, context, evidence, definition, call, *, findings=()):
    def validate(row):
        hypotheses.validate(row, evidence, definition)
        text = row.get("method", "")
        if any(
            token in text
            for token in (
                "product_1",
                "product_2",
                "product_3",
                "初稿曾",
                "上一轮",
                "此前的作品",
            )
        ):
            raise ValueError(
                "Fresh Actor method cannot impersonate another work or its revision history"
            )
        if row.get("status") == "propose":
            targets = row.get("validation_targets")
            if (
                not isinstance(targets, list)
                or not 1 <= len(targets) <= 3
                or len({t.get("id") for t in targets}) != len(targets)
                or any(
                    t.get("id") not in {"hypothesis:1", "hypothesis:2", "hypothesis:3"}
                    or not isinstance(t.get("quote"), str)
                    or not t["quote"].strip()
                    or t["quote"] not in text
                    for t in targets
                )
            ):
                raise ValueError(
                    "Identify up to three consequential exact method passages for validation"
                )
            if findings and {t["id"] for t in targets} != {
                t["id"] for t in method.get("validation_targets", [])
            }:
                raise ValueError(
                    "Repair preserves target IDs so original cases can be rechecked"
                )
        return row

    return await ask(
        call,
        "outcome_hypothesis_actor_projection",
        compact_learning_context(context),
        evidence,
        "Produce the complete fresh-Actor method, not a historical review. Rewrite as forward-looking "
        "Task steps with applicability conditions and falsifiable verification. No product IDs, previous "
        "answers, prior grades or claims that this Actor already made an error. Aim for the largest "
        "Task outcome gain. If verified attacks are supplied, repair only their affected prescriptions "
        "and preserve valid work. Identify up to three consequential factual/causal prescriptions "
        "as exact method quotes, with stable hypothesis:1/2/3 IDs for independent case testing. "
        "After repair keep those IDs and update their quotes to the corrected passages. "
        "Do not invent facts or claim to have executed a test. A keep is allowed.",
        {
            **hypotheses.SCHEMA,
            "validation_targets": [
                {"id": "hypothesis:1/2/3", "quote": "exact method passage"}
            ],
        },
        extra={
            "draft_method": method,
            "current_definition": definition,
            "verified_findings": list(findings),
        },
        validate=validate,
        max_contract_retries=1,
        max_evidence_rounds=2,
        close_evidence=True,
    )
