"""Bounded cross-family attacks with separate public-reference and Pack executions.

An attack is a proposal. A tool receipt proves only its execution. A scoped,
independent reference judgment plus a disagreeing real checker opens a dispute;
neither a majority vote nor a successful compilation proves outcome quality.
"""

import asyncio
from copy import deepcopy
from urllib.parse import quote

from reward_engine.core.batch import BatchPolicy, collect_batch
from reward_engine.core.outcome.clauses import claim_binding
from reward_engine.core.outcome.clauses import targets as clause_targets
from reward_engine.core.outcome.construction import fingerprint
from reward_engine.core.outcome.context import (
    ResponseContractError,
    ask,
    build_context,
    compact_learning_context,
    measurement_test_view,
    require_refs,
)
from reward_engine.core.outcome.harness import measurement_context
from reward_engine.core.outcome.review_policy import require_adversarial

VERSION = "outcome-counterexamples/2026-09-12.3"
STRATEGY = "counterexample_validated"
STATES = {"unverified", "supported_on_cases", "refuted", "inconclusive", "disputed"}

CASE_SCHEMA = {
    "id": "unique local case ID",
    "question": "precise property to assess within the Task",
    "user_context_product_id": "Exact product_id of the actual user dialogue this case tests; omit only if no user amendments exist",
    "kind": "correct_control/error_variant/boundary",
    "scope": "claim for a local property; whole_check only for a complete checker input",
    "artifact": "self-contained short answer or technical claim; NOT expected labels",
    "expected": "Task-grounded material validity: correct_control=pass, error_variant=fail, boundary=pass/fail",
    "why": "derive material validity against the Task, not whether the attacked rule accepts it",
    "probes": [{"tool": "calculate/code.run/document.read", "arguments": {}}],
}
ATTACK_SCHEMA = {
    "issues": [
        {
            "target_id": "exact criterion, test or hypothesis ID",
            "priority": "critical/important/minor",
            "problem": "specific consequential false premise, missing condition or evaluator mistake",
            "evidence_refs": ["exact supplied refs"],
            "suggested_change": "scoped proposed repair",
            "cases": [CASE_SCHEMA],
        }
    ],
    "reason": "Task-grounded assessment, including remaining uncertainty",
}


def dialogue_contract(context):
    """The engine decides applicability; models only select existing branches."""
    ids = sorted(p["product_id"] for p in context["products"] if p["user_updates"])
    schema = deepcopy(CASE_SCHEMA)
    if ids:
        schema["user_context_product_id"] = {"choose_exactly_one": ids}
        instruction = (
            "Each case must select one supplied user_context_product_id. "
            "Never combine requirements from different dialogue branches. "
        )
    else:
        schema.pop("user_context_product_id")
        instruction = (
            "There are no actual user amendments or dialogue branches. "
            "Every case is bound by the engine to the original Task only. "
            "Do not supply user_context_product_id or invent user messages. "
        )
    return schema, instruction


def bind_case_dialogue(case, dialogue_ids):
    """Drop inapplicable metadata, never fabricate or silently switch a real branch.

    Original provider responses remain in the model receipts. With no user
    amendments, this field cannot select any evidence: the engine always uses
    Task-only context. Case material, predictions and evidence still undergo all
    independent validation; this normalization grants no support or repair right.
    """
    if not isinstance(case, dict):
        raise ValueError("A counterexample case must be an object")
    result = deepcopy(case)
    if not dialogue_ids:
        result.pop("user_context_product_id", None)
    elif (
        not isinstance(result.get("user_context_product_id"), str)
        or result["user_context_product_id"] not in dialogue_ids
    ):
        raise ValueError(
            "Case must bind an exact actual user dialogue via user_context_product_id; "
            f"allowed IDs: {sorted(dialogue_ids)}. Do not combine branches."
        )
    return result


def targets_for(artifact, module):
    if module == "hypothesis":
        if artifact.get("validation_targets"):
            return {
                r["id"]: {**deepcopy(r), "standard": artifact["method"]}
                for r in artifact["validation_targets"]
            }
        return {
            "hypothesis:method": {
                "id": "hypothesis:method",
                "standard": artifact.get("method", ""),
                "target": artifact.get("target", ""),
            }
        }
    return {
        **{c["id"]: deepcopy(c) for c in artifact["criteria"]},
        **clause_targets(artifact["criteria"]),
        **{t["test_id"]: deepcopy(t) for t in artifact["tests"]},
    }


def attack_target_view(targets):
    """Review current definitions; drafting history stays in the immutable source Pack."""
    view = {}
    for key, target in targets.items():
        item = (
            measurement_test_view(target) if "test_id" in target else deepcopy(target)
        )
        if "test_id" not in target:
            item.pop("part_revisions", None)
        view[key] = item
    return view


def attack_context(context, evidence, targets):
    """A challenged definition is citable as an object, never as proof of truth.

    Hide unavailable historical provenance handles in the model projection;
    the immutable Pack retains them. References cannot cite absent products.
    """
    view, sources = deepcopy(context), deepcopy(evidence)
    projected = attack_target_view(targets)
    refs = {key: "attack_target:" + quote(key, safe="") for key in targets}

    def visible(value):
        if isinstance(value, dict):
            for key, item in list(value.items()):
                if key in {"evidence_refs", "expected_refs"} and isinstance(item, list):
                    kept = [r for r in item if isinstance(r, str) and r in evidence]
                    if len(kept) != len(item):
                        value[key + "_unavailable_count"] = len(item) - len(kept)
                    value[key] = kept
                else:
                    visible(item)
        elif isinstance(value, list):
            for item in value:
                visible(item)

    visible(projected)
    for key, item in projected.items():
        item["definition_ref"] = refs[key]
        if refs[key] in sources:
            raise ValueError("Attack target reference collides with public evidence")
        sources[refs[key]] = {
            "source_kind": "estimated_definition_under_attack",
            "target": deepcopy(item),
            "authority": "object of review, not truth evidence",
        }
    view["available_evidence_refs"] = list(sources)
    return view, sources, projected


def resolve_target_citations(row, targets, evidence):
    result = deepcopy(row)
    definitions = {"attack_target:" + quote(key, safe=""): key for key in targets}
    for issue in result.get("issues", []) if isinstance(result, dict) else []:
        if not isinstance(issue, dict):
            continue
        target = issue.get("target_id")
        if (
            isinstance(target, str)
            and target not in targets
            and target in definitions
            and target in evidence
        ):
            # This is the exact registered name of this definition, not a fuzzy
            # correction or a semantic reassignment to a different target.
            issue["target_id"] = definitions[target]
            issue.setdefault("reference_resolutions", []).append(
                {
                    "field": "target_id",
                    "provided": target,
                    "resolved": definitions[target],
                }
            )
        if not isinstance(issue.get("evidence_refs"), list):
            continue
        for index, ref in enumerate(issue["evidence_refs"]):
            if isinstance(ref, str) and ref in targets and ref not in evidence:
                canonical = "attack_target:" + quote(ref, safe="")
                if canonical in evidence:
                    issue["evidence_refs"][index] = canonical
                    issue.setdefault("reference_resolutions", []).append(
                        {"provided": ref, "resolved": canonical}
                    )
    return result


def public_reference_context(context, evidence):
    """Keep actual user requirements by dialogue, excluding historical Actor answers."""
    view = compact_learning_context(context)
    view["products"] = [
        {
            "product_id": p["product_id"],
            "user_updates": deepcopy(p["user_updates"]),
            "sources": [],
        }
        for p in view["products"]
        if p["user_updates"]
    ]
    user_refs = {}
    for product in view["products"]:
        for update in product["user_updates"]:
            if evidence.get(update["ref"]) != update["text"]:
                raise ValueError(
                    "Actual user amendment must match its original public evidence"
                )
            user_refs[update["ref"]] = deepcopy(evidence[update["ref"]])
    for key in ("checks", "memories", "gaps"):
        view[key] = []
    view.pop("next_round_focus", None)
    sources = {
        k: deepcopy(v)
        for k, v in evidence.items()
        if k == "task"
        or k.startswith(("task_input:", "reference:task_review:", "skill:task_review:"))
    }
    sources.update(user_refs)
    view["available_evidence_refs"] = list(sources)
    view["reference_documents"] = [
        d for d in view.get("reference_documents", []) if d.get("ref") in sources
    ]
    view["skills"] = [s for s in view.get("skills", []) if s.get("ref") in sources]
    return view, sources


def attack_envelope(row):
    """Recover a redundant summary from supplied issue text, never invent evidence.

    Raw model receipts remain unchanged. Every issue and case still passes the
    same admission and independent verification; this only avoids another paid
    draft for a missing repetition of already supplied problem statements.
    """
    if not isinstance(row, dict) or not isinstance(row.get("issues"), list):
        raise ValueError("Attack needs issues and a Task-grounded reason")
    result = deepcopy(row)
    reason = result.get("reason")
    if reason is None or isinstance(reason, str) and not reason.strip():
        if result["issues"] and all(
            isinstance(issue, dict)
            and isinstance(issue.get("problem"), str)
            and issue["problem"].strip()
            for issue in result["issues"]
        ):
            result["reason"] = "\n".join(issue["problem"] for issue in result["issues"])
            result["envelope_recovery"] = {
                "field": "reason",
                "source": "issues[].problem",
                "boundary": "Verbatim supplied proposals, not verified conclusions",
            }
        else:
            raise ResponseContractError(
                "reason",
                "Supply a nonempty Task-grounded reason; "
                "an empty issues list cannot explain its own assessment",
            )
    elif not isinstance(reason, str):
        raise ResponseContractError(
            "reason", "The top-level reason must be nonempty text"
        )
    return result


def validate_attack(row, targets, evidence):
    row = attack_envelope(row)
    if len(row["issues"]) > 3:
        raise ValueError("At most three consequential targets per seat")
    row = deepcopy(row)
    for issue in row["issues"]:
        if issue.get("target_id") not in targets:
            raise ValueError(
                f"Unknown target_id {issue.get('target_id')!r}; use an exact supplied target key"
            )
        if issue.get("priority") not in {"critical", "important", "minor"}:
            raise ValueError(
                f"Invalid priority {issue.get('priority')!r}; use critical, important or minor"
            )
        if any(
            not isinstance(issue.get(k), str) or not issue[k].strip()
            for k in ("problem", "suggested_change")
        ):
            raise ValueError(
                "Attack must locate a falsifiable problem and scoped proposed change"
            )
        refs = issue.get("evidence_refs")
        if isinstance(refs, list):
            missing = [r for r in refs if not isinstance(r, str) or r not in evidence]
            if missing:
                raise ValueError(
                    f"Unknown evidence_refs for target {issue['target_id']}: {missing!r}. "
                    "Use supplied public source refs or the target definition_ref; "
                    "unavailable historical products are not evidence in this operation."
                )
        require_refs(refs, evidence)
        cases = issue.get("cases")
        if not isinstance(cases, list) or len(cases) != 3:
            raise ValueError(
                "Supply one correct control, error variant and boundary case"
            )
        if {c.get("kind") for c in cases} != {
            "correct_control",
            "error_variant",
            "boundary",
        }:
            raise ValueError("Missing contrast or boundary case")
        if len({c.get("id") for c in cases}) != 3:
            raise ValueError("Case IDs must be distinct")
        available_dialogues = {
            ref.split(":")[0]
            for ref in evidence
            if ref.startswith("product_") and ":public:" in ref
        }
        issue["cases"] = [
            bind_case_dialogue(case, available_dialogues) for case in cases
        ]
        for case in issue["cases"]:
            if case.get("scope", "whole_check") not in {"claim", "whole_check"}:
                raise ValueError("Case scope must be claim or whole_check")
            for field in ("id", "question", "artifact", "why"):
                if not isinstance(case.get(field), str) or not case[field].strip():
                    raise ValueError(
                        "Cases need self-contained material, question and derivation"
                    )
            if case.get("expected") not in {"pass", "fail"}:
                raise ValueError(
                    "Predict a falsifiable pass/fail, not an executed verdict"
                )
            if (
                case["kind"] == "correct_control"
                and case["expected"] != "pass"
                or case["kind"] == "error_variant"
                and case["expected"] != "fail"
            ):
                raise ValueError(
                    f"Case {case['id']}: {case['kind']} has expected={case['expected']}; "
                    "correct_control needs Task-valid material/pass; error_variant needs "
                    "a deliberately Task-invalid mutation/fail. Change the invalid case material, "
                    "not merely its label; a valid answer rejected by a bad rule is still pass."
                )
            probes = case.get("probes", [])
            if not isinstance(probes, list) or len(probes) > 3:
                raise ValueError("At most three bounded probes per case")
            if any(
                p.get("tool") not in {"calculate", "code.run", "document.read"}
                for p in probes
            ):
                raise ValueError("Use registered bounded measurements, never host code")
            if any(
                ref.startswith("product_")
                for p in probes
                for ref in p.get("arguments", {}).get("input_refs", [])
                if isinstance(ref, str)
            ):
                raise ValueError(
                    "Synthetic cases need self-contained inputs, not historical product values"
                )
    return deepcopy(row)


def choose_targets(reviews, limit=3, *, targets=None):
    """Round-robin seats then severity; deduplicate the object, not votes about truth."""
    order = {"critical": 0, "important": 1, "minor": 2}
    queues = {
        seat: sorted(row["issues"], key=lambda i: order[i["priority"]])
        for seat, row in reviews.items()
    }
    chosen, seen = [], set()
    if limit <= 0:
        return chosen

    def identity(issue):
        target = (targets or {}).get(issue["target_id"], {})
        return (
            target.get("criterion_id", issue["target_id"]),
            target["id"] if target.get("target_kind") == "clause" else None,
        )

    while any(queues.values()):
        for seat, rows in queues.items():
            while rows and identity(rows[0]) in seen:
                rows.pop(0)
            if not rows:
                continue
            issue = rows.pop(0)
            chosen.append({**deepcopy(issue), "proposer_seat": seat})
            seen.add(identity(issue))
            if len(chosen) == limit:
                return chosen
    return chosen


def admit_attack_draft(row, targets, evidence):
    """Select bounded well-formed proposals; preserve every rejected draft and its reason."""
    row = attack_envelope(row)
    accepted, excluded = [], []
    for index, issue in enumerate(row["issues"]):
        try:
            checked = validate_attack(
                {"issues": [issue], "reason": row["reason"]}, targets, evidence
            )
        except (ValueError, TypeError, KeyError, AttributeError) as error:
            excluded.append(
                {"index": index, "issue": deepcopy(issue), "reason": str(error)}
            )
        else:
            accepted.append((index, checked["issues"][0]))
    if row["issues"] and not accepted:
        error = ResponseContractError(
            "issues",
            "No complete attack candidate; repair required: "
            + "; ".join(r["reason"] for r in excluded),
        )
        error.details["issues"] = [
            {"index": r["index"], "reason": r["reason"]} for r in excluded
        ]
        error.recovery_instruction = (
            "Repair the listed attack-case contract failures in this SAME draft. Preserve target IDs, "
            "unaffected issues, and the original suspected problem as an unverified proposal. You MAY "
            "reconstruct invalid case material, question, prediction, derivation and probes to supply "
            "a real Task-valid control (pass), a deliberately Task-invalid related mutation (fail), "
            "and an applicability boundary. Predictions concern material validity against the Task, "
            "NOT an attacked evaluator verdict. Never flip a label without making the material and "
            "derivation agree. Read exact supplied evidence when needed; never invent support. "
            "Return the full contract. Independent references still have to test the new cases."
        )
        raise error
    priority = {"critical": 0, "important": 1, "minor": 2}
    accepted.sort(key=lambda pair: priority[pair[1]["priority"]])
    for index, issue in accepted[3:]:
        excluded.append(
            {
                "index": index,
                "issue": deepcopy(issue),
                "reason": "Outside three-candidate seat budget",
            }
        )
    result = {**deepcopy(row), "issues": [issue for _, issue in accepted[:3]]}
    if excluded:
        result["draft_admission"] = {
            "raw_issue_count": len(row["issues"]),
            "selected_indices": [index for index, _ in accepted[:3]],
            "unadmitted": excluded,
            "boundary": "Format and budget selection only; predictions remain unverified",
        }
    return validate_attack(result, targets, evidence)


def validation_record(
    target, state, *, context, evidence, artifact, cases=(), reason=""
):
    if state not in STATES:
        raise ValueError("Unknown validation state")
    binding = claim_binding(
        targets_for(
            artifact, "hypothesis" if target.startswith("hypothesis:") else "reward"
        )[target]
    )
    return {
        "version": VERSION,
        "target_id": target,
        "status": state,
        "task_sha256": fingerprint(context["task"]),
        "artifact_sha256": fingerprint(artifact),
        "target_sha256": fingerprint(
            targets_for(
                artifact, "hypothesis" if target.startswith("hypothesis:") else "reward"
            )[target]
        ),
        "evidence_sha256": fingerprint(evidence),
        "cases": deepcopy(list(cases)),
        "reason": reason,
        "validation_id": fingerprint(
            {"target": target, "artifact": artifact, "cases": cases, "state": state}
        ),
        "claim_binding": binding,
        "boundary": "Scoped public-reference comparison, not a private grader or universal correctness proof",
    }


async def verify_issue(
    issue,
    artifact,
    module,
    context,
    evidence,
    registry,
    reference,
    *,
    record,
    adversarial_review_enabled=False,
):
    """The reference seat never receives the Pack, attack rationale or predicted labels."""
    require_adversarial(adversarial_review_enabled)
    targets = targets_for(artifact, module)
    target = targets[issue["target_id"]]
    material = []
    case_by_id = {}
    public_context, public_evidence = public_reference_context(context, evidence)
    dialogues = {p["product_id"]: p["user_updates"] for p in public_context["products"]}
    issue = deepcopy(issue)
    issue["cases"] = [bind_case_dialogue(case, dialogues) for case in issue["cases"]]
    for case in issue["cases"]:
        dialogue = case.get("user_context_product_id")
        key = (
            "case_"
            + fingerprint(
                {
                    "artifact": case["artifact"],
                    "question": case["question"],
                    **(
                        {"user_context_product_id": dialogue}
                        if dialogue is not None
                        else {}
                    ),
                }
            )[:16]
        )
        if key in case_by_id:
            raise ValueError("Distinct cases must contain distinct material/questions")
        case_by_id[key] = case
        receipts = [
            await registry.tools.run(probe, evidence)
            for probe in case.get("probes", [])
        ]
        material.append(
            {
                "id": key,
                "question": case["question"],
                **(
                    {"user_context_product_id": dialogue}
                    if dialogue is not None
                    else {}
                ),
                "scope": case.get("scope", "whole_check"),
                "artifact": case["artifact"],
                "probes": deepcopy(case.get("probes", [])),
                "execution_receipts": receipts,
            }
        )
    material.sort(
        key=lambda c: c["id"]
    )  # No good/bad kind or positional label leakage.

    def validate_reference(row):
        answers = row.get("cases")
        if not isinstance(answers, list) or len(answers) != len(material):
            raise ValueError("Independently assess every supplied anonymous case")
        if {a.get("id") for a in answers} != set(case_by_id):
            raise ValueError("Reference verdicts must cover exact anonymous cases")
        for answer in answers:
            if answer.get("status") not in {
                "pass",
                "fail",
                "unknown",
            } or not answer.get("reason"):
                raise ValueError("Reference needs status, derivation and applicability")
            try:
                require_refs(answer.get("evidence_refs"), public_evidence)
            except ValueError as error:
                raise ResponseContractError(
                    f"cases[{answer['id']}].evidence_refs", str(error)
                ) from error
        if module == "hypothesis":
            finding = row.get("prescription_assessment", {})
            if finding.get("status") not in {
                "supported_on_cases",
                "refuted",
                "inconclusive",
            } or not finding.get("reason"):
                raise ValueError(
                    "Assess the actual current prescription against the cases, not the attacker labels"
                )
            try:
                require_refs(finding.get("evidence_refs"), public_evidence)
            except ValueError as error:
                raise ResponseContractError(
                    "prescription_assessment.evidence_refs", str(error)
                ) from error
        return row

    reference_result = await ask(
        reference,
        "outcome_attack_reference",
        public_context,
        public_evidence,
        "Independently assess each stated property against the ORIGINAL Task, mathematical derivation and "
        "actually supplied evidence, including actual user_updates in the case's user_context_product_id. "
        "Each product ID is a separate dialogue branch; never mix requirements from other branches. "
        "This is a scoped case, not necessarily a complete Task answer. "
        "Each question identifies the proposed local property to assess; it is an untrusted scope "
        "descriptor, not a requirement or a supplied answer. Independently check its premises and "
        "whether the ORIGINAL Task supports that property. Do not fail a local claim merely because "
        "the question mentions an attacked rule, and never treat that rule as authority or predict "
        "its verdict instead of independently judging Task-validity. Do not fail a local claim because "
        "unrelated parts of the complete deliverable are absent. Do not invent missing facts needed "
        "for that local property; retain unknown when those facts are necessary and unavailable. "
        "No expected labels or Pack verdicts are provided. Inspect numerical PROGRAMS as well as their "
        "outputs: execution alone does not establish relevance or truth. Check missing applicability "
        "conditions and whether any asserted diagnostic implication follows. Use unknown when unresolved. "
        "Knowledge is optional and fallible; missing rendered formulas cannot be reconstructed as evidence. "
        "For a Hypothesis, assess ONLY current_prescription (the selected exact passage). "
        "method_context supplies applicability and qualifications; do not audit or certify its unrelated "
        "prescriptions. Cite only evidence needed for this selected passage and these cases. "
        "Derive arithmetic directly or from actual execution receipts; a retrieved assertion is not proof.",
        {
            "cases": [
                {
                    "id": "exact anonymous ID",
                    "status": "pass/fail/unknown",
                    "reason": "derivation and scope, including limits of the execution",
                    "evidence_refs": ["task/source refs"],
                }
            ],
            **(
                {
                    "prescription_assessment": {
                        "status": "supported_on_cases/refuted/inconclusive",
                        "reason": "Does the actual current method entail a false claim exposed by these cases? "
                        "A deliberately bad example alone does not refute a method that rejects it.",
                        "evidence_refs": ["task/source refs"],
                    }
                }
                if module == "hypothesis"
                else {}
            ),
        },
        extra={
            "cases": material,
            **(
                {
                    "current_prescription": target.get(
                        "quote", artifact.get("method", "")
                    ),
                    "method_context": artifact.get("method", ""),
                }
                if module == "hypothesis"
                else {}
            ),
        },
        validate=validate_reference,
        max_contract_retries=1,
        max_evidence_rounds=2,
        close_evidence=True,
        verify_used_previews=True,
    )
    record(
        "reference",
        {
            "cases": material,
            "assessment": reference_result,
            "proposer_seat": issue["proposer_seat"],
            "reference_blind_to_pack": True,
        },
    )
    reference_by_id = {r["id"]: r for r in reference_result["cases"]}
    if module == "reward":
        tests = (
            [target]
            if "test_id" in target
            else [
                t
                for t in artifact["tests"]
                if t["criterion_id"] == target.get("criterion_id", issue["target_id"])
            ]
        )
        for test in tests:
            if test.get("execution_recipe") and test["checker"] != "evidence.review":
                registry.bind_recipe(test["checker"], test["execution_recipe"])
    else:
        tests = []

    async def execute_case(item):
        candidate = {"artifact": {"final_response": item["artifact"]}}
        if item.get("user_context_product_id"):
            candidate["artifact"]["public_consultation"] = {
                "earlier_exchanges": [
                    {"speaker": "user", "text": update["text"]}
                    for update in dialogues[item["user_context_product_id"]]
                ]
            }
        c, e = build_context(context["task"], [candidate])
        if module == "reward":
            c, e = measurement_context(artifact, c, e)
            local = item["scope"] == "claim" or target.get("target_kind") == "clause"
            selected_tests = [
                t for t in tests if not local or t["checker"] == "evidence.review"
            ]
            if local:
                inquiry = {
                    "questions": [item["question"]],
                    "scope": "Only the supplied claim; missing unrelated full-answer content is not a defect.",
                    "target_clause": deepcopy(target)
                    if target.get("target_kind") == "clause"
                    else None,
                }
                c["measurement_inquiries"] = {
                    t["test_id"]: deepcopy(inquiry) for t in selected_tests
                }
            results = await asyncio.gather(
                *(registry.run(test, c, e) for test in selected_tests)
            )
            full_results = deepcopy(results)
            if local:
                results = [
                    {
                        **r.get(
                            "inquiry_result",
                            {"status": "unknown", "reason": "No scoped inquiry result"},
                        ),
                        **(
                            {
                                "status": "unknown",
                                "reason": r.get("reason", "Checker execution failed"),
                            }
                            if r.get("error")
                            else {}
                        ),
                        **{
                            k: r[k]
                            for k in ("test_id", "criterion_id", "error")
                            if k in r
                        },
                    }
                    for r in results
                ]
            usable = [r for r in results if not r.get("error")]
            observed = (
                "fail"
                if any(r["status"] == "fail" for r in usable)
                else "pass"
                if results
                and len(usable) == len(results)
                and all(r["status"] == "pass" for r in usable)
                else "unknown"
            )
        else:
            # Hypothesis is a technical prescription, not an executable checker.
            # Its proponent's predictions are checked independently on all cases.
            results, observed = [], case_by_id[item["id"]]["expected"]
            full_results = []
        expected = reference_by_id[item["id"]]
        return {
            "case_id": item["id"],
            "material_sha256": fingerprint(item),
            "material": item,
            "reference": expected,
            "current_pack_results": results,
            "full_check_results": full_results,
            "case_kind": case_by_id[item["id"]]["kind"],
            "current_verdict": observed,
            "proposal_prediction": case_by_id[item["id"]]["expected"],
            "disagreement": observed in {"pass", "fail"}
            and expected["status"] in {"pass", "fail"}
            and observed != expected["status"],
        }

    cases = await asyncio.gather(*(execute_case(item) for item in material))
    qualified = all(
        c["reference"]["status"] == case_by_id[c["case_id"]]["expected"]
        for c in cases
        if case_by_id[c["case_id"]]["kind"] in {"correct_control", "error_variant"}
    )
    # A real checker discrepancy is scoped evidence against that binding. For a
    # Hypothesis, a reference disagreement refutes its proposed case prediction,
    # not automatically the whole method; repair must resolve the exact issue.
    state = (
        "disputed"
        if any(c["disagreement"] for c in cases)
        else "supported_on_cases"
        if all(
            c["reference"]["status"] != "unknown" and c["current_verdict"] != "unknown"
            for c in cases
        )
        else "inconclusive"
    )
    if module == "hypothesis":
        state = reference_result["prescription_assessment"]["status"]
        if state == "refuted":
            state = "disputed"
    if not qualified:
        # Agreement on three invalid examples is not a positive/negative discrimination test.
        state = "inconclusive"
    result = validation_record(
        issue["target_id"],
        state,
        context=context,
        evidence=evidence,
        artifact=artifact,
        cases=cases,
        reason="Compare actual public-reference and checker outcomes on these cases only",
    )
    result["issue"] = deepcopy(issue)
    result["contrast_qualified"] = qualified
    result["verification_mode"] = (
        "model_assisted_with_execution"  # Never pretend semantic matching is deterministic.
    )
    record("comparison", result)
    return result


async def attack(
    artifact,
    module,
    context,
    evidence,
    registry,
    members,
    references,
    *,
    record=lambda *_: None,
    target_ids=None,
    proposal_cache=None,
    adversarial_review_enabled=False,
):
    require_adversarial(adversarial_review_enabled)
    if set(members) != {"1", "2", "3"} or set(references) != set(members):
        raise ValueError(
            "Attack requires three independent proposal and reference seats"
        )
    all_targets = targets_for(artifact, module)
    if target_ids is not None and (
        not isinstance(target_ids, list)
        or not target_ids
        or any(not isinstance(key, str) for key in target_ids)
        or len(set(target_ids)) != len(target_ids)
        or not set(target_ids) <= set(all_targets)
    ):
        raise ValueError("Attack focus must name distinct existing targets")
    targets = {
        key: value
        for key, value in all_targets.items()
        if target_ids is None or key in target_ids
    }
    context, evidence = public_reference_context(context, evidence)
    view, map_evidence, target_view = attack_context(context, evidence, targets)
    case_schema, dialogue_instruction = dialogue_contract(context)
    attack_schema = deepcopy(ATTACK_SCHEMA)
    attack_schema["issues"][0]["cases"] = [case_schema]
    # The optional store is session-local: its owner binds model settings and
    # policy. Include all public inputs and tool definitions here, not just IDs.
    binding = fingerprint(
        {
            "version": VERSION,
            "artifact": artifact,
            "module": module,
            "context": context,
            "evidence": evidence,
            "targets": targets,
            "tools": registry.tools.catalog(),
            "schema": attack_schema,
        }
    )
    cached = (
        proposal_cache.load_attack_proposals(binding)
        if proposal_cache is not None
        else {}
    )
    if not isinstance(cached, dict) or not set(cached) <= set(members):
        raise ValueError("Invalid session-local Attack proposal checkpoint")
    rows = {
        seat: admit_attack_draft(
            resolve_target_citations(row, targets, map_evidence), targets, map_evidence
        )
        for seat, row in cached.items()
    }
    record(
        "map_recovery",
        {
            "binding": binding,
            "reused_seats": sorted(rows),
            "requested_seats": sorted(set(members) - set(rows)),
        },
    )

    async def propose(seat):
        return await ask(
            members[seat],
            "outcome_attack_map",
            view,
            map_evidence,
            "Attack the supplied current standards AND actual checking methods, or Hypothesis. "
            "Start from the ORIGINAL Task: locate high-impact false premises, equivalent valid answers "
            "that could be rejected, errors that could pass, and assumptions that break at boundaries. "
            "Do not invent defects to meet a quota. Select up to three consequential targets for validation, "
            "including apparently sound prescriptions; problem may state the falsifiable risk/question "
            "instead of alleging a defect. "
            + dialogue_instruction
            + "For each supply a self-contained "
            "correct control, error variant and boundary case with short material and precise questions. "
            "Prefer the specific clause target when challenging one claim. Set scope=claim for fragments; "
            "use scope=whole_check only for a complete deliverable that can satisfy the whole test. "
            "A local claim must not fail just because unrelated deliverable sections are absent. "
            "Use bounded numerical probes where helpful. Predicted outcomes are hypotheses, NOT executed "
            "results. expected ALWAYS predicts the material's correctness against the ORIGINAL TASK: "
            "correct_control=pass, error_variant=fail, boundary=pass/fail. It is NOT whether the "
            "attacked rule accepts the material or whether a question should be answered yes. "
            "Put a self-contained assertion or deliverable in artifact, not just a question or "
            "commentary about what an evaluator might do. A valid alternative rejected by a bad "
            "rule is a passing control/boundary; also construct a related actually incorrect variant. "
            "For Hypothesis, challenge the actual advice and its applicability; do not treat it as "
            "a grader. The independent reference separately assesses whether that advice entails "
            "an error exposed by the cases. Knowledge is supplemental "
            "and fallible, sources must be read selectively. Cite a target using its definition_ref only "
            "to locate the definition being challenged, not as proof that it is correct. Unavailable "
            "historical provenance is not supplied evidence. No private grading or previous attack findings.",
            attack_schema,
            extra={
                "module": module,
                "targets": target_view,
                "measurement_tools": registry.tools.catalog(),
            },
            validate=lambda row: admit_attack_draft(
                resolve_target_citations(row, targets, map_evidence),
                targets,
                map_evidence,
            ),
            max_contract_retries=1,
            max_evidence_rounds=2,
            close_evidence=True,
        )

    batch = await collect_batch(
        {seat: lambda s=seat: propose(s) for seat in members if seat not in rows},
        BatchPolicy(deadline_s=900),
    )
    record("map_status", batch.summary())
    # Persist each completed semantic proposal even if another provider failed.
    # The strict three-seat boundary still applies; these are recovery inputs,
    # never an implicit vote or permission to declare the review complete.
    rows.update(
        {item.job_id: item.value for item in batch.items if item.status == "ok"}
    )
    record("proposals", rows)
    if proposal_cache is not None:
        proposal_cache.save_attack_proposals(binding, rows)
    batch.require_all()
    if set(rows) != set(members):
        raise ValueError(
            "Attack requires all three proposal seats, including recovered drafts"
        )
    if proposal_cache is not None:
        # Later deliberate reviews may sample again. Only incomplete batches
        # receive recovery; independent references below always execute afresh.
        proposal_cache.clear_attack_proposals(binding)
    rows = {seat: rows[seat] for seat in members}
    selected = choose_targets(rows, targets=targets)
    record("selected_targets", selected)

    async def verify(index, issue):
        reference_seat = str(int(issue["proposer_seat"]) % 3 + 1)
        try:
            return await verify_issue(
                issue,
                artifact,
                module,
                context,
                evidence,
                registry,
                references[reference_seat],
                record=lambda n, d: record(f"target_{index}/{n}", d),
                adversarial_review_enabled=adversarial_review_enabled,
            )
        except Exception as error:
            # A failed bounded reference/check is uncertainty about THIS target,
            # not permission to cancel other targets or call the Pack validated.
            failure = {
                "error_type": type(error).__name__,
                "reason": str(error),
                **(
                    {"details": deepcopy(error.details)}
                    if hasattr(error, "details")
                    else {}
                ),
            }
            record(f"target_{index}/verification_unavailable", failure)
            result = validation_record(
                issue["target_id"],
                "inconclusive",
                context=context,
                evidence=evidence,
                artifact=artifact,
                reason="Bounded case verification unavailable; no support or revision authority granted",
            )
            result.update(
                issue=deepcopy(issue),
                verification_error=failure,
                contrast_qualified=False,
            )
            return result

    findings = await asyncio.gather(
        *(verify(i, issue) for i, issue in enumerate(selected))
    )
    tested = {r["target_id"] for r in findings}
    findings.extend(
        validation_record(
            key,
            "unverified",
            context=context,
            evidence=evidence,
            artifact=artifact,
            reason="Outside this bounded attack coverage",
        )
        for key in all_targets
        if key not in tested
    )
    result = {
        "version": VERSION,
        "status": "partial"
        if any(r.get("verification_error") for r in findings)
        else "completed",
        "module": module,
        "records": findings,
        "proposals": rows,
        "selected_target_count": len(selected),
        "artifact_sha256": fingerprint(artifact),
    }
    record("result", result)
    return result
