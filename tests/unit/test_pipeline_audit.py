"""Synthetic receipt trees, never network calls or changes to real runs."""

import asyncio
import hashlib
import json
import shutil
from copy import deepcopy
from pathlib import Path

import pytest

from experiments.pipeline_audit import (
    IncompleteReceipt,
    audit_run,
    criteria,
    final_candidate_source,
    main,
    measurement,
    receipt_digest,
)
from reward_engine.adapters.workbuddy.evidence import collect
from reward_engine.adapters.workbuddy.io import digest
from reward_engine.core.outcome.context import build_context, check_bundle
from reward_engine.core.outcome.delivery_triage import review as targeted_review
from reward_engine.core.outcome.harness import Registry, measure, measurement_context
from reward_engine.core.outcome.task_progress import TaskProgress
from reward_engine.core.outcome.task_sequence import TaskSequence

TASK = {"task_id": "task-one", "prompt": "Explain the actual command and verify it"}
TASK_DIR = Path("tasks/task-one")


def write(root, path, value, *, lines=False):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(json.dumps(row) + "\n" for row in value) if lines else json.dumps(value)
    )


def read(root, path):
    return json.loads((root / path).read_text())


def edit(root, path, change):
    row = read(root, path)
    change(row)
    write(root, path, row)


def mapping(seats=3):
    return {
        "complete": True,
        "total": seats,
        "succeeded": seats,
        "items": [{"job_id": str(i), "status": "ok"} for i in range(seats)],
    }


def measured(
    pack,
    observed,
    *,
    task=TASK,
    defect=False,
    targeted=False,
    previous=None,
    inquiries=False,
):
    """Actually execute the harness on fixture inputs; never backfill a binding."""

    async def check(_test, _context, evidence):
        assert (
            evidence["product_1:artifact:final_response"]
            == observed["artifact"]["final_response"]
        )
        return {
            "status": "fail" if defect else "pass",
            "reason": "Observed CLI output",
            "gap": "Wrong flag" if defect else "",
            "evidence_refs": ["product_1:artifact:final_response"],
            "error": False,
        }

    registry = Registry()
    registry.register("semantic.judge", check, "Offline fixture check")
    context, evidence = build_context(task, [observed])
    if not targeted:
        return asyncio.run(measure(pack, registry, context, evidence))

    async def overview(_request):
        return {
            "outcome_goal": "Explain the actual command accurately",
            "full_score_gaps": [],
            "improvements": [],
            "status": "needs_improvement" if defect else "target_satisfied",
            "reason": "Checked the command in the delivered explanation",
            "gaps": [
                {
                    "gap": "Wrong flag",
                    "evidence_refs": ["product_1:artifact:final_response"],
                }
            ]
            if defect
            else [],
            "verify": [
                {
                    "test_id": "t1",
                    "reason": "Does the exact command match this delivery?",
                }
            ]
            if inquiries
            else [],
            "claim_checks": [
                {
                    "claim": "Command accepts --help",
                    "claim_ref": "product_1:artifact:final_response",
                    "support_refs": ["product_1:artifact:final_response"],
                    "assessment": "supported",
                    "reason": "Observed in command output",
                    "test_id": "t1" if inquiries else None,
                }
            ],
        }

    registry.review_call = overview
    return asyncio.run(
        targeted_review(pack, registry, context, evidence, previous=previous)
    )


def native_score_fixture(task_id=TASK["task_id"], *, rule_score=0.0, llm_score=0.0):
    """Office native aggregate schema, including authored expected item coverage."""
    verdicts = [
        {
            "item_id": f"{task_id}:rule",
            "judge_name": "rule",
            "judge_type": "rule_script",
            "status": "pass" if rule_score == 1 else "fail",
            "score": rule_score,
            "reason": "Observed authored rule results",
            "evidence_ids": [],
            "confidence": None,
            "metadata": {},
        },
        {
            "item_id": f"{task_id}:llm:quality",
            "judge_name": "llm_judge",
            "judge_type": "llm_judge",
            "status": "pass" if llm_score == 1 else "fail",
            "score": llm_score,
            "reason": "Observed authored rubric result",
            "evidence_ids": [],
            "confidence": None,
            "metadata": {},
        },
    ]
    judges = []
    for index, verdict in enumerate(verdicts):
        judges.append(
            {
                "judge_name": verdict["judge_name"],
                "judge_type": verdict["judge_type"],
                "judge_family": "llm" if index else "rule",
                "status": "pass" if index else verdict["status"],
                "ok": True,
                "verdicts": [deepcopy(verdict)],
                "errors": [],
                "evidence_ids": [],
                "scores": {"llm_judge": llm_score, "rubric_binary_mean": llm_score}
                if index
                else {"pass_rate": rule_score, "overall": rule_score},
                "metadata": {}
                if index
                else {
                    "tests_passed": int(rule_score * 10),
                    "tests_total": 10,
                    "test_status": "full_pass"
                    if rule_score == 1
                    else "partial_pass"
                    if rule_score
                    else "no_pass",
                    "raw": {
                        "overall": rule_score,
                        "test_status": "full_pass"
                        if rule_score == 1
                        else "partial_pass"
                        if rule_score
                        else "no_pass",
                        "tests_passed": int(rule_score * 10),
                        "tests_total": 10,
                    },
                },
            }
        )
    config = {"method": "weighted_sum", "rule_weight": 0.8, "llm_weight": 0.2}
    overall = round(rule_score * 0.8 + llm_score * 0.2, 4)
    merge = {
        **config,
        "applied": True,
        "rule_score": rule_score,
        "llm_score": llm_score,
        "overall": overall,
    }
    return {
        "overall": overall,
        "reward": overall,
        "test_pass_rate": rule_score,
        "llm_judge_component_score": llm_score,
        "test_status": "full_pass"
        if overall == 1
        else "partial_pass"
        if overall
        else "no_pass",
        "tests_passed": int(rule_score * 10),
        "tests_total": 10,
        "verdicts": verdicts,
        "judges": judges,
        "metadata": {"scoring_policy": "pass_rate", "score_merge": merge},
        "diagnostics": {
            "score_merge": {**merge, "configured": config},
            "plan": {
                "task_id": task_id,
                "dataset_id": "wb-bench-office-v1.0",
                "scoring_policy": "pass_rate",
                "items": [
                    {
                        "id": v["item_id"],
                        "type": "rule" if i == 0 else "llm",
                        "weight": 1.0,
                        "metadata": {},
                    }
                    for i, v in enumerate(verdicts)
                ],
                "judges": [
                    {
                        "name": j["judge_name"],
                        "type": j["judge_type"],
                        "family": j["judge_family"],
                        "item_ids": [j["verdicts"][0]["item_id"]],
                        "weight": 1.0,
                    }
                    for j in judges
                ],
                "metadata": {
                    "source_case": task_id,
                    "llm_judge_enabled": True,
                    "score_merge": config,
                },
            },
        },
    }


def native(
    root,
    slot,
    benchmark,
    *,
    pack=None,
    targeted=False,
    review_turns=2,
    unchanged=False,
    unresolved=False,
    inquiries=False,
):
    cid = slot.as_posix().replace("/", "-")
    session = "session-" + cid
    is_b = pack is not None
    turns = review_turns if is_b else 1
    progress = TaskProgress(TASK, pack) if is_b else None
    snapshots = []
    previous = None
    for turn in range(1, turns + 1):
        answer = f"{cid} explanation {1 if unchanged else turn}"
        events = [
            {
                "type": "text",
                "sessionID": session,
                "part": {"text": answer, "messageID": f"m{turn}"},
            },
            {"type": "step_finish", "sessionID": session, "part": {"reason": "stop"}},
        ]
        if benchmark == "claw_eval":
            log, snapshot = (
                slot / f"actor-{turn}-0.jsonl",
                slot / "snapshots" / str(turn),
            )
        else:
            log = slot / "harbor" / cid / "agent" / f"outcome-turn-{turn}.jsonl"
            snapshot = slot / "snapshots" / f"delivery-{turn}"
        write(root, log, events, lines=True)
        workspace = root / snapshot / "workspace"
        workspace.mkdir(parents=True)
        (workspace / "deliverable.txt").write_text(answer)
        sample = collect(
            workspace,
            candidate_id=cid,
            final_response=answer,
            initial_hashes={},
            output=root / snapshot,
        )[0]
        snapshots.append(sample)
        if is_b:
            estimate = measured(
                pack,
                sample,
                defect=turn < turns or unresolved,
                targeted=targeted,
                previous=previous,
                inquiries=inquiries,
            )
            previous_context, previous_evidence = measurement_context(
                pack, *build_context(TASK, [sample])
            )
            previous = (previous_context, previous_evidence, estimate)
            write(
                root,
                slot / "checks" / f"delivery_{turn}.json",
                {"candidate": sample, "measurement": estimate},
            )
            progress.begin(
                build_context(TASK, [sample])[0], turn, f"checks/delivery_{turn}.json"
            )
            progress.finish(estimate)
            status = (
                "revision_requested"
                if turn < turns
                else "budget_exhausted_unresolved"
                if unresolved
                else "full_score_target_satisfied"
                if targeted
                else "satisfied_under_estimator"
            )
            progress.state.update(
                status=status,
                revisions_requested=turn if turn < turns else turn - 1,
                max_revisions=2,
            )
            write(
                root,
                slot / "checks" / f"task_progress_{turn}.json",
                progress.snapshot(),
            )
            write(root, slot / "checks/task_progress.json", progress.snapshot())
    write(root, slot / "actor_state.json", {"session": session, "turn": turns})
    for previous in range(1, turns):
        write(
            root,
            slot / "checks" / f"feedback_{previous}.json",
            {
                "message": "The observed --old flag is incorrect. Use --help and check the resulting explanation."
            },
        )
    row = {
        "status": "completed",
        "candidate": snapshots[-1],
        "session": session,
        "review_status": progress.state["status"] if is_b else "disabled",
        "task_progress": progress.snapshot() if is_b else None,
        "final_measurement": read(root, slot / "checks" / f"delivery_{turns}.json")[
            "measurement"
        ]
        if is_b
        else None,
        "memory": [],
    }
    is_arm = slot.parent.name == "actors"
    if benchmark == "claw_eval":
        row.update(
            identity=digest(
                {
                    "task": TASK["task_id"],
                    "guidance": "",
                    "review": is_b,
                    "protocol": "frozen-config",
                }
            ),
            runtime_directory=str(root / slot),
            native_grader_trial=str(root / slot),
        )
        write(root, slot / "STARTED.json", {"identity": row["identity"]})
        trace = [
            {"type": "trace_start", "task_id": TASK["task_id"], "trace_id": cid},
            {"type": "trace_end", "trace_id": cid},
        ]
        write(root, slot / "trace.jsonl", trace, lines=True)
        write(
            root, slot / "environment-snapshot.json", {"file:answer": "actual content"}
        )
        grade = {
            "status": "graded",
            "protocol": "native-claw-eval-custom-judge",
            "task_id": TASK["task_id"],
            "trace_id": cid,
            "task_score": 0.125,
            "scores": {
                "completion": 0.0,
                "robustness": 0.25,
                "safety": 1.0,
                "efficiency_turns": 7,
            },
            "official_leaderboard_equivalent": False,
            "snapshot_error_keys": [],
        }
        directory = slot / "official.private/grader-id"
        write(root, directory / "result.private.json", grade)
        grade.update(
            record_directory=str(root / directory),
            binding={
                "identity": "frozen-config",
                "trace": hashlib.sha256(
                    (root / slot / "trace.jsonl").read_bytes()
                ).hexdigest(),
                "snapshot": hashlib.sha256(
                    (root / slot / "environment-snapshot.json").read_bytes()
                ).hexdigest(),
            },
        )
    else:
        write(root, slot / "ACTOR_RESULT.json", row)
        row["spec_identity"] = "identity-" + cid
        write(
            root,
            slot / "spec.json",
            {
                "spec_identity": row["spec_identity"],
                "candidate_id": cid,
                "task": TASK,
                "task_path": str(root / "dataset" / TASK["task_id"]),
                "grade": is_arm,
            },
        )
        task_path = root / "dataset" / TASK["task_id"]
        task_path.mkdir(parents=True, exist_ok=True)
        (task_path / "task.toml").write_text(
            '[metadata]\nsource_case = "' + TASK["task_id"] + '"\n'
        )
        write(
            root,
            task_path / "tests/judge.yaml",
            {
                "llm_judge": {"rubrics": [{"id": "quality"}]},
                "score_merge": {
                    "method": "weighted_sum",
                    "rule_weight": 0.8,
                    "llm_weight": 0.2,
                },
            },
        )
        detail = native_score_fixture()
        score_path = slot / "harbor" / cid / "verifier/score.json"
        write(root, score_path, detail)
        rewards = {
            key: detail[key]
            for key in (
                "overall",
                "reward",
                "test_pass_rate",
                "llm_judge_component_score",
                "tests_passed",
                "tests_total",
            )
        }
        write(
            root,
            slot / "TRIAL_RESULT.private.json",
            {
                "trial_name": cid,
                "exception_info": None,
                "verifier_result": {"rewards": rewards},
            },
        )
        grade = {
            "status": "scored",
            "overall": 0.0,
            "raw_overall": 0.0,
            "rewards": rewards,
            "native_test_status": detail["test_status"],
            "native_score": str(root / score_path),
            "trial": str(root / slot / "harbor" / cid),
            "exception": None,
            "scoring_errors": [],
        }
    write(root, slot / "RESULT.json", row)
    if is_arm:
        write(root, slot / "OFFICIAL.private.json", grade)
    return row, grade


def make_run(
    root,
    benchmark="claw_eval",
    *,
    targeted=False,
    no_revision_rounds=(),
    unchanged_rounds=(),
    unresolved_rounds=(),
    inquiries=False,
):
    write(
        root,
        "config.json",
        {
            "validation_benchmark": benchmark,
            "validation_task_id": TASK["task_id"],
            "rounds": 3,
            "full_audit": True,
            "quality_resampling": False,
            "identity": "frozen-config",
        },
    )
    write(
        root,
        "manifest.json",
        {"config_identity": "frozen-config", "actor_slots": 12, "grading_slots": 9},
    )
    write(
        root,
        "validation_protocol.json",
        {
            "full_harness_audit": True,
            "historical_samples_reused": False,
            "quality_resampling": False,
        },
    )
    write(
        root,
        "validation_status.json",
        {"status": "execution_completed", "completed_rounds": 3},
    )
    pool = [
        native(root, TASK_DIR / "seeds" / str(i), benchmark)[0]["candidate"]
        for i in range(1, 4)
    ]
    memory, checks, summaries, decisions = [], [], [], []
    criterion = {
        "id": "c1",
        "standard": "Describe the actual CLI",
        "measurement_question": "Does the observed command work?",
        "kind": "hard",
        "priority": "critical",
        "priority_reason": "Incorrect commands prevent task completion",
        "origin": "task",
        "evidence_refs": ["task"],
        "subrequirements": ["The CLI example matches the actual interface"],
    }
    for number in range(1, 4):
        folder = TASK_DIR / f"round_{number}"
        retrieved = [
            {
                "memory_id": m["memory_id"],
                "key": m["key"],
                "value": m["value"],
                "relevance": 0.8,
            }
            for m in memory
        ]
        context = build_context(TASK, pool, checks=checks, memories=retrieved)[0]
        write(root, folder / "context.json", context)
        jury = {
            "criteria": [criterion],
            "gaps": [{"gap_id": "g1", "gap": "Wrong flag"}],
        }
        write(root, folder / "jury.json", jury)
        write(root, folder / "jury_map.json", mapping())
        pack = {
            "pack_id": f"pack-{number}",
            "task_id": TASK["task_id"],
            "task_prompt": TASK["prompt"],
            "criteria": [
                {
                    **criterion,
                    "anchors": {
                        "fail": "Commands fail",
                        "partial": "Some commands match",
                        "full": "All commands match observed behavior",
                    },
                }
            ],
            "status": "callable",
            "tests": [
                {
                    "test_id": "t1",
                    "criterion_id": "c1",
                    "checker": "semantic.judge",
                    "binding_status": "callable",
                }
            ],
        }
        write(root, folder / "pack.json", pack)
        write(
            root,
            folder / "construction_smoke.json",
            {
                "status": "receipt_path_checked",
                "bindings": {"t1": "callable"},
                "receipt": {
                    "status": "pass",
                    "reason": "Parsed fixture",
                    "evidence_refs": ["product_1:artifact:final_response"],
                },
            },
        )
        write(
            root,
            folder / "hypothesis.json",
            {
                "status": "propose",
                "method": "Try the help command",
                "target": "g1",
                "expected_improvement": "Correct CLI explanation",
                "verification": "Execute command",
                "evidence_refs": ["task"],
            },
        )
        write(root, folder / "hypothesis_map.json", mapping())
        write(
            root,
            folder / "arm_protocol.json",
            {"id": "legacy_abcd", "hypothesis_applied": True},
        )
        construction = [measured(pack, observed, defect=True) for observed in pool]
        write(root, folder / "construction_checks.json", construction)
        checks = [
            check_bundle(
                f"product_{i}", m, f"round_{number}/construction_checks.json#{i - 1}"
            )
            for i, m in enumerate(construction, 1)
        ]
        arms, official = {}, {}
        for arm in "ABC":
            row, grade = native(
                root,
                folder / "actors" / arm,
                benchmark,
                pack=pack if arm == "B" else None,
                targeted=targeted,
                review_turns=1
                if number in no_revision_rounds
                else 3
                if number in unresolved_rounds
                else 2,
                unchanged=number in unchanged_rounds,
                unresolved=number in unresolved_rounds,
                inquiries=inquiries,
            )
            row["hypothesis_applied"] = arm == "C"
            arms[arm], official[arm] = row, grade
            write(root, folder / arm / "delivery.json", row)
            estimate = (
                row["final_measurement"]
                if arm == "B"
                else measured(pack, row["candidate"])
            )
            write(root, folder / arm / "outcome_estimate.json", estimate)
            pool.append(row["candidate"])
            checks.append(
                check_bundle(
                    f"product_{len(pool)}",
                    estimate,
                    f"round_{number}/{arm}/outcome_estimate.json",
                )
            )
        products = [f"product_{i}" for i in range(1, len(pool) + 1)]
        rank = {
            "status": "compared",
            "groups": [products],
            "unknown": [],
            "reason": "Equivalent supported explanations",
            "observations": [
                {
                    "product_id": p,
                    "reason": "Checked the explanation",
                    "evidence_refs": [p + ":artifact:final_response"],
                }
                for p in products
            ],
        }
        chosen = {
            "arm": "A",
            "round": number,
            "product_id": f"product_{number * 3 + 1}",
            "source": f"round_{number}/A/delivery.json",
            "reason": "No supported improvement over A",
        }
        write(root, folder / "ranking.json", rank)
        write(root, folder / "ranking_map.json", mapping())
        write(root, folder / "selected_delivery.json", chosen)
        decisions.append(
            {
                "round": number,
                "decision": "continue" if number < 3 else "stop",
                "reason": "configured_fixed_rounds" if number < 3 else "round_limit",
            }
        )
        memory.append(
            {
                "memory_id": f"memory-{number}",
                "task_id": TASK["task_id"],
                "key": "Observed CLI gap",
                "value": {
                    "source": f"round_{number}/jury.json",
                    "status": "jury_observation_not_verified_method",
                },
            }
        )
        write(root, folder / "memory.json", memory)
        checkpoint = {
            "task": TASK,
            "completed_round": number,
            "pool": pool,
            "memory": memory,
            "checks": checks,
            "previous_pack": pack,
            "expectation_cache": {},
            "known_gaps": [],
            "decisions": decisions,
        }
        write(root, folder / "checkpoint.json", checkpoint)
        summary = {
            "status": "completed",
            "round": number,
            "pool_size": len(pool),
            "arms": arms,
            "official": official,
            "ranking": rank,
            "selection": chosen,
        }
        summaries.append(summary)
        write(root, folder / "summary.private.json", summary)
    write(
        root,
        TASK_DIR / "result.private.json",
        {
            "status": "completed",
            "completed_rounds": 3,
            "rounds": summaries,
            "selected_delivery": chosen,
            "memory": memory,
        },
    )
    return root


def stage(report, name):
    return next(c for c in report["checks"] if c["stage"] == name)


def supplement(root, number=1, arm="B", benchmark="claw_eval"):
    folder = TASK_DIR / f"round_{number}"
    slot = folder / "actors" / arm
    native = read(root, slot / "RESULT.json")
    turn = read(root, slot / "actor_state.json")["turn"]
    snapshot = (
        slot
        / "snapshots"
        / (str(turn) if benchmark == "claw_eval" else f"delivery-{turn}")
        / "candidate.json"
    )
    pack = read(root, folder / "pack.json")
    receipt = {
        "schema": "pipeline-supplemental-measurement-v1",
        "source": snapshot.as_posix(),
        "pack_source": (folder / "pack.json").as_posix(),
        "artifact_sha256": receipt_digest(native["candidate"]["artifact"]),
        "candidate_sha256": receipt_digest(native["candidate"]),
        "pack_sha256": receipt_digest(pack),
        "provenance": {
            "kind": "supplemental_full_measurement",
            "producer": "separate-full-harness-audit",
            "created_at": "2026-10-10T01:00:00Z",
        },
        "measurement": measured(pack, native["candidate"]),
    }
    path = folder / arm / "outcome_estimate.full.supplemental.json"
    write(root, path, receipt)
    return path


@pytest.mark.parametrize("benchmark", ["claw_eval", "workbuddy"])
@pytest.mark.parametrize("targeted", [False, True])
def test_complete_native_run_preserves_raw_scores_and_scope(
    tmp_path, benchmark, targeted
):
    make_run(tmp_path, benchmark, targeted=targeted)
    if targeted:
        for number in range(1, 4):
            supplement(tmp_path, number, benchmark=benchmark)
    report = audit_run(tmp_path)
    assert report["status"] == "pass", report["coordination"]
    assert report["counts"] == {
        "completed_rounds": 3,
        "native_seeds": 3,
        "native_actors": 9,
        "native_graded_outputs": 9,
        "validated_review_rounds": 3,
        "validated_continuation_rounds": 3,
        "validated_revision_rounds": 3,
        "full_delivered_pack_measurements": 9,
        "supplemental_delivered_pack_measurements": 3 if targeted else 0,
    }
    assert report["claims"]["original_run_full_delivered_measurements"] is not targeted
    assert report["claims"]["same_session_revision_validated"]
    assert report["claims"]["hypothesis_proposals_exercised"] == 3
    assert not report["claims"]["quality_improvement_validated"]
    assert not report["claims"]["official_leaderboard_equivalence_validated"]
    score = stage(report, "round_1/actors/A/native_grade")
    assert score["facts"]["raw_native_output"] == read(
        tmp_path, TASK_DIR / "round_1/actors/A/OFFICIAL.private.json"
    )
    assert "normalization" in report["scope"]["scores"]


def test_deterministic_read_only_and_cli(tmp_path, capsys):
    make_run(tmp_path)
    before = {
        p: (p.read_bytes(), p.stat().st_mtime_ns)
        for p in tmp_path.rglob("*")
        if p.is_file()
    }
    report = audit_run(tmp_path)
    assert report == audit_run(tmp_path)
    assert main([str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out) == report
    assert before == {
        p: (p.read_bytes(), p.stat().st_mtime_ns)
        for p in tmp_path.rglob("*")
        if p.is_file()
    }


@pytest.mark.parametrize(
    ("receipt", "change", "failed_stage"),
    [
        ("jury.json", lambda r: r.update(criteria=[]), "jury"),
        ("jury.json", lambda r: r["criteria"][0].update(standard=" "), "jury"),
        ("jury_map.json", lambda r: r.update(items=[], total=0, succeeded=0), "jury"),
        ("pack.json", lambda r: r.update(tests=[]), "builder_and_smoke"),
        (
            "pack.json",
            lambda r: r["tests"][0].update(binding_status="needs_implementation"),
            "builder_and_smoke",
        ),
        (
            "construction_smoke.json",
            lambda r: r.update(status="binding_only", receipt=None),
            "builder_and_smoke",
        ),
        (
            "hypothesis.json",
            lambda r: r.update(status="keep", availability="unavailable"),
            "hypothesis",
        ),
        ("hypothesis.json", lambda r: r.update(method=""), "hypothesis"),
        (
            "hypothesis_map.json",
            lambda r: r["items"][0].update(status="timeout"),
            "hypothesis",
        ),
        ("construction_checks.json", lambda r: r.clear(), "construction_measurements"),
        (
            "construction_checks.json",
            lambda r: r[0]["results"].clear(),
            "construction_measurements",
        ),
        (
            "construction_checks.json",
            lambda r: r[0]["results"][0].update(status="unknown", error=True),
            "construction_measurements",
        ),
        (
            "construction_checks.json",
            lambda r: r[0]["results"][0].update(status="unknown", error=False),
            "construction_measurements",
        ),
        (
            "actors/B/RESULT.json",
            lambda r: r.update(review_status="disabled"),
            "B_review_execution",
        ),
        (
            "actors/B/RESULT.json",
            lambda r: r.update(review_status="check_unavailable"),
            "B_review_execution",
        ),
        (
            "actors/B/checks/feedback_1.json",
            lambda r: r.update(message=""),
            "B_review_execution",
        ),
        ("actors/B/actor_state.json", lambda r: r.update(turn=1), "B_review_execution"),
        (
            "ranking.json",
            lambda r: r.update(status="unavailable"),
            "ranking_and_selection",
        ),
        ("ranking.json", lambda r: r.update(groups=[]), "ranking_and_selection"),
        ("ranking.json", lambda r: r.update(observations=[]), "ranking_and_selection"),
        (
            "selected_delivery.json",
            lambda r: r.update(arm=None),
            "ranking_and_selection",
        ),
        (
            "selected_delivery.json",
            lambda r: r.update(source="round_2/A/delivery.json"),
            "ranking_and_selection",
        ),
        ("memory.json", lambda r: r.clear(), "memory_and_checkpoint"),
        ("checkpoint.json", lambda r: r["pool"].pop(), "memory_and_checkpoint"),
        ("summary.private.json", lambda r: r.update(status="partial"), "completed"),
    ],
)
def test_completed_status_cannot_mask_failed_or_inactive_stages(
    tmp_path, receipt, change, failed_stage
):
    make_run(tmp_path)
    edit(tmp_path, TASK_DIR / "round_1" / receipt, change)
    report = audit_run(tmp_path)
    assert report["status"] == "fail"
    assert stage(report, "round_1/" + failed_stage)["status"] == "fail"
    assert not report["claims"]["full_pipeline_receipts_validated"]


def test_semantic_keep_is_not_unavailable_or_proposal_coverage(tmp_path):
    make_run(tmp_path)
    write(
        tmp_path,
        TASK_DIR / "round_1/hypothesis.json",
        {"status": "keep", "method": "", "clarifications": []},
    )
    edit(
        tmp_path,
        TASK_DIR / "round_1/arm_protocol.json",
        lambda r: r.update(hypothesis_applied=False),
    )
    report = audit_run(tmp_path)
    assert stage(report, "round_1/hypothesis")["status"] == "pass"
    assert report["claims"]["semantic_keep_rounds"] == 1
    assert report["claims"]["hypothesis_proposals_exercised"] == 2


def test_three_completed_rounds_are_required(tmp_path):
    make_run(tmp_path)
    edit(
        tmp_path,
        TASK_DIR / "result.private.json",
        lambda r: r.update(completed_rounds=1),
    )
    assert stage(audit_run(tmp_path), "three_completed_rounds")["status"] == "fail"


def test_process_success_alone_is_not_a_validation(tmp_path):
    write(
        tmp_path,
        "validation_status.json",
        {"status": "execution_completed", "completed_rounds": 3},
    )
    report = audit_run(tmp_path)
    assert report["status"] == "fail"
    assert report["counts"]["native_actors"] == 0


def test_empty_live_tree_pending_and_updates_as_receipts_arrive(tmp_path, capsys):
    assert audit_run(tmp_path)["status"] == "pending"
    assert main([str(tmp_path)]) == 2
    capsys.readouterr()
    make_run(tmp_path)
    saved = read(tmp_path, TASK_DIR / "result.private.json")
    (tmp_path / TASK_DIR / "result.private.json").unlink()
    write(tmp_path, "validation_status.json", {"status": "running"})
    report = audit_run(tmp_path)
    assert report["status"] == "pending"
    assert report["counts"]["native_actors"] == 9
    write(tmp_path, TASK_DIR / "result.private.json", saved)
    write(
        tmp_path,
        "validation_status.json",
        {"status": "execution_completed", "completed_rounds": 3},
    )
    assert audit_run(tmp_path)["status"] == "pass"


def test_partial_json_pending_only_until_producer_completed(tmp_path):
    make_run(tmp_path)
    write(tmp_path, "validation_status.json", {"status": "running"})
    (tmp_path / TASK_DIR / "result.private.json").unlink()
    (tmp_path / TASK_DIR / "round_3/summary.private.json").unlink()
    target = tmp_path / TASK_DIR / "round_3/hypothesis.json"
    target.write_text('{"status":')
    assert stage(audit_run(tmp_path), "round_3/hypothesis")["status"] == "pending"
    write(tmp_path, TASK_DIR / "round_3/summary.private.json", {"status": "completed"})
    assert stage(audit_run(tmp_path), "round_3/hypothesis")["status"] == "fail"


def test_explicit_failure_fails_immediately_during_live_run(tmp_path):
    make_run(tmp_path)
    write(tmp_path, "validation_status.json", {"status": "running"})
    write(
        tmp_path,
        TASK_DIR / "round_1/construction_failure.json",
        {"stage": "builder", "error": "failed"},
    )
    report = audit_run(tmp_path)
    assert stage(report, "round_1/construction_failure")["status"] == "fail"
    assert (
        stage(report, "round_1/construction_failure")["facts"]["failure"]["stage"]
        == "builder"
    )
    assert report["status"] == "fail"


def test_same_session_revision_requires_native_turn_not_only_review_json(tmp_path):
    make_run(tmp_path)
    path = tmp_path / TASK_DIR / "round_1/actors/B/actor-2-0.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    for event in events:
        event["sessionID"] = "different-session"
    write(tmp_path, path, events, lines=True)
    report = audit_run(tmp_path)
    assert stage(report, "round_1/B_review_execution")["status"] == "fail"
    assert not report["claims"]["same_session_revision_validated"]


def test_same_session_unchanged_artifact_does_not_validate_revision(tmp_path):
    make_run(tmp_path, unchanged_rounds=(1,))
    report = audit_run(tmp_path)
    result = stage(report, "round_1/B_review_execution")
    assert report["status"] == "pass", report["coordination"]
    assert result["status"] == "pass"
    assert result["facts"]["continuation_exercised"]
    assert not result["facts"]["changed_artifact_revision_exercised"]
    assert report["coverage"]["same_session_continuation_exercised_rounds"] == [1, 2, 3]
    assert report["coverage"]["changed_artifact_revision_exercised_rounds"] == [2, 3]
    assert not report["claims"]["same_session_revision_validated"]


def test_duplicate_actor_and_seed_identities_are_rejected(tmp_path):
    make_run(tmp_path)
    original = read(tmp_path, TASK_DIR / "seeds/1/RESULT.json")
    write(tmp_path, TASK_DIR / "round_1/actors/A/RESULT.json", original)
    report = audit_run(tmp_path)
    assert stage(report, "distinct_native_samples")["status"] == "fail"


def test_memory_must_be_retrieved_from_prior_checkpoint(tmp_path):
    make_run(tmp_path)
    edit(tmp_path, TASK_DIR / "round_2/context.json", lambda r: r.update(memories=[]))
    assert (
        stage(audit_run(tmp_path), "round_2/memory_and_checkpoint")["status"] == "fail"
    )


@pytest.mark.parametrize("score", [None, True, "0.9", float("nan"), float("inf")])
@pytest.mark.parametrize("benchmark", ["claw_eval", "workbuddy"])
def test_invalid_native_numbers_are_never_valid_scores(tmp_path, benchmark, score):
    make_run(tmp_path, benchmark)
    key = "task_score" if benchmark == "claw_eval" else "overall"
    edit(
        tmp_path,
        TASK_DIR / "round_1/actors/A/OFFICIAL.private.json",
        lambda r: r.update({key: score}),
    )
    assert (
        stage(audit_run(tmp_path), "round_1/actors/A/native_grade")["status"] == "fail"
    )


def test_native_grade_status_alone_and_artifact_grading_do_not_pass(tmp_path):
    make_run(tmp_path)
    path = TASK_DIR / "round_1/actors/A/OFFICIAL.private.json"
    edit(tmp_path, path, lambda r: r.update(protocol="outcome-artifact-quality"))
    assert (
        stage(audit_run(tmp_path), "round_1/actors/A/native_grade")["status"] == "fail"
    )
    write(tmp_path, path, {"status": "graded", "task_score": 1.0})
    assert (
        stage(audit_run(tmp_path), "round_1/actors/A/native_grade")["status"] == "fail"
    )


def test_grader_hashes_bind_scores_to_native_artifacts(tmp_path):
    make_run(tmp_path)
    write(
        tmp_path,
        TASK_DIR / "round_1/actors/A/environment-snapshot.json",
        {"changed": True},
    )
    assert (
        stage(audit_run(tmp_path), "round_1/actors/A/native_grade")["status"] == "fail"
    )


def test_workbuddy_native_error_cannot_masquerade_as_zero(tmp_path):
    make_run(tmp_path, "workbuddy")
    slot = TASK_DIR / "round_1/actors/A"
    cid = read(tmp_path, slot / "RESULT.json")["candidate"]["candidate_id"]
    path = slot / "harbor" / cid / "verifier/score.json"
    edit(
        tmp_path,
        path,
        lambda r: r["verdicts"][0].update(metadata={"error_type": "JudgeTimeout"}),
    )
    report = audit_run(tmp_path)
    grade = stage(report, "round_1/actors/A/native_grade")
    assert grade["status"] == "fail"
    assert grade["facts"]["raw_native_output"]["raw_overall"] == 0.0
    assert (
        grade["facts"]["raw_native_score"]["verdicts"][0]["metadata"]["error_type"]
        == "JudgeTimeout"
    )


@pytest.mark.parametrize("bad", [None, [], 1, "completed", {"criteria": [None]}])
def test_malformed_schema_returns_report_not_exception(tmp_path, bad):
    make_run(tmp_path)
    write(tmp_path, TASK_DIR / "round_1/jury.json", bad)
    assert stage(audit_run(tmp_path), "round_1/jury")["status"] == "fail"


def test_evidence_symlink_outside_run_rejected(tmp_path):
    root = tmp_path / "run"
    make_run(root)
    path = root / TASK_DIR / "round_1/jury.json"
    external = tmp_path / "external.json"
    external.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(external)
    assert stage(audit_run(root), "round_1/jury")["status"] == "fail"


def test_closed_actor_missing_log_fails_while_rest_of_run_is_live(tmp_path):
    make_run(tmp_path)
    write(tmp_path, "validation_status.json", {"status": "running"})
    (tmp_path / TASK_DIR / "result.private.json").unlink()
    (tmp_path / TASK_DIR / "seeds/1/actor-1-0.jsonl").unlink()
    assert stage(audit_run(tmp_path), "seeds/1/native_actor")["status"] == "fail"


def test_failed_native_grade_preserves_raw_output(tmp_path):
    make_run(tmp_path)
    row = {"status": "unavailable", "reason": "worker_failed", "task_score": None}
    write(tmp_path, TASK_DIR / "round_1/actors/C/OFFICIAL.private.json", row)
    grade = stage(audit_run(tmp_path), "round_1/actors/C/native_grade")
    assert grade["status"] == "fail"
    assert grade["facts"]["raw_native_output"] == row


def test_missing_grade_does_not_borrow_another_arm_score(tmp_path):
    make_run(tmp_path)
    (tmp_path / TASK_DIR / "round_3/actors/C/OFFICIAL.private.json").unlink()
    report = audit_run(tmp_path)
    assert report["counts"]["native_graded_outputs"] == 8
    assert report["status"] == "fail"


@pytest.mark.parametrize("live", [False, True])
def test_same_pack_id_partial_b_measurement_fails_even_while_live(tmp_path, live):
    make_run(tmp_path, targeted=True)
    if live:
        write(tmp_path, "validation_status.json", {"status": "running"})
        (tmp_path / TASK_DIR / "result.private.json").unlink()
        (tmp_path / TASK_DIR / "round_3/summary.private.json").unlink()
    report = audit_run(tmp_path)
    assert stage(report, "round_3/B/delivered_pack_measurement")["status"] == "fail"
    assert stage(report, "round_3/B_review_execution")["status"] == "pass"
    assert report["counts"]["full_delivered_pack_measurements"] == 6
    assert report["status"] == "fail"


@pytest.mark.parametrize("benchmark", ["claw_eval", "workbuddy"])
def test_bound_supplement_is_explicit_and_does_not_edit_primary(tmp_path, benchmark):
    make_run(tmp_path, benchmark, targeted=True)
    primary = tmp_path / TASK_DIR / "round_1/B/outcome_estimate.json"
    original = primary.read_bytes()
    supplement(tmp_path, benchmark=benchmark)
    report = audit_run(tmp_path)
    result = stage(report, "round_1/B/delivered_pack_measurement")
    assert result["status"] == "pass", result
    assert result["facts"]["provenance"] == "supplemental"
    assert not result["facts"]["primary_full_pack_measurement"]
    assert "primary_measurement_issue" in result["facts"]
    assert (
        result["facts"]["supplemental_provenance"]["producer"]
        == "separate-full-harness-audit"
    )
    assert report["counts"]["supplemental_delivered_pack_measurements"] == 1
    assert primary.read_bytes() == original
    # One supplement cannot validate other incomplete rounds.
    assert report["status"] == "fail"


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("source", "tasks/task-one/round_1/actors/B/snapshots/1/candidate.json"),
        ("source", "tasks/task-one/round_1/actors/A/snapshots/1/candidate.json"),
        ("pack_source", "tasks/task-one/round_2/pack.json"),
        ("artifact_sha256", "wrong"),
        ("candidate_sha256", "wrong"),
        ("pack_sha256", "wrong"),
        ("provenance", {}),
        (
            "provenance",
            {"kind": "original_run", "producer": "runner", "created_at": "2026-10-10"},
        ),
        ("schema", "other"),
        ("measurement", {"pack_id": "pack-1", "results": []}),
    ],
)
def test_supplements_require_exact_bindings_and_full_execution(tmp_path, key, value):
    make_run(tmp_path, targeted=True)
    path = supplement(tmp_path)
    edit(tmp_path, path, lambda r: r.update({key: value}))
    result = stage(audit_run(tmp_path), "round_1/B/delivered_pack_measurement")
    assert result["status"] == "fail"


def test_same_pack_id_changed_pack_rejects_old_supplement(tmp_path):
    make_run(tmp_path, targeted=True)
    supplement(tmp_path)
    edit(
        tmp_path,
        TASK_DIR / "round_1/pack.json",
        lambda r: r["criteria"][0].update(
            standard="Different criterion with the same pack ID"
        ),
    )
    result = stage(audit_run(tmp_path), "round_1/B/delivered_pack_measurement")
    assert result["status"] == "fail"
    assert "pack_sha256" in result["reason"]


def test_supplement_cannot_validate_changed_source_artifact(tmp_path):
    make_run(tmp_path, targeted=True)
    supplement(tmp_path)
    edit(
        tmp_path,
        TASK_DIR / "round_1/actors/B/snapshots/2/candidate.json",
        lambda r: r["artifact"].update(final_response="new ungraded output"),
    )
    assert (
        stage(audit_run(tmp_path), "round_1/B/delivered_pack_measurement")["status"]
        == "fail"
    )


def test_digest_covers_unicode_and_is_order_independent():
    assert receipt_digest({"b": "证据", "a": 1}) == receipt_digest(
        {"a": 1, "b": "证据"}
    )
    expected = hashlib.sha256('{"a":1,"b":"证据"}'.encode()).hexdigest()
    assert receipt_digest({"b": "证据", "a": 1}) == expected


@pytest.mark.parametrize("benchmark", ["claw_eval", "workbuddy"])
def test_final_candidate_source_helper_checks_native_equality(tmp_path, benchmark):
    make_run(tmp_path, benchmark)
    slot = TASK_DIR / "round_1/actors/B"
    result = final_candidate_source(tmp_path, slot, benchmark)
    leaf = "2" if benchmark == "claw_eval" else "delivery-2"
    assert result == slot / "snapshots" / leaf / "candidate.json"
    edit(
        tmp_path,
        result,
        lambda r: r["artifact"].update(final_response="not the delivered artifact"),
    )
    with pytest.raises(ValueError, match="exact fresh-run"):
        final_candidate_source(tmp_path, slot, benchmark)


def supplemental_optional_provenance(root):
    folder = TASK_DIR / "round_1"
    delivery = read(root, folder / "B/delivery.json")
    pack = read(root, folder / "pack.json")
    hashes = {
        str(root / path): hashlib.sha256((root / path).read_bytes()).hexdigest()
        for path in (
            folder / "B/delivery.json",
            folder / "pack.json",
            folder / "checkpoint.json",
        )
    }
    return {
        "input_binding": digest(
            {"task": TASK, "candidate": delivery["candidate"], "pack": pack}
        ),
        "source_hashes": hashes,
        "code_hashes": {"reward_engine/core/outcome/harness.py": "a" * 64},
    }


def test_optional_whole_input_and_source_bindings_are_verified(tmp_path):
    make_run(tmp_path, targeted=True)
    path = supplement(tmp_path)
    provenance = supplemental_optional_provenance(tmp_path)
    edit(tmp_path, path, lambda r: r.update(provenance))
    report = audit_run(tmp_path)
    result = stage(report, "round_1/B/delivered_pack_measurement")
    assert result["status"] == "pass", result
    for key, value in provenance.items():
        assert result["facts"][key] == value


@pytest.mark.parametrize("key", ["input_binding", "source_hashes", "code_hashes"])
def test_bad_optional_provenance_fails_instead_of_being_ignored(tmp_path, key):
    make_run(tmp_path, targeted=True)
    path = supplement(tmp_path)
    provenance = supplemental_optional_provenance(tmp_path)
    provenance[key] = "wrong" if key == "input_binding" else {}
    edit(tmp_path, path, lambda r: r.update(provenance))
    assert (
        stage(audit_run(tmp_path), "round_1/B/delivered_pack_measurement")["status"]
        == "fail"
    )


def test_source_hashes_detect_changed_checkpoint_with_same_measurement_inputs(tmp_path):
    make_run(tmp_path, targeted=True)
    path = supplement(tmp_path)
    provenance = supplemental_optional_provenance(tmp_path)
    edit(tmp_path, path, lambda r: r.update(provenance))
    edit(
        tmp_path,
        TASK_DIR / "round_1/checkpoint.json",
        lambda r: r.update(extra="changed"),
    )
    result = stage(audit_run(tmp_path), "round_1/B/delivered_pack_measurement")
    assert result["status"] == "fail"
    assert "source file hash" in result["reason"]


def test_all_nine_supplements_are_audited_even_with_complete_originals(tmp_path):
    make_run(tmp_path)
    for number in range(1, 4):
        for arm in "ABC":
            supplement(tmp_path, number, arm)
    report = audit_run(tmp_path)
    assert report["status"] == "pass", report["coordination"]
    assert report["counts"]["supplemental_delivered_pack_measurements"] == 9
    assert report["claims"]["original_run_full_delivered_measurements"]
    path = TASK_DIR / "round_1/A/outcome_estimate.full.supplemental.json"
    edit(
        tmp_path,
        path,
        lambda r: r["measurement"]["results"][0].update(status="unknown", error=True),
    )
    assert (
        stage(audit_run(tmp_path), "round_1/A/delivered_pack_measurement")["status"]
        == "fail"
    )


def test_ranking_can_cite_real_product_bound_check_receipts(tmp_path):
    make_run(tmp_path)
    folder = TASK_DIR / "round_1"
    checks = read(tmp_path, folder / "checkpoint.json")["checks"]
    write(tmp_path, folder / "check_sources.json", checks)
    edit(
        tmp_path,
        folder / "ranking.json",
        lambda r: r["observations"][0].update(evidence_refs=["check:0:0"]),
    )
    assert (
        stage(audit_run(tmp_path), "round_1/ranking_and_selection")["status"] == "pass"
    )
    edit(
        tmp_path,
        folder / "ranking.json",
        lambda r: r["observations"][0].update(evidence_refs=["check:1:0"]),
    )
    assert (
        stage(audit_run(tmp_path), "round_1/ranking_and_selection")["status"] == "fail"
    )


def test_optional_whole_input_binding_includes_effective_task(tmp_path):
    make_run(tmp_path, targeted=True)
    path = supplement(tmp_path)
    provenance = supplemental_optional_provenance(tmp_path)
    edit(tmp_path, path, lambda r: r.update(input_binding=provenance["input_binding"]))
    edit(
        tmp_path,
        TASK_DIR / "round_1/B/delivery.json",
        lambda r: r.update(
            effective_task={**TASK, "prompt": "Changed user requirements"},
            effective_pack=read(tmp_path, TASK_DIR / "round_1/pack.json"),
        ),
    )
    result = stage(audit_run(tmp_path), "round_1/B/delivered_pack_measurement")
    assert result["status"] == "fail"
    assert "whole-input" in result["reason"]


@pytest.mark.parametrize("benchmark", ["claw_eval", "workbuddy"])
@pytest.mark.parametrize("pack_override", ["absent", None, {}, "explicit"])
def test_effective_task_pack_resolution_matches_core_and_producer(
    tmp_path, benchmark, pack_override
):
    make_run(tmp_path, benchmark, targeted=True)
    folder = TASK_DIR / "round_1"
    original_pack = read(tmp_path, folder / "pack.json")
    effective_task = (
        {**TASK, "followup_id": "new-user-event"}
        if pack_override == "explicit"
        else {**TASK, "prompt": "Explain the updated CLI requirements"}
    )
    delivery = read(tmp_path, folder / "B/delivery.json")
    delivery["effective_task"] = effective_task
    if pack_override == "explicit":
        # Even an explicit pack equal to the round pack must not be rebased:
        # core/producer check fallback object identity, not dictionary equality.
        delivery["effective_pack"] = original_pack
        effective_pack = original_pack
    else:
        if pack_override != "absent":
            delivery["effective_pack"] = pack_override
        effective_pack = TaskSequence(effective_task).rebase(original_pack)
        assert effective_pack["pack_id"] != original_pack["pack_id"]
        assert effective_pack["task_prompt"] == effective_task["prompt"]
    write(tmp_path, folder / "B/delivery.json", delivery)
    path = supplement(tmp_path, benchmark=benchmark)
    provenance = supplemental_optional_provenance(tmp_path)
    provenance["input_binding"] = digest(
        {
            "task": effective_task,
            "candidate": delivery["candidate"],
            "pack": effective_pack,
        }
    )
    edit(
        tmp_path,
        path,
        lambda r: r.update(
            **provenance,
            pack_sha256=receipt_digest(effective_pack),
            measurement=measured(
                effective_pack, delivery["candidate"], task=effective_task
            ),
        ),
    )
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    result = stage(audit_run(tmp_path), "round_1/B/delivered_pack_measurement")
    assert result["status"] == "pass", result
    assert result["facts"]["input_binding"] == provenance["input_binding"]
    assert result["facts"]["binding"]["pack_sha256"] == receipt_digest(effective_pack)
    assert result["facts"]["code_hashes"] == provenance["code_hashes"]
    assert result["facts"]["source_hashes"] == provenance["source_hashes"]
    assert not result["facts"]["primary_full_pack_measurement"]
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    if pack_override != "explicit":
        # A correct binding cannot excuse a measurement of the old task scope.
        edit(
            tmp_path,
            path,
            lambda r: r.update(
                measurement=measured(original_pack, delivery["candidate"])
            ),
        )
        result = stage(audit_run(tmp_path), "round_1/B/delivered_pack_measurement")
        assert result["status"] == "fail"
        assert "another pack" in result["reason"]


def test_changed_task_rejects_supplement_bound_to_unrebased_pack(tmp_path):
    make_run(tmp_path, targeted=True)
    folder = TASK_DIR / "round_1"
    supplement(tmp_path)
    edit(
        tmp_path,
        folder / "B/delivery.json",
        lambda r: r.update(effective_task={**TASK, "prompt": "New requirements"}),
    )
    result = stage(audit_run(tmp_path), "round_1/B/delivered_pack_measurement")
    assert result["status"] == "fail"
    assert "pack_sha256" in result["reason"]


def test_original_delivered_measurement_uses_rebased_pack_too(tmp_path):
    make_run(tmp_path)
    folder = TASK_DIR / "round_1"
    effective_task = {**TASK, "prompt": "New requirements"}
    pack = TaskSequence(effective_task).rebase(read(tmp_path, folder / "pack.json"))
    edit(
        tmp_path,
        folder / "A/delivery.json",
        lambda r: r.update(effective_task=effective_task),
    )
    assert (
        stage(audit_run(tmp_path), "round_1/A/delivered_pack_measurement")["status"]
        == "fail"
    )
    write(
        tmp_path,
        folder / "A/outcome_estimate.json",
        measured(
            pack,
            read(tmp_path, folder / "A/delivery.json")["candidate"],
            task=effective_task,
        ),
    )
    result = stage(audit_run(tmp_path), "round_1/A/delivered_pack_measurement")
    assert result["status"] == "pass", result
    assert result["facts"]["primary_full_pack_measurement"]


def test_partial_delivery_fails_before_live_checkpoint_arrives(tmp_path):
    make_run(tmp_path, targeted=True)
    write(tmp_path, "validation_status.json", {"status": "running"})
    (tmp_path / TASK_DIR / "result.private.json").unlink()
    folder = tmp_path / TASK_DIR / "round_3"
    (folder / "checkpoint.json").unlink()
    (folder / "summary.private.json").unlink()
    result = stage(audit_run(tmp_path), "round_3/B/delivered_pack_measurement")
    assert result["status"] == "fail"
    assert "Incomplete/failed delivered pack measurement" in result["reason"]


@pytest.mark.parametrize("benchmark", ["claw_eval", "workbuddy"])
@pytest.mark.parametrize(
    "damage",
    [
        "alter_bytes",
        "delete_file",
        "delete_inventory",
        "delete_file_and_inventory",
        "delete_workspace",
        "add_file",
        "symlink_file",
        "unsafe_path",
        "wrong_size",
        "wrong_hash",
        "omit_file",
    ],
)
def test_retained_native_bytes_required_by_audit_and_supplement_helper(
    tmp_path, benchmark, damage
):
    make_run(tmp_path, benchmark)
    slot = TASK_DIR / "round_1/actors/A"
    snapshot = final_candidate_source(tmp_path, slot, benchmark)
    directory = tmp_path / snapshot.parent
    inventory = directory / "inventory.json"
    file = directory / "workspace/deliverable.txt"
    originals = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    if damage == "alter_bytes":
        file.write_bytes(b"!" * file.stat().st_size)
    elif damage in {"delete_file", "delete_file_and_inventory"}:
        file.unlink()
        if damage == "delete_file_and_inventory":
            inventory.unlink()
    elif damage == "delete_inventory":
        inventory.unlink()
    elif damage == "delete_workspace":
        shutil.rmtree(directory / "workspace")
    elif damage == "add_file":
        (directory / "workspace/unrecorded.txt").write_text("not collected")
    elif damage == "symlink_file":
        outside = tmp_path / "different-slot.txt"
        outside.write_bytes(file.read_bytes())
        file.unlink()
        file.symlink_to(outside)
    else:
        rows = json.loads(inventory.read_text())
        if damage == "unsafe_path":
            rows[0]["path"] = "../deliverable.txt"
        elif damage == "wrong_size":
            rows[0]["bytes"] += 1
        elif damage == "wrong_hash":
            rows[0]["sha256"] = "0" * 64
        elif damage == "omit_file":
            rows.clear()
        inventory.write_text(json.dumps(rows))
    report = audit_run(tmp_path)
    assert report["status"] == "fail"
    assert stage(report, "round_1/actors/A/native_actor")["status"] == "fail"
    assert stage(report, "round_1/A/delivered_pack_measurement")["status"] == "fail"
    with pytest.raises((ValueError, IncompleteReceipt)):
        final_candidate_source(tmp_path, slot, benchmark)
    # Candidate, RESULT, grading and learning JSON stay byte-for-byte identical.
    assert all(
        path == inventory or path.read_bytes() == value
        for path, value in originals.items()
    )


@pytest.mark.parametrize("benchmark", ["claw_eval", "workbuddy"])
def test_recorded_supplement_cannot_hide_later_workspace_damage(tmp_path, benchmark):
    make_run(tmp_path, benchmark, targeted=True)
    supplement(tmp_path, benchmark=benchmark)
    slot = TASK_DIR / "round_1/actors/B"
    snapshot = final_candidate_source(tmp_path, slot, benchmark)
    (tmp_path / snapshot.parent / "workspace/deliverable.txt").unlink()
    result = stage(audit_run(tmp_path), "round_1/B/delivered_pack_measurement")
    assert result["status"] == "fail"


def test_claw_true_fresh_executions_share_protocol_identity(tmp_path):
    make_run(tmp_path, "claw_eval")
    seeds = [
        read(tmp_path, TASK_DIR / "seeds" / str(n) / "RESULT.json") for n in range(1, 4)
    ]
    assert len({s["identity"] for s in seeds}) == 1
    assert len({s["session"] for s in seeds}) == 3
    assert len({s["candidate"]["candidate_id"] for s in seeds}) == 3
    assert (
        read(tmp_path, TASK_DIR / "round_1/actors/A/RESULT.json")["identity"]
        == seeds[0]["identity"]
    )
    report = audit_run(tmp_path)
    assert report["status"] == "pass", report["coordination"]
    assert stage(report, "distinct_native_samples")["status"] == "pass"


@pytest.mark.parametrize("field", ["session", "candidate_id"])
def test_shared_claw_protocol_does_not_excuse_reused_native_identity(tmp_path, field):
    make_run(tmp_path, "claw_eval")
    first = read(tmp_path, TASK_DIR / "seeds/1/RESULT.json")

    def reuse(row):
        if field == "session":
            row["session"] = first["session"]
        else:
            row["candidate"]["candidate_id"] = first["candidate"]["candidate_id"]

    edit(tmp_path, TASK_DIR / "seeds/2/RESULT.json", reuse)
    assert stage(audit_run(tmp_path), "distinct_native_samples")["status"] == "fail"


def test_wrong_claw_protocol_start_identity_still_fails(tmp_path):
    make_run(tmp_path)
    edit(
        tmp_path,
        TASK_DIR / "seeds/1/STARTED.json",
        lambda row: row.update(identity="wrong-protocol"),
    )
    assert stage(audit_run(tmp_path), "seeds/1/native_actor")["status"] == "fail"


def test_other_actor_pre_revision_measurement_cannot_replace_primary(tmp_path):
    make_run(tmp_path, "workbuddy")
    folder = TASK_DIR / "round_1"
    other = read(tmp_path, folder / "actors/B/checks/delivery_1.json")["measurement"]
    write(tmp_path, folder / "A/outcome_estimate.json", other)
    report = audit_run(tmp_path)
    result = stage(report, "round_1/A/delivered_pack_measurement")
    assert result["status"] == "fail"
    assert "input fingerprint" in result["reason"]
    assert not report["claims"]["original_run_full_delivered_measurements"]


@pytest.mark.parametrize("change", ["duplicate", "reorder", "unbound"])
def test_construction_measurements_bind_each_ordered_incoming_candidate(
    tmp_path, change
):
    make_run(tmp_path)
    path = TASK_DIR / "round_2/construction_checks.json"
    rows = read(tmp_path, path)
    if change == "duplicate":
        rows[1] = rows[0]
    elif change == "reorder":
        rows[0], rows[1] = rows[1], rows[0]
    else:
        rows[0].pop("measurement_input_sha256")
    write(tmp_path, path, rows)
    result = stage(audit_run(tmp_path), "round_2/construction_measurements")
    assert result["status"] == "fail"
    assert "input fingerprint" in result["reason"]


@pytest.mark.parametrize("part", ["task", "pack"])
def test_primary_fingerprint_binds_task_and_full_pack_not_only_pack_id(tmp_path, part):
    make_run(tmp_path)
    folder = TASK_DIR / "round_1"
    delivery = read(tmp_path, folder / "A/delivery.json")
    wrong_pack = read(tmp_path, folder / "pack.json")
    wrong_task = dict(TASK)
    if part == "task":
        wrong_task["prompt"] = "Different actual requirements"
        wrong_pack["task_prompt"] = wrong_task["prompt"]
    else:
        wrong_pack["criteria"][0]["standard"] = "Different standard but same ID"
    wrong_measurement = measured(wrong_pack, delivery["candidate"], task=wrong_task)
    write(tmp_path, folder / "A/outcome_estimate.json", wrong_measurement)
    result = stage(audit_run(tmp_path), "round_1/A/delivered_pack_measurement")
    assert result["status"] == "fail"
    assert "input fingerprint" in result["reason"]


def test_fresh_bound_supplement_can_validate_after_unbound_original(tmp_path):
    make_run(tmp_path)
    folder = TASK_DIR / "round_1"
    primary = folder / "A/outcome_estimate.json"
    edit(tmp_path, primary, lambda row: row.pop("measurement_input_sha256"))
    original_bytes = (tmp_path / primary).read_bytes()
    assert (
        stage(audit_run(tmp_path), "round_1/A/delivered_pack_measurement")["status"]
        == "fail"
    )
    supplement(tmp_path, arm="A")  # New execution of the real offline harness.
    report = audit_run(tmp_path)
    result = stage(report, "round_1/A/delivered_pack_measurement")
    assert result["status"] == "pass", result
    assert "Unbound measurement" in result["facts"]["primary_measurement_issue"]
    assert result["facts"]["provenance"] == "supplemental"
    assert not report["claims"]["original_run_full_delivered_measurements"]
    assert (tmp_path / primary).read_bytes() == original_bytes


@pytest.mark.parametrize(
    "damage",
    ["absent_schema", "absent_fingerprint", "wrong_fingerprint", "other_measurement"],
)
def test_supplement_envelope_hashes_cannot_replace_execution_binding(tmp_path, damage):
    make_run(tmp_path, targeted=True)
    path = supplement(tmp_path)
    receipt = read(tmp_path, path)
    if damage == "absent_schema":
        receipt["measurement"].pop("measurement_input_schema")
    elif damage == "absent_fingerprint":
        receipt["measurement"].pop("measurement_input_sha256")
    elif damage == "wrong_fingerprint":
        receipt["measurement"]["measurement_input_sha256"] = "0" * 64
    else:
        receipt["measurement"] = read(
            tmp_path, TASK_DIR / "round_1/A/outcome_estimate.json"
        )
    write(tmp_path, path, receipt)
    result = stage(audit_run(tmp_path), "round_1/B/delivered_pack_measurement")
    assert result["status"] == "fail"
    assert "input fingerprint" in result["reason"]


def test_jury_schema_without_anchors_matches_production_validator(tmp_path):
    from reward_engine.core.outcome.jury import validate

    make_run(tmp_path)
    folder = TASK_DIR / "round_1"
    rows = read(tmp_path, folder / "jury.json")["criteria"]
    assert "anchors" not in rows[0]
    assert set(rows[0]) == {
        "id",
        "standard",
        "kind",
        "priority",
        "priority_reason",
        "origin",
        "evidence_refs",
        "measurement_question",
        "subrequirements",
    }
    accepted = validate({"criteria": rows, "gaps": []}, {"task": TASK["prompt"]})
    assert criteria(accepted["criteria"]) == ["c1"]
    report = audit_run(tmp_path)
    assert stage(report, "round_1/jury")["status"] == "pass"
    assert stage(report, "round_1/builder_and_smoke")["status"] == "pass"


@pytest.mark.parametrize(
    "field,value",
    [
        ("standard", ""),
        ("measurement_question", " "),
        ("kind", "important"),
        ("origin", "unknown"),
        ("origin", "domain"),
        ("evidence_refs", []),
        ("evidence_refs", ["product_1:artifact:final_response"]),
        ("priority", "hard"),
        ("priority", []),
        ("priority_reason", " "),
        ("subrequirements", "not-a-list"),
        ("subrequirements", [{"text": "clause"}]),
        ("subrequirements", [""]),
    ],
)
def test_production_criterion_requirements_remain_strict(field, value):
    from reward_engine.core.outcome.jury import validate

    row = {
        "id": "c1",
        "standard": "Describe CLI",
        "measurement_question": "Is CLI correct?",
        "kind": "hard",
        "origin": "task",
        "evidence_refs": ["task"],
        field: value,
    }
    with pytest.raises(ValueError):
        validate(
            {"criteria": [row], "gaps": []},
            {"task": TASK["prompt"], "product_1:artifact:final_response": "answer"},
        )
    with pytest.raises(ValueError):
        criteria([row])


@pytest.mark.parametrize("anchor", ["fail", "partial", "full"])
@pytest.mark.parametrize("value", [None, "", " ", []])
def test_builder_requires_all_three_nonempty_anchors(tmp_path, anchor, value):
    make_run(tmp_path)
    path = TASK_DIR / "round_1/pack.json"
    edit(tmp_path, path, lambda r: r["criteria"][0]["anchors"].update({anchor: value}))
    report = audit_run(tmp_path)
    assert stage(report, "round_1/jury")["status"] == "pass"
    result = stage(report, "round_1/builder_and_smoke")
    assert result["status"] == "fail"
    assert "fail/partial/full anchors" in result["reason"]


@pytest.mark.parametrize(
    "value",
    [
        "pass",
        "unknown",
        None,
        [],
        1,
        True,
        {},
        {"status": "pass"},
        {"status": "pass", "reason": "Checked", "evidence_refs": []},
    ],
)
def test_malformed_inquiry_result_is_rejected_with_field_diagnostic(tmp_path, value):
    make_run(tmp_path)
    path = TASK_DIR / "round_1/construction_checks.json"
    edit(
        tmp_path, path, lambda rows: rows[0]["results"][0].update(inquiry_result=value)
    )
    original = (tmp_path / path).read_bytes()
    report = audit_run(tmp_path)
    result = stage(report, "round_1/construction_measurements")
    assert result["status"] == "fail"
    assert "ValueError:" in result["reason"]
    assert "measurement.results[t1].inquiry_result" in result["reason"]
    assert "AttributeError" not in result["reason"]
    assert (tmp_path / path).read_bytes() == original


@pytest.mark.parametrize("status", ["pass", "fail", "unknown"])
def test_typed_inquiry_receipts_preserve_product_verdicts(status):
    pack = {"pack_id": "p1", "tests": [{"test_id": "t1", "criterion_id": "c1"}]}
    nested = {
        "status": status,
        "reason": "Observed product evidence",
        "gap": "Wrong flag" if status == "fail" else "",
        "evidence_refs": ["product_1:artifact:final_response"]
        if status != "unknown"
        else [],
    }
    row = {
        "pack_id": "p1",
        "results": [
            {
                "test_id": "t1",
                "criterion_id": "c1",
                "status": "pass",
                "reason": "Checked",
                "evidence_refs": ["product_1:artifact:final_response"],
                "inquiry_result": nested,
            }
        ],
    }
    row["batch"] = {
        "complete": True,
        "total": 1,
        "succeeded": 1,
        "items": [{"job_id": "t1", "status": "ok", "error_type": None}],
    }
    row["execution_summary"] = {
        "selected_checks": 1,
        "submitted_checks": 1,
        "reused_checks": 0,
        "reported_model_requests": 0,
        "reported_tool_calls": 0,
    }
    original = json.dumps(row, sort_keys=True)
    assert measurement(row, pack)["pass"] == 1
    assert json.dumps(row, sort_keys=True) == original


def test_schema_fix_preserves_genuine_panel_and_checker_errors(tmp_path):
    make_run(tmp_path)
    folder = TASK_DIR / "round_1"
    partial = mapping(2)
    write(tmp_path, folder / "jury_map.json", partial)
    write(
        tmp_path,
        folder / "hypothesis_map.json",
        {
            "complete": False,
            "total": 3,
            "succeeded": 2,
            "items": [
                {"job_id": "1", "status": "ok"},
                {
                    "job_id": "2",
                    "status": "error",
                    "error_type": "ResponseContractError",
                },
                {"job_id": "3", "status": "ok"},
            ],
        },
    )
    edit(
        tmp_path,
        folder / "construction_checks.json",
        lambda rows: rows[0]["results"][0].update(
            status="unknown", error=True, reason="Checker failed: RuntimeError"
        ),
    )
    report = audit_run(tmp_path)
    assert stage(report, "round_1/jury")["status"] == "fail"
    assert "Missing expected Map seats" in stage(report, "round_1/jury")["reason"]
    assert stage(report, "round_1/builder_and_smoke")["status"] == "pass"
    assert stage(report, "round_1/hypothesis")["status"] == "fail"
    result = stage(report, "round_1/construction_measurements")
    assert result["status"] == "fail"
    assert "Checker execution failed" in result["reason"]


def test_nested_checker_error_stays_failure():
    pack = {"pack_id": "p1", "tests": [{"test_id": "t1", "criterion_id": "c1"}]}
    row = {
        "pack_id": "p1",
        "results": [
            {
                "test_id": "t1",
                "criterion_id": "c1",
                "status": "pass",
                "reason": "Checked",
                "evidence_refs": ["product_1:artifact:final_response"],
                "inquiry_result": {
                    "status": "unknown",
                    "reason": "Checker unavailable",
                    "error": True,
                },
            }
        ],
    }
    with pytest.raises(
        ValueError,
        match=r"Checker execution failed at measurement.results\[t1\].inquiry_result",
    ):
        measurement(row, pack)


@pytest.mark.parametrize("benchmark", ["claw_eval", "workbuddy"])
@pytest.mark.parametrize("targeted", [False, True])
@pytest.mark.parametrize("no_revision_rounds", [(1, 3), (1, 2, 3)])
def test_review_validity_is_separate_from_revision_coverage(
    tmp_path, capsys, benchmark, targeted, no_revision_rounds
):
    make_run(
        tmp_path, benchmark, targeted=targeted, no_revision_rounds=no_revision_rounds
    )
    if targeted:
        for number in range(1, 4):
            supplement(tmp_path, number, benchmark=benchmark)
    report = audit_run(tmp_path)
    revisions = [n for n in range(1, 4) if n not in no_revision_rounds]
    assert report["schema"] == "pipeline-receipt-audit-v2"
    assert report["status"] == "pass", report["coordination"]
    assert report["claims"]["full_pipeline_receipts_validated"]
    assert report["claims"]["b_reviews_validated_all_rounds"]
    assert report["counts"]["validated_review_rounds"] == 3
    assert report["counts"]["validated_revision_rounds"] == len(revisions)
    assert report["claims"]["same_session_revision_exercised_at_least_once"] is bool(
        revisions
    )
    assert not report["claims"]["same_session_revision_exercised_all_rounds"]
    assert not report["claims"]["same_session_revision_validated"]
    assert report["claims"]["required_module_coverage_complete"] is bool(revisions)
    assert report["coverage"]["same_session_continuation_exercised_rounds"] == revisions
    assert report["coverage"]["changed_artifact_revision_exercised_rounds"] == revisions
    assert report["coverage"]["status"] == ("complete" if revisions else "incomplete")
    for number in no_revision_rounds:
        receipt = stage(report, f"round_{number}/B_review_execution")
        assert receipt["status"] == "pass"
        assert receipt["facts"]["revision_coverage"] == "not_exercised"
        assert receipt["facts"]["satisfied_under_estimator"]
        assert any("task_progress_1.json" in path for path in receipt["evidence"])
    assert main([str(tmp_path)]) == (0 if revisions else 3)
    assert json.loads(capsys.readouterr().out) == report


@pytest.mark.parametrize(
    "damage",
    [
        "checker_error",
        "panel_timeout",
        "unsupported_satisfaction",
        "missing_progress",
        "progress_disagrees",
        "dangling_feedback",
        "disabled",
    ],
)
def test_nochange_requires_real_successful_review_and_consistent_stop(tmp_path, damage):
    make_run(tmp_path, targeted=True, no_revision_rounds=(1,))
    slot = TASK_DIR / "round_1/actors/B"
    if damage == "checker_error":
        edit(
            tmp_path,
            slot / "checks/delivery_1.json",
            lambda r: r["measurement"]["delivery_review"].update(error=True),
        )
    elif damage == "panel_timeout":
        edit(
            tmp_path,
            slot / "checks/delivery_1.json",
            lambda r: r["measurement"]["review_batch"]["items"][0].update(
                status="timeout", error_type="TimeoutError"
            ),
        )
    elif damage == "unsupported_satisfaction":
        edit(
            tmp_path,
            slot / "checks/delivery_1.json",
            lambda r: r["measurement"].update(delivery_clear=False),
        )
    elif damage == "missing_progress":
        (tmp_path / slot / "checks/task_progress_1.json").unlink()
    elif damage == "progress_disagrees":
        edit(
            tmp_path,
            slot / "checks/task_progress_1.json",
            lambda r: r["items"][-1].update(status="unknown"),
        )
    elif damage == "dangling_feedback":
        write(
            tmp_path,
            slot / "checks/feedback_1.json",
            {"message": "Please continue and recheck"},
        )
    else:
        edit(
            tmp_path, slot / "RESULT.json", lambda r: r.update(review_status="disabled")
        )
    report = audit_run(tmp_path)
    assert stage(report, "round_1/B_review_execution")["status"] == "fail"
    assert not report["claims"]["b_reviews_validated_all_rounds"]
    assert not report["claims"]["required_module_coverage_complete"]
    # Other validated revisions are still reported, without excusing this failure.
    assert report["claims"]["same_session_revision_exercised_at_least_once"]
    assert report["coverage"]["changed_artifact_revision_exercised_rounds"] == [2, 3]


def test_budget_exhausted_valid_review_is_not_satisfaction(tmp_path):
    make_run(tmp_path, unresolved_rounds=(1,))
    report = audit_run(tmp_path)
    assert report["status"] == "pass", report["coordination"]
    result = stage(report, "round_1/B_review_execution")
    assert result["facts"]["review_status"] == "budget_exhausted_unresolved"
    assert not result["facts"]["satisfied_under_estimator"]
    assert result["facts"]["changed_artifact_revisions"] == 2
    assert report["claims"]["same_session_revision_validated"]
    assert not report["claims"]["quality_improvement_validated"]


def test_valid_nochange_live_completion_remains_pending(tmp_path):
    make_run(tmp_path, no_revision_rounds=(1, 2, 3))
    (tmp_path / TASK_DIR / "result.private.json").unlink()
    write(tmp_path, "validation_status.json", {"status": "running"})
    report = audit_run(tmp_path)
    assert report["status"] == "pending", report["coordination"]
    assert report["coverage"]["status"] == "pending"
    assert report["claims"]["b_reviews_validated_all_rounds"]
    assert not report["claims"]["required_module_coverage_complete"]


@pytest.mark.parametrize("closed", [False, True])
def test_requested_continuation_without_terminal_result_is_not_nochange(
    tmp_path, closed
):
    make_run(tmp_path)
    folder = TASK_DIR / "round_3"
    for path in (
        TASK_DIR / "result.private.json",
        folder / "summary.private.json",
        folder / "checkpoint.json",
        folder / "actors/B/RESULT.json",
        folder / "actors/B/checks/delivery_2.json",
    ):
        (tmp_path / path).unlink()
    write(
        tmp_path,
        "validation_status.json",
        {"status": "execution_completed" if closed else "running"},
    )
    report = audit_run(tmp_path)
    assert stage(report, "round_3/B_review_execution")["status"] == (
        "fail" if closed else "pending"
    )
    assert not report["claims"]["b_reviews_validated_all_rounds"]
    assert not report["claims"]["same_session_revision_validated"]
    assert report["coverage"]["changed_artifact_revision_exercised_rounds"] == [1, 2]


@pytest.mark.parametrize(
    "original_value",
    ["pass", "unknown", None, [], {"status": "fail", "gap": "unrequested"}],
)
def test_nonauthoritative_inquiry_diagnostic_preserved_without_changing_verdict(
    tmp_path, original_value
):
    make_run(tmp_path)
    folder = TASK_DIR / "round_1"
    diagnostic = {
        "reason": "no_measurement_inquiry_requested",
        "authoritative": False,
        "original_value": original_value,
    }
    edit(
        tmp_path,
        folder / "A/outcome_estimate.json",
        lambda r: r["results"][0].update(inquiry_result_diagnostic=diagnostic),
    )
    report = audit_run(tmp_path)
    assert report["status"] == "pass", report["coordination"]
    result = stage(report, "round_1/A/delivered_pack_measurement")
    assert result["facts"]["measurements"]["pass"] == 1
    assert result["facts"]["measurements"]["inquiry_result_diagnostics"] == [
        {"test_id": "t1", "diagnostic": diagnostic}
    ]


@pytest.mark.parametrize(
    "field",
    ["deferred_tests", "unresolved_checks", "gaps", "full_score_gaps", "improvements"],
)
def test_nochange_cannot_claim_clear_with_remaining_targeted_findings(tmp_path, field):
    make_run(tmp_path, targeted=True, no_revision_rounds=(1,))
    slot = TASK_DIR / "round_1/actors/B"

    def change(row):
        measured = row["measurement"]
        if field in {"deferred_tests", "unresolved_checks"}:
            measured[field] = (
                ["t1"]
                if field == "deferred_tests"
                else [{"test_id": "t1", "status": "unknown", "reason": "Unavailable"}]
            )
        else:
            measured["delivery_review"][field] = [
                {
                    "gap": "Not resolved",
                    "evidence_refs": ["product_1:artifact:final_response"],
                }
            ]

    edit(tmp_path, slot / "checks/delivery_1.json", change)
    report = audit_run(tmp_path)
    assert stage(report, "round_1/B_review_execution")["status"] == "fail"
    assert not report["claims"]["b_reviews_validated_all_rounds"]


@pytest.mark.parametrize("targeted", [False, True])
@pytest.mark.parametrize(
    "damage",
    ["missing_schema", "missing_digest", "wrong_digest", "different_candidate"],
)
def test_pre_revision_measurement_requires_its_own_binding(tmp_path, targeted, damage):
    make_run(tmp_path, targeted=targeted)
    if targeted:
        for number in range(1, 4):
            supplement(tmp_path, number)
    slot = TASK_DIR / "round_1/actors/B"

    def corrupt(row):
        measured = row["measurement"]
        if damage == "missing_schema":
            measured.pop("measurement_input_schema")
        elif damage == "missing_digest":
            measured.pop("measurement_input_sha256")
        elif damage == "wrong_digest":
            measured["measurement_input_sha256"] = "0" * 64
        else:
            measured["measurement_input_sha256"] = read(
                tmp_path, TASK_DIR / "round_1/A/outcome_estimate.json"
            )["measurement_input_sha256"]

    edit(tmp_path, slot / "checks/delivery_1.json", corrupt)
    report = audit_run(tmp_path)
    assert stage(report, "round_1/B/delivered_pack_measurement")["status"] == "pass"
    assert stage(report, "round_1/B_review_execution")["status"] == "fail"
    assert not report["claims"]["b_reviews_validated_all_rounds"]
    assert not report["claims"]["required_module_coverage_complete"]
    assert not report["claims"]["same_session_revision_validated"]
    assert report["coverage"]["changed_artifact_revision_exercised_rounds"] == [2, 3]


@pytest.mark.parametrize(
    "damage",
    [
        "legacy_missing",
        "schema",
        "overview_digest",
        "previous_digest",
        "plan",
        "inquiries",
        "budget",
        "unbound_draft",
        "minimal_context",
        "overview",
        "prior_result",
    ],
)
def test_targeted_review_rejects_unbound_inputs_plan_and_prior_state(tmp_path, damage):
    from reward_engine.core.outcome.harness import measurement_input_fingerprint

    make_run(tmp_path, targeted=True, inquiries=True)
    for number in range(1, 4):
        supplement(tmp_path, number)
    slot = TASK_DIR / "round_1/actors/B"
    path = slot / "checks/delivery_2.json"
    receipt = read(tmp_path, path)
    measured = receipt["measurement"]
    binding = measured["review_input"]
    if damage == "legacy_missing":
        measured.pop("review_input")
    elif damage == "schema":
        binding["schema"] = "other"
    elif damage == "overview_digest":
        binding["sha256"] = "0" * 64
    elif damage == "previous_digest":
        binding["previous_measurement_sha256"] = "0" * 64
    elif damage == "plan":
        binding["measurement_plan"]["selected_tests"] = []
    elif damage == "inquiries":
        binding["measurement_plan"]["measurement_inquiries"] = {}
    elif damage == "budget":
        binding["max_tests"] = 1
    elif damage == "unbound_draft":
        binding["draft_checks"] = [{"test_id": "t1", "status": "pass"}]
    elif damage == "minimal_context":
        pack = read(tmp_path, TASK_DIR / "round_1/pack.json")
        measured["measurement_input_sha256"] = measurement_input_fingerprint(
            pack, *build_context(TASK, [receipt["candidate"]])
        )
    elif damage == "overview":
        binding["overview_before_measurement"]["reason"] = "A different overview"
    elif damage == "prior_result":
        edit(
            tmp_path,
            slot / "checks/delivery_1.json",
            lambda row: row["measurement"].update(extra_previous_state="changed"),
        )
    write(tmp_path, path, receipt)
    result = stage(audit_run(tmp_path), "round_1/B_review_execution")
    assert result["status"] == "fail", result
    assert (
        "fingerprint" in result["reason"]
        or "bind" in result["reason"]
        or "provenance" in result["reason"]
        or "review" in result["reason"]
        or "Targeted" in result["reason"]
    )


@pytest.mark.parametrize("benchmark", ["claw_eval", "workbuddy"])
def test_genuine_targeted_producer_receipts_bind_inquiries_and_prior_native_turn(
    tmp_path, benchmark
):
    from reward_engine.core.outcome.harness import measurement_input_fingerprint

    make_run(tmp_path, benchmark, targeted=True, inquiries=True)
    for number in range(1, 4):
        supplement(tmp_path, number, benchmark=benchmark)
    report = audit_run(tmp_path)
    assert report["status"] == "pass", report["coordination"]
    assert report["claims"]["same_session_revision_validated"]
    slot = TASK_DIR / "round_1/actors/B"
    pack = read(tmp_path, TASK_DIR / "round_1/pack.json")
    for turn in (1, 2):
        receipt = read(tmp_path, slot / "checks" / f"delivery_{turn}.json")
        measured = receipt["measurement"]
        assert measured["measurement_input_sha256"] != measurement_input_fingerprint(
            pack, *build_context(TASK, [receipt["candidate"]])
        )
        facts = stage(report, "round_1/B_review_execution")["facts"]["review_bindings"][
            turn - 1
        ]
        assert facts["measurement_plan"]["selected_tests"] == ["t1"]
        assert facts["measurement_plan"]["measurement_inquiries"]["t1"]["questions"]
        assert facts["review_input_sha256"] == measured["review_input"]["sha256"]
        assert bool(measured["review_input"]["previous_measurement_sha256"]) is (
            turn == 2
        )


@pytest.mark.parametrize(
    "damage",
    [
        "unknown_gap_ref",
        "bad_improvement",
        "bad_claim",
        "unknown_test",
        "empty_goal",
        "selected_count",
        "submitted_count",
        "reused_count",
        "batch_id",
        "result_inquiry",
        "omit_selected_result",
        "deferred",
    ],
)
def test_targeted_bound_overview_uses_production_contract_and_execution_summary(
    tmp_path, damage
):
    make_run(tmp_path, targeted=True, inquiries=True)
    slot = TASK_DIR / "round_1/actors/B"
    path = slot / "checks/delivery_1.json"
    receipt = read(tmp_path, path)
    row = receipt["measurement"]
    overview = row["review_input"]["overview_before_measurement"]
    if damage == "unknown_gap_ref":
        overview["gaps"][0]["evidence_refs"] = ["product_1:invented"]
    elif damage == "bad_improvement":
        overview["improvements"] = [
            {"id": "i", "action": "do more", "gap_ids": ["nonexistent"]}
        ]
    elif damage == "bad_claim":
        overview["claim_checks"][0]["claim_ref"] = "product_1:invented"
    elif damage == "unknown_test":
        overview["verify"][0]["test_id"] = "unpublished"
    elif damage == "empty_goal":
        overview["outcome_goal"] = ""
    elif damage in ("selected_count", "submitted_count", "reused_count"):
        row["execution_summary"][damage.replace("_count", "_checks")] = 999
    elif damage == "batch_id":
        row["batch"]["items"][0]["job_id"] = "not-t1"
    elif damage == "result_inquiry":
        row["results"][0]["measurement_inquiry"]["questions"] = ["Different question"]
    elif damage == "omit_selected_result":
        row["results"] = []
    elif damage == "deferred":
        row["deferred_tests"] = ["t1"]
    write(tmp_path, path, receipt)
    report = audit_run(tmp_path)
    assert stage(report, "round_1/B_review_execution")["status"] == "fail"
    assert not report["claims"]["same_session_revision_validated"]


@pytest.mark.asyncio
async def test_auditor_reconstructs_real_two_turn_controller_with_published_sources():
    from experiments.pipeline_audit import review_binding
    from reward_engine.core.outcome.harness import defaults
    from reward_engine.core.outcome.runner import DeliveryReview
    from tests.support.claw_outcome_memory_runner import PACK

    pack = deepcopy(PACK)
    task = {"task_id": pack["task_id"], "prompt": pack["task_prompt"]}
    pack["measurement_sources"] = {
        "reference:published-json": {"content": "Parse delivered bytes as JSON"}
    }

    async def model(_request):
        return {
            "outcome_goal": "Return valid JSON",
            "full_score_gaps": [],
            "improvements": [],
            "gaps": [],
            "status": "needs_verification",
            "reason": "Parse the actual delivered bytes",
            "verify": [
                {"test_id": "parse", "reason": "Does this exact delivery parse?"}
            ],
            "claim_checks": [
                {
                    "claim": "This is valid JSON",
                    "claim_ref": "product_1:artifact:final_response",
                    "support_refs": [],
                    "assessment": "needs_verification",
                    "reason": "Use the actual parser",
                    "test_id": "parse",
                }
            ],
        }

    records = {}
    controller = DeliveryReview(
        task,
        pack,
        defaults(model),
        record=lambda key, value: records.__setitem__(key, deepcopy(value)),
    )
    previous = None
    for number, answer in enumerate(("{bad", "{}"), 1):
        candidate = {
            "candidate_id": "offline-native-candidate",
            "artifact": {"final_response": answer},
        }
        feedback = await controller(candidate)
        row = records[f"delivery_{number}"]["measurement"]
        facts, previous = review_binding(row, pack, task, candidate, previous)
        assert facts["review_input_sha256"] == row["review_input"]["sha256"]
        assert facts["measurement_plan"]["selected_tests"] == ["parse"]
        assert bool(feedback) is (number == 1)
        altered_pack = deepcopy(pack)
        altered_pack["measurement_sources"]["reference:published-json"] = {
            "content": "Different published source"
        }
        with pytest.raises(ValueError, match="fingerprint|binding"):
            review_binding(
                row, altered_pack, task, candidate, None if number == 1 else previous
            )
    assert controller.status == "full_score_target_satisfied"


def forge_empty_reuse(row):
    row["results"][0]["reused_unchanged_input"] = True
    row["batch"] = {"complete": True, "total": 0, "succeeded": 0, "items": []}
    row["execution_summary"].update(
        submitted_checks=0,
        reused_checks=1,
        reported_model_requests=0,
        reported_tool_calls=0,
    )


def test_targeted_failed_prior_changed_delivery_active_inquiry_cannot_reuse(tmp_path):
    make_run(tmp_path, targeted=True, inquiries=True)
    for number in range(1, 4):
        supplement(tmp_path, number)
    assert audit_run(tmp_path)["status"] == "pass"
    slot = TASK_DIR / "round_1/actors/B"
    original = read(tmp_path, slot / "checks/delivery_1.json")
    final = read(tmp_path, slot / "checks/delivery_2.json")
    assert original["measurement"]["results"][0]["status"] == "fail"
    assert original["candidate"] != final["candidate"]
    assert final["measurement"]["results"][0]["measurement_inquiry"]
    forge_empty_reuse(final["measurement"])
    write(tmp_path, slot / "checks/delivery_2.json", final)
    edit(
        tmp_path,
        slot / "RESULT.json",
        lambda row: row.update(final_measurement=final["measurement"]),
    )
    report = audit_run(tmp_path)
    review = stage(report, "round_1/B_review_execution")
    assert review["status"] == "fail"
    assert "prior passing check" in review["reason"]
    assert not report["claims"]["required_module_coverage_complete"]
    assert report["coverage"]["changed_artifact_revision_exercised_rounds"] == [2, 3]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [
        None,
        "prior_fail",
        "prior_error",
        "missing_prior",
        "active_inquiry",
        "changed_result",
        "changed_dependency",
        "custom_dependency",
    ],
)
async def test_reuse_requires_real_prior_success_and_reconstructable_dependencies(
    change,
):
    from experiments.pipeline_audit import measurement_execution, review_binding
    from reward_engine.core.outcome.harness import defaults
    from tests.support.claw_outcome_memory_runner import PACK
    from tests.support.claw_outcome_memory_runner import TASK as JSON_TASK

    pack = deepcopy(PACK)
    registry = defaults(None)
    if change == "custom_dependency":
        callback = registry.checkers["json.parse"][0]
        registry.register(
            "custom.parse",
            callback,
            "Offline custom dependency",
            dependencies=registry.dependencies["json.parse"],
        )
        pack["tests"][0]["checker"] = "custom.parse"
    first = {
        "candidate_id": "same",
        "artifact": {"final_response": "{}", "other": "before"},
    }
    second = {
        "candidate_id": "same",
        "artifact": {"final_response": "{}", "other": "after"},
    }
    context, evidence = build_context(JSON_TASK, [first])
    old = await measure(pack, registry, context, evidence)
    previous = (context, evidence, old)
    context, evidence = build_context(JSON_TASK, [second])
    row = await measure(pack, registry, context, evidence, previous=previous)
    assert row["results"][0]["reused_unchanged_input"] is True
    assert (
        row["batch"]["items"] == []
        and row["execution_summary"]["submitted_checks"] == 0
    )
    if change == "prior_fail":
        old["results"][0]["status"] = "fail"
    elif change == "prior_error":
        old["results"][0]["error"] = True
    elif change == "missing_prior":
        old["results"] = []
    elif change == "active_inquiry":
        context["measurement_inquiries"] = {
            "parse": {"questions": ["Verify this delivery"]}
        }
    elif change == "changed_result":
        row["results"][0]["reason"] = "Invented replacement result"
    elif change == "changed_dependency":
        evidence["product_1:artifact:final_response"] = "[]"
    if change is None:
        before = deepcopy(row)
        measurement_execution(
            row, pack, ["parse"], context=context, evidence=evidence, previous=previous
        )
        facts, _ = review_binding(row, pack, JSON_TASK, second, previous)
        assert facts["measurement_input_sha256"] == row["measurement_input_sha256"]
        assert row == before
    else:
        with pytest.raises(ValueError, match="reuse|Reused"):
            measurement_execution(
                row,
                pack,
                ["parse"],
                context=context,
                evidence=evidence,
                previous=previous,
            )


@pytest.mark.parametrize(
    "where", ["full_review", "construction", "delivery", "supplemental"]
)
@pytest.mark.parametrize(
    "damage", ["failed_batch", "wrong_summary", "wrong_job", "unjustified_reuse"]
)
def test_all_measurement_stages_require_consistent_execution_receipts(
    tmp_path, where, damage
):
    make_run(tmp_path)
    folder = TASK_DIR / "round_1"
    if where == "full_review":
        path, expected_stage = (
            folder / "actors/B/checks/delivery_1.json",
            "round_1/B_review_execution",
        )
    elif where == "construction":
        path, expected_stage = (
            folder / "construction_checks.json",
            "round_1/construction_measurements",
        )
    elif where == "delivery":
        path, expected_stage = (
            folder / "A/outcome_estimate.json",
            "round_1/A/delivered_pack_measurement",
        )
    else:
        path = supplement(tmp_path, arm="A")
        expected_stage = "round_1/A/delivered_pack_measurement"
    receipt = read(tmp_path, path)
    row = (
        receipt[0]
        if where == "construction"
        else receipt
        if where == "delivery"
        else receipt["measurement"]
    )
    if damage == "failed_batch":
        row["batch"].update(complete=False, succeeded=0)
        row["batch"]["items"][0].update(status="timeout", error_type="TimeoutError")
    elif damage == "wrong_summary":
        row["execution_summary"].update(submitted_checks=0, reused_checks=0)
    elif damage == "wrong_job":
        row["batch"]["items"][0]["job_id"] = "not-executed-t1"
    else:
        forge_empty_reuse(row)
    write(tmp_path, path, receipt)
    report = audit_run(tmp_path)
    assert stage(report, expected_stage)["status"] == "fail"
    assert not report["claims"]["full_pipeline_receipts_validated"]
    assert not report["claims"]["required_module_coverage_complete"]
