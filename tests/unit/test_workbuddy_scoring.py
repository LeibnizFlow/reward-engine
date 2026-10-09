"""Native grading contract checks, with no grader calls or score rewriting."""

import json
from copy import deepcopy

import pytest

from experiments.workbuddy.scoring import native_contract, scoring_errors
from experiments.workbuddy.worker import official_result
from tests.unit.test_pipeline_audit import (
    TASK,
    TASK_DIR,
    audit_run,
    make_run,
    native,
    native_score_fixture,
    read,
    stage,
    write,
)


def mutate(detail, damage):
    if damage == "empty_verdict":
        detail["verdicts"] = [{}]
    elif damage == "nonobject_verdict":
        detail["verdicts"] = ["pass"]
    elif damage == "empty_list":
        detail["verdicts"] = []
    elif damage.startswith("field_"):
        detail["verdicts"][0].pop(damage.removeprefix("field_"))
    elif damage in ("judge_error", "build_error", "unknown", "abstain", "review"):
        detail["verdicts"][0].update(status=damage, score=0.0, metadata={})
        detail["judges"][0]["verdicts"][0].update(status=damage, score=0.0, metadata={})
    elif damage == "missing_item":
        detail["verdicts"].pop()
        detail["judges"][1]["verdicts"].clear()
    elif damage == "shrink_plan":
        detail["verdicts"].pop()
        detail["judges"].pop()
        detail["diagnostics"]["plan"]["items"].pop()
        detail["diagnostics"]["plan"]["judges"].pop()
    elif damage == "duplicate_item":
        detail["verdicts"].append(deepcopy(detail["verdicts"][0]))
    elif damage == "unexpected_item":
        detail["verdicts"][0]["item_id"] = "not-authored"
        detail["judges"][0]["verdicts"][0]["item_id"] = "not-authored"
    elif damage == "aggregate":
        detail["overall"] = detail["reward"] = 0.5
    elif damage == "status":
        detail["test_status"] = "full_pass"
    elif damage == "component":
        detail["judges"][0]["scores"]["pass_rate"] = 0.5
    elif damage == "judge_envelope":
        detail["judges"][0]["status"] = "judge_error"
    elif damage == "merge_weights":
        detail["diagnostics"]["plan"]["metadata"]["score_merge"]["llm_weight"] = 1
    elif damage == "unknown_policy":
        detail["diagnostics"]["plan"]["scoring_policy"] = "other"
    elif damage == "test_counts":
        detail["tests_total"] = 100
    elif damage == "duplicate_disagrees":
        detail["judges"][0]["verdicts"][0]["reason"] = "Different result"
    elif damage == "raw_empty":
        detail["judges"][0]["metadata"]["raw"] = {}
    elif damage == "raw_status_only":
        detail["judges"][0]["metadata"]["raw"] = {
            "test_status": "no_pass",
            "reason": "No score file",
        }
    elif damage == "raw_counts_disagree":
        detail["judges"][0]["metadata"]["raw"].update(tests_passed=6, tests_total=7)
    elif damage == "raw_zero_counts_fallback":
        detail["judges"][0]["metadata"]["raw"].update(tests_passed=0, tests_total=0)
    elif damage == "raw_tests_counts_disagree":
        detail["judges"][0]["metadata"]["raw"] = {
            "overall": 0.0,
            "tests": [{"passed": "yes"}, {"passed": False}, {"passed": 1}],
        }
    else:
        raise AssertionError(damage)


DAMAGES = [
    "empty_verdict",
    "nonobject_verdict",
    "empty_list",
    "field_item_id",
    "field_judge_name",
    "field_judge_type",
    "field_score",
    "field_status",
    "field_metadata",
    "field_evidence_ids",
    "judge_error",
    "build_error",
    "unknown",
    "abstain",
    "review",
    "missing_item",
    "shrink_plan",
    "duplicate_item",
    "unexpected_item",
    "aggregate",
    "status",
    "component",
    "judge_envelope",
    "merge_weights",
    "unknown_policy",
    "test_counts",
    "duplicate_disagrees",
    "raw_empty",
    "raw_status_only",
    "raw_counts_disagree",
    "raw_zero_counts_fallback",
    "raw_tests_counts_disagree",
]


def contract():
    return native_contract(
        TASK["task_id"],
        b'[metadata]\nsource_case="task-one"\n',
        json.dumps(
            {
                "llm_judge": {"rubrics": [{"id": "quality"}]},
                "score_merge": {
                    "method": "weighted_sum",
                    "rule_weight": 0.8,
                    "llm_weight": 0.2,
                },
            }
        ).encode(),
    )


@pytest.mark.parametrize("damage", DAMAGES)
def test_invalid_native_grades_rejected_by_validator_worker_and_auditor(
    tmp_path, damage
):
    make_run(tmp_path, "workbuddy")
    slot = TASK_DIR / "round_1/actors/A"
    spec = read(tmp_path, slot / "spec.json")
    path = slot / "harbor" / spec["candidate_id"] / "verifier/score.json"
    detail = read(tmp_path, path)
    mutate(detail, damage)
    before = deepcopy(detail)
    assert scoring_errors(detail, contract=contract())
    assert detail == before
    write(tmp_path, path, detail)
    serialized = read(tmp_path, slot / "TRIAL_RESULT.private.json")
    result = official_result(spec, serialized, tmp_path / slot)
    assert result["status"] == "failed"
    assert result["overall"] is None
    assert result["raw_overall"] == 0.0
    assert result["scoring_errors"]
    report = audit_run(tmp_path)
    assert stage(report, "round_1/actors/A/native_grade")["status"] == "fail"
    assert not report["claims"]["full_pipeline_receipts_validated"]
    assert not report["claims"]["required_module_coverage_complete"]


@pytest.mark.parametrize("field", ["score", "confidence"])
@pytest.mark.parametrize("value", [True, "0.0", -0.1, 1.1, float("inf"), float("nan")])
def test_native_verdict_numeric_fields_are_finite_unit_numbers(field, value):
    detail = native_score_fixture()
    detail["verdicts"][0][field] = value
    assert scoring_errors(detail, contract=contract())


@pytest.mark.parametrize(
    "rule_score,llm_score", [(0.0, 0.0), (0.5, 0.0), (0.0, 1.0), (0.9, 1.0), (1.0, 1.0)]
)
def test_real_product_failure_partial_credit_and_zero_are_preserved(
    tmp_path, rule_score, llm_score
):
    slot = TASK_DIR / "round_1/actors/A"
    native(tmp_path, slot, "workbuddy")
    spec = read(tmp_path, slot / "spec.json")
    detail = native_score_fixture(rule_score=rule_score, llm_score=llm_score)
    before = deepcopy(detail)
    assert not scoring_errors(detail, contract=contract())
    assert detail == before
    write(
        tmp_path, slot / "harbor" / spec["candidate_id"] / "verifier/score.json", detail
    )
    serialized = read(tmp_path, slot / "TRIAL_RESULT.private.json")
    serialized["verifier_result"]["rewards"] = {
        k: detail[k]
        for k in (
            "overall",
            "reward",
            "test_pass_rate",
            "llm_judge_component_score",
            "tests_passed",
            "tests_total",
        )
    }
    result = official_result(spec, serialized, tmp_path / slot)
    assert result["status"] == "scored"
    assert result["overall"] == result["raw_overall"] == detail["overall"]
    assert not result["scoring_errors"]


@pytest.mark.parametrize("value", [None, [], {}, "pass", 0])
def test_worker_handles_malformed_native_envelope(tmp_path, value):
    slot = TASK_DIR / "round_1/actors/A"
    native(tmp_path, slot, "workbuddy")
    spec = read(tmp_path, slot / "spec.json")
    write(
        tmp_path, slot / "harbor" / spec["candidate_id"] / "verifier/score.json", value
    )
    result = official_result(
        spec, read(tmp_path, slot / "TRIAL_RESULT.private.json"), tmp_path / slot
    )
    assert result["status"] == "failed"
    assert result["raw_overall"] == 0.0


def test_trial_components_must_match_native_score(tmp_path):
    slot = TASK_DIR / "round_1/actors/A"
    native(tmp_path, slot, "workbuddy")
    spec = read(tmp_path, slot / "spec.json")
    serialized = read(tmp_path, slot / "TRIAL_RESULT.private.json")
    serialized["verifier_result"]["rewards"]["test_pass_rate"] = 0.75
    assert official_result(spec, serialized, tmp_path / slot)["status"] == "failed"


@pytest.mark.parametrize(
    "value", [True, "0.0", -1, 2, 10**400, float("inf"), float("nan")]
)
def test_worker_rejects_invalid_trial_reward_without_coercion(tmp_path, value):
    slot = TASK_DIR / "round_1/actors/A"
    native(tmp_path, slot, "workbuddy")
    spec = read(tmp_path, slot / "spec.json")
    serialized = read(tmp_path, slot / "TRIAL_RESULT.private.json")
    serialized["verifier_result"]["rewards"]["overall"] = value
    result = official_result(spec, serialized, tmp_path / slot)
    assert result["status"] == "failed"
    assert result["overall"] is None
    assert result["raw_overall"] is value


def test_malformed_authored_contract_fails_explicitly(tmp_path):
    slot = TASK_DIR / "round_1/actors/A"
    native(tmp_path, slot, "workbuddy")
    spec = read(tmp_path, slot / "spec.json")
    (tmp_path / "dataset" / TASK["task_id"] / "tests/judge.yaml").write_text(
        "llm_judge: [broken"
    )
    result = official_result(
        spec, read(tmp_path, slot / "TRIAL_RESULT.private.json"), tmp_path / slot
    )
    assert result["status"] == "failed"
    assert (
        result["scoring_errors"][0]["reason"] == "Malformed authored native judge.yaml"
    )


@pytest.mark.parametrize("location", ["metadata", "raw"])
@pytest.mark.parametrize("status", ["build_error", "judge_error", "full_pass"])
def test_native_rule_source_status_cannot_be_overridden_by_success_envelopes(
    tmp_path, location, status
):
    make_run(tmp_path, "workbuddy")
    slot = TASK_DIR / "round_1/actors/A"
    spec = read(tmp_path, slot / "spec.json")
    path = slot / "harbor" / spec["candidate_id"] / "verifier/score.json"
    detail = read(tmp_path, path)
    source = detail["judges"][0]["metadata"]
    if location == "raw":
        source = source["raw"]
    source["test_status"] = status
    original = deepcopy(detail)
    errors = scoring_errors(detail, contract=contract())
    assert errors and (
        "source" in errors[0]["reason"] or "payload" in errors[0]["reason"]
    )
    assert detail == original
    write(tmp_path, path, detail)
    worker = official_result(
        spec, read(tmp_path, slot / "TRIAL_RESULT.private.json"), tmp_path / slot
    )
    assert worker["status"] == "failed" and worker["raw_overall"] == 0.0
    report = audit_run(tmp_path)
    result = stage(report, "round_1/actors/A/native_grade")
    assert result["status"] == "fail"
    assert "Native grading invalid" in result["reason"]
    assert not report["claims"]["required_module_coverage_complete"]


def test_rule_pass_at_zero_is_rejected_by_entire_audit(tmp_path):
    make_run(tmp_path, "workbuddy")
    slot = TASK_DIR / "round_1/actors/A"
    spec = read(tmp_path, slot / "spec.json")
    path = slot / "harbor" / spec["candidate_id"] / "verifier/score.json"
    detail = read(tmp_path, path)
    detail["verdicts"][0]["status"] = "pass"
    detail["judges"][0]["verdicts"][0]["status"] = "pass"
    write(tmp_path, path, detail)
    worker = official_result(
        spec, read(tmp_path, slot / "TRIAL_RESULT.private.json"), tmp_path / slot
    )
    assert worker["status"] == "failed"
    result = stage(audit_run(tmp_path), "round_1/actors/A/native_grade")
    assert result["status"] == "fail"
    assert "Rule verdict status/score disagreement" in result["reason"]


@pytest.mark.parametrize("rule_score", [0.0, 0.9016, 1.0])
def test_native_rule_verdict_mapping_preserves_fractional_credit_and_count_mismatch(
    tmp_path, rule_score
):
    slot = TASK_DIR / "round_1/actors/A"
    native(tmp_path, slot, "workbuddy")
    spec = read(tmp_path, slot / "spec.json")
    path = slot / "harbor" / spec["candidate_id"] / "verifier/score.json"
    detail = native_score_fixture(rule_score=rule_score, llm_score=1.0)
    # Authored aggregation can disagree with raw test counts; preserve that.
    detail["tests_passed"] = 2
    detail["judges"][0]["metadata"]["tests_passed"] = 2
    detail["judges"][0]["metadata"]["raw"]["tests_passed"] = 2
    for row in (detail["verdicts"][0], detail["judges"][0]["verdicts"][0]):
        row["metadata"].update(
            aggregate_score_verdict=True,
            aggregate_counts_mismatch=True,
            counts_score=0.2,
        )
    original = deepcopy(detail)
    assert not scoring_errors(detail, contract=contract())
    assert detail == original
    write(tmp_path, path, detail)
    serialized = read(tmp_path, slot / "TRIAL_RESULT.private.json")
    serialized["verifier_result"]["rewards"] = {
        key: detail[key]
        for key in (
            "reward",
            "overall",
            "test_pass_rate",
            "llm_judge_component_score",
            "tests_passed",
            "tests_total",
        )
    }
    result = official_result(spec, serialized, tmp_path / slot)
    assert result["status"] == "scored"
    assert result["overall"] == result["raw_overall"] == detail["overall"]
    wrong_status = "fail" if rule_score == 1.0 else "pass"
    detail["verdicts"][0]["status"] = wrong_status
    detail["judges"][0]["verdicts"][0]["status"] = wrong_status
    write(tmp_path, path, detail)
    result = official_result(spec, serialized, tmp_path / slot)
    assert result["status"] == "failed"
    assert result["raw_overall"] == original["overall"]


def test_rule_payload_status_follows_native_pass_alias_and_aggregate_precedence():
    detail = native_score_fixture(rule_score=1.0, llm_score=0.0)
    raw = detail["judges"][0]["metadata"]["raw"]
    raw["test_status"] = "pass"
    raw["reward"] = 1.0  # Native source precedence: reward before overall.
    raw["overall"] = 0.25
    assert not scoring_errors(detail, contract=contract())
    raw.pop("test_status")  # Native inference uses aggregate, not test counts.
    assert not scoring_errors(detail, contract=contract())
    raw["reward"] = 0.5
    assert scoring_errors(detail, contract=contract())


@pytest.mark.parametrize(
    "raw",
    [
        {},
        {"test_status": "no_pass", "reason": "No score file"},
        {"test_status": "partial_pass"},
        {"overall": "invalid", "tests_passed": True, "tests_total": False},
        {"tests": []},
        {"tests": [None, "pass", 1]},
        {"tests_passed": "invalid", "tests_total": None, "reward": False},
    ],
)
@pytest.mark.parametrize("rule_score", [0.0, 0.9016])
def test_unscorable_raw_cannot_borrow_outer_counts_or_score(raw, rule_score):
    detail = native_score_fixture(rule_score=rule_score)
    detail["judges"][0]["metadata"]["raw"] = deepcopy(raw)
    before = deepcopy(detail)
    errors = scoring_errors(detail, contract=contract())
    assert errors[0]["reason"] == "Unscorable native rule source payload"
    assert detail == before


@pytest.mark.parametrize("rule_score", [0.0, 0.9016, 1.0])
@pytest.mark.parametrize(
    "raw_counts,expected_counts",
    [
        pytest.param({}, None, id="aggregate-only"),
        pytest.param({"tests_passed": 0, "tests_total": 0}, None, id="zero-fallback"),
        pytest.param({"tests_passed": "2", "tests_total": "7"}, (2, 7), id="strings"),
        pytest.param(
            {"tests_passed": 2.9, "tests_total": 7.9}, (2, 7), id="int-conversion"
        ),
        pytest.param(
            {"tests_passed": -1, "tests_total": 7}, (0, 7), id="negative-passed"
        ),
        pytest.param({"tests_passed": 9, "tests_total": 7}, (7, 7), id="clamp-passed"),
        pytest.param({"tests_passed": 2, "tests_total": -1}, None, id="negative-total"),
        pytest.param({"tests_total": 7}, (0, 7), id="missing-passed"),
        pytest.param(
            {
                "tests": [
                    {"passed": "yes"},
                    {"passed": "false"},
                    {"passed": -1},
                    "ignored",
                ]
            },
            (2, 3),
            id="tests-list",
        ),
        pytest.param(
            {"tests_passed": 99, "tests": [{"passed": "passed"}, {}]},
            (1, 2),
            id="missing-total-replaces-both",
        ),
        pytest.param(
            {
                "tests_passed": True,
                "tests_total": 99,
                "tests": [{"passed": True}, {"passed": False}],
            },
            (1, 2),
            id="bool-count-replaced-by-tests",
        ),
        pytest.param(
            {"tests_passed": 2, "tests_total": 7, "tests": [{"passed": True}]},
            (2, 7),
            id="explicit-counts-before-tests",
        ),
        pytest.param(
            {"tests_passed": 99, "tests": ["ignored"]},
            None,
            id="empty-filtered-tests-aggregate-fallback",
        ),
    ],
)
def test_native_raw_count_rules_preserve_authored_aggregate(
    tmp_path, rule_score, raw_counts, expected_counts
):
    slot = TASK_DIR / "round_1/actors/A"
    native(tmp_path, slot, "workbuddy")
    spec = read(tmp_path, slot / "spec.json")
    detail = native_score_fixture(rule_score=rule_score, llm_score=1.0)
    source = detail["judges"][0]["metadata"]
    # The first usable aggregate wins even if other aggregates/count fractions
    # differ. Counts follow the pinned reader, not that authored score.
    source["raw"] = {**raw_counts, "reward": str(rule_score), "overall": 0.25}
    passed, total = expected_counts or (int(rule_score >= 1.0), 1)
    source.update(tests_passed=passed, tests_total=total)
    detail.update(tests_passed=passed, tests_total=total)
    before = deepcopy(detail)
    assert not scoring_errors(detail, contract=contract())
    assert detail == before
    path = slot / "harbor" / spec["candidate_id"] / "verifier/score.json"
    write(tmp_path, path, detail)
    serialized = read(tmp_path, slot / "TRIAL_RESULT.private.json")
    serialized["verifier_result"]["rewards"] = {
        key: detail[key] for key in serialized["verifier_result"]["rewards"]
    }
    result = official_result(spec, serialized, tmp_path / slot)
    assert result["status"] == "scored"
    assert result["overall"] == result["raw_overall"] == detail["overall"]
    assert read(tmp_path, path) == before
    # Consistent outer counts cannot override the counts extracted from raw.
    source["tests_total"] = detail["tests_total"] = total + 1
    errors = scoring_errors(detail, contract=contract())
    assert errors[0]["reason"] == "Native rule source/metadata count disagreement"


@pytest.mark.parametrize("passed", [False, True])
def test_native_tests_only_source_remains_scorable(passed):
    detail = native_score_fixture(rule_score=float(passed))
    judge = detail["judges"][0]
    judge["metadata"]["raw"] = {"tests": [{"name": "task-one:rule", "passed": passed}]}
    judge["metadata"].update(tests_passed=int(passed), tests_total=1)
    detail.update(tests_passed=int(passed), tests_total=1)
    judge["scores"].pop("overall", None)
    assert not scoring_errors(detail, contract=contract())
