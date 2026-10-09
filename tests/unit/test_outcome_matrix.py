"""Matrix regressions adapted from archived test_claw_eval_outcome_matrix.py.

Use explicit protocol inputs instead of the removed legacy runner fixtures.
"""

import json
from copy import deepcopy

import pytest

from reward_engine.core.shared.outcome_matrix import (
    PROTOCOL_ID,
    normalize_matrix_reply,
    prepare_matrix_request,
)


def request_body(calibration=False):
    decision = "preference" if calibration else "verdict"
    body = {
        "schema_version": "reward_builder_v4.overall_calibration_judge_request.v1"
        if calibration
        else "reward_builder_v4.final_outcome_battle.v1",
        "task": {
            "prompt": "Prepare the report.\n1. Include totals.\n   Explain the calculation.\n2. Draft only; do not send."
        },
        "candidate_1": {
            "evidence": [{"evidence_ref": "terminal:a", "content": "First report"}]
        },
        "candidate_2": {
            "evidence": [{"evidence_ref": "terminal:b", "content": "Second report"}]
        },
        "instruction": "Judge the public Task outcome independently.",
        "response": {
            "reason": "Explain",
            "evidence_refs": ["Known aliases"],
            decision: "Allowed label",
        },
        "projection": {"criteria": ["delivery"]},
        "harness_evidence_by_criterion": [],
    }
    if calibration:
        body["evidence_contract"] = {"allowed_refs": ["terminal:a", "terminal:b"]}
    else:
        body["evidence_refs"] = ["terminal:a", "terminal:b"]
    return body


def reply(request, label="tie", *, missing=False):
    def cells():
        return {
            slot: {
                "fact": "未见可核实的材料。"
                if missing
                else "Observed product content.",
                "evidence_refs": []
                if missing
                else [row["evidence_ref"] for row in request[slot]["evidence"]],
            }
            for slot in ("candidate_1", "candidate_2")
        }

    decision = "verdict" if "verdict" in request["response"] else "preference"
    return {
        "requirement_comparisons": [
            {"requirement_id": row["requirement_id"], **cells()}
            for row in request["outcome_comparison_protocol"]["task_units"]
        ],
        "action_format_check": {
            "requested_boundary": "Prepare a draft without sending it.",
            **cells(),
        },
        "reason": "Independent judgment, not a sum of matrix rows.",
        "evidence_refs": [] if missing else ["terminal:a", "terminal:b"],
        decision: label,
    }


@pytest.mark.parametrize("calibration", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_request_enrichment_is_lossless_idempotent_and_candidate_order_agnostic(
    calibration, reverse
):
    body = request_body(calibration)
    if reverse:
        body["candidate_1"], body["candidate_2"] = (
            body["candidate_2"],
            body["candidate_1"],
        )
    before = deepcopy(body)
    request = prepare_matrix_request(body)
    assert body == before
    assert prepare_matrix_request(request) == request
    for key in body.keys() - {"instruction", "response"}:
        assert request[key] == body[key]
    assert request["outcome_comparison_protocol"] == {
        "id": PROTOCOL_ID,
        "task_units": [
            {"requirement_id": "task-1", "text": "Prepare the report."},
            {
                "requirement_id": "task-2",
                "text": "1. Include totals.\n   Explain the calculation.",
            },
            {"requirement_id": "task-3", "text": "2. Draft only; do not send."},
        ],
    }
    decision = "preference" if calibration else "verdict"
    assert list(request["response"]) == [
        "requirement_comparisons",
        "action_format_check",
        "reason",
        "evidence_refs",
        decision,
    ]
    assert request["response"][decision] == body["response"][decision]
    assert (
        "Extra actions or stronger commitments receive no automatic credit"
        in request["instruction"]
    )
    assert "Do not invent a required file" in request["instruction"]
    # Deep copy isolation includes candidate evidence, not just the outer request.
    request["candidate_1"]["evidence"][0]["content"] = "Changed copy"
    assert body == before


@pytest.mark.parametrize(
    "prompt,texts",
    [
        ("Do this.\n\nAnd that.", ["Do this.", "And that."]),
        ("Do this without executing it.", ["Do this without executing it."]),
        (
            "Context\n1) First\n   continuation\n2) Second",
            ["Context", "1) First\n   continuation", "2) Second"],
        ),
        (
            "\n\n  First paragraph.\n \n Second paragraph.\n\n",
            ["First paragraph.", "Second paragraph."],
        ),
    ],
)
def test_task_units_preserve_all_public_text_and_continuation_lines(prompt, texts):
    body = request_body()
    body["task"]["prompt"] = prompt
    units = prepare_matrix_request(body)["outcome_comparison_protocol"]["task_units"]
    assert units == [
        {"requirement_id": f"task-{i}", "text": text} for i, text in enumerate(texts, 1)
    ]


@pytest.mark.parametrize("prompt", [None, "", " \n", 3])
def test_missing_public_prompt_is_rejected(prompt):
    body = request_body()
    body["task"]["prompt"] = prompt
    with pytest.raises(ValueError, match="original public Task prompt"):
        prepare_matrix_request(body)


@pytest.mark.parametrize("calibration", [False, True])
@pytest.mark.parametrize("label", ["first", "second", "tie", "uncertain"])
def test_normalization_preserves_entire_response_and_independent_label(
    calibration, label
):
    request = prepare_matrix_request(request_body(calibration))
    if label == "uncertain":
        label = "abstain" if calibration else "unknown"
    raw = reply(request, label, missing=label in {"unknown", "abstain"})
    before = deepcopy(raw)
    normalized, notes = normalize_matrix_reply(request, raw)
    decision = "preference" if calibration else "verdict"
    assert raw == before
    assert normalized == {
        "reason": json.dumps(raw, ensure_ascii=False, separators=(",", ":")),
        "evidence_refs": raw["evidence_refs"],
        decision: label,
    }
    assert json.loads(normalized["reason"]) == raw
    assert notes == [
        {
            "operation": "preserve_task_evidence_matrix_in_reason",
            "protocol_id": PROTOCOL_ID,
        }
    ]
    normalized["evidence_refs"].append("change-to-copy")
    assert raw == before


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("missing", "cover every"),
        ("duplicate", "unique known"),
        ("foreign", "unique known"),
        ("wrong_candidate", "correct candidate"),
        ("boundary_wrong_candidate", "correct candidate"),
        ("invented_alias", "correct candidate"),
        ("duplicate_alias", "unique known"),
        ("non_text_alias", "unique known"),
        ("string_aliases", "unique known"),
        ("extra_field", "fields differ"),
        ("missing_action_check", "fields differ"),
        ("blank_fact", "nonempty text"),
        ("blank_reason", "nonempty text"),
        ("blank_boundary", "nonempty text"),
        ("invalid_verdict", "Invalid independent"),
        ("rows_not_list", "must be a list"),
        ("cell_not_object", "fields differ"),
        ("unknown_final_ref", "known evidence"),
    ],
)
def test_invalid_structure_and_evidence_ownership_are_rejected_without_mutation(
    mutation, match
):
    request = prepare_matrix_request(request_body())
    raw = reply(request)
    row = raw["requirement_comparisons"][0]
    cell = row["candidate_1"]
    if mutation == "missing":
        raw["requirement_comparisons"] = []
    elif mutation == "duplicate":
        raw["requirement_comparisons"].append(deepcopy(row))
    elif mutation == "foreign":
        row["requirement_id"] = "invented"
    elif mutation == "wrong_candidate":
        cell["evidence_refs"] = ["terminal:b"]
    elif mutation == "boundary_wrong_candidate":
        raw["action_format_check"]["candidate_2"]["evidence_refs"] = ["terminal:a"]
    elif mutation == "invented_alias":
        cell["evidence_refs"] = ["invented"]
    elif mutation == "duplicate_alias":
        cell["evidence_refs"] *= 2
    elif mutation == "non_text_alias":
        cell["evidence_refs"] = [3]
    elif mutation == "string_aliases":
        cell["evidence_refs"] = "terminal:a"
    elif mutation == "extra_field":
        row["winner"] = "candidate_2"
    elif mutation == "missing_action_check":
        del raw["action_format_check"]
    elif mutation == "blank_fact":
        cell["fact"] = " "
    elif mutation == "blank_reason":
        raw["reason"] = " "
    elif mutation == "blank_boundary":
        raw["action_format_check"]["requested_boundary"] = " "
    elif mutation == "invalid_verdict":
        raw["verdict"] = "candidate_1"
    elif mutation == "rows_not_list":
        raw["requirement_comparisons"] = {}
    elif mutation == "cell_not_object":
        row["candidate_1"] = "Observed"
    elif mutation == "unknown_final_ref":
        raw["evidence_refs"].append("unmeasured")
    before = deepcopy((request, raw))
    with pytest.raises(ValueError, match=match):
        normalize_matrix_reply(request, raw)
    assert (request, raw) == before


def test_calibration_final_citations_respect_measured_allowlist():
    request = prepare_matrix_request(request_body(calibration=True))
    request["evidence_refs"] = ["unmeasured"]
    raw = reply(request)
    assert normalize_matrix_reply(request, raw)[0]["preference"] == "tie"
    raw["evidence_refs"].append("unmeasured")
    with pytest.raises(ValueError, match="known evidence"):
        normalize_matrix_reply(request, raw)


def test_legacy_inputs_are_unchanged_and_not_enriched():
    original = request_body()
    legacy = {"reason": "Existing judgment", "verdict": "second", "evidence_refs": []}
    assert normalize_matrix_reply(original, legacy) == (legacy, [])
    unrelated = {
        "schema_version": "criterion-battle",
        "response": {"criterion_findings": []},
    }
    result = prepare_matrix_request(unrelated)
    assert result == unrelated
    result["response"]["criterion_findings"].append("copy")
    assert unrelated["response"]["criterion_findings"] == []


def test_protocol_conflicts_are_rejected():
    request = prepare_matrix_request(request_body())
    request["task"]["prompt"] = "A different task"
    with pytest.raises(ValueError, match="Conflicting Outcome"):
        prepare_matrix_request(request)


@pytest.mark.parametrize("field", ["schema_version", "protocol_id"])
def test_unknown_protocols_cannot_normalize_a_reply(field):
    request = prepare_matrix_request(request_body())
    raw = reply(request)
    if field == "schema_version":
        request[field] = "unsupported"
    else:
        request["outcome_comparison_protocol"]["id"] = "unsupported"
    with pytest.raises(ValueError, match="Unsupported Outcome"):
        normalize_matrix_reply(request, raw)


def measurements_body(values, *, calibration=True):
    body = request_body(calibration)
    body["harness_evidence_by_criterion"] = [
        {
            "criterion_id": f"criterion-{index}",
            "harness_evidence": {
                "test_results": [
                    {
                        "test_id": f"test-{index}",
                        "status": "pass" if index == 0 else "fail",
                        "evidence_refs": ["terminal:a" if index == 0 else "terminal:b"],
                        "measurements": {"observed": deepcopy(value)},
                    }
                ]
            },
        }
        for index, value in enumerate(values)
    ]
    return body


def test_repeated_large_measurements_expand_losslessly_with_each_receipt_intact():
    value = {
        "report": "完整观察\n" * 400,
        "typed": [None, False, 0, {"literal": "No summary"}],
    }
    other = {"different": "Not duplicated"}
    body = measurements_body([value, other, value, value])
    before = deepcopy(body)
    request = prepare_matrix_request(body)
    assert body == before
    assert prepare_matrix_request(request) == request
    assert request["shared_measurements"] == {"measurement-1": value}
    expanded = deepcopy(request["harness_evidence_by_criterion"])
    for row in expanded:
        for test in row["harness_evidence"]["test_results"]:
            measured = test["measurements"]["observed"]
            if set(measured) == {"shared_measurement_ref"}:
                test["measurements"]["observed"] = deepcopy(
                    request["shared_measurements"][measured["shared_measurement_ref"]]
                )
    assert expanded == before["harness_evidence_by_criterion"]
    assert "Expand it to the complete original JSON value" in request["instruction"]
    assert len(json.dumps(request)) < len(json.dumps(body))
    raw = reply(request)
    normalized, _ = normalize_matrix_reply(request, raw)
    assert json.loads(normalized["reason"]) == raw


@pytest.mark.parametrize(
    "values",
    [
        [{"body": "x" * 2048}, {"body": "x" * 2047 + "y"}],
        [{"body": "short"}, {"body": "short"}],
        [{"shared_measurement_ref": "x" * 2048}] * 2,
        [{"body": "x" * 2048}] * 2 + [{"shared_measurement_ref": "reserved"}],
    ],
)
def test_unequal_small_or_reserved_measurements_are_not_factored(values):
    body = measurements_body(values)
    request = prepare_matrix_request(body)
    assert "shared_measurements" not in request
    assert (
        request["harness_evidence_by_criterion"]
        == body["harness_evidence_by_criterion"]
    )


def test_terminal_jury_does_not_factor_measurements():
    body = measurements_body([{"body": "x" * 2048}] * 2, calibration=False)
    request = prepare_matrix_request(body)
    assert "shared_measurements" not in request
    assert (
        request["harness_evidence_by_criterion"]
        == body["harness_evidence_by_criterion"]
    )


def test_factoring_rejects_existing_shared_table_without_overwriting_input():
    body = measurements_body([{"body": "x" * 2048}] * 2)
    body["shared_measurements"] = {"measurement-1": "existing"}
    before = deepcopy(body)
    with pytest.raises(ValueError, match="Conflicting shared_measurements"):
        prepare_matrix_request(body)
    assert body == before
