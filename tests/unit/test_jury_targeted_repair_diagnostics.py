"""Captured WB failure shapes, strict edit parity and offline canonical Roles."""

from copy import deepcopy

import pytest

from reward_engine.core.outcome import jury
from reward_engine.core.outcome.context import build_context
from reward_engine.core.outcome.jury_drafts import expectations
from reward_engine.core.outcome.jury_repair import apply_edits, diagnostics
from reward_engine.core.outcome.reducer_projection import expand_texts
from tests.unit.test_claw_outcome_jury_details import criterion, decision
from tests.unit.test_outcome_role_contract_recovery import roles as roles


def edit(path, value=None, *, op="replace"):
    return {
        "op": op,
        "path": path,
        "value": value,
        "reason": "Explicit fixture correction",
    }


def captured_shape():
    """Synthetic content with the retained R2 indices, scopes and wire shapes."""
    context, evidence = build_context(
        {"task_id": "json", "prompt": "Deliver valid JSON and explain it."},
        [{"artifact": {"final_response": "{"}} for _ in range(6)],
    )
    row = criterion()
    row["subrequirements"][0] = (
        "Valid JSON with the following explicit requirements. " * 5
    )
    expected = expectations([row] * 22, "seat", "before_products")
    gaps = []
    # Product omissions observed at gaps 0, 1, 2, 4; no host-authored repair refs.
    for i, missing in enumerate(([2, 3, 4, 6], [1, 2], [2], [], [2, 6], [])):
        gaps.append(
            {
                "gap_id": f"syntax_{i}",
                "product_ids": [f"product_{j}" for j in range(1, 7)],
                "expected": "Valid JSON",
                "expected_basis": "task",
                "expected_refs": ["task"],
                "observed": "Responses contain an opening brace",
                "gap": "Possibly invalid JSON",
                "evidence_refs": [
                    f"product_{j}:artifact:final_response"
                    for j in range(1, 7)
                    if j not in missing
                ],
                "verification": "Parse each actual response",
                "status": "needs_check",
            }
        )
    draft = {
        "criteria": [row],
        "gaps": gaps,
        "coverage": [decision(item) for item in expected],
        "uncertainties": [],
    }
    draft["coverage"][21]["subrequirements"].append(
        {
            **deepcopy(draft["coverage"][21]["subrequirements"][0]),
            "part_id": "seat:21/3",
        }
    )
    inputs = {
        "independent_expectations": expected,
        "proposal_expectations": [],
        "proposals": [{"member": "a"}, {"member": "b"}],
        "revision_protocol": True,
    }
    return context, evidence, inputs, draft


def test_all_missing_product_scopes_and_coverage_are_diagnostic_only():
    _, evidence, inputs, draft = captured_shape()
    # A recorded check binds only its own product, and an unavailable product-like
    # handle does not count merely because its spelling has the right prefix.
    evidence["check:2:2"] = {
        "product_id": "product_2",
        "status": "unknown",
        "error": True,
    }
    draft["gaps"][2]["evidence_refs"].extend(["check:2:2", "task"])
    draft["gaps"][4]["evidence_refs"].extend(["task", "product_2:nonexistent"])
    before = deepcopy((draft, evidence, inputs))
    issues = diagnostics(
        draft,
        inputs["independent_expectations"],
        evidence=evidence,
        product_ref=jury.product_ref,
    )
    scoped = {tuple(i["path"]): i for i in issues if "missing_product_ids" in i}
    assert {path[1]: i["missing_product_ids"] for path, i in scoped.items()} == {
        0: ["product_2", "product_3", "product_4", "product_6"],
        1: ["product_1", "product_2"],
        4: ["product_2", "product_6"],
    }
    assert any(i["path"] == ["coverage", 21, "subrequirements", 3] for i in issues)
    assert all("allowed_refs" not in i and "value" not in i for i in scoped.values())
    assert (draft, evidence, inputs) == before


@pytest.mark.parametrize(
    "draft", [None, [], {"criteria": None}, {"gaps": None}, {"coverage": None}]
)
def test_diagnostics_do_not_mask_malformed_definition_errors(draft):
    before = deepcopy(draft)
    assert diagnostics(draft, [], evidence={}, product_ref=jury.product_ref)
    assert draft == before


@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize("repair_succeeds", [False, True])
@pytest.mark.asyncio
async def test_actual_role_captured_repair_chain_is_bounded_atomic_and_grounded(
    roles, bounded, repair_succeeds
):
    make, calls = roles
    context, evidence, inputs, draft = captured_shape()
    before = deepcopy((context, evidence, inputs, draft))
    records = {}
    # Same failing index/op as retained repair 1, after a valid prefix including removal.
    first = {
        "edits": [edit(["criteria", 0], draft["criteria"][0])] * 7
        + [
            edit(["gaps", 5], op="remove"),
            edit(["gaps", 5], draft["gaps"][5], op="add"),
        ]
    }
    # Same first missing path as retained repair 2; the later coverage[21] edit
    # is never reached. No prefix may be published or used as the next base.
    second = {
        "edits": [edit(["criteria", 0], draft["criteria"][0])] * 9
        + [
            edit(["coverage", 3, "subrequirements", 3], {}),
            edit(["coverage", 21, "subrequirements", 3], {}),
        ]
    }

    def reply(wire):
        request = expand_texts(wire)
        if request["mode"] == "outcome_jury_reduce":
            return deepcopy(draft)
        repair = request["validation_repair"]
        assert repair["invalid_draft"] == draft
        assert request["independent_expectations"] == inputs["independent_expectations"]
        assert "shared_texts" in wire
        assert isinstance(
            wire["validation_repair"]["invalid_draft"]["criteria"][0][
                "subrequirements"
            ][0],
            dict,
        )
        assert "$reducer_text" in request["instruction"]
        assert "no edits were applied" in request["instruction"]
        scopes = [i for i in repair["field_issues"] if "missing_product_ids" in i]
        assert [i["path"][1] for i in scopes] == [0, 1, 2, 4]
        if "invalid_edits" not in repair:
            assert (
                repair["reason"] == "Gap must cite evidence for every affected product"
            )
            return deepcopy(first)
        assert repair["invalid_edits"] == first
        failure = repair["edit_failure"]
        assert failure["edit_index"] == 8 and failure["op"] == "add"
        assert failure["path"] == ["gaps", 5]
        assert failure["parent_length"] == 5
        assert failure["batch_applied"] is False and failure["applied_edit_count"] == 0
        if not repair_succeeds:
            return deepcopy(second)
        # These are explicit model fixture choices; diagnostics do not select refs.
        return {
            "edits": [
                edit(
                    ["gaps", i, "evidence_refs"],
                    [f"product_{j}:artifact:final_response" for j in range(1, 7)],
                )
                for i in (0, 1, 2, 4)
            ]
            + [edit(["coverage", 21, "subrequirements", 3], op="remove")]
        }

    reducer = make("reducer", reply, bounded=bounded)
    assert reducer.jury_repair_mode == "targeted"
    run = jury.reduce_proposals(
        context,
        evidence,
        inputs,
        reducer,
        record=lambda key, value: records.update({key: deepcopy(value)}),
    )
    if repair_succeeds:
        output = await run
        assert output["criteria"] == draft["criteria"]
        assert len(output["gaps"]) == 6
        assert all(g["status"] == "needs_check" for g in output["gaps"])
        assert len(output["coverage"]) == 22
        assert all(len(row["subrequirements"]) == 3 for row in output["coverage"])
        assert jury.validate(output, evidence, context) == output
    else:
        with pytest.raises(
            ValueError, match="Repair path must already exist"
        ) as caught:
            await run
        failure = caught.value.details
        assert failure["edit_index"] == 9
        assert failure["path"] == ["coverage", 3, "subrequirements", 3]
        assert failure["parent_length"] == 3 and failure["parent_type"] == "list"
        assert records["jury_reduce_validation_2"]["edit_failure"] == failure
        assert "jury" not in records
    assert len(calls) == 3
    assert [req["mode"] for _, req in calls] == [
        "outcome_jury_reduce",
        "outcome_jury_repair",
        "outcome_jury_repair",
    ]
    assert all("response_recovery" not in req for _, req in calls)
    assert records["jury_reduce_raw"] == draft
    assert "jury_reduce_repair_1_raw" not in records
    assert (context, evidence, inputs, draft) == before


@pytest.mark.parametrize(
    "bad",
    [
        None,
        [],
        "edit",
        {},
        edit(["gaps", 0], op="add"),
        edit(["gaps", 0], op=True),
        edit(["gaps", 0], op=[]),
        edit([]),
        edit("gaps"),
        edit([[]]),
        edit(["private", 0]),
        edit(["gaps", -1]),
        edit(["gaps", True]),
        edit(["gaps", False]),
        edit(["gaps", 0.0]),
        edit(["gaps", 1]),
        edit(["gaps", 0, "missing"]),
        edit(["gaps", 0, "missing", "child"]),
        edit(["criteria", 0, "id", "child"]),
        {"op": "replace", "path": ["gaps", 0], "reason": "Missing value"},
        {**edit(["gaps", 0]), "reason": " "},
        {**edit(["gaps", 0]), "reason": False},
    ],
)
def test_rejections_keep_valid_prefix_atomic_and_report_parent_shape(bad):
    draft = {"criteria": [{"id": "x"}], "gaps": [{"gap_id": "original"}]}
    reply = {"edits": [edit(["criteria", 0, "id"], "temporary"), bad]}
    before = deepcopy((draft, reply))
    with pytest.raises((ValueError, TypeError)) as caught:
        apply_edits(draft, reply)
    details = caught.value.details
    assert details["edit_index"] == 1
    assert details["batch_applied"] is False and details["applied_edit_count"] == 0
    assert details["repair_base"] == "validation_repair.invalid_draft"
    assert (draft, reply) == before


@pytest.mark.parametrize(
    "reply", [{}, {"edits": []}, {"edits": None}, {"edits": [edit(["gaps"])] * 65}]
)
def test_malformed_batches_keep_existing_limit(reply):
    with pytest.raises(ValueError, match="one to 64 explicit edits") as caught:
        apply_edits({"gaps": []}, reply)
    assert caught.value.details["edit_index"] is None
    assert caught.value.details["batch_applied"] is False


@pytest.mark.parametrize(
    "path,parent_path,parent_type,shape",
    [
        (["gaps", 0, "missing"], ["gaps", 0], "dict", {"parent_keys": ["gap_id"]}),
        (
            ["gaps", 0, "missing", "child"],
            ["gaps", 0],
            "dict",
            {"unresolved_component_index": 2},
        ),
        (["gaps", 1], ["gaps"], "list", {"parent_length": 1}),
        (["gaps", 0, "gap_id", "child"], ["gaps", 0, "gap_id"], "str", {}),
    ],
)
def test_failing_path_reports_nearest_existing_parent(
    path, parent_path, parent_type, shape
):
    with pytest.raises(ValueError, match="Repair path must already exist") as caught:
        apply_edits({"gaps": [{"gap_id": "g"}]}, {"edits": [edit(path)]})
    details = caught.value.details
    assert details["path"] == path and details["op"] == "replace"
    assert (
        details["parent_path"] == parent_path and details["parent_type"] == parent_type
    )
    assert all(details[key] == value for key, value in shape.items())


@pytest.mark.parametrize(
    "value", [None, False, True, 0, 0.25, "", "text", [], {}, {"a": [False, None]}]
)
def test_arbitrary_json_replacement_is_unchanged_and_copied(value):
    draft = {"gaps": [{"gap_id": "g"}]}
    reply = {"edits": [edit(["gaps", 0], value)]}
    before = deepcopy((draft, reply))
    result = apply_edits(draft, reply)
    assert result == {"gaps": [value]}
    if isinstance(value, (dict, list)):
        assert result["gaps"][0] is not value
    assert (draft, reply) == before


def test_existing_parent_replacement_and_descending_removals_remain_accepted():
    draft = {"gaps": [{"gap_id": "old"}], "coverage": ["a", "b", "c", "d"]}
    reply = {
        "edits": [
            edit(["gaps", 0], {"gap_id": "old", "evidence_refs": []}),
            edit(["gaps", 0, "evidence_refs"], ["model-selected-ref"]),
            edit(["coverage", 3], op="remove"),
            edit(["coverage", 1], op="remove"),
        ]
    }
    before = deepcopy((draft, reply))
    assert apply_edits(draft, reply) == {
        "gaps": [{"gap_id": "old", "evidence_refs": ["model-selected-ref"]}],
        "coverage": ["a", "c"],
    }
    assert (draft, reply) == before
