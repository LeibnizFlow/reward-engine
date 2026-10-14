"""Native-v4-shaped missing-why diagnostics with offline model-authored repairs."""

import socket
from collections import Counter
from copy import deepcopy

import pytest

from reward_engine.core.outcome import jury
from reward_engine.core.outcome.context import ResponseContractError, build_context
from reward_engine.core.outcome.jury_drafts import expectations
from reward_engine.core.outcome.jury_revision import materialize
from reward_engine.core.outcome.reducer_projection import expand_texts
from tests.unit.test_outcome_jury_revision import criterion
from tests.unit.test_outcome_role_contract_recovery import roles as roles


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Retain controls permit only offline MockTransport")

    monkeypatch.setattr(socket.socket, "connect", forbidden)


def captured_shape():
    row = {
        **criterion(),
        "why": "The Task explicitly requires complete JSON with every exception.",
        "priority_reason": "Both syntax and exceptions are necessary for use.",
    }
    initial = expectations([row], "seat", "before_products")
    decision = {
        "source_parts": [p["part_id"] for p in initial[0]["subrequirements"]],
        "action": "retain",
        "targets": [{"criterion_id": row["id"], "part_index": i} for i in range(2)],
    }
    return initial, {"criteria": [row], "revision_decisions": [decision]}


def reject(raw, initial):
    before = deepcopy((raw, initial))
    with pytest.raises(ResponseContractError, match="Retain must preserve") as caught:
        materialize(raw, initial, {"task": "Task", "source:1": "Observation"})
    assert (raw, initial) == before
    assert caught.value.details["decision_index"] == 0
    return caught.value.details


def test_missing_why_reports_each_source_and_exact_absence_without_filling():
    initial, raw = captured_shape()
    why = raw["criteria"][0].pop("why")
    details = reject(raw, initial)
    assert details["source_count"] == details["target_count"] == 2
    for i, mismatch in enumerate(details["retain_mismatches"]):
        assert mismatch == {
            "source_part_id": f"seat:0/{i}",
            "target": {"criterion_id": "json", "part_index": i},
            "field_differences": [
                {
                    "field": "why",
                    "source_present": True,
                    "target_present": False,
                    "source_value": why,
                }
            ],
        }
    assert "why" not in raw["criteria"][0]
    # Explicitly restoring the model's original value is a positive control,
    # not a host default or a repair to any historical native receipt.
    raw["criteria"][0]["why"] = why
    result = materialize(raw, initial, {"task": "Task"})
    assert result["criteria"] == raw["criteria"]
    assert result["revision_decisions"] == raw["revision_decisions"]


@pytest.mark.parametrize(
    ("field", "value", "reported"),
    [
        ("why", "Only syntax matters.", "why"),
        ("evidence_refs", ["task", "source:1"], "evidence_refs"),
        ("subrequirements", ["Valid JSON", "Keep some exceptions"], "text"),
        ("priority", "minor", "priority"),
    ],
)
def test_changed_meaning_still_rejects_with_exact_values(field, value, reported):
    initial, raw = captured_shape()
    raw["criteria"][0][field] = value
    details = reject(raw, initial)
    changes = [d for m in details["retain_mismatches"] for d in m["field_differences"]]
    assert {d["field"] for d in changes} == {reported}
    assert all(d["source_present"] and d["target_present"] for d in changes)
    assert all(d["source_value"] != d["target_value"] for d in changes)
    # Existing explicit rewrites still require a model-authored reason/evidence.
    raw["revision_decisions"][0]["action"] = "rewrite"
    with pytest.raises(ResponseContractError, match="explicit reason"):
        materialize(raw, initial, {"task": "Task"})
    raw["revision_decisions"][0].update(
        reason="Explicit proposed change", evidence_refs=["task"]
    )
    assert materialize(raw, initial, {"task": "Task"})["criteria"] == raw["criteria"]


@pytest.mark.parametrize("added", [False, True])
def test_absence_is_distinct_from_authored_null(added):
    initial, raw = captured_shape()
    if added:
        raw["criteria"][0]["condition"] = None
    else:
        initial[0]["condition"] = None
    details = reject(raw, initial)
    diff = details["retain_mismatches"][0]["field_differences"][0]
    assert diff["field"] == "condition"
    assert diff["source_present"] is not added
    assert diff["target_present"] is added
    assert ("source_value" in diff) is not added
    assert ("target_value" in diff) is added


def test_mismatched_cardinality_still_rejects_without_invented_pair():
    initial, raw = captured_shape()
    raw["revision_decisions"][0]["targets"].pop()
    details = reject(raw, initial)
    assert (details["source_count"], details["target_count"]) == (2, 1)
    assert details["retain_mismatches"] == []


@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize("repair", ["restore", "inherit", "unchanged"])
@pytest.mark.asyncio
async def test_actual_role_map_schema_recovery_and_existing_fallback(
    roles, bounded, repair
):
    make, calls = roles
    initial, raw = captured_shape()
    row = raw["criteria"][0]
    saved = deepcopy(row)
    records = {}
    context, evidence = build_context(
        {"task_id": "json", "prompt": "Deliver complete JSON and every exception."}
    )

    def member(request):
        if request["mode"] == "outcome_expectations":
            assert "why" in request["response"]["criteria"][0]
            return {"criteria": [deepcopy(row)]}
        assert request["mode"] == "outcome_jury_map"
        assert "why" in request["response"]["criteria"][0]
        before = request["independent_expectations"][0]
        answer = deepcopy(raw)
        answer["revision_decisions"][0]["source_parts"] = [
            p["part_id"] for p in before["subrequirements"]
        ]
        if "response_recovery" not in request or repair == "unchanged":
            answer["criteria"][0].pop("why")
        if "response_recovery" in request:
            recovery = request["response_recovery"]
            assert recovery["attempt"] == 1
            assert "why" not in recovery["rejected_response"]["criteria"][0]
            assert all(
                [d["field"] for d in m["field_differences"]] == ["why"]
                for m in recovery["details"]["retain_mismatches"]
            )
            if repair == "inherit":
                return {"criteria": [{"inherit_from": before["expectation_id"]}]}
        return answer

    def reduce(wire):
        request = expand_texts(wire)
        for proposal in request["proposals"]:
            if repair == "unchanged":
                assert proposal["revision_ledger_failure"]["retain_mismatches"]
                assert all(
                    d["action"] == "defer" for d in proposal["revision_decisions"]
                )
                assert len(proposal["unclaimed_additions"]) == 2
            else:
                assert "revision_ledger_failure" not in proposal
                assert all(
                    d["action"] == "retain" for d in proposal["revision_decisions"]
                )
        return {"criteria": [deepcopy(row)]}

    result = await jury.propose(
        context,
        evidence,
        {str(i): make(f"member_{i}", member, bounded=bounded) for i in range(3)},
        make("reducer", reduce, bounded=bounded),
        revision_protocol=True,
        record=lambda name, value: records.update({name: deepcopy(value)}),
    )
    assert result["criteria"] == [saved]
    assert row == saved
    assert Counter(r["mode"] for _, r in calls) == {
        "outcome_expectations": 3,
        "outcome_jury_map": 6,
        "outcome_jury_reduce": 1,
    }
    assert sum("response_recovery" in r for _, r in calls) == 3
    assert records["jury_map"]["succeeded"] == 3
