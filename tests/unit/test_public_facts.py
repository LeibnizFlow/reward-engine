"""Real public fact computations, observation provenance, and registry integrity."""

import asyncio
from copy import deepcopy

import pytest

from reward_engine.core.outcome.context import build_context
from reward_engine.core.outcome.harness import Registry
from reward_engine.core.outcome.public_facts import (
    CHECKER,
    bind_observations,
    check_fact,
    install,
    run,
    select,
    validate_binding,
    validate_predicate,
)


def inputs(value):
    return build_context(
        {"task_id": "delivery", "prompt": "Save report.md; do not send it."},
        [{"source_documents": [{"value": value}]}],
    )


def descriptor(**changes):
    return {
        "id": "observation-1",
        "source_ref": "product_1:source:0",
        "path": ["value"],
        "object": "delivery",
        "property": "contents",
        "phase": "terminal",
        "sequence": 0,
        "complete": True,
        "status": "observed",
        "observed_at": "2026-09-30T10:00:00+00:00",
        **changes,
    }


def predicate(**changes):
    return {
        "object": "delivery",
        "property": "contents",
        "phase": "terminal",
        "op": "contains",
        "expected": "report.md",
        **changes,
    }


@pytest.mark.parametrize(
    "actual,expected,op,complete,status",
    [
        (["report.md"], "report.md", "contains", True, "pass"),
        (["output/report.md"], "report.md", "contains", True, "fail"),
        (["report.md"], "report.md", "contains", False, "pass"),
        ([], "report.md", "contains", False, "unknown"),
        (["sent"], "sent", "forbids", False, "fail"),
        ([], "sent", "forbids", True, "pass"),
        ([], "sent", "forbids", False, "unknown"),
        (
            [{"path": "report.md", "size": 3}],
            {"size": 3, "path": "report.md"},
            "contains",
            True,
            "pass",
        ),
        (
            [{"path": "report.md", "size": 3}],
            {"path": "report.md"},
            "contains",
            True,
            "fail",
        ),
        ({"total": 3, "ok": True}, {"ok": True, "total": 3}, "equals", True, "pass"),
        (True, 1, "equals", True, "fail"),
        (3, 3, "equals", False, "unknown"),
        (3, 4, "equals", False, "unknown"),
        (None, None, "equals", True, "pass"),
    ],
)
def test_structural_comparison_respects_types_and_observation_coverage(
    actual, expected, op, complete, status
):
    context, evidence = inputs(actual)
    bound = bind_observations(context, evidence, [descriptor(complete=complete)])
    result = check_fact(predicate(expected=expected, op=op), bound, evidence)
    assert result["status"] == status
    assert result["error"] is False
    assert result["evidence_refs"] == ["product_1:source:0"]
    if status != "unknown":
        assert result["observed"] == actual
        assert result["expected"] == expected
        assert result["observation_ids"] == ["observation-1"]
        assert bool(result["gap"]) is (status == "fail")
    else:
        assert result["unknown_kind"] == "evidence"


def test_membership_does_not_treat_free_text_as_an_inventory():
    context, evidence = inputs("report.md")
    context = bind_observations(context, evidence, [descriptor()])
    result = check_fact(predicate(), context, evidence)
    assert result["status"] == "unknown"
    assert result["error"] is True
    assert result["unknown_kind"] == "execution"


@pytest.mark.parametrize(
    "status,kind", [("error", "execution"), ("unavailable", "evidence")]
)
def test_failed_observations_do_not_require_a_value_at_the_path(status, kind):
    context, evidence = inputs({})
    context = bind_observations(
        context, evidence, [descriptor(path=["not_present"], status=status)]
    )
    result = check_fact(predicate(), context, evidence)
    assert result["status"] == "unknown"
    assert result["unknown_kind"] == kind


def test_select_traverses_nested_lists_and_dicts_without_coercion():
    value = {"files": [{"path": "report.md"}]}
    assert select(value, ["files", 0, "path"]) == "report.md"
    assert select(value, []) == value


@pytest.mark.parametrize(
    "path",
    [
        "files",
        ["files", -1],
        ["files", True],
        ["files", "0"],
        ["files", 1],
        ["files", 0, "path", 0],
        ["files"] * 17,
    ],
)
def test_select_rejects_invalid_or_unbounded_paths(path):
    with pytest.raises(ValueError, match="Observation path"):
        select({"files": [{"path": "report.md"}]}, path)


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"id": ""}, "IDs must be unique"),
        ({"id": 1}, "IDs must be unique"),
        ({"source_ref": "task"}, "candidate's public"),
        ({"source_ref": "product_2:source:0"}, "candidate's public"),
        ({"object": ""}, "explicit coverage"),
        ({"property": None}, "explicit coverage"),
        ({"phase": "future"}, "explicit coverage"),
        ({"sequence": True}, "explicit coverage"),
        ({"sequence": -1}, "explicit coverage"),
        ({"complete": 1}, "explicit coverage"),
        ({"status": "pass"}, "explicit coverage"),
        ({"observed_at": ""}, "explicit coverage"),
        ({"observed_at": "2026-09-30T10:00:00"}, "timezone"),
    ],
)
def test_invalid_observation_descriptors_are_rejected(changes, match):
    context, evidence = inputs(["report.md"])
    with pytest.raises(ValueError, match=match):
        bind_observations(context, evidence, [descriptor(**changes)])


def test_binding_copies_descriptors_and_context_and_cannot_be_rebound():
    context, evidence = inputs(["report.md"])
    rows = [descriptor()]
    before = deepcopy((context, evidence, rows))
    bound = bind_observations(context, evidence, rows)
    assert (context, evidence, rows) == before
    rows[0]["path"] = ["different"]
    assert check_fact(predicate(), bound, evidence)["status"] == "pass"
    with pytest.raises(ValueError, match="already frozen"):
        bind_observations(bound, evidence, [])
    with pytest.raises(ValueError, match="IDs must be unique"):
        bind_observations(context, evidence, [descriptor(), descriptor()])


@pytest.mark.parametrize("count", [0, 2])
def test_observations_require_exactly_one_candidate(count):
    context, evidence = inputs([])
    context["products"] *= count
    with pytest.raises(ValueError, match="exactly one"):
        bind_observations(context, evidence, [])


@pytest.mark.parametrize(
    "mutation", ["task", "product", "evidence", "missing_evidence"]
)
def test_binding_rejects_changed_candidate_or_public_inputs(mutation):
    context, evidence = inputs(["report.md"])
    bound = bind_observations(context, evidence, [descriptor()])
    if mutation == "task":
        bound["task"]["prompt"] = "Different task"
    elif mutation == "product":
        bound["products"][0]["sources"] = []
    elif mutation == "evidence":
        evidence["product_1:source:0"]["value"] = []
    else:
        del evidence["product_1:source:0"]
    with pytest.raises(ValueError, match="candidate binding changed"):
        check_fact(predicate(), bound, evidence)


def test_source_digest_is_checked_before_a_fact_is_accepted():
    context, evidence = inputs(["report.md"])
    bound = bind_observations(context, evidence, [descriptor()])
    bound["public_reward_observations"]["rows"][0]["source_sha256"] = "invalid"
    with pytest.raises(ValueError, match="Observation source changed"):
        check_fact(predicate(), bound, evidence)


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        {},
        predicate(object=""),
        predicate(property=1),
        predicate(phase="intermediate"),
        predicate(op="substring"),
        predicate(phase="history", op="equals"),
    ],
)
def test_invalid_predicates_cannot_be_scored(invalid):
    with pytest.raises(ValueError, match="Fact check needs|History checks"):
        validate_predicate(invalid)


def test_unbound_or_wrong_property_and_missing_history_are_unknown():
    context, evidence = inputs(["report.md"])
    assert validate_binding(context, evidence) == []
    assert check_fact(predicate(), context, evidence)["status"] == "unknown"
    bound = bind_observations(context, evidence, [descriptor()])
    assert (
        check_fact(predicate(property="other"), bound, evidence)["status"] == "unknown"
    )
    result = check_fact(predicate(phase="history", op="forbids"), bound, evidence)
    assert result["status"] == "unknown"
    assert "phase is unavailable" in result["reason"]


def test_environment_observation_and_async_registry_execute_real_checker():
    context, evidence = inputs([])
    ref = context["products"][0]["environment_ref"]
    evidence[ref] = {"value": ["report.md"]}
    bound = bind_observations(context, evidence, [descriptor(source_ref=ref)])
    registry = Registry()
    install(registry)
    catalog = registry.catalog()
    install(registry)
    assert registry.catalog() == catalog
    assert registry.checkers[CHECKER][0] is run
    test = {
        "test_id": "delivery-check",
        "criterion_id": "delivery",
        "checker": CHECKER,
        "arguments": predicate(),
    }
    result = asyncio.run(registry.run(test, bound, evidence))
    assert result["status"] == "pass"
    assert result["measurement_kind"] == "deterministic"
    assert result["evidence_refs"] == [ref]


def test_install_rejects_preexisting_checker_with_same_name():
    registry = Registry()
    # Use a real, different callable; it must never be invoked by install().
    registry.register(CHECKER, check_fact, "Conflicting implementation")
    with pytest.raises(ValueError, match="cannot be replaced"):
        install(registry)
