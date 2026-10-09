"""Reduce selects clauses; code carries their exact words and source identity."""

from copy import deepcopy

import pytest

from reward_engine.core.outcome import builder, jury
from reward_engine.core.outcome.context import build_context
from reward_engine.core.outcome.harness import defaults
from reward_engine.core.outcome.jury_drafts import (
    apply_part_decisions,
    selection_inputs,
)


def sample():
    criterion = {"id": "correctness", "standard": "Respect the Task contract", "kind": "soft",
                 "origin": "domain", "evidence_refs": ["task"], "measurement_question": "Is the result valid?"}
    exp = {**criterion, "expectation_id": "seat:observed:0", "subrequirements": [
        {"part_id": "seat:observed:0/0", "text": "Use `a+b`, equivalent to `b+a`; only for commuting values."},
        {"part_id": "seat:observed:0/1", "text": "Always double every result."},
        {"part_id": "seat:observed:0/2", "text": "Optional chart."}]}
    inputs = {"proposals": [{}, {}], "independent_expectations": [], "proposal_expectations": [exp]}
    return criterion, exp, inputs


def test_selection_ids_are_short_unique_and_do_not_mutate_source():
    _, exp, inputs = sample()
    original = deepcopy(inputs)
    inputs["independent_expectations"] = [{**exp, "expectation_id": "before"}]
    view = selection_inputs(inputs)
    ids = [p["selection_id"] for r in view["independent_expectations"] + view["proposal_expectations"]
           for p in r["subrequirements"]]
    assert ids == ["p0", "p1", "p2", "p3", "p4", "p5"]
    assert inputs["proposal_expectations"] == original["proposal_expectations"]


def test_retain_is_exact_rewrite_is_explicit_and_omission_stays_pending():
    c, exp, inputs = sample()
    raw = {"criteria": [c], "part_decisions": [
        {"parts": ["p0"], "action": "retain", "criterion_id": c["id"]},
        {"parts": ["p1"], "action": "rewrite", "criterion_id": c["id"],
         "text": "Double only when the Task asks for it.", "reason": "No unconditional request",
         "evidence_refs": ["task"]}]}
    original = deepcopy(raw)
    result = apply_part_decisions(raw, inputs, {"task": "Add two values"})
    assert result["criteria"][0]["subrequirements"] == [exp["subrequirements"][0]["text"],
                                                         "Double only when the Task asks for it."]
    assert result["criteria"][0]["part_revisions"][0]["evidence_refs"] == ["task"]
    rows = jury.reconcile_coverage([exp], result)
    assert rows[0]["status"] == "deferred"
    assert [p["status"] for p in rows[0]["subrequirements"]] == ["mapped", "mapped", "deferred"]
    assert raw == original


@pytest.mark.parametrize("bad", [
    [{"parts": ["invented"], "action": "retain", "criterion_id": "correctness"}],
    [{"parts": ["p0", "p0"], "action": "retain", "criterion_id": "correctness"}],
    [{"parts": ["p0"], "action": "retain", "criterion_id": "missing"}],
    [{"parts": ["p0"], "action": "rewrite", "criterion_id": "correctness", "text": "Changed"}],
    [{"parts": ["p0"], "action": "rewrite", "criterion_id": "correctness", "text": "Changed",
      "reason": "because", "evidence_refs": ["made-up-source"]}],
    [{"parts": ["p0"], "action": "exclude"}],
])
def test_invalid_selection_does_not_guess_a_clause_or_source(bad):
    c, _, inputs = sample()
    with pytest.raises(ValueError):
        apply_part_decisions({"criteria": [c], "part_decisions": bad}, inputs, {"task": "Actual task"})


def test_duplicate_across_groups_and_mixed_protocol_reject():
    c, _, inputs = sample()
    d = {"parts": ["p0"], "action": "retain", "criterion_id": c["id"]}
    with pytest.raises(ValueError, match="duplicate"):
        apply_part_decisions({"criteria": [c], "part_decisions": [d, d]}, inputs, {})
    with pytest.raises(ValueError, match="not both"):
        apply_part_decisions({"criteria": [c], "part_decisions": [], "coverage": []}, inputs, {})


@pytest.mark.asyncio
async def test_selected_clauses_reach_builder_without_model_recopy_or_parent_bookkeeping():
    c, exp, inputs = sample()
    context, evidence = build_context({"task_id": "x", "prompt": "Add two commuting values"})
    async def reducer(request):
        assert "part_decisions" in request["response"] and "coverage" not in request["response"]
        return {"criteria": [c], "part_decisions": [
            {"parts": ["p0"], "action": "retain", "criterion_id": c["id"]},
            {"parts": ["p1", "p2"], "action": "exclude", "reason": "Unrequested extras"}]}
    definition = await jury.reduce_proposals(context, evidence, inputs, reducer)
    assert definition["coverage"][0]["status"] == "mapped"
    async def compile(request):
        # The compiler receives the original wording, not a reducer paraphrase.
        assert exp["subrequirements"][0]["text"] in str(request)
        return {"rubrics": [{"criterion_id": c["id"], "anchors": {"fail": "Contradiction", "partial": "Unknown", "full": "Consistent"}}],
                "tests": [{"test_id": "check", "criterion_id": c["id"], "checker": "evidence.review",
                           "question": "Check contract", "arguments": {}}]}
    pack = await builder.build(context, evidence, definition, defaults(None), compile)
    assert pack["criteria"][0]["subrequirements"] == [exp["subrequirements"][0]["text"]]
    assert pack["tests"][0]["rubric"]["subrequirements"] == pack["criteria"][0]["subrequirements"]
