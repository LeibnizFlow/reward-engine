"""A collected Map can be reduced again without generating new member judgments."""

from copy import deepcopy

import pytest

from reward_engine.core.outcome import jury
from reward_engine.core.outcome.context import build_context


@pytest.mark.asyncio
async def test_reduce_only_keeps_task_subrequirements_without_new_map_calls():
    context, evidence = build_context({"task_id": "x", "prompt": "Fix calculation; preserve units"})
    criterion = {"id": "units", "standard": "Preserve units", "kind": "hard", "origin": "task",
                 "measurement_question": "Are units preserved?", "evidence_refs": ["task"]}
    expected = {**criterion, "expectation_id": "a:0",
                "subrequirements": [{"part_id": "a:0/0", "text": "Preserve units"}]}
    inputs = {"proposals": [{"criterion_refs": ["a:0"]}, {"criterion_refs": ["a:0"]}],
              "independent_expectations": [expected], "proposal_expectations": []}
    saved = deepcopy(inputs)
    calls = []

    async def reducer(request):
        calls.append(request["mode"])
        supplied = deepcopy(request["independent_expectations"])
        assert supplied[0]["subrequirements"][0].pop("selection_id") == "p0"
        assert supplied == [expected]
        return {"criteria": [criterion], "coverage": [{"expectation_id": "a:0",
            "status": "mapped", "criterion_ids": ["units"], "reason": "Required by task",
            "subrequirements": [{"part_id": "a:0/0", "status": "mapped", "criterion_ids": ["units"],
                                 "reason": "Explicitly retained", "retained_text": "Preserve units"}]}]}

    result = await jury.reduce_proposals(context, evidence, inputs, reducer)
    assert calls == ["outcome_jury_reduce"]
    assert result["coverage"][0]["status"] == "mapped"
    assert inputs == saved


@pytest.mark.asyncio
async def test_reduce_only_does_not_turn_one_member_into_a_committee():
    async def never(_):
        pytest.fail("Cannot call reducer without enough collected proposals")

    with pytest.raises(ValueError, match="at least two"):
        await jury.reduce_proposals({}, {}, {"proposals": [{}]}, never)
