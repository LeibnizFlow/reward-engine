
from tests.support.claw_outcome_hypothesis_questions import (
    proposal,
)
import asyncio
from copy import deepcopy

import pytest

from reward_engine.core.outcome import hypotheses
from reward_engine.core.outcome.context import build_context




def test_three_maps_and_reducer_preserve_proactive_question_to_actor():
    context, evidence = build_context({"task_id": "generic", "prompt": "Draft an explanation."})
    calls = []

    async def call(request):
        calls.append(request)
        assert "Proactively consider" in request["instruction"]
        if request["mode"].endswith("reduce"):
            assert len(request["proposals"]) == 3
        return proposal()

    row = asyncio.run(hypotheses.propose(context, evidence, {}, {str(i): call for i in range(3)}, call))
    text = hypotheses.actor_prompt(row)
    assert len(calls) == 4 and "Who is the intended audience?" in text
    assert "NOT confirmed requirements" in text and "never simulate" in text
    row["clarifications"][0]["expected_answer"] = "PRIVATE_OLD_ANSWER"
    assert "PRIVATE_OLD_ANSWER" not in hypotheses.actor_prompt(row)


@pytest.mark.parametrize("mutation", ["keep", "no_value", "no_fallback", "fake_ref", "too_many"])
def test_invalid_questions_do_not_silently_become_action_prompts(mutation):
    row = deepcopy(proposal())
    if mutation == "keep":
        row["status"] = "keep"
    elif mutation == "no_value":
        row["clarifications"][0]["why"] = ""
    elif mutation == "no_fallback":
        del row["clarifications"][0]["if_unanswered"]
    elif mutation == "fake_ref":
        row["clarifications"][0]["evidence_refs"] = ["invented"]
    else:
        row["clarifications"] *= 4
    with pytest.raises(ValueError):
        hypotheses.validate(row, {"task": "Draft an explanation."})


def test_no_question_remains_a_valid_method_or_keep():
    row = proposal()
    del row["clarifications"]
    assert row["method"] in hypotheses.actor_prompt(row)
    assert row["verification"] in hypotheses.actor_prompt(row)
    assert hypotheses.actor_prompt({"status": "keep", "method": ""}) == ""


def test_method_history_is_labeled_as_external_without_rewriting_the_evidence():
    row = proposal()
    row["method"] = "Earlier candidates disagreed about a calculation. Recompute it."
    original = deepcopy(row)
    text = hypotheses.actor_prompt(row)
    assert text.startswith("Source: an exploration proposal")
    assert "NOT this fresh Actor's conversation history" in text
    assert "not your prior actions, observed results or confirmed user statements" in text
    assert row["method"] in text and row["verification"] in text
    assert row == original  # Frozen proposal and its provenance stay intact.
