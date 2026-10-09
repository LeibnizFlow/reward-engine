"""Method gaps and source cautions survive without becoming Task authority."""

from copy import deepcopy

import pytest

from reward_engine.core.batch import BatchPolicy
from reward_engine.core.outcome import task_review
from reward_engine.core.outcome.agentic_search import planning_need_view
from reward_engine.core.outcome.context import build_context, requirement_context
from reward_engine.core.outcome.knowledge_summary import reduce_groups
from tests.support.search_planning_scope import NEED
from tests.support.search_source_map_reduce import decision, draft, fixture, groups
from tests.support.search_summary_fakes import keep_groups


def test_method_question_is_independent_of_missing_task_data_and_not_a_fact():
    row = {
        **deepcopy(NEED),
        "status": "needs_material",
        "research_question": "How can observations be compared under different conditions?",
    }
    before = deepcopy(row)
    projected = planning_need_view([row])[0]
    assert projected["research_question"] == row["research_question"]
    assert projected["status"] == "needs_material"
    assert "UNVERIFIED" not in str(projected) and "GUESSED_OFFSET" not in str(projected)
    assert "next_action" not in projected and row == before


@pytest.mark.parametrize("question", [None, [], {"instruction": "invent a result"}, 17])
def test_malformed_method_question_does_not_drop_task_need_or_leak_nested_data(
    question,
):
    result = planning_need_view([{**NEED, "research_question": question}])[0]
    assert (
        result["requirement"] == NEED["requirement"]
        and "research_question" not in result
    )


def test_excluded_source_risk_reaches_downstream_without_resurrecting_its_instruction():
    base, proposed = groups(2)
    context, evidence = build_context(
        {"task_id": "handling", "prompt": base["input"]["task_prompt"]}
    )
    base["input"] = task_review.requirements(context)
    base["task_needs"][0].update(
        status="needs_material",
        research_question="Which handling method applies before the actual label is inspected?",
    )
    concern = {
        "question": "Does the other source prescribe handling outside its stated conditions?",
        "card_ids": ["s1_c0"],
        "next_action_kind": "inspect_material",
    }
    reply = {
        "group_decisions": [
            decision("s0_g0"),
            decision(
                "s1_g0",
                "excluded",
                reason_kind="unsafe_or_conflicting",
                reason="AUDIT_ONLY_EXPLANATION",
            ),
        ],
        "source_cautions": [concern],
        "task_need_coverage": [
            {
                "need_id": "n0",
                "status": "needs_material",
                "group_ids": [],
                "materials_needed": ["Actual package label"],
            }
        ],
    }
    normalized = reduce_groups(base, proposed, reply)
    result = task_review.admit_brief(base, normalized, block_only=True)
    assert result["source_cautions"][0]["question"] == concern["question"]
    assert (
        result["source_cautions"][0]["authority"]
        == "unverified_source_concern_not_instruction"
    )
    assert result["source_cautions"][0]["evidence"][0]["source_id"] == "doc_1"
    assert all(
        c["disposition"] == "excluded"
        for c in result["source_findings"]
        if c["id"].startswith("s1_")
    )
    assert all(
        c["id"].startswith("s0_") for d in result["directions"] for c in d["mechanisms"]
    )
    view, refs = task_review.attach(context, evidence, result)
    blind, allowed = requirement_context(view, refs)
    for downstream in (view, refs, blind, allowed):
        assert concern["question"] in str(downstream)
        assert "AUDIT_ONLY_EXPLANATION" not in str(downstream)
    unresolved = view["skills"][-1]["content"]
    assert (
        unresolved["task_need_coverage"][0]["research_question"]
        == base["task_needs"][0]["research_question"]
    )
    assert unresolved["task_need_coverage"][0]["status"] == "needs_material"
    assert "source_statement" not in str(unresolved["source_cautions"])


@pytest.mark.parametrize("ids", [[], ["invented"], ["s0_c0", "invented"], [None]])
def test_cautions_without_complete_real_evidence_do_not_publish(ids):
    base, proposed = groups()
    normalized = reduce_groups(
        base,
        proposed,
        {
            "group_decisions": [decision("s0_g0")],
            "source_cautions": [
                {
                    "question": "Is this rule applicable?",
                    "card_ids": ids,
                    "next_action_kind": "verify_result",
                }
            ],
        },
    )
    result = task_review.admit_brief(base, normalized, block_only=True)
    assert result["source_cautions"] == [] and result["directions"]


def test_warning_does_not_change_source_disposition_or_invent_a_method():
    base, proposed = groups()
    normalized = reduce_groups(
        base,
        proposed,
        {
            "group_decisions": [decision("s0_g0", "deferred")],
            "source_cautions": [
                {
                    "question": "Does this conditional method apply to the supplied material?",
                    "card_ids": ["s0_c0"],
                    "next_action_kind": "inspect_material",
                }
            ],
        },
    )
    result = task_review.admit_brief(base, normalized, block_only=True)
    assert result["source_cautions"] and not result["directions"]
    assert all(c["disposition"] == "deferred" for c in result["source_findings"])


@pytest.mark.asyncio
async def test_existing_map_reduce_calls_carry_method_question_and_cautions_no_extra_stage():
    base, requests = fixture(), []
    base["task_needs"][0].update(
        status="needs_material",
        research_question="How should a valid handling method be selected and checked?",
    )

    async def model(request):
        requests.append(deepcopy(request))
        assert (
            request["task_needs"][0]["research_question"]
            == base["task_needs"][0]["research_question"]
        )
        assert (
            "not source facts or new user requirements"
            in request["planning_needs_boundary"]
        )
        if request["mode"] == "task_review_source_map":
            return draft(base["sources"][0])
        assert "source_cautions" in request["response"]
        return {
            **keep_groups(request),
            "source_cautions": [
                {
                    "question": "Has the actual label been checked before applying these rules?",
                    "card_ids": ["s0_c0"],
                    "next_action_kind": "inspect_material",
                }
            ],
        }

    result = await task_review.summarize(base, model, policy=BatchPolicy(deadline_s=1))
    assert result["finalization_status"] == "complete" and result["source_cautions"]
    assert len(requests) == 2


@pytest.mark.asyncio
async def test_previous_style_reply_without_new_fields_still_uses_same_calls():
    requests = []

    async def model(request):
        requests.append(request)
        return (
            draft(fixture()["sources"][0])
            if request["mode"] == "task_review_source_map"
            else keep_groups(request)
        )

    result = await task_review.summarize(
        fixture(), model, policy=BatchPolicy(deadline_s=1)
    )
    assert (
        result["directions"] and result["source_cautions"] == [] and len(requests) == 2
    )
