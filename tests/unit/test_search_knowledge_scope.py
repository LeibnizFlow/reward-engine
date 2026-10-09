"""A research receipt records useful sources, never Task completion."""

import asyncio
from copy import deepcopy

import pytest

from reward_engine.core.outcome.knowledge_evidence import (
    SOURCE_CONTRACT,
    normalize_source_brief,
    reading_units,
)
from reward_engine.core.outcome.task_review import (
    admit_brief,
    attach,
    requirements,
    summarize,
)
from tests.support.search_summary_fakes import keep_groups


def setup():
    context = {
        "task": {
            "task_id": "generic",
            "prompt": "Inspect supplied records and export verified results.",
        },
        "products": [],
        "skills": [],
    }
    base = {
        "input": requirements(context),
        "sources": [
            {
                "source_id": "source_0",
                "text": "# Verification method\n\nInspect the records, then check the requested output.\n",
                "title": "Method",
                "url": "https://example.test/method",
                "read_kind": "body",
            }
        ],
        "task_needs": [
            {
                "need_id": "need_0",
                "requirement": context["task"]["prompt"],
                "task_basis": context["task"]["prompt"],
                "status": "needs_material",
                "next_action": "UNVERIFIED_PLANNER_FACT",
                "external_knowledge_needed": "PLANNER_GUESS",
            }
        ],
        "search_batch": {"complete": True},
        "read_batch": {"complete": True},
        "research": {"history": []},
    }
    brief = {
        "source_findings": [
            {
                "id": "f0",
                "source_id": "source_0",
                "block_ids": reading_units(base["sources"][0]["text"])[0]["block_ids"],
                "need_ids": ["need_0"],
                "disposition": "retained",
                "direction_id": "k0",
            }
        ],
        "task_need_coverage": [
            {
                "need_id": "need_0",
                "status": "related_knowledge",
                "finding_ids": ["f0"],
                "open_questions": [
                    {
                        "question": "Do the actual records meet the requested conditions?",
                        "card_ids": ["f0"],
                        "next_action_kind": "inspect_material",
                    }
                ],
                "materials_needed": [
                    "Actual records and the generated export for checking."
                ],
            }
        ],
    }
    return context, base, brief


def admit(base, brief):
    return admit_brief(base, normalize_source_brief(base, brief), block_only=True)


def test_related_method_is_not_task_completion_and_agenda_survives_attachment():
    context, base, brief = setup()
    result = admit(base, brief)
    coverage = result["task_need_coverage"][0]
    assert coverage["status"] == "related_knowledge"
    assert coverage["task_completion"] == "not_assessed"
    assert coverage["finding_ids"] == ["f0"] and coverage["citations"]
    assert result["status"] == "partial"
    view, evidence = attach(context, {}, result)
    agenda = evidence["skill:task_review:unresolved"]["task_need_coverage"][0]
    assert agenda["materials_needed"] and agenda["open_questions"]
    assert "Unverified" in agenda["agenda_boundary"]
    assert "UNVERIFIED_PLANNER_FACT" not in str(view)
    assert "PLANNER_GUESS" not in str(view)
    assert "Actual records" not in str(evidence["skill:task_review:k0"]["mechanisms"])


@pytest.mark.parametrize("status", ["supported", "pass", "complete", {}, [], None])
def test_current_contract_cannot_claim_verified_completion(status):
    _, base, brief = setup()
    brief["task_need_coverage"][0].update(status=status, task_completion="pass")
    row = admit(base, brief)["task_need_coverage"][0]
    assert row["status"] == "unknown" and row["task_completion"] == "not_assessed"


@pytest.mark.parametrize("ids", [["invented"], [], ["f0", "invented"], {}, "f0"])
def test_related_knowledge_requires_real_retained_cards_for_this_need(ids):
    _, base, brief = setup()
    brief["task_need_coverage"][0]["finding_ids"] = ids
    row = admit(base, brief)["task_need_coverage"][0]
    assert (
        row["status"] == "unknown" and not row["finding_ids"] and not row["citations"]
    )


def test_material_need_can_keep_related_sources_without_promising_sufficiency():
    _, base, brief = setup()
    brief["task_need_coverage"][0]["status"] = "needs_material"
    row = admit(base, brief)["task_need_coverage"][0]
    assert row["status"] == "needs_material" and row["finding_ids"] == ["f0"]
    assert row["task_completion"] == "not_assessed"


def test_malformed_or_duplicate_coverage_cannot_erase_need_or_pollute_facts():
    _, base, brief = setup()
    question = {
        "question": "Which data needs verification?",
        "card_ids": ["f0"],
        "next_action_kind": "verify_result",
    }
    brief["task_need_coverage"][0].update(
        open_questions=[
            {},
            question,
            question,
            {**question, "card_ids": ["invented"]},
            {**question, "next_action_kind": "assert_done"},
        ],
        materials_needed={"invented": 1},
    )
    row = admit(base, brief)["task_need_coverage"][0]
    assert row["open_questions"] == [
        question,
        {**question, "card_ids": [], "reference_status": "unresolved"},
    ]
    assert row["materials_needed"] == []
    brief["task_need_coverage"].append(deepcopy(brief["task_need_coverage"][0]))
    row = admit(base, brief)["task_need_coverage"][0]
    assert row["status"] == "unknown" and row["open_questions"] == []
    assert row["requirement"] == base["task_needs"][0]["requirement"]


def test_summary_keeps_task_but_not_planner_assertions_and_logs_original_base():
    _, base, brief = setup()
    original = deepcopy(base)
    calls, records = [], []

    async def model(request):
        calls.append(request)
        return (
            keep_groups(request)
            if request["mode"] == "task_review_group_reduce"
            else brief
        )

    result = asyncio.run(
        summarize(base, model, record=lambda *row: records.append(row))
    )
    assert [r["mode"] for r in calls] == [
        "task_review_source_map",
        "task_review_group_reduce",
    ]
    request = calls[0]
    assert request["context"]["task_prompt"] == original["input"]["task_prompt"]
    assert "UNVERIFIED_PLANNER_FACT" not in str(request) and "PLANNER_GUESS" not in str(
        request
    )
    assert "not source facts" in request["planning_needs_boundary"]
    assert result["task_needs"] == original["task_needs"]
    assert result["source_contract"] == SOURCE_CONTRACT
    assert (
        records[-1][0] == "review"
        and records[-1][1]["task_needs"] == original["task_needs"]
    )
