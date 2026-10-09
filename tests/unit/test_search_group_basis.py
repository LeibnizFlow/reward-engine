"""Generic decision/evidence binding, not a semantic correctness oracle."""

from copy import deepcopy

import pytest

from reward_engine.core.batch import BatchPolicy
from reward_engine.core.outcome import task_review
from reward_engine.core.outcome.knowledge_evidence import evidence_blocks
from reward_engine.core.outcome.knowledge_summary import reduce_groups, source_groups
from tests.support.search_source_map_reduce import decision, draft, fixture, groups
from tests.support.search_summary_fakes import keep_groups


@pytest.mark.parametrize(
    "basis", [None, [], "s0_c0", ["s1_c0"], ["s0_g0"], ["s0_c0", "s0_c0"], [None], [{}]]
)
def test_invalid_basis_defers_only_its_group_without_rewriting_evidence(basis):
    base, bundles = groups(2)
    original = deepcopy(bundles)
    result = reduce_groups(
        base,
        bundles,
        {
            "group_decisions": [
                decision("s0_g0", basis_card_ids=basis),
                decision("s1_g0", basis_card_ids=["s1_c0"]),
            ]
        },
        require_basis=True,
    )
    ledger = result["audit"]["group_decisions"]
    assert [r["disposition"] for r in ledger] == ["deferred", "retained"]
    assert "this exact group" in ledger[0]["decision_issue"]
    assert bundles == original
    assert all(
        c["source_statement"] == "".join(q["quote"] for q in c["citations"])
        for c in result["source_findings"]
    )


def test_group_and_card_numbers_are_not_interchangeable():
    base = fixture()
    proposal = draft(base["sources"][0])
    first = proposal["source_findings"][0]
    # First group splits into two distant cards, so group 1 actually owns card 2.
    proposal["source_findings"].append(
        {
            **first,
            "id": "second",
            "direction_id": "history",
            "block_ids": [evidence_blocks(base["sources"][0]["text"])[1]["block_id"]],
        }
    )
    bundles = source_groups(base, proposal, 0)["groups"]
    assert [c["id"] for c in bundles[1]["cards"]] == ["s0_c2"]
    for basis, expected in [("s0_c1", "deferred"), ("s0_c2", "retained")]:
        result = reduce_groups(
            base,
            bundles,
            {
                "group_decisions": [
                    decision("s0_g0", basis_card_ids=["s0_c0"]),
                    decision("s0_g1", basis_card_ids=[basis]),
                ]
            },
            require_basis=True,
        )
        assert result["audit"]["group_decisions"][1]["disposition"] == expected


@pytest.mark.parametrize("context_only", [False, True])
def test_dropped_or_context_only_card_cannot_ground_the_decision(context_only):
    base, bundles = groups()
    row = decision("s0_g0", basis_card_ids=["s0_c1"])
    if context_only:
        bundles[0]["cards"][1]["context_only"] = True
    else:
        row["omit_cards"] = [{"card_id": "s0_c1", "reason": "Omitted by test model."}]
    result = reduce_groups(
        base, bundles, {"group_decisions": [row]}, require_basis=True
    )
    assert result["audit"]["group_decisions"][0]["disposition"] == "deferred"
    assert "Decision basis" in result["audit"]["group_decisions"][0]["decision_issue"]


def test_basis_does_not_strip_unselected_conditions_or_change_historical_replay():
    base, bundles = groups()
    legacy = {"group_decisions": [decision("s0_g0")]}
    assert reduce_groups(base, bundles, legacy)["directions"]
    assert not reduce_groups(base, bundles, legacy, require_basis=True)["directions"]
    reply = {"group_decisions": [decision("s0_g0", basis_card_ids=["s0_c0"])]}
    result = reduce_groups(base, bundles, reply, require_basis=True)
    assert len(result["source_findings"]) == 2
    assert all(c["disposition"] == "retained" for c in result["source_findings"])
    assert "Except" in result["source_findings"][1]["source_statement"]


@pytest.mark.asyncio
async def test_runtime_requires_basis_without_another_model_stage():
    calls = []

    async def model(request):
        calls.append(request["mode"])
        if request["mode"] == "task_review_source_map":
            return draft(fixture()["sources"][0])
        fields = list(request["response"]["group_decisions"][0])
        assert fields.index("basis_card_ids") < fields.index("reason")
        reply = keep_groups(request)
        del reply["group_decisions"][0]["basis_card_ids"]
        return reply

    result = await task_review.summarize(
        fixture(), model, policy=BatchPolicy(deadline_s=1)
    )
    assert calls == ["task_review_source_map", "task_review_group_reduce"]
    assert not result["directions"]
    assert "Decision basis" in result["audit"]["group_decisions"][0]["decision_issue"]
