"""Bounded Reduce recovery, exact-source compression, and preserved loop results."""

import asyncio
import json
from copy import deepcopy

import pytest

from reward_engine.core.outcome import jury, loop
from reward_engine.core.outcome.harness import defaults
from tests.support.claw_outcome_jury_reliability import (
    apply_part_decisions,
    compact_selection_inputs,
    expectations,
    fixture,
    resolve_parts,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "broken", ["criterion", "selector", "coverage", "missing_field"]
)
async def test_reduce_repairs_once_and_records_both_replies(broken):
    ctx, evidence, c, inputs = fixture()
    records, calls = {}, []
    raw = {
        "criteria": [deepcopy(c)],
        "part_decisions": [
            {"parts": ["p0"], "action": "retain", "criterion_id": "json"}
        ],
    }
    bad = deepcopy(raw)
    if broken == "criterion":
        bad["criteria"][0]["origin"] = "invented"
    elif broken == "selector":
        bad["part_decisions"][0]["criterion_id"] = "typo"
    elif broken == "coverage":
        bad = {
            "criteria": [deepcopy(c)],
            "coverage": [{"expectation_id": "nonexistent"}],
        }
    else:
        del bad["criteria"][0]["id"]

    async def reducer(request):
        calls.append(deepcopy(request))
        if len(calls) == 1:
            return deepcopy(bad)
        assert request["repair"]["previous_output"] == bad
        assert request["repair"]["validation_error"]
        return deepcopy(raw)

    result = await jury.reduce_proposals(
        ctx, evidence, inputs, reducer, record=records.__setitem__
    )
    assert len(calls) == 2
    assert result["criteria"][0]["origin"] == "task"
    assert records["jury_reduce_attempt_1/jury_reduce_raw"] == bad
    assert records["jury_reduce_attempt_2/jury_reduce_raw"] == raw
    checkpoint = records["jury_reduce_checkpoint"]
    assert checkpoint["inputs"] == inputs and checkpoint["evidence"] == evidence
    assert result == records["jury"]


@pytest.mark.asyncio
async def test_failed_repair_never_resamples_maps_or_publishes():
    ctx, evidence, c, _ = fixture()
    records, calls = {}, []

    async def member(request):
        calls.append(request["mode"])
        return {"criteria": [deepcopy(c)]}

    async def broken(request):
        calls.append(request["mode"])
        return {"criteria": [{**c, "origin": "wrong"}]}

    with pytest.raises(jury.ReduceValidationError):
        await jury.propose(
            ctx,
            evidence,
            {"a": member, "b": member, "c": member},
            broken,
            record=records.__setitem__,
        )
    assert calls.count("outcome_expectations") == 3
    assert calls.count("outcome_jury_map") == 3
    assert calls.count("outcome_jury_reduce") == 2
    assert "jury" not in records
    checkpoint = records["jury_reduce_checkpoint"]

    async def fixed(request):
        return {"criteria": [c]}

    result = await jury.reduce_proposals(
        checkpoint["context"], checkpoint["evidence"], checkpoint["inputs"], fixed
    )
    assert result["criteria"] == [c]


@pytest.mark.asyncio
async def test_transport_error_is_not_format_repair():
    ctx, evidence, _, inputs = fixture()
    calls = []

    async def offline(request):
        calls.append(request)
        raise ConnectionError("offline")

    with pytest.raises(ConnectionError):
        await jury.reduce_proposals(ctx, evidence, inputs, offline)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_oversized_input_saves_checkpoint_without_calling_or_truncating():
    ctx, evidence, _, inputs = fixture()
    ctx["task"]["prompt"] = "x" * 170000
    records = {}

    async def never(_):
        pytest.fail("Oversize must stop before paid request")

    with pytest.raises(ValueError, match="budget"):
        await jury.reduce_proposals(
            ctx, evidence, inputs, never, record=records.__setitem__
        )
    assert (
        records["jury_reduce_checkpoint"]["context"]["task"]["prompt"] == "x" * 170000
    )
    assert "jury" not in records


@pytest.mark.asyncio
async def test_cancellation_does_not_become_a_model_repair():
    ctx, evidence, _, inputs = fixture()

    async def cancel(_):
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await jury.reduce_proposals(ctx, evidence, inputs, cancel)


@pytest.mark.asyncio
async def test_missing_alias_protocol_is_repaired_not_silently_lost():
    ctx, evidence, c, inputs = fixture()
    inputs["proposal_expectations"] = expectations([c], "b", "after")
    calls = []

    async def reducer(request):
        calls.append(request)
        reply = {
            "criteria": [deepcopy(c)],
            "part_decisions": [
                {"parts": ["p0"], "action": "retain", "criterion_id": "json"}
            ],
        }
        if len(calls) == 2:
            reply["selection_protocol"] = "exact-alias-v1"
        return reply

    result = await jury.reduce_proposals(ctx, evidence, inputs, reducer)
    assert len(calls) == 2 and all(r["status"] == "mapped" for r in result["coverage"])


def test_exact_duplicate_clauses_expand_to_every_original_source():
    _, evidence, c, inputs = fixture()
    inputs["proposal_expectations"] = expectations([c], "b", "after")
    original = deepcopy(inputs)
    view = compact_selection_inputs(inputs)
    assert view["selection_aliases"] == {"p0": ["p1"]}
    assert view["proposal_expectations"][0]["subrequirements"] == []
    result = apply_part_decisions(
        {
            "criteria": [c],
            "selection_protocol": "exact-alias-v1",
            "part_decisions": [
                {"parts": ["p0"], "action": "retain", "criterion_id": "json"}
            ],
        },
        inputs,
        evidence,
    )
    assert all(r["status"] == "mapped" for r in result["coverage"])
    assert {r["subrequirements"][0]["part_id"] for r in result["coverage"]} == {
        "a:0/0",
        "b:0/0",
    }
    assert inputs == original


def test_exact_duplicate_compaction_reduces_payload_without_losing_originals():
    _, _, c, inputs = fixture()
    c["subrequirements"] = ["Preserve this precise conditional statement. " * 30]
    inputs["independent_expectations"] = expectations([c], "a", "before")
    inputs["proposal_expectations"] = expectations([c], "b", "after") + expectations(
        [c], "c", "after"
    )
    compact = compact_selection_inputs(inputs)
    assert len(json.dumps(compact)) < len(json.dumps(inputs))
    assert compact["selection_aliases"] == {"p0": ["p1", "p2"]}
    assert len(inputs["proposal_expectations"]) == 2


def test_conflicting_duplicate_decisions_are_not_silently_resolved():
    _, evidence, c, inputs = fixture()
    inputs["proposal_expectations"] = expectations([c], "b", "after")
    with pytest.raises(ValueError, match="duplicate"):
        apply_part_decisions(
            {
                "criteria": [c],
                "selection_protocol": "exact-alias-v1",
                "part_decisions": [
                    {"parts": ["p0"], "action": "retain", "criterion_id": "json"},
                    {
                        "parts": ["p1"],
                        "action": "exclude",
                        "reason": "Conflicting opinion",
                    },
                ],
            },
            inputs,
            evidence,
        )


@pytest.mark.parametrize(
    "change",
    [
        {"origin": "user"},
        {"kind": "soft"},
        {"evidence_refs": ["product_1:user"]},
        {"standard": "Valid JSON only if requested"},
        {"subrequirements": ["valid json"]},
        {"subrequirements": ["Valid JSON", "Only if requested"]},
    ],
)
def test_similar_text_or_different_scope_is_not_merged(change):
    _, _, c, inputs = fixture()
    inputs["proposal_expectations"] = expectations([{**c, **change}], "b", "after")
    assert not compact_selection_inputs(inputs).get("selection_aliases")


def test_short_substring_does_not_prove_coverage_but_legacy_exact_text_does():
    _, _, c, inputs = fixture()
    e = inputs["independent_expectations"][0]
    row = {
        "subrequirements": [
            {
                "part_id": "a:0/0",
                "status": "mapped",
                "criterion_ids": ["json"],
                "reason": "retained",
                "retained_text": "Valid",
            }
        ]
    }
    assert resolve_parts(e, row, {"json": c})[0]["status"] == "deferred"
    row["subrequirements"][0]["retained_text"] = "Valid JSON"
    legacy = {k: v for k, v in c.items() if k != "subrequirements"}
    assert resolve_parts(e, row, {"json": legacy})[0]["status"] == "mapped"


def test_check_refs_are_accepted_only_for_the_right_product():
    ctx = {"products": [{"product_id": "product_1"}]}
    result = {
        "groups": [["product_1"]],
        "unknown": [],
        "reason": "Observed",
        "observations": [
            {
                "product_id": "product_1",
                "reason": "Parse failed",
                "evidence_refs": ["check:p"],
            }
        ],
    }
    evidence = {"check:p": {"product_id": "product_1", "status": "fail"}}
    assert jury.validate_ranking(result, ctx, evidence)
    for invalid in ({}, {"product_id": "product_2"}, "not a check record"):
        with pytest.raises(ValueError, match="actual product"):
            jury.validate_ranking(result, ctx, {"check:p": invalid})


def test_user_criterion_requires_context_explicitly():
    _, evidence, c, _ = fixture()
    with pytest.raises(ValueError, match="scoped context"):
        jury.validate({"criteria": [{**c, "origin": "user"}]}, evidence)


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_round", [1, 2])
async def test_loop_stops_without_losing_seeds_or_prior_completed_round(
    monkeypatch, fail_round
):
    ctx, _, c, _ = fixture()
    records, sampled = {}, []
    seeds = [
        {"candidate_id": str(i), "artifact": {"final_response": "{}"}} for i in range(3)
    ]

    async def propose(*args, **kwargs):
        if len(sampled) + 1 == fail_round:
            raise jury.ReduceValidationError(
                ValueError("invalid reducer"), {"criteria": []}
            )
        return {"criteria": [c], "gaps": []}

    async def build(*args, **kwargs):
        return {
            "pack_id": "pack",
            "criteria": [{**c, "anchors": {"full": "Valid"}}],
            "tests": [],
        }

    async def method(*args, **kwargs):
        return {"status": "keep", "method": ""}

    async def smoke(*args, **kwargs):
        return {"status": "callable"}

    async def rank(*args, **kwargs):
        return {
            "status": "compared",
            "groups": [["product_4"], ["product_1", "product_2", "product_3"]],
            "continuation": {
                "decision": "continue",
                "reason": "New gap",
                "product_ids": ["product_4"],
            },
        }

    async def sample(arm, number, *args):
        sampled.append(arm)
        return {
            "candidate": {
                "candidate_id": "fresh",
                "artifact": {"final_response": "{}"},
            },
            "final_measurement": {"pack_id": "pack", "results": []},
        }

    monkeypatch.setattr(jury, "propose", propose)
    monkeypatch.setattr(jury, "rank", rank)
    monkeypatch.setattr(loop.builder, "build", build)
    monkeypatch.setattr(loop.hypotheses, "propose", method)
    monkeypatch.setattr(loop, "smoke", smoke)
    result = await loop.run_task(
        ctx["task"],
        seeds,
        roles=lambda *_: None,
        registry=defaults(None),
        sample=sample,
        rounds=2,
        enabled_arms=("A",),
        record=records.__setitem__,
    )
    assert result["status"] == ("failed" if fail_round == 1 else "partial")
    assert result["completed_rounds"] == fail_round - 1
    assert result["stop_reason"] == "jury_unavailable"
    state = records[f"round_{fail_round}/recovery_state"]
    assert state["pool"][:3] == seeds and len(state["pool"]) == 3 + fail_round - 1
    assert result["selected_delivery"] == (
        None if fail_round == 1 else result["rounds"][0]["selection"]
    )
