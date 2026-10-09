"""Independent final decisions preserve evidence, isolation, ties and recovery."""

import json
from copy import deepcopy

import pytest

from reward_engine.core.outcome.context import RequestBudgetExceeded
from reward_engine.core.outcome.pool_selector import (
    close_evidence_ranking,
    decide,
    prepare,
    select,
)

TASK = {"task_id": "t", "prompt": "Deliver the requested correct result"}


def candidates():
    return {
        f"slot-{i}": {
            "candidate_id": f"run-private-{i}",
            "artifact": {"final_response": f"Content {i}"},
            "environment": {"workspace": "/workspace"},
        }
        for i in range(12)
    }


def ranking(context, winner=None, unknown=()):
    ids = [r["product_id"] for r in context["products"]]
    ordered = ([winner] if winner else []) + [
        p for p in ids if p != winner and p not in unknown
    ]
    return {
        "groups": [[p] for p in ordered],
        "unknown": list(unknown),
        "reason": "Actual content comparison",
        "observations": [
            {
                "product_id": p,
                "reason": "Observed content or unresolved interpretation",
                "evidence_refs": [p + ":artifact:final_response"],
            }
            for p in ids
        ],
    }


@pytest.mark.parametrize("winner_index", range(12))
def test_every_seed_or_historical_round_can_win(winner_index):
    prepared = prepare(TASK, candidates(), runtime={}, shuffle=lambda _: None)
    winner = f"product_{winner_index + 1}"
    result = decide(ranking(prepared["context"], winner), prepared)
    assert result["candidate_key"] == prepared["mapping"][winner]
    assert result["status"] == "selected"


def test_private_metadata_is_not_in_context_or_input_identity():
    pool = candidates()
    for c in pool.values():
        c.update(
            official="PRIVATE_GRADE",
            final_measurement="OLD_CHECK",
            memory="MEMORY",
            arm="C",
            round=3,
        )
        c["artifact"]["rewardpack"] = "REWARD_PACK"
        c["environment"]["run_path"] = "/private/run/round_3/C"
    prepared = prepare(TASK, pool, runtime={}, shuffle=lambda _: None)
    wire = json.dumps(prepared["context"]) + json.dumps(prepared["evidence"])
    for forbidden in [
        "PRIVATE_GRADE",
        "OLD_CHECK",
        "MEMORY",
        "REWARD_PACK",
        "run-private-",
        "/private/run/",
        "slot-",
    ]:
        assert forbidden not in wire
    for c in pool.values():
        c["official"] = "CHANGED_SCORE"
    assert prepare(TASK, pool, runtime={}, previous=prepared) == prepared
    assert not prepared["context"]["checks"] and not prepared["context"]["memories"]


def test_scoped_user_updates_and_exact_public_deliveries_are_retained():
    pool = candidates()
    pool["slot-0"]["artifact"]["public_consultation"] = {
        "earlier_exchanges": [
            {"speaker": "user", "text": "Use exact units"},
            {"speaker": "assistant", "text": "Earlier reply"},
        ]
    }
    prepared = prepare(TASK, pool, runtime={}, shuffle=lambda _: None)
    first, second = prepared["context"]["products"][:2]
    assert first["user_updates"][0]["text"] == "Use exact units"
    assert not second["user_updates"]
    assert first["artifact"]["earlier_public_deliveries"][0]["text"] == "Earlier reply"


def test_ties_use_content_hash_and_unknowns_remain_visible():
    prepared = prepare(TASK, candidates(), runtime={}, shuffle=lambda _: None)
    raw = ranking(prepared["context"])
    raw["groups"] = [[p[0] for p in raw["groups"]]]
    first = decide(raw, prepared)
    raw["groups"][0].reverse()
    assert decide(raw, prepared)["candidate_key"] == first["candidate_key"]
    assert first["tie_break"] == "public_content_hash"
    unknown = list(prepared["mapping"])[-1:]
    assert (
        decide(ranking(prepared["context"], unknown=unknown), prepared)["status"]
        == "selected_with_unknowns"
    )
    result = decide(
        ranking(prepared["context"], unknown=list(prepared["mapping"])), prepared
    )
    assert result["status"] == "unavailable" and result["candidate_key"] is None


@pytest.mark.parametrize("change", ["duplicate", "missing", "foreign_ref"])
def test_invalid_ranking_is_rejected(change):
    prepared = prepare(TASK, candidates(), runtime={}, shuffle=lambda _: None)
    raw = ranking(prepared["context"])
    if change == "duplicate":
        raw["groups"][1] = raw["groups"][0]
    elif change == "missing":
        raw["groups"].pop()
    else:
        raw["observations"][0]["evidence_refs"] = ["check:old_reward"]
    with pytest.raises(ValueError):
        decide(raw, prepared)


@pytest.mark.asyncio
async def test_one_assessment_reads_repairs_and_resumes_without_repeating_completed_calls():
    pool, records, calls = candidates(), {}, []
    pool["slot-0"]["source_documents"] = [
        {"source": "actual file", "text": "Exact full file " * 300}
    ]
    fail = True

    async def model(request):
        nonlocal fail
        calls.append(deepcopy(request))
        product = next(
            p
            for p in request["context"]["products"]
            if p["artifact"]["final_response"] == "Content 0"
        )
        ref = product["sources"][0]["ref"]
        if "requested_evidence" not in request:
            return {"need_evidence": [ref]}
        if fail:
            fail = False
            raise ConnectionError("Temporary transport failure")
        assert (
            request["requested_evidence"][ref] == pool["slot-0"]["source_documents"][0]
        )
        raw = ranking(request["context"], product["product_id"])
        if "response_recovery" not in request:
            raw["groups"].pop()
        return raw

    kwargs = {
        "runtime": {"model": "fixed"},
        "record": lambda k, v: records.__setitem__(k, deepcopy(v)),
        "restore": records.get,
    }
    with pytest.raises(ConnectionError):
        await select(TASK, pool, model, **kwargs)
    result = await select(TASK, pool, model, **kwargs)
    assert result["candidate_key"] == "slot-0"
    assert (
        len(calls) == 4
    )  # initial read, failed transport, invalid reply, contract repair
    assert await select(TASK, pool, model, **kwargs) == result and len(calls) == 4
    with pytest.raises(ValueError, match="different public inputs"):
        await select(TASK, pool, model, **{**kwargs, "runtime": {"model": "changed"}})


@pytest.mark.asyncio
async def test_budget_rejects_before_model_call_without_truncating_or_extra_compression():
    records = {}

    async def forbidden(_):
        pytest.fail("No model call within this budget")

    with pytest.raises(RequestBudgetExceeded):
        await select(
            TASK,
            candidates(),
            forbidden,
            runtime={},
            record=records.__setitem__,
            restore=records.get,
            max_bytes=100,
        )
    assert len(records["inputs"]["mapping"]) == 12
    assert records["status"]["status"] == "failed"


@pytest.mark.asyncio
async def test_contract_repair_budget_is_global_across_evidence_stages():
    records, calls = {}, []

    async def model(request):
        calls.append(request)
        if len(calls) == 2:
            return {"need_evidence": ["task"]}
        return {"groups": [], "unknown": []}  # Invalid before AND after a read.

    with pytest.raises(ValueError):
        await select(
            TASK,
            candidates(),
            model,
            runtime={},
            record=records.__setitem__,
            restore=records.get,
        )
    assert len(calls) == 3  # No second contract repair after the evidence expansion.


@pytest.mark.asyncio
async def test_cited_file_previews_are_expanded_before_finalizing():
    pool, records, calls = candidates(), {}, []
    pool["slot-0"]["source_documents"] = [
        {"source": "delivery", "text": "Actual file content " * 100}
    ]

    async def model(request):
        calls.append(request)
        raw = ranking(request["context"])
        target = next(p for p in request["context"]["products"] if p["sources"])
        next(o for o in raw["observations"] if o["product_id"] == target["product_id"])[
            "evidence_refs"
        ] = [target["sources"][0]["ref"]]
        return raw

    await select(
        TASK, pool, model, runtime={}, record=records.__setitem__, restore=records.get
    )
    assert len(calls) == 2 and calls[-1]["requested_evidence"]


@pytest.mark.asyncio
@pytest.mark.parametrize("position", ["last", "first", "tied_first", "read_again"])
async def test_evidence_closure_withdraws_unsupported_claims_without_promoting_runner_up(
    position,
):
    pool, records, calls = candidates(), {}, []
    for c in pool.values():
        c["source_documents"] = [
            {"source": f"file-{i}", "text": "Full file content " * 100}
            for i in range(3)
        ]

    async def model(request):
        calls.append(deepcopy(request))
        raw = ranking(request["context"])
        target = raw["groups"][-1 if position == "last" else 0][0]

        def ref(i):
            return target + f":source:{i}"

        if len(calls) < 3:
            return {"need_evidence": [ref(len(calls) - 1)]}
        if position == "read_again":
            return {"need_evidence": [ref(2)]}
        if position == "tied_first":
            raw["groups"] = [raw["groups"][0] + raw["groups"][1], *raw["groups"][2:]]
        row = next(o for o in raw["observations"] if o["product_id"] == target)
        row.update(reason="UNVERIFIED QUALITY CLAIM", evidence_refs=[ref(2)])
        return {
            "response": raw
        }  # Closing normalization also handles the supported envelope.

    kwargs = dict(
        runtime={},
        record=lambda k, v: records.__setitem__(k, deepcopy(v)),
        restore=records.get,
    )
    result = await select(TASK, pool, model, **kwargs)
    assert len(calls) == 4
    assert (
        calls[-1]["evidence_access_closed"]
        and len(calls[-1]["requested_evidence"]) == 2
    )
    assert not any(r.endswith(":source:2") for r in calls[-1]["allowed_support_refs"])
    assert "UNVERIFIED QUALITY CLAIM" not in json.dumps(result)
    if position == "last":
        assert (
            result["product_id"] == "product_1"
            and result["status"] == "selected_with_unknowns"
        )
        assert result["ranking"]["unknown"] == ["product_12"]
    else:
        assert result["status"] == "unavailable" and result["candidate_key"] is None
        assert len(result["ranking"]["unknown"]) == 12
    assert any(k.startswith("evidence_closure/") for k in records)
    assert await select(TASK, pool, model, **kwargs) == result and len(calls) == 4


def test_closure_preserves_verified_output_and_does_not_repair_bad_contracts():
    pool = candidates()
    pool["slot-0"]["source_documents"] = [
        {"source": "file", "text": "Long evidence " * 100}
    ]
    prepared = prepare(TASK, pool, runtime={}, shuffle=lambda _: None)
    raw = ranking(prepared["context"])
    raw["observations"][0]["evidence_refs"] = ["product_1:source:0"]
    result, audit = close_evidence_ranking(
        raw, {"requested_evidence": {"product_1:source:0": {}}}, prepared
    )
    assert result == raw and audit is None
    raw["groups"].pop()
    assert close_evidence_ranking(raw, {}, prepared) == (raw, None)
    invalid = {"need_evidence": ["invented-ref"]}
    assert close_evidence_ranking(invalid, {}, prepared) == (invalid, None)


@pytest.mark.asyncio
async def test_controlled_replay_order_is_validated_and_frozen():
    pool, records = candidates(), {}
    order = list(reversed(pool))

    async def model(request):
        assert (
            request["context"]["products"][0]["artifact"]["final_response"]
            == "Content 11"
        )
        return ranking(request["context"])

    kwargs = dict(runtime={}, record=records.__setitem__, restore=records.get)
    result = await select(TASK, pool, model, order=order, **kwargs)
    assert result["candidate_key"] == order[0]
    with pytest.raises(ValueError, match="differs from frozen"):
        await select(TASK, pool, model, order=list(pool), **kwargs)
    with pytest.raises(ValueError, match="every candidate exactly once"):
        await select(TASK, pool, model, order=order[:-1], **kwargs)
