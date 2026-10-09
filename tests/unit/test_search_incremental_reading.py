"""Reader findings drive the next research action; finalization reuses the actual Maps."""
import asyncio
import json
from copy import deepcopy
from time import monotonic

import pytest
from tests.support.search_summary_fakes import keep_groups

from reward_engine.core.batch import BatchPolicy
from reward_engine.core.outcome import task_review
from reward_engine.core.outcome.agentic_search import (
    ResearchBudget,
    decision_history,
    research,
)


def map_reply(request, *, missing=False):
    source = request["sources"][0]
    return {"source_findings": [{"id": "f0", "source_id": source["source_id"],
        "block_ids": [b["block_id"] for b in source["blocks"] if b["kind"] not in {"heading", "whitespace"}],
        "direction_id": "handling", "need_ids": ["need_0"], "disposition": "retained"}],
        "task_need_coverage": [{"need_id": "need_0", "status": "related_knowledge", "finding_ids": ["f0"],
            "open_questions": [{"question": "What exceptions apply to this handling rule?",
                "card_ids": ["f0"], "next_action_kind": "read_source"}] if missing else [],
            "materials_needed": []}]}


def base():
    return {"input": {"task_prompt": "Inspect handling requirements and exceptions.", "user_updates": []},
        "task_needs": [{"need_id": "need_0", "requirement": "Inspect requirements and exceptions."}],
        "sources": [{"source_id": "source_0", "text": "Use cold handling for this class of shipment.",
                     "read_kind": "body", "title": "Handling", "url": "https://docs.example/handling"}],
        "search_batch": {"complete": True}, "read_batch": {"complete": True}}


@pytest.mark.asyncio
async def test_source_reader_gap_reaches_next_decision_and_maps_are_not_repeated_at_finalization():
    requests, reads, records = [], [], {}
    context = {"task": {"task_id": "private-id", "prompt": base()["input"]["task_prompt"]}, "products": []}
    async def search(_):
        return {"results": [{"url": "https://docs.example/handling"}, {"url": "https://docs.example/exceptions"}]}
    async def extract(url):
        reads.append(url)
        return {"results": [{"url": url, "raw_content": "Use cold handling." if url.endswith("handling")
                              else "Except for shipments labelled room-temperature."}]}
    async def model(request):
        requests.append(deepcopy(request))
        assert "private-id" not in json.dumps(request)
        if request["mode"] == "task_review_queries":
            return {"queries": ["handling instructions"], "task_needs": [{"requirement": "Inspect requirements and exceptions."}]}
        if request["mode"] == "task_review_source_map":
            return map_reply(request, missing=request["sources"][0]["source_id"] == "source_0")
        if request["mode"] == "task_review_group_reduce":
            assert len(reads) == 2 and request["missing_sources"] == []
            return keep_groups(request)
        if not reads:
            assert "reader_feedback" not in request
            return {"reads": [{"candidate_id": "doc_0", "gap": "Read actual handling method"}]}
        findings = request["reader_feedback"]["sources"]
        if len(reads) == 1:
            assert findings[0]["task_need_coverage"][0]["open_questions"][0]["question"].startswith("What exceptions")
            assert "not sufficient coverage" in request["instruction"]
            return {"reads": [{"candidate_id": "doc_1", "gap": "Read missing exceptions"}]}
        assert len(findings) == 2 and findings[1]["task_need_coverage"][0]["open_questions"] == []
        return {"stop": True}
    result = await task_review.review(context, model, search, extract, enabled=True,
        policy=BatchPolicy(deadline_s=1), research_budget=ResearchBudget(), record=records.__setitem__)
    assert result["finalization_status"] == "complete" and result["summary_reused_maps"] == 2
    assert [r["mode"] for r in requests] == ["task_review_queries", "task_review_research", "task_review_source_map",
        "task_review_research", "task_review_source_map", "task_review_research", "task_review_group_reduce"]
    assert "reader_feedback/1" in records and "reader_feedback/2" in records
    assert any("room-temperature" in c["source_statement"] for c in result["source_findings"])
    assert all("reader_feedback" not in r for r in requests if r["mode"] == "task_review_group_reduce")


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["body", "task", "user", "needs"])
async def test_map_reuse_is_bound_to_exact_task_user_needs_and_source(change):
    current, cache, retries, calls = base(), {}, [], []
    async def model(request):
        calls.append(request)
        return map_reply(request)
    await task_review._map_sources(current, model, lambda *_: None, BatchPolicy(deadline_s=1), cache, retries)
    original = deepcopy(current)
    await task_review._map_sources(current, model, lambda *_: None, BatchPolicy(deadline_s=1), cache, retries)
    assert len(calls) == 1 and current == original
    if change == "body":
        current["sources"][0]["text"] += " Use different handling for marked containers."
    elif change == "task":
        current["input"]["task_prompt"] += " Report seal damage too."
    elif change == "user":
        current["input"]["user_updates"].append({"text": "Do not open containers."})
    else:
        current["task_needs"][0]["requirement"] += " Check seals."
    await task_review._map_sources(current, model, lambda *_: None, BatchPolicy(deadline_s=1), cache, retries)
    assert len(calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["timeout", "invalid_json", "runtime"])
async def test_live_map_failure_is_kept_and_final_reduce_does_not_pay_for_it_again(failure):
    current, cache, retries, calls = base(), {}, [], []
    current["sources"].append({**current["sources"][0], "source_id": "source_1", "url": "https://docs.example/second"})
    async def model(request):
        calls.append(request)
        if request["mode"] == "task_review_group_reduce":
            assert request["missing_sources"][0]["source_id"] == "source_0"
            return keep_groups(request)
        if request["sources"][0]["source_id"] == "source_0":
            if failure == "invalid_json":
                raise json.JSONDecodeError("Incomplete response", "{", 1)
            if failure == "runtime":
                raise RuntimeError("Unavailable")
            await asyncio.Event().wait()
        return map_reply(request)
    await task_review._map_sources(current, model, lambda *_: None, BatchPolicy(deadline_s=.02), cache, retries)
    before = len(calls)
    result = await task_review.summarize(current, model, deadline_s=.1, _map_cache=cache, _format_retries=retries)
    assert len(calls) == before + 1 and result["summary_map_batch"]["succeeded"] == 1
    assert result["summary_reused_maps"] == 2 and result["status"] == "partial"
    assert len(result["summary_format_retries"]) == (1 if failure == "invalid_json" else 0)


@pytest.mark.asyncio
async def test_reader_callback_timeout_does_not_discard_real_body_or_claim_coverage():
    calls = []
    async def search(_):
        return {"results": [{"url": "https://docs.example/method"}]}
    async def extract(url):
        return {"results": [{"url": url, "raw_content": "Inspect seals before release."}]}
    async def observer(sources, policy):
        sources[0]["text"] = "Observer cannot mutate actual evidence"
        await asyncio.Event().wait()
    async def model(request):
        calls.append(request)
        if not request["sources"]:
            return {"reads": [{"candidate_id": "doc_0", "gap": "Inspect method"}]}
        assert request["reader_feedback"]["status"] == "unavailable"
        assert request["sources"][0]["preview"] == "Inspect seals before release."
        return {"stop": True}
    started = monotonic()
    result = await research({}, {"queries": ["handling"]}, model, search, extract,
        policy=BatchPolicy(deadline_s=.02), record=lambda *_: None, on_read=observer)
    assert len(calls) == 2 and monotonic() - started < .4
    assert result["sources"][0]["text"] == "Inspect seals before release."


def test_history_projection_keeps_actual_positions_and_drops_duplicated_link_payloads():
    history = [{"reason": "old unverified guess", "gaps": [{"status": "supported"}],
        "read_source_ids": ["source_0"], "actions": [{"status": "read", "offset": 8000, "new_chars": 1200,
            "physical_requests": 1, "fetched_url": "https://docs.example/method",
            "observed_links": [{"url": "https://docs.example/" + str(i)} for i in range(1000)],
            "locator": {"method": "keyword_windows", "query_used": True, "requested_offset": 8000,
                        "candidates": [{"preview": "duplicated noise" * 1000}]}}]}]
    before = deepcopy(history)
    slim = decision_history(history)
    assert history == before and len(json.dumps(slim)) < len(json.dumps(history)) / 20
    assert slim[0]["read_source_ids"] == ["source_0"]
    assert slim[0]["actions"][0]["offset"] == 8000 and slim[0]["actions"][0]["locator"]["query_used"]


@pytest.mark.asyncio
async def test_one_slow_live_reader_preserves_other_completed_source_for_reduce():
    """A late reader must not erase evidence another reader already returned."""
    async def search(_):
        return {"results": [{"url": "https://docs.example/slow"}, {"url": "https://docs.example/fast"}]}
    async def extract(url):
        return {"results": [{"url": url, "raw_content": "Inspect seals before release."}]}
    async def model(request):
        if request["mode"] == "task_review_queries":
            return {"queries": ["handling"], "task_needs": [{"requirement": "Inspect requirements and exceptions."}]}
        if request["mode"] == "task_review_research":
            return {"stop": True} if request["history"] else {"reads": [
                {"candidate_id": row["candidate_id"], "gap": "Inspect handling method"}
                for row in request["candidates"]]}
        if request["mode"] == "task_review_source_map":
            if request["sources"][0]["source_id"] == "source_0":
                await asyncio.Event().wait()
            return map_reply(request)
        assert request["mode"] == "task_review_group_reduce"
        return keep_groups(request)
    result = await task_review.review(
        {"task": {"task_id": "isolated", "prompt": base()["input"]["task_prompt"]}, "products": []},
        model, search, extract, enabled=True, policy=BatchPolicy(deadline_s=.04),
        research_budget=ResearchBudget())
    assert result["summary_map_batch"]["succeeded"] == 1
    assert result["summary_missing_sources"] == [{"source_id": "source_0", "status": "timeout"}]
    assert result["directions"] and result["finalization_status"] == "complete"
