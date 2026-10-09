"""No IO/model calls: structural provenance is not semantic entailment."""

from copy import deepcopy
from dataclasses import asdict

import pytest

from reward_engine.core.outcome import task_review
from reward_engine.core.outcome.agentic_search import ResearchBudget
from reward_engine.core.outcome.context import build_context, requirement_context
from reward_engine.core.outcome.knowledge_evidence import (
    citation,
    evidence_blocks,
    normalize_source_brief,
    reading_units,
    spans,
    summary_sources,
)
from tests.support.search_summary_fakes import keep_groups

# These v25 regression cases intentionally retain model-authored source facts.
# New production cards have their own contract tests; never reinterpret saved v2.
SOURCE_CONTRACT = "source-facts-v2"


def base():
    context, _ = build_context(
        {"task_id": "example", "prompt": "Inspect the input and report limitations."}
    )
    texts = [
        "The input width is 6 units.\n\nThe checksum covers the payload only.\n",
        "```c\n/* Width is 10 units. */\nconsume(input, 6);\n```\n",
    ]
    batch = {"complete": True, "items": [], "total": 0, "succeeded": 0}
    return {
        "input": task_review.requirements(context),
        "sources": [
            {
                "source_id": f"source_{i}",
                "text": text,
                "title": f"Reference {i}",
                "url": f"https://example.test/{i}",
                "offset": 100,
                "total_chars": 100 + len(text),
                "excerpt_only": True,
                "source_kind": "web_document",
                "read_kind": "body",
            }
            for i, text in enumerate(texts)
        ],
        "search_batch": deepcopy(batch),
        "read_batch": deepcopy(batch),
        "research": {
            "budget": asdict(ResearchBudget()),
            "stop_reason": "model_stop",
            "history": [],
        },
        "task_needs": [
            {
                "need_id": "need_0",
                "requirement": "Inspect the input",
                "task_basis": "Inspect the input",
                "next_action": "UNSUPPORTED_PLAN_ADVICE",
            },
            {
                "need_id": "need_1",
                "requirement": "Report limitations",
                "task_basis": "report limitations",
                "next_action": "UNSUPPORTED_PLAN_ADVICE",
            },
        ],
        "skill_search": False,
        "source_contract": SOURCE_CONTRACT,
    }


def fact(original, source=0, **extra):
    body = original["sources"][source]
    return {
        "id": f"finding_{source}",
        "source_id": body["source_id"],
        "block_ids": [evidence_blocks(body["text"])[0]["block_id"]],
        "finding": "The documented input width is 6 units.",
        "applicability": "Documented format only.",
        "need_ids": ["need_0"],
        "disposition": "retained",
        "direction_id": "knowledge_0",
        "reason": "This source establishes a width; no target-specific edit is established.",
        **extra,
    }


def draft(original):
    return {
        "source_findings": [fact(original)],
        "directions": [{"id": "knowledge_0", "direction": "Input format"}],
        "source_notes": [
            {"source_id": "source_1", "reason": "Damaged transcript not used."}
        ],
        "task_need_coverage": [
            {"need_id": "need_0", "status": "supported", "finding_ids": ["finding_0"]},
            {"need_id": "need_1", "status": "needs_material", "finding_ids": []},
        ],
    }


def admit(original, proposed):
    normalized = normalize_source_brief(original, proposed, contract=SOURCE_CONTRACT)
    return task_review.admit_brief(original, normalized, block_only=True)


def test_structures_preserve_complete_tables_conditions_lists_and_code():
    table = "| Field | Offset | Meaning |\n| --- | --- | --- |\n" + "".join(
        f"| field_{i} | {i * 4} | A complete field description |\n" for i in range(12)
    )
    condition = (
        "A normal close flushes changes. "
        + "Context matters. " * 28
        + ("However, a read-only final client leaves the files present.\n\n")
    )
    listing = "Applicable only when:\n\n1. The version matches.\n\n2. All inputs are present.\n\n"
    code = "```c\n/* A misleading count of 10. */\n\nconsume(input, 6);\n```\n"
    text = "\n# Layout\n\n" + table + "\n" + condition + listing + code
    blocks = evidence_blocks(text)
    assert "".join(b["text"] for b in blocks) == text
    assert all(
        text[b["start"] : b["end"]] == b["text"] and len(b["text"]) <= 2000
        for b in blocks
    )
    assert sum(table in b["text"] for b in blocks) == 1
    assert sum(condition in b["text"] for b in blocks) == 1
    assert sum(listing in b["text"] for b in blocks) == 1
    assert sum(code in b["text"] for b in blocks) == 1
    assert not any(b["continuation"] or b["fragmented_line"] for b in blocks)
    assert evidence_blocks(text) == blocks


@pytest.mark.parametrize(
    "text",
    [
        "",
        " \n\n",
        "One paragraph without a newline.",
        "| a | b |\n" * 1000,
        "| a | b |\r" * 1000,
        "正文" * 2500 + "\nend",
        "```text\n" + "a bounded complete line\n" * 400 + "```",
    ],
)
def test_all_characters_survive_bounded_structural_or_long_line_splits(text):
    blocks = evidence_blocks(text)
    assert "".join(b["text"] for b in blocks) == text
    assert all(0 < len(b["text"]) <= 2000 for b in blocks)
    assert all(b["text"] == text[b["start"] : b["end"]] for b in blocks)
    assert all(
        a["end"] == b["start"] for a, b in zip(blocks[:-1], blocks[1:], strict=True)
    )
    if len(text) > 2000:
        assert any(b["continuation"] for b in blocks)
    if "正文" in text:
        assert any(b["fragmented_line"] for b in blocks)
    elif len(text) > 2000:
        assert not any(b["fragmented_line"] for b in blocks)


def test_unclosed_fence_is_explicitly_incomplete_and_bad_limits_rejected():
    block = evidence_blocks("```c\nconsume(input, 6);")[0]
    assert block["kind"] == "code" and block["continuation"]["incomplete_excerpt"]
    for bad in (0, -1, True, 2.5):
        with pytest.raises(ValueError):
            evidence_blocks("text", bad)


def test_sources_keep_original_metadata_once_and_omit_navigation():
    original = base()
    before = deepcopy(original)
    nav = {**original["sources"][0], "source_id": "nav", "read_kind": "navigation"}
    request = summary_sources(original["sources"] + [nav])
    assert len(request) == 2
    for body, public in zip(original["sources"], request, strict=True):
        assert public["offset"] == 100 and public["total_chars"] == body["total_chars"]
        assert "spans" not in public and "text" not in public
        assert "".join(b["text"] for b in public["blocks"]) == body["text"]
        assert "structural_groups" in public and "reading_units" not in public
        assert all("text" not in group for group in public["structural_groups"])
    assert original == before


def test_code_copies_bound_blocks_and_ignores_model_quotes_and_other_source_citations():
    original, proposed = base(), draft(base())
    proposed["source_findings"][0]["citations"] = [
        {"source_id": "source_1", "quote": "INVENTED"}
    ]
    result = admit(original, proposed)
    finding = result["directions"][0]["mechanisms"][0]
    c = finding["citations"][0]
    assert c == {
        "source_id": "source_0",
        "block_id": "block_0_29",
        "quote": original["sources"][0]["text"][:29],
        "span_start": 0,
        "span_end": 29,
    }
    assert c["quote"].endswith("\n\n")
    assert result["task_need_coverage"][0]["citations"] == finding["citations"]
    assert result["task_need_coverage"][0]["finding_ids"] == ["finding_0"]
    assert result["source_admission"] == "evidence-block-v1"
    assert "reason" not in finding


@pytest.mark.parametrize(
    "binding",
    [
        [],
        ["fake"],
        ["block_0_1"],
        None,
        "block_0_29",
        [True],
        [{"source_id": "source_1", "block_id": "block_0_29"}],
    ],
)
def test_invalid_binding_is_pending_and_cannot_fall_back_to_supplied_quote(binding):
    original, proposed = base(), draft(base())
    proposed["source_findings"][0].update(
        block_ids=binding,
        citations=[{"source_id": "source_0", "quote": original["sources"][0]["text"]}],
    )
    result = admit(original, proposed)
    assert not result["directions"]
    assert result["pending_findings"][0]["disposition"] == "deferred"
    assert result["task_need_coverage"][0]["status"] == "unknown"


def test_only_one_to_three_same_source_adjacent_blocks_can_bind_a_fact():
    original = base()
    original["sources"][0]["text"] = (
        "First rule.\n\nSecond rule.\n\nThird rule.\n\nFourth rule.\n"
    )
    ids = [b["block_id"] for b in evidence_blocks(original["sources"][0]["text"])]
    for binding, valid in [
        (ids[:3], True),
        (ids, False),
        ([ids[0], ids[2]], False),
        ([ids[1], ids[0]], False),
        ([ids[0], ids[0]], False),
    ]:
        proposed = draft(original)
        proposed["source_findings"][0]["block_ids"] = binding
        result = admit(original, proposed)
        assert bool(result["directions"]) == valid


def test_old_span_and_quote_apis_remain_available_but_block_only_never_falls_back():
    source = base()["sources"][0]
    lookup = {source["source_id"]: source}
    old = {
        "source_id": source["source_id"],
        "span_id": spans(source["text"])[0]["span_id"],
    }
    quote = {"source_id": source["source_id"], "quote": "input width is 6 units."}
    assert citation(old, lookup, span_only=True)
    assert citation(quote, lookup)
    assert citation(old, lookup, block_only=True) is None
    assert citation(quote, lookup, block_only=True) is None
    assert citation({**old, **quote, "block_id": "invented"}, lookup) is None


def test_local_deferred_transcript_cannot_revoke_independent_specification():
    original, proposed = base(), draft(base())
    proposed["source_findings"].append(
        fact(
            original,
            1,
            disposition="deferred",
            finding="This transcript has a comment/operation mismatch.",
            reason="Only this transcript is uncertain.",
        )
    )
    proposed["uncertainties"] = ["GLOBAL_FALSE: the specification count is unknown."]
    proposed["source_notes"] = []
    result = admit(original, proposed)
    assert result["source_findings"][0]["disposition"] == "retained"
    assert len(result["directions"][0]["mechanisms"]) == 1
    assert result["uncertainties"][0]["scope"] == {
        "kind": "source_interpretation",
        "source_id": "source_1",
        "block_ids": proposed["source_findings"][1]["block_ids"],
        "finding_id": "finding_1",
    }
    assert result["uncertainties"][1]["scope"] == {
        "kind": "task_need",
        "need_id": "need_1",
    }
    assert "GLOBAL_FALSE" not in str(result)
    context, evidence = build_context(
        {"task_id": "example", "prompt": "Inspect the input and report limitations."}
    )
    view, refs = task_review.attach(context, evidence, result)
    blind, _ = requirement_context(view, refs)
    assert "source_interpretation" in str(
        blind["skills"]
    ) and "Documented format only" in str(blind["skills"])
    assert "UNSUPPORTED_PLAN_ADVICE" not in str(blind["skills"])


@pytest.mark.parametrize(
    "ids", [["unknown"], ["finding_1"], ["finding_0", "unknown"], [], None]
)
def test_coverage_requires_retained_findings_bound_to_the_same_need(ids):
    original, proposed = base(), draft(base())
    proposed["source_findings"].append(fact(original, 1, need_ids=["need_1"]))
    proposed["task_need_coverage"][0]["finding_ids"] = ids
    result = admit(original, proposed)
    assert result["task_need_coverage"][0]["status"] == "unknown"
    assert not result["task_need_coverage"][0]["citations"]


def test_duplicate_finding_or_missing_group_cannot_support_coverage():
    original, proposed = base(), draft(base())
    proposed["source_findings"].append(fact(original, disposition="deferred"))
    result = admit(original, proposed)
    assert all(r["disposition"] == "deferred" for r in result["source_findings"])
    assert result["task_need_coverage"][0]["status"] == "unknown"
    proposed = draft(original)
    proposed["directions"] = []
    result = admit(original, proposed)
    assert result["task_need_coverage"][0]["status"] == "unknown"


def test_block_binding_proves_traceability_not_semantic_entailment():
    original, proposed = base(), draft(base())
    proposed["source_findings"][0]["finding"] = (
        "The width is 99, a deliberately unsupported interpretation."
    )
    result = admit(original, proposed)
    row = result["directions"][0]["mechanisms"][0]
    assert row["grounding"] == "quotes_located"
    assert "99" in row["finding"] and "6 units" in row["citations"][0]["quote"]
    assert "not exhaustive or semantic verification" in result["coverage_boundary"]


@pytest.mark.asyncio
async def test_summary_forces_cards_without_extra_call_or_model_contract_downgrade():
    original = base()
    requests = []

    async def call(request):
        requests.append(request)
        if request["mode"] == "task_review_group_reduce":
            return keep_groups(request)
        sid = request["sources"][0]["source_id"]
        assert request["sources"] == summary_sources(
            [s for s in original["sources"] if s["source_id"] == sid]
        )
        assert "block_ids" in request["response"]["source_findings"][0]
        assert "unit_id" not in request["response"]["source_findings"][0]
        assert "citations" not in request["response"]["source_findings"][0]
        assert "uncertainties" not in request["response"]
        assert "finding_ids" in request["response"]["task_need_coverage"][0]
        if sid == "source_1":
            return {"source_findings": []}
        response = draft(original)
        card = response["source_findings"][0]
        card.pop("block_ids")
        card["block_ids"] = reading_units(original["sources"][0]["text"])[0][
            "block_ids"
        ]
        response["task_need_coverage"][0]["status"] = "related_knowledge"
        response["source_contract"] = "source-facts-v1"
        return response

    result = await task_review.summarize(original, call)
    assert (
        len(requests) == 3 and result["source_contract"] == task_review.SOURCE_CONTRACT
    )
    assert result["version"] == task_review.VERSION
    assert result["directions"][0]["mechanisms"][0]["citations"][0]["block_id"]
    assert (
        result["directions"][0]["mechanisms"][0]["source_statement"]
        == original["sources"][0]["text"]
    )
    assert result["task_need_coverage"][0]["status"] == "related_knowledge"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value",
    [
        ("version", "task-review-v24-skill-reader"),
        ("source_admission", "source-spans-v2-adjacent"),
        ("source_contract", "source-facts-v1"),
    ],
)
async def test_old_contract_cache_is_missed_without_relabeling_cached_input(
    monkeypatch, field, value
):
    original = base()
    cached = {
        **admit(original, draft(original)),
        "version": task_review.VERSION,
        "source_admission": task_review.ADMISSION,
        "cache_identity": None,
        "discovery_config": None,
        "summary_status": "complete",
        field: value,
    }
    before = deepcopy(cached)
    calls = []

    async def research(*args, **kwargs):
        return {
            k: deepcopy(original[k])
            for k in ("sources", "search_batch", "read_batch", "research")
        }

    async def call(request):
        calls.append(request["mode"])
        if request["mode"] == "task_review_queries":
            return {"queries": [], "task_needs": original["task_needs"]}
        if request["mode"] == "task_review_group_reduce":
            return keep_groups(request)
        if request["sources"][0]["source_id"] == "source_1":
            return {"source_findings": []}
        response = draft(original)
        response["source_findings"][0].pop("block_ids")
        response["source_findings"][0]["block_ids"] = reading_units(
            original["sources"][0]["text"]
        )[0]["block_ids"]
        response["task_need_coverage"][0]["status"] = "related_knowledge"
        return response

    async def no_io(*args):
        raise AssertionError("No external IO is permitted")

    monkeypatch.setattr(task_review, "research", research)
    context, _ = build_context(
        {"task_id": "example", "prompt": "Inspect the input and report limitations."}
    )
    result = await task_review.review(
        context, call, no_io, no_io, enabled=True, cached=cached
    )
    assert (
        calls
        == [
            "task_review_queries",
            "task_review_source_map",
            "task_review_source_map",
            "task_review_group_reduce",
        ]
        and not result["cache_hit"]
    )
    assert cached == before
