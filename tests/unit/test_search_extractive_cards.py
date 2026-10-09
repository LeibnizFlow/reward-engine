"""Extractive fidelity is testable; source truth/relevance is not code-verified."""

from copy import deepcopy
from dataclasses import asdict

import pytest

from reward_engine.core.outcome import task_review
from reward_engine.core.outcome.agentic_search import ResearchBudget
from reward_engine.core.outcome.context import (
    build_context,
    measurement_prior_view,
    requirement_context,
)
from reward_engine.core.outcome.knowledge_evidence import (
    SOURCE_CONTRACT,
    citation,
    evidence_blocks,
    normalize_source_brief,
    reading_units,
    spans,
)
from tests.support.search_summary_fakes import keep_groups


def base(text=None, *, contract=SOURCE_CONTRACT):
    context, _ = build_context(
        {"task_id": "cards", "prompt": "Inspect the format and report limitations."}
    )
    text = text or (
        "# Format\n\nThe documented width is 6 units.\n\n"
        "The default close removes files; a read-only final client leaves them present.\n"
    )
    batch = {"complete": True, "items": [], "total": 0, "succeeded": 0}
    return {
        "input": task_review.requirements(context),
        "sources": [
            {
                "source_id": "source_0",
                "text": text,
                "title": "Original reference",
                "url": "https://example.test/reference",
                "offset": 700,
                "total_chars": 700 + len(text),
                "excerpt_only": True,
                "source_kind": "web_document",
                "read_kind": "body",
            }
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
                "requirement": "Inspect the format",
                "task_basis": "Inspect the format",
            },
            {
                "need_id": "need_1",
                "requirement": "Report limitations",
                "task_basis": "report limitations",
            },
        ],
        "source_contract": contract,
        "skill_search": False,
    }


def draft(original, **extra):
    production = original["source_contract"] == SOURCE_CONTRACT
    if production:
        binding = {
            "block_ids": reading_units(original["sources"][0]["text"])[0]["block_ids"]
        }
    else:
        blocks = evidence_blocks(original["sources"][0]["text"])
        chosen = next(b for b in blocks if b["kind"] not in {"heading", "whitespace"})
        binding = {"block_ids": [chosen["block_id"]]}
    return {
        "source_findings": [
            {
                "id": "finding_0",
                "source_id": "source_0",
                **binding,
                "need_ids": ["need_0"],
                "disposition": "retained",
                "direction_id": "knowledge_0",
                **extra,
            }
        ],
        "task_need_coverage": [
            {
                "need_id": "need_0",
                "status": "related_knowledge" if production else "supported",
                "finding_ids": ["finding_0"],
            },
            {"need_id": "need_1", "status": "needs_material", "finding_ids": []},
        ],
    }


def admit(original, proposed):
    return task_review.admit_brief(
        original,
        normalize_source_brief(
            original, proposed, contract=original["source_contract"]
        ),
        block_only=True,
    )


def test_model_factual_rewrite_is_audit_only_not_a_claim_bound_to_real_quote():
    original = base()
    proposed = draft(
        original,
        finding="FABRICATED: the width is 99 units.",
        source_statement="FABRICATED statement.",
        applicability="FABRICATED always.",
        reason="FABRICATED repair.",
        title="FABRICATED title.",
        citations=[{"source_id": "source_0", "quote": "FABRICATED quote."}],
    )
    result = admit(original, proposed)
    card = result["directions"][0]["mechanisms"][0]
    chosen = reading_units(original["sources"][0]["text"])[0]
    assert (
        card["source_statement"]
        == original["sources"][0]["text"][chosen["start"] : chosen["end"]]
    )
    assert "".join(c["quote"] for c in card["citations"]) == card["source_statement"]
    assert card["source_offset"] == 700
    assert (
        card["blocks"][0]["start"] == chosen["start"]
        and "text" not in card["blocks"][0]
    )
    assert card["block_ids"] == chosen["block_ids"]
    assert "FABRICATED" not in str(card)
    assert "FABRICATED" in str(result["source_findings"][0]["audit"])
    assert all(
        k not in card for k in ("finding", "applicability", "reason", "audit", "title")
    )
    assert result["source_admission"] == task_review.ADMISSION == "extractive-card-v3"


def test_heading_only_cannot_publish_or_support_need_but_heading_with_body_can():
    original = base(contract="extractive-cards-v1")
    ids = [b["block_id"] for b in evidence_blocks(original["sources"][0]["text"])]
    rejected = admit(original, draft(original, block_ids=ids[:1]))
    assert rejected["source_findings"][0]["disposition"] == "deferred"
    assert not rejected["directions"]
    assert rejected["task_need_coverage"][0]["status"] == "unknown"
    assert "Heading-only" in rejected["source_findings"][0]["projection_issue"]
    accepted = admit(original, draft(original, block_ids=ids[:2]))
    assert accepted["directions"][0]["mechanisms"][0]["source_statement"] == "".join(
        b["text"] for b in evidence_blocks(original["sources"][0]["text"])[:2]
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"source_id": "unknown"},
        {"source_id": []},
        {"block_ids": "invented"},
        {"block_ids": None},
        {"block_ids": []},
        {"need_ids": ["unknown"]},
        {"need_ids": None},
        {"id": ""},
        {"id": "Unbounded prose is not an ID."},
        {"disposition": "published"},
        {"direction_id": None},
        {"direction_id": "free written factual heading"},
    ],
)
def test_invalid_selection_is_deferred_without_quote_fallback(changes):
    original = base()
    result = admit(original, draft(original, **changes))
    assert not result["directions"]
    assert result["pending_findings"][0]["disposition"] == "deferred"
    assert result["task_need_coverage"][0]["status"] == "unknown"


@pytest.mark.parametrize("malformed", [[], {}])
def test_malformed_card_and_coverage_status_preserve_other_valid_cards(malformed):
    original = base()
    proposed = draft(original)
    proposed["source_findings"].append(
        {
            **proposed["source_findings"][0],
            "id": "malformed_card",
            "disposition": malformed,
            "audit_note": "AUDIT_ONLY malformed card",
        }
    )
    proposed["task_need_coverage"][1].update(
        status=malformed, finding_ids=[{"claim": "AUDIT_ONLY invalid ID"}]
    )
    result = admit(original, proposed)
    assert [card["id"] for card in result["directions"][0]["mechanisms"]] == [
        "finding_0"
    ]
    assert result["pending_findings"][0]["disposition"] == "deferred"
    assert result["task_need_coverage"][0]["status"] == "related_knowledge"
    assert result["task_need_coverage"][1]["status"] == "unknown"
    assert result["task_need_coverage"][1]["finding_ids"] == []
    context, evidence = build_context(
        {"task_id": "cards", "prompt": "Inspect the format and report limitations."}
    )
    view, refs = task_review.attach(context, evidence, result)
    assert "AUDIT_ONLY" not in str(view) + str(refs)


def test_only_one_to_three_same_source_adjacent_blocks_are_accepted():
    original = base(
        "One fact.\n\nSecond fact.\n\nThird fact.\n\nFourth fact.\n",
        contract="extractive-cards-v1",
    )
    ids = [b["block_id"] for b in evidence_blocks(original["sources"][0]["text"])]
    for binding, accepted in [
        (ids[:3], True),
        (ids, False),
        ([ids[0], ids[2]], False),
        ([ids[1], ids[0]], False),
        ([ids[0], ids[0]], False),
    ]:
        result = admit(original, draft(original, block_ids=binding))
        assert bool(result["directions"]) == accepted


def test_continuation_and_fragmentation_are_preserved_not_rewritten_as_complete():
    original = base(
        "```text\n" + "a whole line with scope\r" * 210 + "```\n",
        contract="extractive-cards-v1",
    )
    offered = evidence_blocks(original["sources"][0]["text"])
    assert len(offered) == 3
    result = admit(
        original, draft(original, block_ids=[b["block_id"] for b in offered])
    )
    card = result["directions"][0]["mechanisms"][0]
    assert card["source_statement"] == original["sources"][0]["text"]
    assert [b["continuation"] for b in card["blocks"]] == [
        b["continuation"] for b in offered
    ]
    assert "".join(c["quote"] for c in card["citations"]) == card["source_statement"]
    long_line = base("正文" * 1500, contract="extractive-cards-v1")
    result = admit(long_line, draft(long_line))
    card = result["directions"][0]["mechanisms"][0]
    assert card["blocks"][0]["fragmented_line"]
    assert card["blocks"][0]["continuation"]["parts"] == 2
    assert len(card["source_statement"]) == 2000


def test_duplicate_ids_navigation_and_unrelated_coverage_cannot_be_supported():
    original = base()
    proposed = draft(original)
    proposed["source_findings"].append(deepcopy(proposed["source_findings"][0]))
    result = admit(original, proposed)
    assert (
        not result["directions"]
        and result["task_need_coverage"][0]["status"] == "unknown"
    )
    original["sources"][0]["read_kind"] = "navigation"
    result = admit(original, draft(original))
    assert not result["directions"]
    original = base()
    proposed = draft(original)
    proposed["task_need_coverage"][1] = {
        "need_id": "need_1",
        "status": "related_knowledge",
        "finding_ids": ["finding_0"],
    }
    result = admit(original, proposed)
    assert result["task_need_coverage"][1]["status"] == "unknown"


def test_attach_never_leaks_audit_via_directions_pending_uncertainties_or_history():
    original = base()
    original["research"]["history"] = [
        {"gaps": [{"status": "unknown", "reason": "AUDIT_ONLY history claim"}]}
    ]
    proposed = draft(original, audit_note="AUDIT_ONLY retained explanation")
    proposed["source_findings"].append(
        {
            **proposed["source_findings"][0],
            "id": "finding_1",
            "disposition": "deferred",
            "reason": "AUDIT_ONLY pending claim",
            "finding": "AUDIT_ONLY false fact",
        }
    )
    proposed.update(
        directions=[{"id": "knowledge_0", "direction": "AUDIT_ONLY factual heading"}],
        uncertainties=["AUDIT_ONLY global uncertainty"],
        source_notes=[
            {"source_id": "source_0", "reason": "AUDIT_ONLY no useful source"}
        ],
    )
    result = admit(original, proposed)
    assert "AUDIT_ONLY" in str(result)
    context, evidence = build_context(
        {"task_id": "cards", "prompt": "Inspect the format and report limitations."}
    )
    view, refs = task_review.attach(context, evidence, result)
    blind, allowed = requirement_context(view, refs)
    assert "AUDIT_ONLY" not in str(view) + str(refs) + str(blind) + str(allowed)
    knowledge = view["skills"][0]["content"]
    assert (
        "citations" not in knowledge
    )  # Quotes are not copied at the direction level again.
    assert knowledge["mechanisms"][0]["citations"][0]["quote"]
    unresolved = view["skills"][-1]["content"]
    assert unresolved["pending_findings"][0]["disposition"] == "deferred"
    assert unresolved["research_open_questions"] == []
    assert "citations" not in unresolved["task_need_coverage"][0]
    assert unresolved["task_need_coverage"][0]["finding_ids"] == ["finding_0"]
    assert all("audit" not in row for row in unresolved["pending_findings"])


def test_prior_previews_distinguish_copied_task_needs_without_audit_factual_titles():
    original = base()
    proposed = draft(original, audit_note="AUDIT_ONLY first group")
    first = proposed["source_findings"][0]
    proposed["source_findings"].extend(
        [
            {**first, "id": "finding_duplicate_need"},
            {
                **first,
                "id": "finding_other",
                "need_ids": ["need_1"],
                "direction_id": "knowledge_1",
                "block_ids": reading_units(original["sources"][0]["text"])[-1][
                    "block_ids"
                ],
                "title": "AUDIT_ONLY factual title",
                "reason": "AUDIT_ONLY causal explanation",
            },
            {
                **first,
                "id": "finding_deferred",
                "need_ids": ["need_1"],
                "disposition": "deferred",
            },
        ]
    )
    proposed["directions"] = [
        {"id": "knowledge_0", "direction": "AUDIT_ONLY generated title"}
    ]
    result = admit(original, proposed)
    context, evidence = build_context(
        {"task_id": "cards", "prompt": "Inspect the format and report limitations."}
    )
    view, refs = task_review.attach(context, evidence, result)
    preview = measurement_prior_view(view, refs, {})
    prior = [
        row
        for row in preview["skills"]
        if row["content"]["kind"] == "task_review_prior"
    ]
    assert len(prior) == 2 and all(row["preview_only"] for row in prior)
    assert [row["content"]["relevance"] for row in prior] == [
        "Inspect the format",
        "Report limitations",
    ]
    assert all("mechanisms" not in row["content"] for row in prior)
    assert "AUDIT_ONLY" not in str(preview) + str(refs)
    assert view["skills"][0]["content"][
        "mechanisms"
    ]  # Previewing did not mutate attached evidence.


def test_false_or_irrelevant_source_is_still_only_a_copy_not_semantic_verification():
    original = base("A deliberately incorrect source says the width is 99.\n")
    result = admit(original, draft(original))
    card = result["directions"][0]["mechanisms"][0]
    assert card["source_statement"] == original["sources"][0]["text"]
    assert card["grounding"] == "quotes_located"
    assert result["task_need_coverage"][0]["status"] == "related_knowledge"
    assert "not exhaustive or semantic verification" in result["coverage_boundary"]


def test_wrong_neighbor_cannot_publish_the_models_uncited_extra_procedure():
    original = base(
        "A failed validation discards this record.\n\n"
        "Recovery scans once and stops at the first invalid checksum.\n",
        contract="extractive-cards-v1",
    )
    proposed = draft(
        original,
        finding="Recovery scans once and stops at the first invalid checksum.",
        applicability="Every recovery path.",
    )
    result = admit(original, proposed)
    card = result["directions"][0]["mechanisms"][0]
    assert card["source_statement"] == "A failed validation discards this record.\n\n"
    assert "Recovery scans once" not in str(result["directions"])
    assert (
        card["grounding"] == "quotes_located"
    )  # Wrong selection is still possible, not semantic PASS.


@pytest.mark.parametrize("contract", ["source-facts-v1", "source-facts-v2"])
def test_historical_contracts_keep_generated_finding_and_legacy_quote_apis(contract):
    original = base(contract=contract)
    proposed = draft(
        original,
        finding="Historical statement.",
        applicability="Historical conditions.",
        reason="Historical reason.",
    )
    proposed["source_contract"] = contract
    proposed["directions"] = [{"id": "knowledge_0", "direction": "Historical topic"}]
    if contract == "source-facts-v1":
        proposed["source_findings"][0]["citations"] = [
            {
                "source_id": "source_0",
                "span_id": spans(original["sources"][0]["text"])[0]["span_id"],
            }
        ]
    normalized = normalize_source_brief(original, proposed, contract=contract)
    assert normalized["source_findings"][0]["finding"] == "Historical statement."
    assert "source_statement" not in normalized["source_findings"][0]
    result = task_review.admit_brief(
        original, normalized, block_only=contract == "source-facts-v2"
    )
    assert (
        result["directions"][0]["mechanisms"][0]["finding"] == "Historical statement."
    )
    source = original["sources"][0]
    lookup = {"source_0": source}
    assert citation(
        {"source_id": "source_0", "span_id": spans(source["text"])[0]["span_id"]},
        lookup,
    )
    assert citation(
        {"source_id": "source_0", "quote": "The documented width is 6 units."}, lookup
    )


@pytest.mark.asyncio
async def test_production_summary_forces_extractive_contract_in_two_stages():
    original = base()
    requests = []

    async def call(request):
        requests.append(deepcopy(request))
        if request["mode"] == "task_review_group_reduce":
            return keep_groups(request)
        response = draft(original, finding="AUDIT_ONLY ungrounded rewrite")
        response["source_contract"] = "source-facts-v2"
        return response

    result = await task_review.summarize(original, call)
    assert [r["mode"] for r in requests] == [
        "task_review_source_map",
        "task_review_group_reduce",
    ]
    shape = requests[0]["response"]["source_findings"][0]
    assert all(
        k not in shape
        for k in ("finding", "source_statement", "applicability", "reason")
    )
    assert "block_ids" in shape and "unit_id" not in shape
    assert (
        "structural_groups" in requests[0]["sources"][0]
        and "reading_units" not in requests[0]["sources"][0]
    )
    assert (
        "".join(block["text"] for block in requests[0]["sources"][0]["blocks"])
        == original["sources"][0]["text"]
    )
    assert "directions" not in requests[0]["response"]
    assert result["version"] == task_review.VERSION
    assert result["source_contract"] == SOURCE_CONTRACT == "extractive-cards-v3"
    assert result["directions"][0]["mechanisms"][0]["source_statement"]
    assert "AUDIT_ONLY" not in str(result["directions"])


@pytest.mark.asyncio
async def test_cache_requires_new_version_admission_and_contract_without_relabeling_old_cards():
    original = base()
    cached = admit(original, draft(original))
    cached.update(
        version=task_review.VERSION,
        summary_status="complete",
        produced_by={"summary": "original-role"},
    )
    context, _ = build_context(
        {"task_id": "cards", "prompt": "Inspect the format and report limitations."}
    )
    calls = []

    async def no_call(request):
        calls.append(request)
        raise ValueError("cache miss reaches the planner")

    hit = await task_review.review(
        context, no_call, no_call, no_call, enabled=True, cached=cached
    )
    assert (
        hit["cache_hit"] and hit["produced_by"] == cached["produced_by"] and not calls
    )
    for key, old in [
        ("version", "task-review-v26-extractive-cards"),
        ("source_admission", "extractive-card-v1"),
        ("source_contract", "extractive-cards-v1"),
    ]:
        stale = {**cached, key: old}
        with pytest.raises(ValueError, match="cache miss"):
            await task_review.review(
                context, no_call, no_call, no_call, enabled=True, cached=stale
            )
        assert stale[key] == old
    assert len(calls) == 3
