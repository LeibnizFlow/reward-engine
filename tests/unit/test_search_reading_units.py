"""Bounded syntax/context units preserve source bytes, not semantic completeness."""
# Unicode escapes preserve CJK coverage and original UTF-8/character boundaries.

from copy import deepcopy

import pytest

from reward_engine.core.outcome.knowledge_evidence import (
    citation,
    evidence_blocks,
    extractive_card,
    reading_units,
    spans,
    summary_sources,
)
from reward_engine.core.outcome.knowledge_evidence import (
    normalize_source_brief as normalize_current_brief,
)

SOURCE_CONTRACT = (
    "extractive-cards-v2"  # Historical v27 replay remains byte-compatible.
)


def normalize_source_brief(base, brief, *, contract=SOURCE_CONTRACT):
    return normalize_current_brief(base, brief, contract=contract)


def fixture(text):
    return {
        "sources": [
            {
                "source_id": "source_0",
                "title": "Reference",
                "text": text,
                "url": "https://example.test/reference",
                "offset": 400,
                "total_chars": len(text) + 900,
                "excerpt_only": True,
                "read_kind": "body",
            }
        ],
        "task_needs": [{"need_id": "need_0", "requirement": "Inspect the format."}],
    }


def draft(original, **extra):
    unit = reading_units(original["sources"][0]["text"])[0]
    return {
        "source_findings": [
            {
                "id": "finding_0",
                "source_id": "source_0",
                "unit_id": unit["unit_id"],
                "need_ids": ["need_0"],
                "disposition": "retained",
                "direction_id": "knowledge_0",
                **extra,
            }
        ]
    }


def unit_text(text, unit):
    blocks = {b["block_id"]: b for b in evidence_blocks(text)}
    result = "".join(blocks[identity]["text"] for identity in unit["block_ids"])
    assert result == text[unit["start"] : unit["end"]]
    return result


@pytest.mark.parametrize("quote", ["> ", "> > ", ""])
def test_lead_and_fenced_algorithm_travel_together_without_changing_legacy_blocks(
    quote,
):
    lead = "The calculation is defined as follows:\n\n"
    code = (
        "".join(quote + line + "\n" for line in ["```c", "consume(input, 6);", "```"])
        + "\n"
    )
    text = lead + code + "The inputs must match the documented version.\n"
    old = evidence_blocks(text)
    units = reading_units(text)
    assert any(lead + code in unit_text(text, unit) for unit in units)
    assert evidence_blocks(text) == old
    if quote:
        assert old[0]["text"] == lead  # Legacy splitter remains unchanged.
        assert old[1]["text"] == code
        assert all(
            set(unit["block_ids"]) <= {b["block_id"] for b in old} for unit in units
        )


@pytest.mark.parametrize("marker", ["\\*", "*", "†", "‡", "[1]", "[^a]:", "(2)", "3."])
def test_table_and_immediate_explicit_footnote_are_bound_by_syntax(marker):
    table = f"| Field | Rule |\n| --- | --- |\n| width | six {marker} |\n\n"
    note = f"{marker} Only the documented release has this behavior.\n\n"
    text = "Format:\n\n" + table + note
    units = reading_units(text)
    assert any(table + note in unit_text(text, unit) for unit in units)
    selected = next(unit for unit in units if table + note in unit_text(text, unit))
    assert set(selected["block_ids"]) == {b["block_id"] for b in evidence_blocks(text)}


def test_nearby_short_section_context_keeps_conditions_without_remote_exception_claims():
    text = (
        "## Lifecycle\n\nA normal close flushes data and removes temporary files.\n\n"
        "## Variations\n\nIn exclusive mode the temporary index is not used.\n\n"
        "A persistence option flushes data but leaves the temporary files present.\n\n"
    )
    units = reading_units(text)
    assert any(
        "A normal close" in unit_text(text, u)
        and "A persistence option" in unit_text(text, u)
        for u in units
    )
    assert all("text" not in u for u in units)
    distant = (
        text
        + "## Unrelated large section\n\n"
        + "Bounded unrelated line.\n" * 500
        + "\n## Remote\n\nAnother exception.\n"
    )
    selected = next(
        u for u in reading_units(distant) if "A normal close" in unit_text(distant, u)
    )
    assert "Another exception." not in unit_text(distant, selected)
    assert selected["context_limited"]


@pytest.mark.parametrize("ending", ["\n", "\r\n", "\r"])
def test_large_structure_units_are_capped_contiguous_and_explicitly_continued(ending):
    text = ending.join(
        ["# Long specification", "", "```text"]
        + ["A complete bounded source line."] * 700
        + ["```", ""]
    )
    blocks = evidence_blocks(text)
    units = reading_units(text)
    assert len(units) > 1 and len({u["unit_id"] for u in units}) == len(units)
    assert {bid for u in units for bid in u["block_ids"]} == {
        b["block_id"] for b in blocks
    }
    for unit in units:
        assert len(unit_text(text, unit)) <= 6000
        assert unit["continuation"]
        assert all(
            row["structure_start"]
            <= row["included_start"]
            < row["included_end"]
            <= row["structure_end"]
            for row in unit["continuation"]
        )
        assert not unit["incomplete_excerpt"]


def test_unclosed_quoted_fence_and_long_line_keep_incompleteness_metadata():
    text = "The routine is:\n\n> ```c\n> " + "\u6570\u636e" * 4100
    units = reading_units(text)
    assert all(u["incomplete_excerpt"] for u in units)
    assert all(len(unit_text(text, u)) <= 6000 for u in units)
    original = fixture(text)
    result = normalize_source_brief(original, draft(original))
    card = result["source_findings"][0]
    assert card["reading_unit"]["incomplete_excerpt"]
    assert any(b["fragmented_line"] for b in card["blocks"])


@pytest.mark.parametrize("value", [0, 1, 1999, -1, True, 2.5, "6000", 6001])
def test_invalid_cap_cannot_split_or_rename_historical_blocks(value):
    with pytest.raises(ValueError, match="MAX_BLOCK_CHARS"):
        reading_units("Body.", value)


@pytest.mark.parametrize(
    "structure",
    [
        "| Field | Value |\n| --- | --- |\n| width | six \\* |\n\n\\* Version-specific exception.\n\n",
        "The calculation is:\n\n> ```c\n> consume(input, 6);\n> ```\n\n",
    ],
)
def test_short_bound_structure_is_not_split_at_a_long_sections_unit_boundary(structure):
    prefix = "# Long section\n\n" + ("Context sentence. " * 330) + "\n\n"
    text = prefix + structure + "Trailing context. " * 100
    units = reading_units(text)
    affected = [
        unit_text(text, unit)
        for unit in units
        if unit["start"] < len(prefix) + len(structure) and unit["end"] > len(prefix)
    ]
    assert affected and all(structure in content for content in affected)


def test_summary_exposes_body_once_and_only_id_metadata_for_units():
    original = fixture("A UNIQUE_BODY_SENTINEL.\n\nAnother paragraph.\n")
    before = deepcopy(original)
    navigation = {
        **original["sources"][0],
        "source_id": "navigation",
        "read_kind": "navigation",
    }
    public = summary_sources(original["sources"] + [navigation])
    assert len(public) == 1
    assert public[0]["offset"] == 400 and public[0]["excerpt_only"]
    assert (
        "".join(b["text"] for b in public[0]["blocks"])
        == original["sources"][0]["text"]
    )
    assert str(public).count("UNIQUE_BODY_SENTINEL") == 1
    assert "text" not in public[0] and all(
        "text" not in u for u in public[0]["structural_groups"]
    )
    assert original == before


def test_production_unit_resolves_code_owned_blocks_and_ignores_model_override():
    original = fixture(
        "A documented method:\n\n> ```c\n> consume(input, 6);\n> ```\n\nA condition.\n"
    )
    proposed = draft(
        original,
        block_ids=["invented"],
        source_statement="AUDIT_ONLY invented source fact",
        reading_unit={"block_ids": ["invented"]},
        citations=[{"quote": "AUDIT_ONLY"}],
    )
    result = normalize_source_brief(original, proposed)
    row = result["source_findings"][0]
    expected = reading_units(original["sources"][0]["text"])[0]
    assert result["source_contract"] == SOURCE_CONTRACT == "extractive-cards-v2"
    assert row["disposition"] == "retained"
    assert row["reading_unit"] == expected
    assert row["block_ids"] == expected["block_ids"]
    assert row["source_statement"] == unit_text(
        original["sources"][0]["text"], expected
    )
    assert row["source_offset"] == 400
    assert "".join(c["quote"] for c in row["citations"]) == row["source_statement"]
    card = extractive_card(row)
    assert card["reading_unit"] == expected and "AUDIT_ONLY" not in str(card)


@pytest.mark.parametrize("identity", [None, [], {}, "invented", "unit_0_1"])
def test_missing_or_invalid_unit_cannot_fall_back_to_real_raw_blocks(identity):
    original = fixture("An actual source statement.\n")
    proposed = draft(
        original,
        unit_id=identity,
        block_ids=[evidence_blocks(original["sources"][0]["text"])[0]["block_id"]],
    )
    row = normalize_source_brief(original, proposed)["source_findings"][0]
    assert row["disposition"] == "deferred"
    assert (
        row["block_ids"] == []
        and row["citations"] == []
        and row["source_statement"] == ""
    )


def test_unit_cannot_cross_source_or_use_navigation_and_good_card_survives_bad_one():
    original = fixture("First source is longer.\n\nMore context.\n")
    original["sources"].append(
        {**original["sources"][0], "source_id": "source_1", "text": "Short.\n"}
    )
    proposed = draft(original)
    proposed["source_findings"].append(
        {
            **proposed["source_findings"][0],
            "id": "wrong_source",
            "source_id": "source_1",
        }
    )
    rows = normalize_source_brief(original, proposed)["source_findings"]
    assert [r["disposition"] for r in rows] == ["retained", "deferred"]
    original["sources"][0]["read_kind"] = "navigation"
    assert (
        normalize_source_brief(original, proposed)["source_findings"][0]["disposition"]
        == "deferred"
    )


def test_heading_only_cannot_publish_and_whitespace_context_does_not_erase_body():
    original = fixture("# Only a heading\n")
    assert (
        normalize_source_brief(original, draft(original))["source_findings"][0][
            "disposition"
        ]
        == "deferred"
    )
    original = fixture("\n\n# Heading\n\nA real source paragraph.\n")
    row = normalize_source_brief(original, draft(original))["source_findings"][0]
    assert row["disposition"] == "retained"
    assert row["source_statement"] == original["sources"][0]["text"]
    assert row["citations"] and all(c["quote"].strip() for c in row["citations"])


def test_saved_v26_block_selection_still_works_without_units_and_old_citations_locate():
    original = fixture("The stored source rule.\n\nIts condition.\n")
    proposed = draft(original)
    row = proposed["source_findings"][0]
    row.pop("unit_id")
    row["block_ids"] = [evidence_blocks(original["sources"][0]["text"])[0]["block_id"]]
    historical = normalize_source_brief(
        original, proposed, contract="extractive-cards-v1"
    )
    assert historical["source_contract"] == "extractive-cards-v1"
    card = historical["source_findings"][0]
    assert card["disposition"] == "retained" and "reading_unit" not in card
    assert card["source_statement"] == "The stored source rule.\n\n"
    assert (
        normalize_source_brief(original, proposed)["source_findings"][0]["disposition"]
        == "deferred"
    )
    source = original["sources"][0]
    lookup = {"source_0": source}
    assert citation(card["citations"][0], lookup, block_only=True)
    assert citation(
        {"source_id": "source_0", "span_id": spans(source["text"])[0]["span_id"]},
        lookup,
    )
    assert citation(
        {"source_id": "source_0", "quote": "The stored source rule."}, lookup
    )


def test_empty_body_produces_no_units():
    assert reading_units("") == []
