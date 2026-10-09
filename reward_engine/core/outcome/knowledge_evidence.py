"""Deterministic evidence handles; copying a passage does not validate its meaning."""

import re
from bisect import bisect_right
from collections import Counter
from copy import deepcopy

SOURCE_CONTRACT = "extractive-cards-v3"
MAX_BLOCK_CHARS = 2000
MAX_UNIT_CHARS = 6000


def evidence_blocks(text, max_chars=MAX_BLOCK_CHARS):
    """Lossless structural slices, not a claim that their contents are trustworthy."""
    if type(max_chars) is not int or max_chars <= 0:
        raise ValueError("max_chars must be a positive integer")
    lines = text.splitlines(keepends=True)
    offsets, position = [], 0
    for line in lines:
        offsets.append(position)
        position += len(line)
    offsets.append(position)
    line_boundaries = set(offsets)

    def kind(line):
        stripped = line.strip()
        if not stripped:
            return "whitespace"
        if re.match(r"(`{3,}|~{3,})", stripped):
            return "code"
        if re.match(r"#{1,6}\s", stripped):
            return "heading"
        if "|" in stripped and (stripped.startswith("|") or stripped.count("|") >= 2):
            return "table"
        if re.match(r"(?:[-+*]|\d+[.)])\s", stripped):
            return "list"
        return "paragraph"

    regions, i = [], 0
    while i < len(lines):
        start, category, incomplete = i, kind(lines[i]), False
        i += 1
        if category == "code":
            marker = re.match(r"(`{3,}|~{3,})", lines[start].strip()).group()
            incomplete = True
            while i < len(lines):
                close = lines[i].strip()
                i += 1
                if len(close) >= len(marker) and set(close) == {marker[0]}:
                    incomplete = False
                    break
        elif category != "heading":
            while i < len(lines):
                following = kind(lines[i])
                if following == "whitespace":
                    while i < len(lines) and not lines[i].strip():
                        i += 1
                    # Blank lines inside a list do not separate its items.
                    if (
                        category == "list"
                        and i < len(lines)
                        and kind(lines[i]) == "list"
                    ):
                        continue
                    break
                if following != category and not (
                    category == "list"
                    and lines[i][:1].isspace()
                    and following == "paragraph"
                ):
                    break
                i += 1
        # Keep following whitespace, without stripping or manufacturing text.
        while i < len(lines) and not lines[i].strip():
            i += 1
        region = (offsets[start], offsets[i], category, incomplete)
        if (
            regions
            and category in {"table", "list", "code"}
            and regions[-1][2] in {"paragraph", "heading"}
            and (
                regions[-1][2] == "heading"
                or text[regions[-1][0] : regions[-1][1]].rstrip().endswith(":")
            )
        ):
            previous = regions.pop()
            region = (previous[0], region[1], category, incomplete)
        regions.append(region)

    blocks = []
    for begin, end, category, incomplete in regions:
        pieces, cursor = [], begin
        while cursor < end:
            cut = min(cursor + max_chars, end)
            if cut < end:
                line_end = offsets[bisect_right(offsets, cut) - 1]
                if line_end > cursor:
                    cut = line_end
            fragmented = cursor not in line_boundaries or cut not in line_boundaries
            pieces.append((cursor, cut, fragmented))
            cursor = cut
        for part, (start, stop, fragmented) in enumerate(pieces):
            continuation = (
                None
                if len(pieces) == 1 and not incomplete
                else {
                    "part": part + 1,
                    "parts": len(pieces),
                    "structure_start": begin,
                    "structure_end": end,
                    "incomplete_excerpt": incomplete,
                }
            )
            blocks.append(
                {
                    "block_id": f"block_{start}_{stop}",
                    "start": start,
                    "end": stop,
                    "kind": category,
                    "text": text[start:stop],
                    "continuation": continuation,
                    "fragmented_line": fragmented,
                }
            )
    return blocks


def reading_units(text, max_chars=MAX_UNIT_CHARS, *, atomic_only=False):
    """Bounded contiguous reading context over immutable historical block IDs.

    Markdown syntax supplies grouping, not semantic entailment. Small sections,
    code lead-ins and explicit table footnotes travel together where the budget
    permits. Nearby context cannot establish that remote exceptions are absent.
    The original text is present only in evidence_blocks, never repeated here.
    """
    if type(max_chars) is not int or not MAX_BLOCK_CHARS <= max_chars <= MAX_UNIT_CHARS:
        raise ValueError(
            "max_chars must be an integer between MAX_BLOCK_CHARS and MAX_UNIT_CHARS"
        )
    blocks = evidence_blocks(text)
    if not blocks:
        return []
    starts = [b["start"] for b in blocks]
    joins = set()

    def join(first, last):
        joins.update(range(first, last))

    def unquote(line):
        return re.sub(r"^(?:\s*>)+\s?", "", line).strip()

    # Restore split structures without changing their legacy slices.
    for i, block in enumerate(blocks[:-1]):
        continued = block["continuation"]
        if continued and blocks[i + 1]["continuation"] == {
            **continued,
            "part": continued["part"] + 1,
        }:
            join(i, i + 1)

    # Recognize fenced code even inside Markdown blockquotes. Only inspect a
    # normalized line for syntax; copy the untouched source text later.
    fence, begin, cursor, code_ranges = None, None, 0, []
    for line in text.splitlines(keepends=True):
        stripped = unquote(line)
        marker = re.match(r"(`{3,}|~{3,})", stripped)
        if fence is None and marker:
            fence, begin = marker.group(), cursor
        elif fence and re.fullmatch(
            re.escape(fence[0]) + "{" + str(len(fence)) + ",}", stripped
        ):
            code_ranges.append((begin, cursor + len(line), False))
            fence, begin = None, None
        cursor += len(line)
    if fence:
        code_ranges.append((begin, len(text), True))
    for begin, end, _ in code_ranges:
        first, last = (
            bisect_right(starts, begin) - 1,
            bisect_right(starts, max(begin, end - 1)) - 1,
        )
        if (
            first
            and not text[blocks[first]["start"] : begin].strip()
            and blocks[first - 1]["kind"] in {"paragraph", "heading"}
        ):
            first -= 1
        join(first, last)

    footnote = re.compile(
        r"^(?:\\?[*†‡]+|\[\^?[A-Za-z0-9_-]+\]|\(\d+\)|\d+[.)])(?::|\s)"
    )
    for i, block in enumerate(blocks[:-1]):
        if block["kind"] != "table":
            continue
        following = i + 1
        while following < len(blocks) and footnote.match(
            unquote(blocks[following]["text"])
        ):
            join(following - 1, following)
            following += 1

    atomic_joins = set(joins)
    # A short headed section is a coherent reading scope; a long one is still
    # bounded below. Headings are syntax, not a domain-specific exception rule.
    headings = (
        []
        if atomic_only
        else [i for i, b in enumerate(blocks) if b["kind"] == "heading"]
    )
    for position, first in enumerate(headings):
        last = (
            headings[position + 1] - 1
            if position + 1 < len(headings)
            else len(blocks) - 1
        )
        join(first, last)
    groups, first = [], 0
    for i in range(len(blocks)):
        if i not in joins:
            groups.append((first, i + 1))
            first = i + 1
    chunks = []
    for first, stop in groups:
        pieces, begin, i = [], first, first
        while i < stop:
            if blocks[i]["end"] - blocks[begin]["start"] > max_chars:
                cut = i
                while cut > begin and cut - 1 in atomic_joins:
                    cut -= 1
                # An atomic structure larger than the unit limit must itself
                # continue; otherwise move the complete structure to next unit.
                cut = i if cut == begin else cut
                pieces.append((begin, cut))
                begin, i = cut, cut
            else:
                i += 1
        pieces.append((begin, stop))
        chunks.extend({"first": begin, "stop": end} for begin, end in pieces)
    units, seen = [], set()
    for i, chunk in enumerate(chunks):
        first, stop = chunk["first"], chunk["stop"]
        # At most one neighbouring structural chunk per side, never a skip.
        for candidate_first, candidate_stop in (
            ()
            if atomic_only
            else (
                (first, chunks[i + 1]["stop"])
                if i + 1 < len(chunks)
                else (first, stop),
                (chunks[i - 1]["first"], stop) if i else (first, stop),
            )
        ):
            candidate_first, candidate_stop = (
                min(first, candidate_first),
                max(stop, candidate_stop),
            )
            if (
                blocks[candidate_stop - 1]["end"] - blocks[candidate_first]["start"]
                <= max_chars
            ):
                first, stop = candidate_first, candidate_stop
        begin, end = blocks[first]["start"], blocks[stop - 1]["end"]
        unit_id = f"unit_{begin}_{end}"
        if unit_id in seen:
            continue
        seen.add(unit_id)
        selected = blocks[first:stop]
        continuation = []
        for group_first, group_stop in groups:
            group_start, group_end = (
                blocks[group_first]["start"],
                blocks[group_stop - 1]["end"],
            )
            if (
                group_start < end
                and group_end > begin
                and (group_start < begin or group_end > end)
            ):
                continuation.append(
                    {
                        "structure_start": group_start,
                        "structure_end": group_end,
                        "included_start": max(begin, group_start),
                        "included_end": min(end, group_end),
                    }
                )
        units.append(
            {
                "unit_id": unit_id,
                "start": begin,
                "end": end,
                "block_ids": [b["block_id"] for b in selected],
                "core_block_ids": [
                    b["block_id"] for b in blocks[chunk["first"] : chunk["stop"]]
                ],
                "context_block_ids": [
                    b["block_id"]
                    for b in selected
                    if not chunk["first"]
                    <= bisect_right(starts, b["start"]) - 1
                    < chunk["stop"]
                ],
                "continuation": continuation,
                "incomplete_excerpt": any(
                    a < end and z > begin and incomplete
                    for a, z, incomplete in code_ranges
                )
                or any(
                    (b["continuation"] or {}).get("incomplete_excerpt")
                    for b in selected
                ),
                "context_limited": first > 0 or stop < len(blocks),
            }
        )
    return units


def need_next_action(status):
    return {
        "supported": "Check source applicability against actual Task materials; knowledge is not task completion.",
        "related_knowledge": "Related sources were found. Check their applicability, remaining questions and actual Task materials; Task completion is not assessed.",
        "no_research_needed": "Verify the original Task obligation directly; it is not marked complete.",
        "needs_material": "Obtain or read the actual Task materials before drawing a task-specific conclusion.",
        "needs_user": "Ask the user for the missing information; do not invent their answer.",
        "needs_research": "Research the unresolved knowledge for this Task need.",
    }.get(
        status,
        "Revisit this Task need; no factual or task-completion conclusion is available.",
    )


def normalize_source_brief(base, brief, *, contract=None):
    """Project source statements, not a second model-written repair plan.

    Citations are still checked by admission. This removes redundant advice
    fields; it cannot establish that a source statement was interpreted correctly.
    """

    def rows(value):
        return value if isinstance(value, list) else []

    contract = contract or brief.get("source_contract", SOURCE_CONTRACT)
    if contract in {SOURCE_CONTRACT, "extractive-cards-v2", "extractive-cards-v1"}:
        return _extractive_brief(base, brief, contract=contract)
    if contract not in {"source-facts-v2", "source-facts-v1"}:
        raise ValueError("Unknown source-facts contract")
    legacy = contract == "source-facts-v1"
    needs = {
        r["need_id"]: r["requirement"]
        for r in rows(base.get("task_needs"))
        if isinstance(r, dict)
        and isinstance(r.get("need_id"), str)
        and isinstance(r.get("requirement"), str)
    }
    sources = {s["source_id"]: s for s in body_sources(base.get("sources", []))}
    verification = (
        "Check these source statements and their applicability against the actual Task evidence; "
        "no task-specific repair is established here."
    )
    result = {k: deepcopy(brief[k]) for k in ("source_notes",) if k in brief}
    if legacy and "uncertainties" in brief:
        result["uncertainties"] = deepcopy(brief["uncertainties"])
    result.update(source_contract=contract, source_findings=[], directions=[])
    counts = Counter(
        r.get("id")
        for r in rows(brief.get("source_findings"))
        if isinstance(r, dict) and isinstance(r.get("id"), str)
    )
    for raw in rows(brief.get("source_findings")):
        if not isinstance(raw, dict):
            result["source_findings"].append({"raw": deepcopy(raw)})
            continue
        row = {
            k: deepcopy(raw[k])
            for k in (
                "id",
                "source_id",
                "block_ids",
                "finding",
                "applicability",
                "need_ids",
                "disposition",
                "direction_id",
                "reason",
            )
            if k in raw
        }
        sid, block_ids = row.get("source_id"), row.get("block_ids")
        source = sources.get(sid) if isinstance(sid, str) else None
        offered = evidence_blocks(source["text"]) if source else []
        positions = {b["block_id"]: i for i, b in enumerate(offered)}
        bound = (
            isinstance(block_ids, list)
            and 1 <= len(block_ids) <= 3
            and all(isinstance(b, str) and b in positions for b in block_ids)
        )
        selected = [positions[b] for b in block_ids] if bound else []
        bound = bool(
            bound and selected == list(range(selected[0], selected[0] + len(selected)))
        )
        row["citations"] = (
            deepcopy(raw.get("citations", []))
            if legacy
            else [{"source_id": sid, "block_id": b} for b in block_ids]
            if bound
            else []
        )
        if legacy:
            bound = bool(rows(row["citations"]))
        ids = row.get("need_ids")
        valid = (
            isinstance(ids, list)
            and bool(ids)
            and all(isinstance(i, str) and i in needs for i in ids)
        )
        row.update(
            relevance="; ".join(dict.fromkeys(needs[i] for i in ids))
            if valid
            else "Task connection is unresolved; do not treat this as task-specific advice.",
            verification=verification,
        )
        if (
            not isinstance(row.get("id"), str)
            or not row["id"].strip()
            or counts[row["id"]] != 1
        ):
            row.update(
                disposition="deferred",
                projection_issue="Missing or duplicate finding ID",
            )
        elif not valid:
            row.update(
                disposition="deferred",
                projection_issue="Missing or unknown Task need IDs",
            )
        elif (
            not bound
            or not all(
                isinstance(row.get(k), str) and row[k].strip()
                for k in ("finding", "applicability", "reason")
            )
            or any(
                citation(c, sources, span_only=True, block_only=not legacy) is None
                for c in rows(row.get("citations"))
            )
        ):
            row.update(
                disposition="deferred",
                projection_issue="Need a source statement, conditions and 1-3 same-source adjacent body blocks",
            )
        result["source_findings"].append(row)
    for raw in rows(brief.get("directions")):
        if not isinstance(raw, dict) or not isinstance(raw.get("id"), str):
            continue
        members = [
            r
            for r in result["source_findings"]
            if r.get("disposition") == "retained" and r.get("direction_id") == raw["id"]
        ]
        if not members:
            continue
        citations = []
        for member in members:
            for c in rows(member.get("citations")):
                if c not in citations:
                    citations.append(deepcopy(c))
        # A grouping label is not another place to independently prescribe a method.
        result["directions"].append(
            {
                "id": raw["id"],
                "direction": raw.get("direction", ""),
                "relevance": "; ".join(dict.fromkeys(r["relevance"] for r in members)),
                "applicability": "\n".join(
                    dict.fromkeys(
                        r.get("applicability", "")
                        for r in members
                        if isinstance(r.get("applicability"), str)
                    )
                ),
                "verification": verification,
                "citations": citations,
            }
        )
    coverage = []
    by_id = {
        r["id"]: r
        for r in result["source_findings"]
        if isinstance(r.get("id"), str) and counts[r["id"]] == 1
    }
    for raw in rows(brief.get("task_need_coverage")):
        if not isinstance(raw, dict):
            continue
        if legacy:
            row = {
                k: deepcopy(raw[k])
                for k in ("need_id", "status", "citations")
                if k in raw
            }
            row["next_action"] = need_next_action(row.get("status"))
            coverage.append(row)
            continue
        row = {
            k: deepcopy(raw[k])
            for k in ("need_id", "status", "finding_ids")
            if k in raw
        }
        ids = row.get("finding_ids")
        valid = (
            isinstance(ids, list)
            and bool(ids)
            and all(
                isinstance(i, str)
                and i in by_id
                and by_id[i].get("disposition") == "retained"
                and row.get("need_id") in by_id[i].get("need_ids", [])
                for i in ids
            )
        )
        row["citations"] = []
        if row.get("status") == "supported":
            if not valid:
                row["status"] = "unknown"
            else:
                for identity in ids:
                    for c in by_id[identity]["citations"]:
                        if c not in row["citations"]:
                            row["citations"].append(deepcopy(c))
        row["next_action"] = need_next_action(row.get("status"))
        coverage.append(row)
    result["task_need_coverage"] = coverage
    return result


def _extractive_brief(base, brief, *, contract=SOURCE_CONTRACT):
    """Select existing passages; never pair a generated factual sentence with them.

    Selection/relevance can still be wrong and sources can be false. This only
    removes model-written facts from the authoritative knowledge projection.
    """

    def rows(value):
        return value if isinstance(value, list) else []

    def identity(value):
        return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,80}", value)

    needs = {
        r["need_id"]: r["requirement"]
        for r in rows(base.get("task_needs"))
        if isinstance(r, dict)
        and isinstance(r.get("need_id"), str)
        and isinstance(r.get("requirement"), str)
        and r["requirement"].strip()
    }
    sources = {s["source_id"]: s for s in body_sources(base.get("sources", []))}
    raw_rows = rows(brief.get("source_findings"))
    split_refs = {}
    if contract == SOURCE_CONTRACT:
        raw_rows, split_refs = _split_selections(raw_rows, sources)
    counts = Counter(
        r.get("id")
        for r in raw_rows
        if isinstance(r, dict) and isinstance(r.get("id"), str)
    )
    result = {
        "source_contract": contract,
        "source_findings": [],
        "directions": [],
        "source_notes": [],
        "task_need_coverage": [],
        "audit": {
            k: deepcopy(v)
            for k, v in brief.items()
            if k not in {"source_findings", "task_need_coverage", "source_contract"}
        },
    }
    unit_contract = contract == "extractive-cards-v2"
    current = contract == SOURCE_CONTRACT
    modern = current or unit_contract
    allowed = {
        "id",
        "source_id",
        "need_ids",
        "disposition",
        "direction_id",
        "unit_id" if unit_contract else "block_ids",
    }
    for index, raw in enumerate(raw_rows):
        raw = raw if isinstance(raw, dict) else {"raw": raw}
        sid = raw.get("source_id")
        source = sources.get(sid) if isinstance(sid, str) else None
        offered = evidence_blocks(source["text"]) if source else []
        positions = {b["block_id"]: i for i, b in enumerate(offered)}
        unit_id = raw.get("unit_id")
        unit = (
            next(
                (u for u in reading_units(source["text"]) if u["unit_id"] == unit_id),
                None,
            )
            if source and isinstance(unit_id, str)
            else None
        )
        block_ids = (
            unit["block_ids"]
            if unit_contract and unit
            else None
            if unit_contract
            else raw.get("block_ids")
        )
        requested_blocks = deepcopy(block_ids)
        if (
            current
            and isinstance(block_ids, list)
            and block_ids
            and all(isinstance(b, str) and b in positions for b in block_ids)
        ):
            selected_ids = set(block_ids)
            groups = reading_units(source["text"], atomic_only=True)
            selected_ids = {
                b
                for group in groups
                if selected_ids.intersection(group["block_ids"])
                for b in group["block_ids"]
            }
            block_ids = [
                b["block_id"] for b in offered if b["block_id"] in selected_ids
            ]
        bound = (
            isinstance(block_ids, list)
            and 1 <= len(block_ids) <= (len(offered) if modern else 3)
            and all(isinstance(b, str) and b in positions for b in block_ids)
        )
        selected = [positions[b] for b in block_ids] if bound else []
        bound = bool(
            bound and selected == list(range(selected[0], selected[0] + len(selected)))
        )
        blocks = [offered[i] for i in selected] if bound else []
        substantive = any(
            b["text"].strip() and b["kind"] not in {"heading", "whitespace"}
            for b in blocks
        )
        ids = raw.get("need_ids")
        valid_needs = (
            isinstance(ids, list)
            and bool(ids)
            and all(isinstance(i, str) and i in needs for i in ids)
        )
        status, group = raw.get("disposition"), raw.get("direction_id")
        excluded = current and status == "excluded"
        issue = None
        if not identity(raw.get("id")) or counts[raw["id"]] != 1:
            issue = "Missing, invalid or duplicate card ID."
        elif not valid_needs and not excluded:
            issue = "Missing or unknown Task need IDs."
        elif (
            not bound
            or (not modern and any(not b["text"].strip() for b in blocks))
            or (current and sum(len(b["text"]) for b in blocks) > MAX_UNIT_CHARS)
        ):
            issue = (
                "Need one supplied same-source reading unit."
                if unit_contract
                else "Need contiguous supplied blocks within 6000 characters, including structural companions."
                if current
                else "Need 1-3 supplied same-source adjacent nonblank body blocks."
            )
        elif not substantive and not excluded:
            issue = "Heading-only selection is not a substantive source statement."
        elif not isinstance(status, str) or status not in {
            "retained",
            "deferred",
            "excluded",
        }:
            issue = "Unknown card disposition."
        elif status == "retained" and not identity(group):
            issue = "Retained card needs a bounded direction ID."
        row = {
            "id": raw["id"] if identity(raw.get("id")) else f"invalid_card_{index}",
            "source_id": sid if source else None,
            "block_ids": [b["block_id"] for b in blocks],
            "source_statement": "".join(b["text"] for b in blocks),
            "blocks": [
                {k: deepcopy(v) for k, v in b.items() if k != "text"} for b in blocks
            ],
            "source_offset": source.get("offset", 0) if source else 0,
            "need_ids": list(
                dict.fromkeys(i for i in ids if isinstance(i, str) and i in needs)
            )
            if isinstance(ids, list)
            else [],
            "disposition": "deferred" if issue else status,
            "direction_id": group if identity(group) else None,
            "citations": [
                citation(
                    {"source_id": sid, "block_id": b["block_id"]},
                    sources,
                    block_only=True,
                )
                for b in blocks
                if b["text"].strip()
            ],
            "audit": {k: deepcopy(v) for k, v in raw.items() if k not in allowed},
        }
        if unit_contract:
            row["reading_unit"] = deepcopy(unit)
        if current:
            row["requested_block_ids"] = requested_blocks
        if issue:
            row["projection_issue"] = issue
            row["audit"]["selection"] = {
                k: deepcopy(raw[k]) for k in allowed if k in raw
            }
        result["source_findings"].append(row)
    # A section label is valid CONTEXT for a selected substantive passage in
    # the same source/group, not an independent fact. Keep the label verbatim
    # so conditions expressed in headings are not erased. A title alone, an
    # invalid reference or another source's body can never satisfy this rule.
    if current:
        body_groups = {
            (r["source_id"], r["direction_id"])
            for r in result["source_findings"]
            if r["disposition"] == "retained" and not r.get("projection_issue")
        }
        for row in result["source_findings"]:
            selection = row.get("audit", {}).get("selection", {})
            if (
                row.get("projection_issue")
                == "Heading-only selection is not a substantive source statement."
                and selection.get("disposition") == "retained"
                and (row["source_id"], row["direction_id"]) in body_groups
                and any(b["kind"] == "heading" for b in row["blocks"])
            ):
                row.update(disposition="retained", context_only=True)
                row["audit"]["context_binding"] = row.pop("projection_issue")
    aliases = _deduplicate_cards(result["source_findings"]) if current else {}
    if aliases:
        result["audit"]["card_aliases"] = aliases
    if split_refs:
        result["audit"]["split_cards"] = deepcopy(split_refs)

    def mapped_refs(ids):
        return [
            aliases.get(child, child) if isinstance(child, str) else child
            for identity in ids
            for child in (
                split_refs.get(identity, [identity])
                if isinstance(identity, str)
                else [identity]
            )
        ]

    for row in result["source_findings"]:
        if row["disposition"] != "retained":
            continue
        if not any(d["id"] == row["direction_id"] for d in result["directions"]):
            result["directions"].append(
                {
                    "id": row["direction_id"],
                    "direction": "Selected source passages",
                    "applicability": "Conditions and exceptions remain in the copied source passages; excerpts may be incomplete.",
                    "verification": "Check these fallible sources against actual Task materials; no task-specific conclusion is established.",
                    "citations": [],
                }
            )
        direction = next(
            d for d in result["directions"] if d["id"] == row["direction_id"]
        )
        for c in row["citations"]:
            if c not in direction["citations"]:
                direction["citations"].append(deepcopy(c))
    for direction in result["directions"]:
        # Keep previews distinguishable without a model-written factual title or
        # causal explanation. These are copied Task needs, not source assertions.
        direction["relevance"] = "; ".join(
            dict.fromkeys(
                needs[need_id]
                for row in result["source_findings"]
                if row["disposition"] == "retained"
                and row["direction_id"] == direction["id"]
                for need_id in row["need_ids"]
            )
        )
    for raw in rows(brief.get("source_notes")):
        if (
            isinstance(raw, dict)
            and isinstance(raw.get("source_id"), str)
            and raw["source_id"] in sources
        ):
            result["source_notes"].append(
                {
                    "source_id": raw["source_id"],
                    "reason": "Source considered; no retained extractive card selected.",
                }
            )
    by_id = {
        r["id"]: r for r in result["source_findings"] if r["disposition"] == "retained"
    }
    for raw in rows(brief.get("task_need_coverage")):
        if not isinstance(raw, dict):
            continue
        row = {
            k: deepcopy(raw[k])
            for k in ("need_id", "status", "finding_ids")
            if k in raw
        }
        if modern:
            row.update(
                {
                    k: deepcopy(raw[k])
                    for k in ("open_questions", "materials_needed")
                    if k in raw
                }
            )
        if current:
            if isinstance(row.get("finding_ids"), list):
                row["finding_ids"] = mapped_refs(row["finding_ids"])
            for question in rows(row.get("open_questions")):
                if isinstance(question, dict) and isinstance(
                    question.get("card_ids"), list
                ):
                    question["card_ids"] = mapped_refs(question["card_ids"])
        ids = row.get("finding_ids")
        if isinstance(ids, list):
            # Labels remain available for reading context, but cannot be the
            # only evidence counted as knowledge support for a Task need.
            ids = [
                i
                for i in ids
                if not isinstance(i, str)
                or i not in by_id
                or not by_id[i].get("context_only")
            ]
        valid = (
            isinstance(ids, list)
            and bool(ids)
            and all(
                isinstance(i, str)
                and i in by_id
                and row.get("need_id") in by_id[i]["need_ids"]
                for i in ids
            )
        )
        status = row.get("status")
        linked_status = "related_knowledge" if modern else "supported"
        if not isinstance(status, str) or status not in {
            linked_status,
            "no_research_needed",
            "needs_research",
            "needs_material",
            "needs_user",
            "unknown",
        }:
            row["status"] = "unknown"
        row["finding_ids"] = []
        row["citations"] = []
        if row.get("status") == linked_status or modern and valid:
            if not valid:
                row["status"] = "unknown"
            else:
                row["finding_ids"] = list(dict.fromkeys(ids))
                for fid in ids:
                    for c in by_id[fid]["citations"]:
                        if c not in row["citations"]:
                            row["citations"].append(deepcopy(c))
        row["next_action"] = need_next_action(row.get("status"))
        result["task_need_coverage"].append(row)
    return result


def _split_selections(cards, sources):
    """Partition valid selections, never fill omitted prose or glue distant claims."""
    counts = Counter(
        row.get("id")
        for row in cards
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    )
    reserved, result, links = set(counts), [], {}
    for row in cards:
        if not isinstance(row, dict):
            result.append(row)
            continue
        identity, sid, ids = row.get("id"), row.get("source_id"), row.get("block_ids")
        source = sources.get(sid) if isinstance(sid, str) else None
        blocks = evidence_blocks(source["text"]) if source else []
        known = {b["block_id"]: b for b in blocks}
        if (
            not isinstance(identity, str)
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", identity)
            or counts[identity] != 1
            or not isinstance(ids, list)
            or not ids
            or not all(isinstance(i, str) and i in known for i in ids)
        ):
            result.append(row)  # Invalid IDs remain invalid; no fabricated recovery.
            continue
        groups = [
            g
            for g in reading_units(source["text"], atomic_only=True)
            if set(ids).intersection(g["block_ids"])
        ]
        chunks, end, size = [], None, 0
        for group in groups:
            length = group["end"] - group["start"]
            if end != group["start"] or size + length > MAX_UNIT_CHARS:
                chunks.append([])
                size = 0
            chunks[-1].extend(group["block_ids"])
            end, size = group["end"], size + length
        if len(chunks) <= 1:
            result.append(row)
            continue
        links[identity] = []
        for index, selection in enumerate(chunks):
            child = f"{identity[:55]}__part_{index + 1}"
            suffix = 0
            while child in reserved:
                suffix += 1
                child = f"{identity[:48]}__part_{index + 1}_{suffix}"
            reserved.add(child)
            links[identity].append(child)
            result.append(
                {
                    **deepcopy(row),
                    "id": child,
                    "block_ids": selection,
                    "split_from": {
                        "id": identity,
                        "requested_block_ids": deepcopy(ids),
                    },
                }
            )
    return result, links


def _deduplicate_cards(cards):
    """Merge same-direction retained duplicates/subsets; never infer semantic equivalence."""
    aliases = {}
    ordered = sorted(
        (r for r in cards if r["disposition"] == "retained"),
        key=lambda r: -len(r["source_statement"]),
    )
    kept = []
    for row in ordered:
        parent = next(
            (
                other
                for other in kept
                if row["source_id"] == other["source_id"]
                and row["direction_id"] == other["direction_id"]
                and set(row["block_ids"]) <= set(other["block_ids"])
            ),
            None,
        )
        if parent is None:
            kept.append(row)
        else:
            aliases[row["id"]] = parent["id"]
            parent["need_ids"] = list(
                dict.fromkeys(parent["need_ids"] + row["need_ids"])
            )
            parent.setdefault("audit", {}).setdefault("merged_cards", []).append(
                deepcopy(row)
            )
    cards[:] = [row for row in cards if row["id"] not in aliases]
    return aliases


def extractive_card(row):
    """Allowlist the downstream card: no audit/model explanation can hitchhike."""
    fields = (
        "id",
        "source_id",
        "block_ids",
        "source_statement",
        "blocks",
        "source_offset",
        "reading_unit",
        "need_ids",
        "disposition",
        "direction_id",
        "citations",
        "accounting_status",
        "grounding",
        "context_only",
    )
    return {k: deepcopy(row[k]) for k in fields if k in row}


def scoped_uncertainties(findings, coverage, *, extractive=False):
    """Local receipts cannot revoke another source's retained statement."""
    result = []
    for row in findings:
        if row.get("disposition") != "deferred":
            continue
        result.append(
            {
                "scope": {
                    "kind": "source_selection"
                    if extractive
                    else "source_interpretation",
                    "source_id": row.get("source_id"),
                    "block_ids": deepcopy(row.get("block_ids", [])),
                    "finding_id": row.get("id"),
                },
                "reason": (
                    "Source selection remains unresolved; this does not establish a defect in the source."
                    if extractive
                    else row.get("projection_issue")
                    or row.get("reason")
                    or "Unresolved source interpretation."
                ),
                "boundary": "Local to this source statement; does not revoke other retained findings.",
            }
        )
    for row in coverage:
        if (
            row["status"]
            not in {"supported", "related_knowledge", "no_research_needed"}
            or row.get("open_questions")
            or row.get("materials_needed")
        ):
            result.append(
                {
                    "scope": {"kind": "task_need", "need_id": row["need_id"]},
                    "status": row["status"],
                    "reason": row["next_action"],
                    "boundary": "Missing knowledge/material/user input is not a task-specific conclusion.",
                }
            )
    return result


def body_sources(sources):
    """Legacy excerpts are bodies; explicitly marked navigation is never evidence."""
    return [s for s in sources if s.get("read_kind", "body") == "body"]


def spans(text, width=400):
    """Bounded, lossless slices of the supplied excerpt, with stable local offsets."""
    rows, start = [], 0
    while start < len(text):
        end = min(start + width, len(text))
        if end < len(text):
            boundary = text.rfind("\n", start + width // 4, end)
            if boundary < 0:
                boundary = text.rfind(" ", start + width // 4, end)
            if boundary >= 0:
                end = boundary + 1
        if text[start:end].strip():
            rows.append(
                {
                    "span_id": f"span_{start}_{end}",
                    "start": start,
                    "end": end,
                    "text": text[start:end],
                }
            )
        start = end
    return rows


def summary_sources(sources):
    """Expose each body once, with complete structural blocks instead of fine slices."""
    fields = (
        "source_id",
        "title",
        "url",
        "offset",
        "total_chars",
        "excerpt_only",
        "source_kind",
        "source_class",
        "read_kind",
    )
    return [
        {
            **{k: deepcopy(s[k]) for k in fields if k in s},
            "blocks": evidence_blocks(s["text"]),
            "structural_groups": [
                {
                    k: deepcopy(u[k])
                    for k in ("block_ids", "continuation", "incomplete_excerpt")
                }
                for u in reading_units(s["text"], atomic_only=True)
                if len(u["block_ids"]) > 1
                or u["continuation"]
                or u["incomplete_excerpt"]
            ],
        }
        for s in body_sources(sources)
    ]


def citation(c, sources, *, span_only=False, block_only=False):
    """Resolve a span by code, or strictly locate an unchanged legacy quote."""
    if not isinstance(c, dict) or not isinstance(c.get("source_id"), str):
        return None
    source = sources.get(c["source_id"])
    if source is None or source.get("read_kind", "body") != "body":
        return None
    if "block_id" in c:
        block = next(
            (
                b
                for b in evidence_blocks(source["text"])
                if b["block_id"] == c["block_id"]
            ),
            None,
        )
        if block is None or not block["text"].strip():
            return None  # No invented range, alternate span or model-quote fallback.
        return {
            "source_id": c["source_id"],
            "block_id": block["block_id"],
            "quote": block["text"],
            "span_start": block["start"],
            "span_end": block["end"],
        }
    if block_only:
        return None
    if "span_id" in c:
        offered = spans(source["text"])
        span = next((s for s in offered if s["span_id"] == c["span_id"]), None)
        resolution = {}
        if span is None:
            # Recover a model's bounded union of supplied neighbouring handles,
            # not an invented offset or an unseen part of the fetched document.
            for i in range(len(offered)):
                for count in (2, 3):
                    parts = offered[i : i + count]
                    if (
                        len(parts) != count
                        or c["span_id"]
                        != f"span_{parts[0]['start']}_{parts[-1]['end']}"
                        or any(
                            a["end"] != b["start"]
                            for a, b in zip(parts[:-1], parts[1:], strict=True)
                        )
                    ):
                        continue
                    text = "".join(p["text"] for p in parts)
                    if (
                        len(text) > 1200
                        or text != source["text"][parts[0]["start"] : parts[-1]["end"]]
                    ):
                        continue
                    span = {
                        "text": text,
                        "start": parts[0]["start"],
                        "end": parts[-1]["end"],
                    }
                    resolution = {
                        "resolution": "adjacent_spans",
                        "component_span_ids": [p["span_id"] for p in parts],
                    }
        if span is None:
            return None  # An invalid span cannot fall back to a model-supplied quote.
        return {
            "source_id": c["source_id"],
            "span_id": c["span_id"],
            **resolution,
            "quote": span["text"],
            "span_start": span["start"],
            "span_end": span["end"],
        }
    if span_only:
        return None
    quote = c.get("quote")
    if (
        isinstance(quote, str)
        and quote.strip()
        and " ".join(quote.split()) in " ".join(source["text"].split())
    ):
        return deepcopy(c)
    return None
