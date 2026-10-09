"""Compress only optional navigation text; Task, clauses, cases and receipts stay exact."""

import json
from copy import deepcopy

from reward_engine.core.outcome.context import RequestBudgetExceeded


def knowledge_navigation(context, evidence):
    """Index optional knowledge; retain complete originals at their existing evidence refs."""
    result = deepcopy(context)
    for skill in result.get("skills", []):
        ref, content = skill.get("ref"), skill.get("content")
        if (
            not isinstance(content, dict)
            or not isinstance(ref, str)
            or not ref.startswith("skill:task_review:")
            or evidence.get(ref) != content
        ):
            continue
        navigation = {
            k: deepcopy(content[k])
            for k in (
                "kind",
                "authority",
                "direction",
                "relevance",
                "applicability",
                "verification",
                "grounding",
                "use",
                "conditions_unstated",
                "source_class",
            )
            if k in content
        }
        mechanisms = content.get("mechanisms", [])
        if mechanisms:
            navigation["passage_previews"] = [
                {
                    "id": m.get("id"),
                    "source_id": m.get("source_id"),
                    "preview": str(m.get("source_statement", ""))[:300],
                    "preview_only": True,
                }
                for m in mechanisms
            ]
        navigation["available_sections"] = [
            key for key in content if key not in navigation
        ]
        navigation["details_ref"] = ref
        navigation["boundary"] = (
            "Optional navigation only; read the complete evidence ref before relying on its claims."
        )
        skill["content"], skill["preview_only"] = navigation, True
    return result


async def compress(request, call, cap, record):
    result = deepcopy(request)
    fields = []

    # A field name alone cannot distinguish navigation from authored material.
    # Require explicit preview metadata and never descend into canonical bodies,
    # even when they contain their own preview_only/summary/preview/rationale keys.
    def visit(value, path=()):
        if isinstance(value, dict):
            for key, item in value.items():
                if (
                    value.get("preview_only") is True
                    and key in {"preview", "summary", "rationale"}
                    and isinstance(item, str)
                    and len(item.encode()) > 512
                ):
                    fields.append({"path": list(path + (key,)), "text": item})
                    value[key] = (
                        "[navigation summary pending; canonical source remains available]"
                    )
                elif (
                    key == "content"
                    and value.get("preview_only") is True
                    and isinstance(value.get("ref"), str)
                    and value["ref"].startswith("skill:task_review:")
                    and isinstance(item, dict)
                    and item.get("details_ref") == value["ref"]
                ):
                    # knowledge_navigation's generated index is optional; only
                    # its passage previews may be shortened, not its conditions.
                    visit(
                        item.get("passage_previews", []),
                        path + (key, "passage_previews"),
                    )
                elif key not in {
                    "task",
                    "artifact",
                    "environment",
                    "source_documents",
                    "evidence",
                    "requested_evidence",
                    "content",
                    "text",
                    "result",
                    "results",
                    "checks",
                    "user_updates",
                    "earlier_public_deliveries",
                    "material",
                    "cases",
                    "criteria",
                    "clauses",
                    "verified_findings",
                    "disputed_findings",
                    "response_recovery",
                }:
                    visit(item, path + (key,))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, path + (index,))

    visit(result.get("context", {}), ("context",))

    def size(value):
        return len(json.dumps(value, ensure_ascii=False).encode())

    record(
        {
            "before_bytes": size(request),
            "limit_bytes": cap,
            "status": "preparing",
            "original": request,
        }
    )
    if not fields or size(result) >= cap:
        raise RequestBudgetExceeded(
            result, cap, stage="immutable_context_exceeds_capacity"
        )
    available = cap - size(result)
    # One operation, bounded by the compressor's own declared request capacity.
    operation = {
        "mode": "outcome_navigation_compress",
        "task": request["context"]["task"],
        "instruction": "Summarize optional navigation fields only. Preserve conditions, exceptions "
        "and uncertainty; never decide criteria or invent missing formulas. Return one text per "
        "supplied field in order, within the total encoded byte budget.",
        "fields": fields,
        "max_output_bytes": available,
        "response": {"summaries": ["one nonempty summary per field"]},
    }
    if size(operation) > getattr(call, "max_request_bytes", cap):
        raise RequestBudgetExceeded(
            operation,
            getattr(call, "max_request_bytes", cap),
            stage="compression_input",
        )
    row = await call(operation)
    summaries = row.get("summaries")
    if (
        not isinstance(summaries, list)
        or len(summaries) != len(fields)
        or any(not isinstance(s, str) or not s.strip() for s in summaries)
    ):
        raise ValueError("Compression must preserve every navigation field")
    for field, summary in zip(fields, summaries, strict=True):
        parent = result
        for key in field["path"][:-1]:
            parent = parent[key]
        parent[field["path"][-1]] = summary
    if size(result) > cap:
        raise RequestBudgetExceeded(result, cap, stage="compression_output")
    record(
        {
            "before_bytes": size(request),
            "after_bytes": size(result),
            "limit_bytes": cap,
            "status": "completed",
            "projection": result,
            "canonical_evidence_unchanged": True,
        }
    )
    return result
