"""Shared fixtures and reusable scenarios extracted without changing assertions."""

from reward_engine.core.outcome.knowledge_evidence import (
    evidence_blocks,
    summary_sources,
)


from reward_engine.core.outcome.knowledge_summary import (
    group_view,
    reduce_groups,
    source_groups,
)


def fixture(n=1):
    return {"input": {"task_prompt": "Inspect the supplied shipment and report safe handling.",
                      "user_updates": [{"text": "Keep a record of temperature excursions."}]},
        "sources": [{"source_id": f"doc_{i}", "title": f"Manual {i}", "url": f"https://example.test/{i}",
            "text": "Use cold transport.\n\nUnrelated history.\n\nExcept for containers labelled room-temperature.\n",
            "read_kind": "body"} for i in range(n)],
        "task_needs": [{"need_id": "n0", "requirement": "Check safe handling."}],
        "search_batch": {"complete": True}, "read_batch": {"complete": True}}


def draft(source):
    blocks = evidence_blocks(source["text"])
    return {"source_findings": [{"id": "local_0", "source_id": source["source_id"],
        "block_ids": [blocks[0]["block_id"], blocks[2]["block_id"]], "need_ids": ["n0"],
        "direction_id": "method", "disposition": "retained"}],
        "task_need_coverage": [{"need_id": "n0", "status": "related_knowledge", "finding_ids": ["local_0"]}]}


def groups(n=1):
    base = fixture(n)
    return base, [g for i, source in enumerate(base["sources"])
                  for g in source_groups({**base, "sources": [source]}, draft(source), i)["groups"]]


def decision(gid, disposition="retained", **kw):
    return {"group_id": gid, "disposition": disposition, "reason": "Example reason.", **kw}
