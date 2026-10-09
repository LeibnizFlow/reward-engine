"""Original public Task attachments, distinct from both products and knowledge priors."""

import json
from copy import deepcopy
from urllib.parse import quote


def attach(context, evidence, documents):
    seen = set()
    for document in documents:
        path = document.get("path")
        if not isinstance(path, str) or not path or path in seen:
            raise ValueError("Original Task input needs a distinct public path")
        seen.add(path)
        ref = "task_input:" + quote(path, safe="")
        if ref in evidence and evidence[ref] != document:
            raise ValueError("Original Task input changed within a frozen context")
        evidence[ref] = deepcopy(document)
        docs = context.setdefault("reference_documents", [])
        if not any(row["ref"] == ref for row in docs):
            text = json.dumps(document, ensure_ascii=False)
            docs.append(
                {
                    "ref": ref,
                    "title": path,
                    "source_kind": "original_task_input",
                    "preview": text[:600],
                    "preview_only": len(text) > 600,
                    "boundary": "Original Task-provided input; not an Actor-modified file or optional knowledge",
                }
            )
    context["available_evidence_refs"] = list(evidence)
    return context, evidence


def refs(context):
    return {
        d["ref"]
        for d in context.get("reference_documents", [])
        if d.get("source_kind") == "original_task_input"
        and d["ref"].startswith("task_input:")
    }


def instruction(context):
    return (
        " Original task_input: documents are part of the public Task, not optional knowledge. "
        "Read their contents selectively; do not invent a schema/codebook or treat an Actor-modified "
        "copy as the original input. A task_constraint may cite an exact original Task-input quote. "
        "Partial extraction or uninspected binary content remains unknown."
        if refs(context)
        else ""
    )
