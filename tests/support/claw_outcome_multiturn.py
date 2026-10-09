"""Shared fixtures and reusable scenarios extracted without changing assertions."""

from reward_engine.core.outcome.construction import fingerprint
from reward_engine.core.outcome.validated_construction import publish_validation

TASK = {"task_id": "dialogue", "prompt": "Prepare a report in USD."}


def pack():
    c = {
        "id": "currency",
        "standard": TASK["prompt"],
        "kind": "hard",
        "origin": "task",
        "evidence_refs": ["task"],
        "measurement_question": "Check currency",
        "clauses": [
            {
                "id": "unit",
                "text": TASK["prompt"],
                "use": "task_constraint",
                "evidence_refs": ["task"],
                "source_quotes": [{"ref": "task", "quote": TASK["prompt"]}],
                "conditions": [],
                "alternative_group": "",
            }
        ],
    }
    value = {
        "pack_id": "initial",
        "task_id": TASK["task_id"],
        "task_prompt": TASK["prompt"],
        "criteria": [c],
        "tests": [],
    }
    return publish_validation(
        value,
        {"status": "completed", "artifact_sha256": fingerprint(value), "records": []},
    )
