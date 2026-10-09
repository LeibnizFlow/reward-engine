"""Shared fixtures and reusable scenarios extracted without changing assertions."""

from reward_engine.core.outcome.context import SHARED as SHARED
from reward_engine.core.outcome.context import ask as ask
from reward_engine.core.outcome.context import build_context


def sample():
    return build_context(
        {"task_id": "x", "prompt": "Prepare a draft"},
        [
            {
                "artifact": {
                    "final_response": "draft",
                    "public_consultation": {
                        "earlier_exchanges": [
                            {"speaker": "user", "text": "Do not send"},
                            {"speaker": "assistant", "text": "Could we send?"},
                            {"speaker": "user", "text": "No. Save only."},
                        ]
                    },
                },
                "environment": {"draft": None},
                "source_documents": [{"text": "source"}],
                "raw_trace": "SECRET",
                "grader": "SECRET",
                "persona": "SECRET",
            }
        ],
    )


def definition():
    return {
        "criteria": [
            {
                "id": "saved",
                "standard": "Requested draft saved",
                "kind": "hard",
                "origin": "task",
                "evidence_refs": ["task"],
                "measurement_question": "Draft exists?",
            }
        ],
        "gaps": [
            {
                "gap_id": "missing",
                "product_ids": ["product_1"],
                "expected": "Saved draft",
                "expected_basis": "task",
                "expected_refs": ["task"],
                "observed": "No draft in environment",
                "gap": "No saved draft",
                "verification": "Read actual saved drafts; reject suspicion if present",
                "status": "needs_check",
                "evidence_refs": ["product_1:environment"],
            }
        ],
    }
