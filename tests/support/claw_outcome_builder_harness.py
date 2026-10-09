"""Shared fixtures and reusable scenarios extracted without changing assertions."""

from reward_engine.core.outcome.context import build_context


def setup():
    context, evidence = build_context(
        {"task_id": "x", "prompt": "Deliver JSON"},
        [{"artifact": {"final_response": "{}"}}],
    )
    jury = {
        "criteria": [
            {"id": "syntax", "standard": "Valid JSON", "kind": "hard", "origin": "task"}
        ]
    }
    reply = {
        "rubrics": [
            {
                "criterion_id": "syntax",
                "weight": 1,
                "anchors": {
                    "fail": "invalid",
                    "partial": "invalid fragment",
                    "full": "valid",
                },
            }
        ],
        "tests": [
            {
                "test_id": "custom-id",
                "criterion_id": "syntax",
                "checker": "json.parse",
                "question": "Does JSON parse?",
                "arguments": {"ref": "product_1:artifact:final_response"},
            }
        ],
    }
    return context, evidence, jury, reply
