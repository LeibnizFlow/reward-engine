"""Shared fixtures and reusable scenarios extracted without changing assertions."""


def candidate(value="{}", *, pending=False):
    return {
        "artifact": {
            "final_response": value,
            "public_consultation": {
                "latest_reply_pending": pending,
                "earlier_exchanges": [],
            },
        }
    }


def pack():
    return {
        "task_id": "x",
        "task_prompt": "Deliver JSON",
        "pack_id": "v1",
        "criteria": [{"id": "syntax", "kind": "hard", "standard": "Valid JSON"}],
        "tests": [
            {
                "test_id": key,
                "criterion_id": "syntax",
                "checker": "json.parse",
                "arguments": {"ref": "product_1:artifact:final_response"},
            }
            for key in ("parse", "other", "third", "fourth")
        ],
    }


TASK = {"task_id": "x", "prompt": "Deliver JSON"}
