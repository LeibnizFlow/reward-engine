"""Shared fixtures and reusable scenarios extracted without changing assertions."""

TASK = {"task_id": "x", "prompt": "Return valid JSON"}


PACK = {
    "pack_id": "v1",
    "task_id": "x",
    "task_prompt": TASK["prompt"],
    "criteria": [{"id": "json", "kind": "hard", "weight": 1}],
    "tests": [
        {
            "test_id": "parse",
            "criterion_id": "json",
            "checker": "json.parse",
            "arguments": {"ref": "product_1:artifact:final_response"},
        }
    ],
}
