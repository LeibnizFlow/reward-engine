"""Shared fixtures and reusable scenarios extracted without changing assertions."""

TASK = {"task_id": "x", "prompt": "Deliver valid JSON"}


PACK = {"task_id": "x", "task_prompt": TASK['prompt'], "pack_id": "v1",
        "criteria": [{"id": "syntax", "kind": "hard", "standard": "Valid JSON"}],
        "tests": [{"test_id": "parse", "criterion_id": "syntax", "checker": "json.parse",
                   "arguments": {"ref": "product_1:artifact:current_draft"}}]}


def answer(**kw):
    return {'outcome_goal': 'Fully satisfy the declared task with a correct, complete result.', 'full_score_gaps': [], 'improvements': [], "status": "suggest", "global_assessment": "Verify the current draft's syntax",
            "local_guidance": "Fix the JSON before delivery", "gaps": [], "verify": [], "memory_ids": [], **kw}
