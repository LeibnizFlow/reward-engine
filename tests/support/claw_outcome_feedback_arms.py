"""Shared fixtures and reusable scenarios extracted without changing assertions."""

TASK = {"task_id": "json-task", "prompt": "Return valid JSON"}


PACK = {
    "task_id": TASK["task_id"],
    "task_prompt": TASK["prompt"],
    "pack_id": "v1",
    "criteria": [
        {
            "id": "syntax",
            "kind": "hard",
            "origin": "task",
            "standard": "Valid JSON",
            "measurement_question": "Does the delivered output parse as JSON?",
            "evidence_refs": ["task"],
            "anchors": {"fail": "Invalid", "partial": "Fragment", "full": "Parseable"},
        }
    ],
    "tests": [
        {
            "test_id": "parse",
            "criterion_id": "syntax",
            "checker": "json.parse",
            "question": "Parse actual output",
            "arguments": {"ref": "product_1:artifact:final_response"},
        }
    ],
}


METHOD = {
    "status": "propose",
    "method": "Check serialization before delivering",
    "target": "Syntax",
}


def candidate(text="{}"):
    return {"artifact": {"final_response": text}}


async def model(request):
    assert "PRIVATE_GRADE" not in str(request)
    if request["mode"].startswith("outcome_policy_"):
        return {
            "outcome_goal": "Fully satisfy the declared task with a correct, complete result.",
            "full_score_gaps": [],
            "improvements": [],
            "status": "suggest",
            "global_assessment": "Check syntax",
            "local_guidance": "Repair JSON",
            "gaps": [],
            "verify": [{"test_id": "parse", "reason": "Current draft is uncertain"}]
            if request["mode"] == "outcome_policy_critic"
            else [],
            "memory_ids": [],
        }
    return {
        "outcome_goal": "Fully satisfy the declared task with a correct, complete result.",
        "full_score_gaps": [],
        "improvements": [],
        "status": "target_satisfied",
        "reason": "Use parser for syntax",
        "claim_checks": [],
        "gaps": [],
        "verify": [{"test_id": "parse", "reason": "Inspect actual syntax"}],
    }
