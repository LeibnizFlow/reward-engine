"""Shared fixtures and reusable scenarios extracted without changing assertions."""

from reward_engine.core.outcome.context import build_context


def fixture():
    task = {"task_id": "generic", "prompt": "Produce the requested artifact without publishing it."}
    candidates = [
        {"artifact": {"final_response": "I finished"}, "environment": {"artifact": None}} for _ in range(3)
    ]
    context, evidence = build_context(task, candidates)
    gap = {
        "gap_id": "missing_product",
        "product_ids": ["product_1", "product_2", "product_3"],
        "expected": "Requested artifact exists",
        "expected_basis": "task",
        "expected_refs": ["task"],
        "observed": "Each environment contains no artifact",
        "gap": "Claimed delivery has no observed artifact",
        "evidence_refs": [f"product_{i}:environment" for i in range(1, 4)],
        "verification": "Read the actual artifact storage; reject suspicion if artifact is found",
        "status": "needs_check",
    }
    result = {
        "criteria": [
            {
                "id": "actual_delivery",
                "standard": "Requested artifact is actually delivered",
                "kind": "hard",
                "origin": "task",
                "evidence_refs": ["task"],
                "measurement_question": "Read artifact storage",
            }
        ],
        "gaps": [gap],
        "uncertainties": [],
    }
    return context, evidence, result
