"""Shared fixtures and reusable scenarios extracted without changing assertions."""

NEED = {
    "need_id": "need_0",
    "requirement": "Verify the supplied format",
    "task_basis": "Check the actual byte range before changing the artifact.",
    "status": "needs_research",
    "next_action": "GUESSED_OFFSET_8_11: change those bytes and always recompute a digest",
    "external_knowledge_needed": "UNVERIFIED_REPAIR_ASSUMPTION",
    "already_given_or_directly_checkable": "UNVERIFIED_DIAGNOSIS",
    "technical_parameters": {"offset": 9999},
    "new_requirement": "INVENTED_USER_REQUEST",
    "query_indices": [0],
    "citations": [{"quote": "INVENTED_PLAN_FACT"}],
}
