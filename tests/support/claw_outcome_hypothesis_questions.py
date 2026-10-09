"""Shared fixtures and reusable scenarios extracted without changing assertions."""


def proposal():
    return {
        "status": "propose",
        "method": "Clarify the target audience before drafting.",
        "target": "Audience-dependent content",
        "expected_improvement": "More relevant explanation",
        "verification": "Compare the draft to the user's real answer",
        "target_gap_ids": [],
        "evidence_refs": ["task"],
        "clarifications": [
            {
                "question": "Who is the intended audience?",
                "why": "Controls terminology and examples",
                "when": "Audience still unspecified",
                "if_unanswered": "Draft a neutral outline and state this uncertainty",
                "evidence_refs": ["task"],
            }
        ],
    }
