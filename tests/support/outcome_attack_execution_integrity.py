"""Shared fixtures and reusable scenarios extracted without changing assertions."""

import json


def issue(scope="claim"):
    return {
        "target_id": "syntax",
        "proposer_seat": "1",
        "priority": "important",
        "problem": "Could reject a valid value",
        "suggested_change": "Check actual syntax",
        "evidence_refs": ["task"],
        "cases": [
            {
                "id": kind,
                "kind": kind,
                "artifact": text,
                "question": "Valid JSON?",
                "scope": scope,
                "expected": status,
                "why": "Parse the supplied text",
            }
            for kind, text, status in [
                ("correct_control", "{}", "pass"),
                ("error_variant", "{", "fail"),
                ("boundary", "null", "pass"),
            ]
        ],
    }


async def parser_reference(request):
    rows = []
    for case in request["cases"]:
        try:
            json.loads(case["artifact"])
            status = "pass"
        except ValueError:
            status = "fail"
        rows.append(
            {
                "id": case["id"],
                "status": status,
                "reason": "Executed parser",
                "evidence_refs": ["task"],
            }
        )
    return {"cases": rows}
