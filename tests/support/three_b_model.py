"""Deterministic local role responses for a synthetic JSON task."""

from copy import deepcopy

PROMPT = "Return valid JSON"


def criterion():
    return {
        "id": "syntax",
        "standard": "Valid JSON",
        "kind": "hard",
        "priority": "important",
        "priority_reason": "The task explicitly requests JSON",
        "origin": "task",
        "evidence_refs": ["task"],
        "measurement_question": "Does the final response parse as JSON?",
        "clauses": [
            {
                "id": "syntax",
                "text": PROMPT,
                "use": "task_constraint",
                "evidence_refs": ["task"],
                "source_quotes": [{"ref": "task", "quote": PROMPT}],
                "conditions": [],
                "alternative_group": "",
            }
        ],
    }


def response(request):
    assert "PRIVATE_SCORE" not in str(request)
    mode = request["mode"]
    if "hypothesis" in mode:
        raise AssertionError("Unexpected hypothesis request")
    if mode == "outcome_expectations":
        return {"criteria": [criterion()], "gaps": [], "uncertainties": []}
    if mode == "outcome_jury_map":
        return {
            "criteria": [
                {"inherit_from": e["expectation_id"]}
                for e in request["independent_expectations"]
            ],
            "gaps": [],
            "uncertainties": [],
            "revision_decisions": [],
        }
    if mode == "outcome_jury_reduce":
        decisions = [
            {
                "parts": [part["selection_id"]],
                "action": "retain",
                "criterion_id": "syntax",
                "clause_id": "syntax",
            }
            for e in request["independent_expectations"]
            + request["proposal_expectations"]
            for part in e["subrequirements"]
        ]
        return {
            "criteria": [criterion()],
            "gaps": [],
            "selection_protocol": "exact-alias-v1",
            "part_decisions": decisions,
        }
    if mode == "outcome_builder":
        return {
            "rubrics": [
                {
                    "criterion_id": c["id"],
                    "anchors": {
                        "fail": "Invalid JSON",
                        "partial": "JSON fragment",
                        "full": "Parses as JSON",
                    },
                }
                for c in request["jury"]["criteria"]
            ],
            "tests": [
                {
                    "test_id": "parse",
                    "criterion_id": request["jury"]["criteria"][0]["id"],
                    "checker": "json.parse",
                    "question": "Parse the final response",
                    "arguments": {"ref": "product_1:artifact:final_response"},
                }
            ],
        }
    if mode == "outcome_attack_map":
        return {
            "issues": [],
            "reason": "Synthetic syntax task has no proposed counterexample; not a proof of completeness",
        }
    if mode in {"outcome_ranking_map", "outcome_ranking_reduce"}:
        ids = [p["product_id"] for p in request["context"]["products"]]
        return {
            "groups": [ids],
            "unknown": [],
            "reason": "All delivered JSON answers are equivalent for this synthetic task",
            "observations": [
                {
                    "product_id": p,
                    "reason": "Final answer is valid JSON",
                    "evidence_refs": [p + ":artifact:final_response"],
                }
                for p in ids
            ],
        }
    if mode == "outcome_delivery_review":
        return {
            "outcome_goal": PROMPT,
            "full_score_gaps": [],
            "improvements": [],
            "status": "target_satisfied",
            "reason": "Actual parser checked the delivered JSON",
            "claim_checks": [],
            "gaps": [],
            "verify": [],
        }
    if mode.startswith("outcome_measure") or mode in {
        "outcome_check",
        "outcome_inquiry",
    }:
        ref = "product_1:artifact:final_response"
        return {
            "status": "pass",
            "reason": "Delivered text parses as JSON",
            "gap": "",
            "evidence_refs": [ref],
            "claims": [
                {
                    "claim": "The answer is valid JSON",
                    "claim_refs": [ref],
                    "evidence_refs": [ref],
                    "observation": "The synthetic answer is an empty JSON object",
                    "judgment": "supported",
                    "verdict_relevance": "material",
                    "relevance_reason": "Explicit task requirement",
                }
            ],
        }
    raise AssertionError("Unsupported synthetic request: " + mode)


async def role(request):
    return deepcopy(response(request))
