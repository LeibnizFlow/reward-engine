"""Scripted teaching roles, never an LLM or native-benchmark validation.

The controllers, compilation, registered checks, review and memory execute real
core code. These small authored role responses deliberately expose one repair.
"""

import json

TASK = {
    "task_id": "minimal-total",
    "prompt": 'Add the public quantities 1, 2 and 3. Return ONLY a JSON object with exactly one key "total" holding their integer sum; no prose or Markdown.',
    "quantities": [1, 2, 3],
}
REF = "product_1:artifact:final_response"
STANDARD = (
    "Return only a JSON object with exactly the integer total of the public quantities."
)


def inspect_text(text):
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        return False
    return (
        isinstance(value, dict)
        and set(value) == {"total"}
        and type(value["total"]) is int
        and value["total"] == sum(TASK["quantities"])
    )


async def total_check(test, context, evidence):
    """Real executable check against this example's public task, not a grader."""
    ref = test.get("arguments", {}).get("ref", REF)
    if any(context["task"].get(k) != TASK[k] for k in ("task_id", "prompt")):
        raise ValueError("example.total only implements the supplied minimal task")
    passed = inspect_text(evidence[ref])
    return {
        "status": "pass" if passed else "fail",
        "reason": "Parsed the delivered text and checked its exact keys, integer type and sum against public quantities.",
        "gap": ""
        if passed
        else "Return JSON with exactly total=6, computed from 1+2+3.",
        "evidence_refs": [ref, "task"],
    }


class Offline:
    mode = "offline_deterministic"

    async def role(self, request):
        mode = request["mode"]
        if "ranking" in mode:
            products = request["context"]["products"]
            good, bad = [], []
            observations = []
            for p in products:
                valid = inspect_text(p["artifact"]["final_response"])
                (good if valid else bad).append(p["product_id"])
                observations.append(
                    {
                        "product_id": p["product_id"],
                        "reason": "Scripted comparison of the actual parsed total: "
                        + ("correct" if valid else "incorrect"),
                        "evidence_refs": [p["product_id"] + ":artifact:final_response"],
                    }
                )
            return {
                "groups": [g for g in [good, bad] if g],
                "unknown": [],
                "observations": observations,
                "reason": "Exact public-task total, not native score.",
            }
        if mode == "outcome_expectations" or mode.startswith("outcome_jury"):
            expected = request.get("independent_expectations", []) + request.get(
                "proposal_expectations", []
            )
            answer = {
                "criteria": [
                    {
                        "id": "total",
                        "standard": STANDARD,
                        "kind": "hard",
                        "origin": "task",
                        "measurement_question": "Parse and add public quantities",
                        "evidence_refs": ["task"],
                    }
                ],
                "gaps": [],
                "coverage": [
                    {
                        "expectation_id": e["expectation_id"],
                        "criterion_ids": ["total"],
                        "status": "mapped",
                        "reason": "The exact task total is checked",
                        "subrequirements": [
                            {
                                "part_id": p["part_id"],
                                "criterion_ids": ["total"],
                                "status": "mapped",
                                "reason": "Task requirement retained",
                                "retained_text": STANDARD,
                            }
                            for p in e["subrequirements"]
                        ],
                    }
                    for e in expected
                ],
            }
            if "selection_aliases" in request:
                answer.pop("coverage")
                answer["selection_protocol"] = "exact-alias-v1"
                answer["part_decisions"] = [
                    {
                        "parts": list(request["selection_aliases"]),
                        "action": "retain",
                        "criterion_id": "total",
                    }
                ]
            return answer
        if "hypothesis" in mode:
            return {
                "status": "propose",
                "method": "Add 1+2+3, then parse the JSON before delivery.",
                "target": "Correct total in valid JSON",
                "expected_improvement": "Avoid arithmetic and formatting mistakes; not yet verified",
                "verification": "Run example.total on the actual delivery",
                "evidence_refs": ["task"],
                "target_gap_ids": [],
                "clarifications": [],
            }
        if mode == "outcome_builder":
            return {
                "rubrics": [
                    {
                        "criterion_id": "total",
                        "weight": 1,
                        "anchors": {
                            "fail": "Wrong total or invalid JSON",
                            "partial": "Unresolved total",
                            "full": STANDARD,
                        },
                    }
                ],
                "tests": [
                    {
                        "test_id": "total_check",
                        "criterion_id": "total",
                        "checker": "example.total",
                        "question": "Does the actual delivery meet the exact public total contract?",
                        "arguments": {"ref": REF},
                    }
                ],
            }
        if mode == "outcome_measure":
            # Full DeliveryReview asks the real semantic harness for completeness.
            # This offline role's verdict is explicitly scripted from parsed text.
            product = request["context"]["products"][0]
            ref = product["product_id"] + ":artifact:final_response"
            valid = inspect_text(product["artifact"]["final_response"])
            return {
                "status": "pass" if valid else "fail",
                "reason": "Scripted completeness check of the only task obligation",
                "gap": "" if valid else "Wrong total or invalid delivery",
                "evidence_refs": [ref, "task"],
                "claims": [
                    {
                        "claim": "The delivery is exact JSON with total=6",
                        "claim_refs": [ref],
                        "evidence_refs": [ref, "task"],
                        "observation": product["artifact"]["final_response"],
                        "judgment": "supported" if valid else "contradicted",
                        "verdict_relevance": "material",
                    }
                ],
            }
        raise ValueError("Offline example has no scripted role for " + mode)

    async def actor(self, messages, *, arm, revision):
        # Deliberately incorrect initial A/B demonstrates actual check feedback.
        # C applies the scripted hypothesis; B fixes its delivery on continuation.
        value = 6 if arm == "C" or revision else 5
        return {"final_response": json.dumps({"total": value})}


def candidate(identity, text):
    return {
        "candidate_id": identity,
        "artifact": {"final_response": text},
        "source_documents": [],
        "environment": {},
    }


def seeds():
    return [candidate(f"seed-{i}", json.dumps({"total": i + 2})) for i in range(1, 4)]
