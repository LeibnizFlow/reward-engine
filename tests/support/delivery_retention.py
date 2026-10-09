"""Synthetic actors/roles/publication support; real review and parser execution."""

from copy import deepcopy

from reward_engine.core.outcome.construction import fingerprint
from reward_engine.core.outcome.harness import defaults
from reward_engine.core.outcome.runner import DeliveryReview, run_delivery
from reward_engine.core.outcome.validated_construction import publish_validation
from tests.support.claw_outcome_feedback_arms import PACK, TASK

REF = "product_1:artifact:final_response"


def published_pack():
    raw = deepcopy(PACK)
    raw["criteria"][0]["clauses"] = [
        {
            "id": "syntax",
            "text": TASK["prompt"],
            "use": "task_constraint",
            "evidence_refs": ["task"],
            "source_quotes": [{"ref": "task", "quote": TASK["prompt"]}],
            "conditions": [],
            "alternative_group": "",
        }
    ]
    return publish_validation(
        raw,
        {
            "status": "completed",
            "artifact_sha256": fingerprint(raw),
            "records": [
                {
                    "target_id": "syntax",
                    "validation_id": "synthetic-fixture-only",
                    "status": "supported_on_cases",
                    "cases": [],
                }
            ],
        },
    )


async def scripted_review(request):
    assert request["mode"] == "outcome_delivery_review"
    return {
        "outcome_goal": TASK["prompt"],
        "status": "needs_improvement",
        "reason": "Scripted optional gain to exercise a real regression check",
        "gaps": [],
        "verify": [],
        "claim_checks": [],
        "full_score_gaps": [
            {"id": "gain", "gap": "Consider a useful field", "evidence_refs": [REF]}
        ],
        "improvements": [
            {
                "id": "trial",
                "gap_ids": ["gain"],
                "kind": "outcome_gain",
                "task_quote": TASK["prompt"],
                "current_evidence_refs": [REF],
                "verification_test_ids": ["parse"],
                "factual_premises": [],
                "action": "Consider adding a useful field, preserving JSON syntax",
                "expected_gain": "Try a more informative response",
                "verification": "Reparse the actual revised answer",
                "evidence_refs": [REF],
            }
        ],
    }


class ScriptedActor:
    def __init__(self, texts=("{}", "{")):
        self.texts = list(texts)
        self.calls = []
        self.current = {"artifact": {}}

    def deliver(self):
        # Deliberately mutate the same actor-owned dictionary across turns.
        self.current["artifact"]["final_response"] = self.texts[len(self.calls) - 1]
        return self.current

    async def start(self, prompt):
        self.calls.append("start")
        return self.deliver()

    async def resume(self, feedback):
        self.calls.append("resume")
        return self.deliver()


async def reviewed_row(texts=("{}", "{"), **options):
    actor = ScriptedActor(texts)
    review = DeliveryReview(
        TASK,
        published_pack(),
        defaults(scripted_review),
        max_revisions=len(texts) - 1,
        **options,
    )
    native = await run_delivery(actor, review)
    row = {
        "candidate": deepcopy(native),
        "final_measurement": deepcopy(review.history[-1]),
        "review_status": review.status,
        **review.result_fields(native),
    }
    return actor, review, row


async def prefer_regressed(request):
    assert request["mode"] in {"outcome_ranking_map", "outcome_ranking_reduce"}
    products = request["context"]["products"]
    preferred = [
        p["product_id"] for p in products if p["artifact"]["final_response"] == "{"
    ]
    others = [p["product_id"] for p in products if p["product_id"] not in preferred]
    return {
        "groups": [group for group in [preferred, others] if group],
        "unknown": [],
        "observations": [
            {
                "product_id": p["product_id"],
                "reason": "Deliberately misleading scripted preference",
                "evidence_refs": [p["product_id"] + ":artifact:final_response"],
            }
            for p in products
        ],
        "reason": "Synthetic preference challenges actual critical-check preservation",
    }
