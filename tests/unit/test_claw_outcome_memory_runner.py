
from tests.support.claw_outcome_memory_runner import (
    PACK,
    TASK,
)
import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest

from reward_engine.core.outcome.context import build_context
from reward_engine.core.outcome.harness import defaults
from reward_engine.core.outcome.memory import Memory
from reward_engine.core.outcome.runner import DeliveryReview, run_delivery



def test_same_actor_revises_and_rechecks_not_fresh_resample():
    class Actor:
        starts = 0
        feedback = []

        async def start(self, prompt):
            self.starts += 1
            return {"artifact": {"final_response": "{bad"}}

        async def resume(self, feedback):
            self.feedback.append(feedback)
            return {"artifact": {"final_response": "{}"}}

    async def forbidden(request):
        raise AssertionError("No LLM for JSON")

    actor, memory = Actor(), Memory()
    review = DeliveryReview(TASK, PACK, defaults(forbidden), memory=memory, audit_mode="full")
    result = asyncio.run(run_delivery(actor, review))
    assert actor.starts == 1 and len(actor.feedback) == 1
    assert "Invalid JSON" in actor.feedback[0]
    assert result["artifact"]["final_response"] == "{}"
    assert review.status == "satisfied_under_estimator" and len(review.history) == 2
    assert memory.rows[0]["value"]["observed_changes"][0]["after"] == "pass"


def test_budget_and_missing_checker_do_not_pretend_success():
    async def forbidden(request):
        raise AssertionError()

    pack = {**PACK, "tests": [{**PACK["tests"][0], "checker": "missing"}]}
    review = DeliveryReview(TASK, pack, defaults(forbidden), max_revisions=1, audit_mode="full")
    candidate = {"artifact": {"final_response": "{}"}}
    assert "verify" in asyncio.run(review(candidate))
    assert asyncio.run(review(candidate)) is None
    assert review.status == "budget_exhausted_unresolved"
    assert review.history[-1]["coverage"] == 0


def test_retrieval_top_k_embedding_keys_only_and_task_scope():
    rows = [
        {"memory_id": "1", "task_id": "x", "key": "JSON parsing", "value": {"secret_value": "V_ONLY"}},
        {"memory_id": "2", "task_id": "other", "key": "JSON", "value": {}},
    ]
    bank = Memory(rows)
    context, _ = build_context(TASK)
    assert [r["memory_id"] for r in asyncio.run(bank.retrieve(context, top_k=5))] == ["1"]
    assert asyncio.run(bank.retrieve(context, top_k=0)) == []

    class Embed:
        calls = []

        async def embed(self, texts, **options):
            self.calls.append((texts, options))
            return SimpleNamespace(vectors=[[1.0, 0.0] for _ in texts])

    embedding = Embed()
    result = asyncio.run(bank.retrieve(context, embedding=embedding))
    assert result[0]["value"]["secret_value"] == "V_ONLY"
    assert "V_ONLY" not in str(embedding.calls)


@pytest.mark.parametrize("gap", [None, ""])
def test_nullable_success_gap_does_not_abort_revision_memory(gap):
    memory = Memory()
    before = {"pack_id": "v1", "results": [{"test_id": "x", "status": "pass",
              "gap": gap, "evidence_refs": ["product_1:artifact:final_response"]}]}
    after = deepcopy(before)
    entry = memory.record(TASK, "Verify uncertain checks", before, after,
                          source={"before": "check1", "after": "check2"}, context_note="Revision")
    assert isinstance(entry["key"], str)
    assert entry["value"]["observed_changes"][0]["gap_after"] == ""
    assert after["results"][0]["gap"] is gap


def test_real_review_recheck_handles_nullable_model_gap():
    async def call(request):
        return {"status": "pass", "reason": "Valid answer", "gap": None,
                "evidence_refs": ["product_1:artifact:final_response"],
                "claims": [{"claim": "Valid answer", "claim_refs": ["task"],
                            "evidence_refs": ["product_1:artifact:final_response"],
                            "observation": "Answer is present", "judgment": "supported"}]}

    pack = deepcopy(PACK)
    pack["tests"].append({"test_id": "semantic", "criterion_id": "json", "checker": "evidence.review"})
    review = DeliveryReview(TASK, pack, defaults(call), memory=Memory(), audit_mode="full")
    assert asyncio.run(review({"artifact": {"final_response": "broken"}}))
    assert asyncio.run(review({"artifact": {"final_response": "{}"}})) is None
    assert review.status == "satisfied_under_estimator" and len(review.history) == 2
