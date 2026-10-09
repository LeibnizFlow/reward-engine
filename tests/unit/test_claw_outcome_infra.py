"""Information flow and failure semantics, independent of any benchmark answer."""

import asyncio
from copy import deepcopy

import pytest

from reward_engine.core.batch import BatchPolicy
from reward_engine.core.outcome import hypotheses, jury
from reward_engine.core.outcome.context import build_context, check_bundle
from reward_engine.core.outcome.harness import Registry, defaults, measure
from reward_engine.core.outcome.memory import Memory
from tests.support.claw_outcome_builder_harness import setup
from tests.support.claw_outcome_context_roles import definition, sample
from tests.support.claw_outcome_memory_runner import PACK, TASK


def test_check_evidence_rebound_and_experiment_labels_hidden():
    measurement = {
        "pack_id": "v1",
        "results": [
            {
                "test_id": "x",
                "status": "pass",
                "reason": "actual read",
                "evidence_refs": ["product_1:environment"],
            }
        ],
    }
    bundle = check_bundle(
        "product_2", measurement, "round_1/SECRET_ARM_NAME/result.json"
    )
    context, evidence = build_context(
        TASK, [{"artifact": {}}, {"artifact": {}}], checks=[bundle]
    )
    assert evidence["check:0:0"]["evidence_refs"] == ["product_2:environment"]
    assert "SECRET_ARM_NAME" not in str(context) + str(evidence)
    assert bundle["source"].endswith("result.json")  # Archive keeps original path.
    with pytest.raises(ValueError, match="supplied product"):
        build_context(TASK, [], checks=[bundle])


def test_hard_requirement_provenance_and_user_scope():
    context, evidence = sample()
    row = definition()
    row["criteria"][0]["evidence_refs"] = ["product_1:environment"]
    with pytest.raises(ValueError, match="cite the Task"):
        jury.validate(row, evidence, context)
    row = definition()
    row["gaps"][0].update(
        expected_basis="user",
        expected_refs=["product_1:public:0"],
        product_ids=["product_1", "product_2"],
        evidence_refs=["product_1:environment", "product_2:environment"],
    )
    context["products"].append({"product_id": "product_2", "user_updates": []})
    evidence["product_2:environment"] = {}
    with pytest.raises(ValueError, match="transfer"):
        jury.validate(row, evidence, context)


def test_map_concurrent_partial_timeout_and_single_reduce():
    context, evidence = sample()

    async def scenario():
        reached, release = [], asyncio.Event()

        async def member(request):
            reached.append(request["mode"])
            if len(reached) == 2:
                release.set()
            await release.wait()  # Would time out if Map silently became serial.
            return definition()

        async def slow(request):
            await asyncio.Event().wait()

        records, reductions = {}, []

        async def reduce(request):
            reductions.append(request)
            return definition()

        result = await jury.propose(
            context,
            evidence,
            {"1": member, "2": member, "3": slow},
            reduce,
            policy=BatchPolicy(deadline_s=0.05),
            record=lambda name, data: records.update({name: data}),
        )
        assert result["criteria"] and len(reductions) == 1
        assert len(reductions[0]["proposals"]) == 2
        assert records["jury_expectations_map"]["items"][2]["status"] == "timeout"
        assert records["jury_map"]["succeeded"] == 2
        assert (
            records["jury_map"]["total"] == 2
        )  # no review without a usable initial proposal

    asyncio.run(scenario())


def test_optional_hypothesis_failure_is_explicit_not_semantic_keep():
    context, evidence = sample()

    async def map_call(request):
        return {"status": "keep", "method": ""}

    async def broken(request):
        raise TimeoutError("provider")

    result = asyncio.run(
        hypotheses.propose(context, evidence, definition(), {"1": map_call}, broken)
    )
    assert result["method"] == "" and result["availability"] == "unavailable"
    assert result["error_type"] == "TimeoutError"


@pytest.mark.parametrize(
    "bad",
    [
        {"status": "pass", "reason": "guess", "evidence_refs": ["task"]},
        {
            "status": "pass",
            "reason": "score",
            "evidence_refs": ["product_1:environment"],
            "score": 0.9,
        },
        {
            "status": "fail",
            "reason": "broken",
            "gap": "error",
            "evidence_refs": ["product_1:environment"],
            "error": True,
        },
    ],
)
def test_harness_cannot_turn_guesses_errors_or_absolute_grades_into_facts(bad):
    registry = Registry()

    async def checker(*args):
        return bad

    registry.register("json.parse", checker, "fault injection")
    context, evidence = build_context(TASK, [{"artifact": {"final_response": "{}"}}])
    result = asyncio.run(measure(PACK, registry, context, evidence))
    assert result["results"][0]["status"] == "unknown" and result["coverage"] == 0


def test_semantic_measurement_identified_and_coverage_not_quality():
    context, evidence = build_context(TASK, [{"artifact": {"final_response": "{bad"}}])

    async def semantic(request):
        return {
            "claims": [
                {
                    "claim": "Output is valid JSON",
                    "claim_refs": ["task"],
                    "evidence_refs": ["product_1:artifact:final_response"],
                    "observation": "Output is an unclosed malformed object.",
                    "judgment": "contradicted",
                }
            ],
            "status": "fail",
            "reason": "observed defect",
            "gap": "malformed",
            "evidence_refs": ["product_1:artifact:final_response"],
        }

    pack = {**PACK, "tests": [{**PACK["tests"][0], "checker": "evidence.review"}]}
    result = asyncio.run(measure(pack, defaults(semantic), context, evidence))
    assert result["coverage"] == 1 and not result["ready_under_estimator"]
    assert result["results"][0]["measurement_kind"] == "model_assisted"


def test_memory_refresh_retrieval_and_consumers_cannot_mutate_bank():
    context, _ = sample()
    task, definition_row = context["task"], definition()
    bank = Memory()
    bank.observe_jury(task, definition_row, source="round_1/jury.json")
    identity = bank.rows[0]["memory_id"]
    definition_row["gaps"][0]["status"] = "check_supported"
    bank.observe_jury(task, definition_row, source="round_2/jury.json")
    assert len(bank.rows) == 1 and bank.rows[0]["memory_id"] == identity
    assert bank.rows[0]["value"]["gap_status"] == "check_supported"
    context["gaps"] = definition_row["gaps"]
    picked = asyncio.run(bank.retrieve(context, top_k=1))
    picked[0]["value"]["expected"] = "POISON"
    assert "POISON" not in str(bank.rows)


def test_author_one_attempt_per_missing_key():
    from reward_engine.core.outcome.builder import build

    context, evidence, definition_row, reply = setup()
    reply["tests"][0]["checker"] = "new.open.key"
    reply["tests"].append({**deepcopy(reply["tests"][0]), "test_id": "another-test"})
    attempts = []

    async def author(test, registry):
        attempts.append(test)
        raise RuntimeError("no tool available")

    async def call(request):
        return reply

    pack = asyncio.run(
        build(context, evidence, definition_row, Registry(), call, author=author)
    )
    assert len(attempts) == 1 and pack["status"] == "needs_implementation"
