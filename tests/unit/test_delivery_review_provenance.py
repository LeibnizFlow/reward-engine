"""Real targeted controller receipts bind review, inquiries and prior delivery state."""

from copy import deepcopy

import pytest

from reward_engine.core.outcome.construction import fingerprint
from reward_engine.core.outcome.context import build_context
from reward_engine.core.outcome.delivery_triage import (
    REVIEW_INPUT_SCHEMA,
    measurement_plan,
    prepare_review_inputs,
    review_input_fingerprint,
)
from reward_engine.core.outcome.harness import (
    defaults,
    measurement_context,
    measurement_input_fingerprint,
)
from reward_engine.core.outcome.runner import DeliveryReview
from tests.support.claw_outcome_memory_runner import PACK, TASK


@pytest.mark.asyncio
async def test_two_real_reviews_bind_exact_inquiries_and_prior_state():
    pack = deepcopy(PACK)
    pack["measurement_sources"] = {
        "reference:json-contract": {"content": "Parse the exact delivered bytes"}
    }
    original = deepcopy(pack)
    requests, records = [], {}
    ref = "product_1:artifact:final_response"

    async def model(request):
        requests.append(deepcopy(request))
        assert request["mode"] == "outcome_delivery_review"
        return {
            "outcome_goal": "Return valid JSON",
            "full_score_gaps": [],
            "improvements": [],
            "gaps": [],
            "status": "needs_verification",
            "reason": "Check the actual delivered bytes",
            "verify": [
                {"test_id": "parse", "reason": "Does this exact delivery parse?"}
            ],
            "claim_checks": [
                {
                    "claim": "This delivery is valid JSON",
                    "claim_ref": ref,
                    "support_refs": [],
                    "assessment": "needs_verification",
                    "reason": "Use the real parser",
                    "test_id": "parse",
                }
            ],
        }

    controller = DeliveryReview(
        TASK,
        pack,
        defaults(model),
        record=lambda k, v: records.__setitem__(k, deepcopy(v)),
    )
    candidates = [{"artifact": {"final_response": value}} for value in ("{bad", "{}")]
    assert await controller(candidates[0])
    assert await controller(candidates[1]) is None
    assert controller.status == "full_score_target_satisfied"
    assert len(requests) == len(controller.history) == 2
    assert [m["results"][0]["status"] for m in controller.history] == ["fail", "pass"]
    previous = None
    for number, (candidate, measured) in enumerate(
        zip(candidates, controller.history, strict=True), 1
    ):
        context, evidence = build_context(TASK, [candidate])
        context, evidence = measurement_context(pack, context, evidence)
        binding = measured["review_input"]
        assert binding["schema"] == REVIEW_INPUT_SCHEMA
        assert binding["sha256"] == review_input_fingerprint(
            pack, context, evidence, previous=previous
        )
        assert binding["previous_measurement_sha256"] == (
            fingerprint(previous[2]) if previous else None
        )
        assert binding["draft_checks"] == [] and binding["max_tests"] == 3
        overview = binding["overview_before_measurement"]
        plan = measurement_plan(pack, overview, previous=previous)
        assert binding["measurement_plan"] == plan
        assert plan["selected_tests"] == ["parse"] and not plan["deferred_tests"]
        assert plan["measurement_inquiries"]["parse"]["questions"] == [
            "Does this exact delivery parse?"
        ]
        reviewed_context, reviewed_evidence, _ = prepare_review_inputs(
            pack, context, evidence
        )
        assert "reference:json-contract" in reviewed_evidence
        actual_measurement_context = {
            **reviewed_context,
            "measurement_inquiries": plan["measurement_inquiries"],
        }
        assert measured["measurement_input_sha256"] == measurement_input_fingerprint(
            pack, actual_measurement_context, reviewed_evidence
        )
        # A minimal reconstruction would certify different inputs and must not match.
        assert measured["measurement_input_sha256"] != measurement_input_fingerprint(
            pack, context, evidence
        )
        assert records[f"delivery_{number}"]["measurement"] == measured
        if previous:
            assert binding["sha256"] != review_input_fingerprint(
                pack, context, evidence
            )
            changed_previous = deepcopy(previous)
            changed_previous[2]["results"][0]["reason"] = "Different observed history"
            assert binding["sha256"] != review_input_fingerprint(
                pack, context, evidence, previous=changed_previous
            )
        previous = (context, evidence, measured)
    assert requests[1]["previous_findings"]["checks"][0]["status"] == "fail"
    assert pack == original


@pytest.mark.parametrize(
    "change", ["artifact", "published_source", "test_budget", "draft_checks"]
)
def test_overview_fingerprint_covers_inputs_that_can_change_review(change):
    pack = deepcopy(PACK)
    pack["measurement_sources"] = {"reference:contract": {"content": "Original source"}}
    context, evidence = build_context(TASK, [{"artifact": {"final_response": "{}"}}])
    original = deepcopy((pack, context, evidence))
    expected = review_input_fingerprint(pack, context, evidence)
    normalized = measurement_context(pack, context, evidence)
    assert expected == review_input_fingerprint(pack, *normalized)
    assert (pack, context, evidence) == original
    kwargs = {}
    if change == "artifact":
        context, evidence = build_context(
            TASK, [{"artifact": {"final_response": "different"}}]
        )
    elif change == "published_source":
        pack["measurement_sources"]["reference:contract"]["content"] = (
            "Different source"
        )
    elif change == "test_budget":
        kwargs["max_tests"] = 1
    else:
        kwargs["draft_checks"] = [
            {
                "test_id": "parse",
                "status": "unknown",
                "reason": "Unresolved earlier draft",
            }
        ]
    assert expected != review_input_fingerprint(pack, context, evidence, **kwargs)
