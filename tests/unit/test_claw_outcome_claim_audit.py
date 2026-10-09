import asyncio
import json
from copy import deepcopy

import pytest

from reward_engine.core.outcome.context import build_context
from reward_engine.core.outcome.harness import (
    claim_locations,
    defaults,
    validate_claims,
)


def fixture():
    context, evidence = build_context(
        {
            "task_id": "generic",
            "prompt": "Prepare a status note based on the supplied records.",
        },
        [
            {
                "artifact": {"final_response": "See note."},
                "environment": {"note": "The shipment arrived."},
                "source_documents": [{"tracking": "label created"}],
            }
        ],
    )
    row = {
        "claims": [
            {
                "claim": "Shipment arrived",
                "claim_refs": ["product_1:environment"],
                "evidence_refs": ["product_1:source:0"],
                "observation": "A label exists; arrival is not confirmed.",
                "judgment": "unverified",
            }
        ],
        "status": "fail",
        "reason": "The note asserts an unsupported completion.",
        "gap": "Distinguish recorded status from an unconfirmed outcome.",
        "evidence_refs": ["product_1:environment", "product_1:source:0"],
    }
    return context, evidence, row


def run(row):
    context, evidence, _ = fixture()
    requests = []

    async def call(request):
        requests.append(request)
        return deepcopy(row)

    result = asyncio.run(
        defaults(call).run(
            {
                "test_id": "ground",
                "criterion_id": "accuracy",
                "checker": "evidence.review",
            },
            context,
            evidence,
        )
    )
    return result, requests


def test_claim_audit_precedes_verdict_and_preserves_evidence():
    _, _, row = fixture()
    result, requests = run(row)
    assert result["status"] == "fail" and result["claims"] == row["claims"]
    assert list(requests[0]["response"])[0] == "claims"
    assert "ALL delivery channels" in requests[0]["instruction"]
    assert "T110" not in requests[0]["instruction"]


@pytest.mark.asyncio
async def test_measurement_wrapper_preserves_capacity_two_reads_and_bounded_repairs():
    context, evidence, valid = fixture()
    # Larger than the legacy 160 KB default, still inside the declared role cap.
    context["products"][0]["artifact"]["final_response"] = "a" * 170000
    evidence["product_1:artifact:final_response"] = "a" * 170000
    evidence["product_1:source:1"] = {"delivery_log": "Only a label has been created"}
    context["available_evidence_refs"] = list(evidence)
    calls = []

    class Role:
        bounded_stage = True
        max_request_bytes = 393216
        stage_timeout_s = 20

        def prepare_request(self, request):
            request["strategy_marker"] = "preserved"
            return request

        async def __call__(self, request):
            calls.append(deepcopy(request))
            assert request["strategy_marker"] == "preserved"
            if len(calls) == 1:
                return {"need_evidence": ["product_1:source:0"]}
            if len(calls) == 2:
                return {"need_evidence": ["product_1:source:1"]}
            assert set(request["requested_evidence"]) == {
                "product_1:source:0",
                "product_1:source:1",
            }
            if len(calls) == 3:
                raise json.JSONDecodeError("Unescaped quote", '{"', 1)
            if len(calls) == 4:
                invalid = deepcopy(valid)
                invalid["status"] = "pass"  # Unverified claims cannot become pass.
                return invalid
            return deepcopy(valid)

    result = await defaults(Role()).run(
        {"test_id": "ground", "criterion_id": "accuracy", "checker": "evidence.review"},
        context,
        evidence,
    )
    assert result["status"] == "fail", result
    assert len(calls) == 5
    assert calls[3]["response_recovery"]["error_type"] == "JSONDecodeError"
    assert "Unresolved material claims" in str(calls[4]["response_recovery"])


def test_one_supported_occurrence_cannot_erase_a_reported_contradiction():
    _, _, row = fixture()
    other = deepcopy(row["claims"][0])
    other.update(
        claim="A different passage accurately says label created",
        judgment="supported",
        observation="The tracking record supports that passage",
    )
    row["claims"][0]["judgment"] = "contradicted"
    row["claims"].append(other)
    row["status"] = "pass"
    result, _ = run(row)
    assert result["status"] == "unknown" and result["error"]
    row["status"] = "fail"
    result, _ = run(row)
    assert result["status"] == "fail" and len(result["claims"]) == 2


@pytest.mark.parametrize(
    "mutation", ["no_audit", "invented_ref", "unverified_pass", "no_support"]
)
def test_invalid_audit_is_checker_unknown_not_product_failure(mutation):
    _, _, row = fixture()
    if mutation == "no_audit":
        del row["claims"]
    elif mutation == "invented_ref":
        row["claims"][0]["evidence_refs"] = ["fictional"]
    elif mutation == "unverified_pass":
        row["status"] = "pass"
    else:
        row["claims"][0].update(judgment="supported", evidence_refs=[])
    result, _ = run(row)
    assert result["status"] == "unknown" and result["error"]


def test_supported_claim_can_pass_and_missing_evidence_can_stay_unknown():
    _, _, row = fixture()
    row.update(status="pass", gap="", reason="Status is represented accurately.")
    row["claims"][0].update(
        claim="A label exists",
        judgment="supported",
        observation="Tracking records label creation.",
    )
    assert run(row)[0]["status"] == "pass"
    row.update(status="unknown", reason="Arrival cannot be established.")
    row["claims"][0].update(judgment="unverified", evidence_refs=[])
    result, _ = run(row)
    assert result["status"] == "unknown" and not result.get("error")


def test_unverified_advisory_does_not_erase_supported_required_outcome():
    _, _, row = fixture()
    row.update(
        status="pass",
        gap="",
        reason="The note correctly reports the recorded shipment status.",
    )
    row["claims"][0].update(
        claim="A label exists",
        judgment="supported",
        observation="The tracking record confirms label creation.",
    )
    advisory = {
        **deepcopy(row["claims"][0]),
        "claim": "A stylistic attribution could be more precise",
        "judgment": "unverified",
        "observation": "The author attribution is not independently confirmed.",
        "verdict_relevance": "advisory",
        "relevance_reason": "This standard checks recorded shipment status; optional author attribution does not change it.",
    }
    row["claims"].append(advisory)
    result, _ = run(row)
    assert result["status"] == "pass"
    assert (
        result["claims"][-1] == advisory
    )  # Preserve uncertainty; no synthetic support.
    row["claims"][0]["judgment"] = "unverified"
    assert (
        run(row)[0]["status"] == "unknown"
    )  # A supported advisory cannot waive a required fact.


@pytest.mark.parametrize(
    "broken", ["no_reason", "invalid_relevance", "all_advisory", "invented_ref"]
)
def test_advisory_classification_keeps_scope_and_evidence_guards(broken):
    _, evidence, row = fixture()
    row.update(status="pass")
    row["claims"][0].update(judgment="supported")
    advisory = {
        **deepcopy(row["claims"][0]),
        "judgment": "unverified",
        "verdict_relevance": "advisory",
        "relevance_reason": "Optional wording beyond the standard.",
    }
    row["claims"].append(advisory)
    if broken == "no_reason":
        advisory["relevance_reason"] = " "
    elif broken == "invalid_relevance":
        advisory["verdict_relevance"] = "ignore"
    elif broken == "all_advisory":
        row["claims"] = [advisory]
    else:
        advisory["evidence_refs"] = ["invented"]
    with pytest.raises(ValueError):
        validate_claims(row, evidence)


@pytest.mark.parametrize("broken", ["audit", "inquiry", "both"])
def test_rubric_and_inquiry_validation_are_independent_without_invented_repairs(broken):
    context, evidence, row = fixture()
    row.update(status="pass", gap="", reason="Broad status requirement covered")
    row["claims"][0].update(judgment="supported")
    row["inquiry_result"] = {
        "status": "fail",
        "reason": "The note asserts an unsupported arrival",
        "gap": "Distinguish label creation from arrival",
        "evidence_refs": ["product_1:environment"],
    }
    if broken in {"audit", "both"}:
        row["claims"][0]["judgment"] = "supported (but contradictory elsewhere)"
    if broken in {"inquiry", "both"}:
        row["inquiry_result"]["evidence_refs"] = ["invented"]
    before = deepcopy(row)

    async def call(_):
        return deepcopy(row)

    context["measurement_inquiries"] = {
        "ground": {"questions": ["Is arrival confirmed?"]}
    }
    result = asyncio.run(
        defaults(call).run(
            {
                "test_id": "ground",
                "criterion_id": "accuracy",
                "checker": "evidence.review",
            },
            context,
            evidence,
        )
    )
    assert result["status"] == ("unknown" if broken in {"audit", "both"} else "pass")
    assert result["inquiry_result"]["status"] == (
        "unknown" if broken in {"inquiry", "both"} else "fail"
    )
    if broken == "audit":
        assert result["error"] and result["inquiry_result"] == row["inquiry_result"]
    else:
        assert result["inquiry_result"]["error"] and not result["inquiry_result"]["gap"]
    assert (
        row == before
    )  # No enum rewriting, fabricated citation or semantic re-judgment.


@pytest.mark.parametrize("suffix", [": 'Status section'", " (paragraph 2)", "#Status"])
def test_claim_locator_preserves_source_and_annotation_without_verifying_quote(suffix):
    _, evidence, row = fixture()
    original = "product_1:environment" + suffix
    row["claims"][0]["claim_refs"] = [original]
    normalized = claim_locations(row, evidence)
    assert normalized["claims"][0]["claim_refs"] == ["product_1:environment"]
    assert normalized["claims"][0]["location_resolution"] == [
        {
            "original_ref": original,
            "source_ref": "product_1:environment",
            "annotation": suffix,
            "location_verified": False,
        }
    ]
    assert row["claims"][0]["claim_refs"] == [original]
    result, _ = run(row)
    assert result["status"] == "fail"  # the model verdict is never rewritten


@pytest.mark.parametrize(
    "ref",
    [
        "product_2:environment#Status",
        "fictional (section)",
        "product_1:environment_fake#Status",
        "task some rule",
    ],
)
def test_locator_does_not_guess_missing_source_or_free_text(ref):
    _, _, row = fixture()
    row["claims"][0]["claim_refs"] = [ref]
    result, _ = run(row)
    assert result["status"] == "unknown" and result["error"]
    assert "Evidence references" in result["diagnostic"]


def test_ambiguous_or_exact_handles_are_not_shortened():
    _, evidence, row = fixture()
    evidence["product_1:environment#Status"] = "nested source"
    row["claims"][0]["claim_refs"] = ["product_1:environment#Status"]
    assert claim_locations(row, evidence) == row
    row["claims"][0]["claim_refs"] = ["product_1:environment#Status#detail"]
    assert claim_locations(row, evidence) == row
    with pytest.raises(ValueError):
        validate_claims(row, evidence)


def test_supplied_rubric_is_citable_but_cannot_replace_product_evidence():
    context, evidence, row = fixture()
    test = {
        "test_id": "check-note",
        "criterion_id": "accuracy",
        "checker": "evidence.review",
        "rubric": {"id": "accuracy", "standard": "Claims must be supported."},
    }
    requests = []
    row["claims"][0]["claim_refs"] = ["accuracy"]

    async def call(request):
        requests.append(request)
        return deepcopy(row)

    registry = defaults(call)
    result = asyncio.run(registry.run(test, context, evidence))
    assert result["status"] == "fail"
    assert {"accuracy", "check-note"} <= set(
        requests[0]["context"]["available_evidence_refs"]
    )
    assert "accuracy" not in evidence  # scope is local, not a new observed fact
    row["evidence_refs"] = ["accuracy", "product_1:environment"]
    assert asyncio.run(registry.run(test, context, evidence))["status"] == "fail"
    row["evidence_refs"] = ["accuracy"]
    assert asyncio.run(registry.run(test, context, evidence))["status"] == "unknown"
    test["rubric"]["id"] = "product_fiction"
    row["claims"][0]["claim_refs"] = ["product_fiction"]
    row["evidence_refs"] = ["product_fiction"]
    assert asyncio.run(registry.run(test, context, evidence))["status"] == "unknown"


def test_criterion_cannot_shadow_existing_evidence_and_contradiction_stays_unknown():
    context, evidence, row = fixture()

    async def call(request):
        return row

    test = {
        "test_id": "check",
        "criterion_id": "task",
        "checker": "evidence.review",
        "rubric": {"id": "task", "standard": "Not the real Task"},
    }
    result = asyncio.run(defaults(call).run(test, context, evidence))
    assert result["status"] == "unknown" and "collides" in result["diagnostic"]
    row["status"] = "pass"
    row["claims"][0]["claim_refs"] = ["product_1:environment#Status"]
    row["claims"][0]["judgment"] = "contradicted"
    assert run(row)[0]["status"] == "unknown"
