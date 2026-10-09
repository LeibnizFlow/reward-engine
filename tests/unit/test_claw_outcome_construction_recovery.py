"""Evidence handle correction uses the same request and never buys another source read."""

from copy import deepcopy

import pytest

from reward_engine.core.outcome.context import (
    EvidenceReadLimit,
    ResponseContractError,
    ask,
    build_context,
)


def inputs():
    return build_context(
        {"task_id": "verify", "prompt": "Compare the actual source statements"},
        skills=[{"claim": "first source"}, {"claim": "second source"}],
    )


@pytest.mark.asyncio
async def test_bad_evidence_handle_after_read_is_corrected_without_losing_existing_evidence():
    context, evidence = inputs()
    before, seen = deepcopy((context, evidence)), []

    async def model(request):
        seen.append(deepcopy(request))
        if len(seen) == 1:
            return {"need_evidence": ["skill:0"]}
        if len(seen) == 2:
            return {"need_evidence": ["invented:source"]}
        if len(seen) == 3:
            assert request["response_recovery"]["details"]["field"] == "need_evidence"
            assert request["requested_evidence"] == {"skill:0": evidence["skill:0"]}
            return {"need_evidence": ["skill:1"]}
        assert request["requested_evidence"] == {
            "skill:0": evidence["skill:0"],
            "skill:1": evidence["skill:1"],
        }
        assert "response_recovery" not in request
        return {"status": "checked"}

    output = await ask(
        model,
        "outcome_hypothesis_audit",
        context,
        evidence,
        "Inspect both sources",
        {},
        max_contract_retries=1,
        max_evidence_rounds=2,
    )
    assert output == {"status": "checked"} and len(seen) == 4
    corrected = deepcopy(seen[2])
    corrected.pop("response_recovery")
    assert corrected == seen[1]
    assert (context, evidence) == before


@pytest.mark.asyncio
async def test_persistently_bad_evidence_handle_retries_once_then_stops():
    context, evidence = inputs()
    seen = []

    async def model(request):
        seen.append(deepcopy(request))
        if len(seen) == 1:
            return {"need_evidence": ["skill:0"]}
        assert request["requested_evidence"] == {"skill:0": evidence["skill:0"]}
        return {"need_evidence": ["invented:source"]}

    with pytest.raises(ResponseContractError) as caught:
        await ask(
            model,
            "outcome_builder_audit",
            context,
            evidence,
            "Inspect",
            {},
            max_contract_retries=1,
            max_evidence_rounds=2,
            close_evidence=True,
        )
    assert caught.value.details["field"] == "need_evidence"
    assert len(seen) == 3
    assert seen[-1]["response_recovery"]["attempt"] == 1


@pytest.mark.asyncio
async def test_correcting_an_evidence_handle_does_not_reset_the_read_budget():
    context, evidence = inputs()
    seen = []

    async def model(request):
        seen.append(deepcopy(request))
        if len(seen) == 1:
            return {"need_evidence": ["skill:0"]}
        if "response_recovery" not in request:
            return {"need_evidence": ["invented:source"]}
        return {"need_evidence": ["skill:1"]}

    with pytest.raises(EvidenceReadLimit):
        await ask(
            model,
            "outcome_jury_audit",
            context,
            evidence,
            "Inspect",
            {},
            max_contract_retries=1,
            max_evidence_rounds=1,
        )
    assert len(seen) == 3
    assert all(
        row["requested_evidence"] == {"skill:0": evidence["skill:0"]}
        for row in seen[1:]
    )


@pytest.mark.parametrize(
    "invalid", [None, [], "skill:0", ["missing"], [{"ref": "skill:0"}]]
)
@pytest.mark.asyncio
async def test_no_retry_default_still_fails_on_first_malformed_evidence_request(
    invalid,
):
    context, evidence = inputs()
    seen = []

    async def model(request):
        seen.append(request)
        return {"need_evidence": invalid}

    with pytest.raises(ResponseContractError) as caught:
        await ask(model, "outcome_hypothesis_map", context, evidence, "Inspect", {})
    assert caught.value.details["field"] == "need_evidence"
    assert len(seen) == 1
