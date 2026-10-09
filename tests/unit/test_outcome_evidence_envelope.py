"""Transport wrappers must preserve the core evidence-inquiry protocol."""

from copy import deepcopy

import pytest

from reward_engine.adapters.claw_eval.reply_format import normalize_reply
from reward_engine.core.outcome.context import ResponseContractError, ask, build_context

SHAPE = {"probes": [], "unavailable_reason": "or empty"}
REQUEST = {
    "mode": "outcome_checker_author",
    "response": SHAPE,
    "evidence_access": "Return need_evidence with exact handles.",
}


@pytest.mark.parametrize("typed", [False, True])
@pytest.mark.parametrize("placement", ["outer", "inner", "inquiry_only"])
@pytest.mark.asyncio
async def test_wrapped_inquiry_expands_evidence_without_format_retry(typed, placement):
    context, evidence = build_context(
        {"task_id": "fixture", "prompt": "Original task text."}
    )
    calls, untouched = [], []

    async def model(request):
        calls.append(deepcopy(request))
        if len(calls) == 1:
            payload = {"probes": [], "unavailable_reason": "Read before authoring."}
            raw = {"response": payload}
            if placement == "outer":
                raw["need_evidence"] = ["task"]
            else:
                raw["response"] = (
                    {"need_evidence": ["task"]}
                    if placement == "inquiry_only"
                    else {**payload, "need_evidence": ["task"]}
                )
            if typed:
                raw["type"] = "json_object"
            before = deepcopy(raw)
            result, _ = normalize_reply(request, raw)
            assert raw == before
            untouched.append(before)
            assert result["need_evidence"] == ["task"]
            return result
        assert request["requested_evidence"]["task"] == "Original task text."
        assert "response_recovery" not in request
        return {"probes": [], "unavailable_reason": "Required runtime is unavailable."}

    result = await ask(
        model, "outcome_checker_author", context, evidence, "Author probes", SHAPE
    )
    assert len(calls) == 2  # Initial inquiry and actual answer; no format repair.
    assert result == {
        "probes": [],
        "unavailable_reason": "Required runtime is unavailable.",
    }
    assert untouched


@pytest.mark.parametrize("refs", [["invented-source"], "task", [None]])
@pytest.mark.asyncio
async def test_transport_does_not_authorize_invalid_evidence(refs):
    context, evidence = build_context({"task_id": "fixture", "prompt": "Task."})
    calls = []

    async def model(request):
        calls.append(request)
        return normalize_reply(
            request, {"type": "json_object", "response": {"need_evidence": refs}}
        )[0]

    with pytest.raises(ResponseContractError):
        await ask(
            model, "outcome_checker_author", context, evidence, "Author probes", SHAPE
        )
    assert len(calls) == 1


@pytest.mark.parametrize(
    "raw",
    [
        {
            "type": "json_object",
            "response": {"probes": [], "unavailable_reason": ""},
            "need_evidence": ["task"],
            "answer": "conflicting",
        },
        {
            "type": "json_object",
            "response": {
                "probes": [],
                "unavailable_reason": "",
                "other": 1,
                "need_evidence": ["task"],
            },
        },
        {
            "type": "json_object",
            "response": {"need_evidence": ["different"]},
            "need_evidence": ["task"],
        },
        {"type": "text", "response": {"need_evidence": ["task"]}},
    ],
)
def test_conflicting_or_unknown_wrapper_fields_are_not_discarded(raw):
    before = deepcopy(raw)
    with pytest.raises(ResponseContractError):
        normalize_reply(REQUEST, raw)
    assert raw == before


def test_inquiry_wrapper_requires_declared_evidence_protocol():
    raw = {"type": "json_object", "response": {"need_evidence": ["task"]}}
    with pytest.raises(ResponseContractError):
        normalize_reply({"response": SHAPE}, raw)
    owned = {"response": {"need_evidence": ["task"]}}
    assert normalize_reply(
        {"response": {"response": {}}, "evidence_access": "Read"}, owned
    ) == (owned, [])
