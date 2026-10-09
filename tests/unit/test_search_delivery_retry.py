"""Retry missing delivery, not inconvenient knowledge judgments. No live API."""

import asyncio
from copy import deepcopy

import pytest

from reward_engine.core.outcome.context import IncompleteModelResponse
from reward_engine.core.outcome.knowledge_summary import (
    COVERAGE_SCHEMA,
    REDUCE_INSTRUCTION,
)
from reward_engine.core.outcome.task_review import _summary_call


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode,field",
    [
        ("task_review_source_map", "source_findings"),
        ("task_review_group_reduce", "group_decisions"),
    ],
)
@pytest.mark.parametrize("failure", ["output_limit", "missing_required_list"])
@pytest.mark.parametrize("always_fails", [False, True])
async def test_delivery_retry_once_without_changing_task_or_evidence(
    mode, field, failure, always_fails
):
    request = {
        "mode": mode,
        "context": {"task_prompt": "Inspect the supplied material."},
        "sources": [{"source_id": "manual", "text": "Exact source with exceptions."}],
    }
    original, calls, records, retries = deepcopy(request), [], {}, []

    async def model(row):
        calls.append(deepcopy(row))
        if len(calls) == 1 or always_fails:
            if failure == "output_limit":
                raise IncompleteModelResponse("length")
            return {"type": "object"}  # Schema echo is not an actual selection.
        return {field: []}  # A valid empty judgment is never redrawn.

    async def run():
        return await _summary_call(
            request, model, lambda k, v: records.update({k: v}), retries
        )

    if always_fails:
        with pytest.raises(ValueError):
            await run()
    else:
        assert await run() == {field: []}
    assert len(calls) == 2 and len(retries) == len(records) == 1
    assert retries[0]["reason"] == failure
    assert (
        request
        == original
        == {k: v for k, v in calls[1].items() if k != "format_correction"}
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        ValueError("Model-mismatched response"),
        IncompleteModelResponse("content_filter"),
        IncompleteModelResponse(None),
        TimeoutError("deadline"),
    ],
)
async def test_other_failures_do_not_trigger_delivery_redraw(error):
    calls, retries = [], []

    async def model(request):
        calls.append(request)
        raise error

    with pytest.raises(type(error)):
        await _summary_call(
            {"mode": "task_review_group_reduce"}, model, lambda *_: None, retries
        )
    assert len(calls) == 1 and retries == []


@pytest.mark.asyncio
async def test_output_limit_retry_cannot_extend_enclosing_deadline():
    calls, retries = [], []

    async def model(request):
        calls.append(request)
        if len(calls) == 1:
            raise IncompleteModelResponse("length")
        await asyncio.Event().wait()

    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.03):
            await _summary_call(
                {"mode": "task_review_group_reduce"}, model, lambda *_: None, retries
            )
    assert len(calls) == 2 and len(retries) == 1


def test_reduce_question_contract_forbids_guessed_answers_without_task_specific_rules():
    assert (
        "unsupported premises, numerical targets or causal relationships"
        in REDUCE_INSTRUCTION
    )
    assert "not to confirm a guessed answer" in REDUCE_INSTRUCTION
    assert (
        "no guessed value or unsupported premise"
        in COVERAGE_SCHEMA[0]["open_questions"][0]["question"]
    )
    assert "T101" not in REDUCE_INSTRUCTION and "WAL" not in REDUCE_INSTRUCTION
