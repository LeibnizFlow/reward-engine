"""Synthetic syntax failures through complete-draft recovery after evidence reads."""

import json
from copy import deepcopy

import pytest

from reward_engine.core.outcome import authored_json
from reward_engine.core.outcome.context import ask, build_context


@pytest.mark.asyncio
async def test_syntax_recovery_identifies_unescaped_quote_after_evidence_expansion():
    context, evidence = build_context({"task_id": "synthetic", "prompt": "Review the note"})
    malformed = '{"reason":"The note says "hello"."}'
    requests = []

    async def model(request):
        requests.append(deepcopy(request))
        if len(requests) == 1:
            return {"need_evidence": ["task"]}
        if len(requests) == 2:
            return authored_json.decode(malformed)
        recovery = request["response_recovery"]
        assert recovery["rejected_raw_response"] == malformed
        assert "task" in request["requested_evidence"]
        syntax = recovery["details"]["syntax"]
        assert syntax["offset"] == 26
        assert syntax["line"] == 1 and syntax["column"] == 27
        assert syntax["context"] == malformed
        assert syntax["context_start"] == 0
        assert "escape" in recovery["instruction"].lower()
        assert r'\"' in recovery["instruction"]
        return authored_json.decode(json.dumps({"reason": 'The note says "hello".'}))

    model.role_output_protocol = authored_json.VERSION
    model.role_recovery_policy = authored_json.RECOVERY
    model.role_input_projection = "canonical_sources_v1"
    model.role_contract_policy = "canonical_recovery_v1"
    model.bounded_stage = True
    result = await ask(model, "outcome_measure", context, evidence, "Review", {"reason": "text"})
    assert result == {"reason": 'The note says "hello".'}
    assert len(requests) == 3


@pytest.mark.parametrize("raw", [
    '{\n"reason":"中文 "quoted""}',
    '{"reason":"first\nsecond"}',
    r'{"reason":"bad\q"}',
    '{"reason":"' + "a" * 200 + '"bad""}',
    '{"reason":',
])
def test_syntax_coordinates_refer_to_exact_parser_document(raw):
    with pytest.raises(json.JSONDecodeError) as caught:
        authored_json.decode(raw)
    error = caught.value
    detail = authored_json.syntax_diagnostics(error)
    syntax = detail["syntax"]
    assert (syntax["line"], syntax["column"], syntax["offset"]) == (
        error.lineno, error.colno, error.pos)
    assert syntax["location_kind"] == "parser_reported"
    assert syntax["context"] == raw[syntax["context_start"]:error.pos + 80]
    assert len(syntax["context"]) <= 160
    assert error.doc == raw


@pytest.mark.asyncio
@pytest.mark.parametrize("canonical", [True, False])
async def test_syntax_guidance_is_canonical_only_and_budget_stays_one(canonical):
    context, evidence = build_context({"task_id": "synthetic", "prompt": "Review"})
    requests = []

    async def model(request):
        requests.append(deepcopy(request))
        assert (authored_json.JSON_SYNTAX_INSTRUCTION in request["instruction"]) == canonical
        if len(requests) > 1:
            assert ("syntax" in request["response_recovery"]["details"]) == canonical
        return authored_json.decode('{"text":"bad "quoted""}')

    model.role_output_protocol = authored_json.VERSION
    model.role_recovery_policy = authored_json.RECOVERY
    model.bounded_stage = True
    if canonical:
        model.role_input_projection = "canonical_sources_v1"
    with pytest.raises(json.JSONDecodeError) as caught:
        await ask(model, "outcome_measure", context, evidence, "Review", {"text": "text"})
    assert len(requests) == 2
    assert caught.value.operation_budget["syntax_repairs"] == 1


@pytest.mark.parametrize("raw", ['{"x":1,"x":2}', '{} {}', 'Answer: {}', '{"x":'])
def test_diagnostics_do_not_salvage_ambiguous_or_truncated_json(raw):
    with pytest.raises(json.JSONDecodeError) as caught:
        authored_json.decode(raw)
    authored_json.syntax_diagnostics(caught.value)
    with pytest.raises(json.JSONDecodeError):
        authored_json.decode(raw)
