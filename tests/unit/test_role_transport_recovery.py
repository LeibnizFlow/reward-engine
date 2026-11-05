"""Recover a rejected transport via a new complete response, never by merging streams."""

import json

import httpx
import pytest

from reward_engine.adapters.workbuddy.io import read
from tests.support.role_transport import encode, event
from tests.support.role_transport import transport as transport


@pytest.mark.asyncio
async def test_configured_recovery_reissues_identical_request_after_conflicting_ids(transport):
    role, responses, calls = transport
    role.config["role_transport_recovery"] = True
    frames = [event({"content": '{"status":"discard"}'}), event({}, "stop")]
    frames[1]["id"] = "another-completion"
    responses.extend([
        httpx.Response(200, text=encode(frames)),
        httpx.Response(200, json={"id": "fresh-completion", "model": "test-model-a",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"content": '{"status":"unknown"}'}}]}),
    ])
    assert await role({"mode": "test"}) == {"status": "unknown"}
    assert await role({"mode": "test"}) == {"status": "unknown"}
    assert len(calls) == 2
    assert calls[0]["stream"] is True and calls[1]["stream"] is False
    assert {k: v for k, v in calls[0].items() if k not in {"stream", "stream_options"}} == {
        k: v for k, v in calls[1].items() if k != "stream"}
    failure = read(next(role.output.glob("*/failure.json")))
    assert len(failure["completion_ids"]) == 2
    assert failure["partial_output"] == '{"status":"discard"}'
    assert len(list(role.output.glob("*/transport.json"))) == 1


@pytest.mark.asyncio
async def test_recovery_does_not_retry_malformed_semantic_json(transport):
    role, responses, calls = transport
    role.config["role_transport_recovery"] = True
    responses.append(httpx.Response(200, text=encode([event({"content": '{"status":'}, "stop")])))
    with pytest.raises(json.JSONDecodeError):
        await role({"mode": "test"})
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_primary_nonstream_role_keeps_full_response_without_stream_attempt(transport):
    role, responses, calls = transport
    role.config["role_stream"] = False
    responses.append(httpx.Response(200, json={"id": "complete", "model": "test-model-a",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"content": '{"status":"unknown"}'}}]}))
    assert await role({"mode": "test"}) == {"status": "unknown"}
    assert await role({"mode": "test"}) == {"status": "unknown"}
    assert len(calls) == 1 and calls[0]["stream"] is False
    assert "stream_options" not in calls[0]
    assert not list(role.output.glob("*/failure.json"))
    assert read(next(role.output.glob("*/transport.json"))) == {"mode": "nonstream"}
    assert len(list(role.output.glob("*/raw_response.json"))) == 1


@pytest.mark.asyncio
async def test_primary_nonstream_rejects_truncation_and_retains_raw_receipt(transport):
    role, responses, calls = transport
    role.config["role_stream"] = False
    raw = {"id": "incomplete", "model": "test-model-a", "usage": {"completion_tokens": 17},
           "choices": [{"index": 0, "finish_reason": "length", "message": {"content": ""}}]}
    responses.append(httpx.Response(200, json=raw))
    with pytest.raises(ValueError, match="Incomplete"):
        await role({"mode": "test"})
    assert len(calls) == 1
    assert read(next(role.output.glob("*/raw_response.json"))) == raw
    assert len(list(role.output.glob("*/failure.json"))) == 1
    failure = read(next(role.output.glob("*/failure.json")))
    assert failure["usage"] == raw["usage"] and failure["finish_reason"] == "length"
    assert not list(role.output.glob("*/response.json"))


@pytest.mark.asyncio
async def test_unexpected_native_tool_response_gets_bounded_contract_correction(transport):
    from reward_engine.core.outcome.context import ask
    role, responses, calls = transport
    role.config["role_stream"] = False
    native = {"role": "assistant", "content": "", "tool_calls": [{"id": "call-synthetic",
              "type": "function", "function": {"name": "workspace.inspect", "arguments": '{"paths":["*"]}'}}]}
    responses.extend([
        httpx.Response(200, json={"model": "test-model-a", "choices": [
            {"index": 0, "finish_reason": "tool_calls", "message": native}]}),
        httpx.Response(200, json={"model": "test-model-a", "choices": [
            {"index": 0, "finish_reason": "stop", "message": {"content": '{"tests":[]}'}}]}),
    ])
    result = await ask(role, "outcome_harness", {}, {}, "Design only", {"tests": []}, max_contract_retries=1)
    assert result == {"tests": []} and len(calls) == 2
    recovery = json.loads(calls[1]["messages"][1]["content"])["response_recovery"]
    assert recovery["details"]["rejected_message"] == native
    assert recovery["details"]["executed"] is False
    assert len(list(role.output.glob("*/failure.json"))) == 1
    assert len(list(role.output.glob("*/response.json"))) == 1


@pytest.mark.asyncio
async def test_native_tool_contract_correction_cannot_retry_forever(transport):
    from reward_engine.core.outcome.context import ResponseContractError, ask
    role, responses, calls = transport
    role.config["role_stream"] = False
    raw = {"model": "test-model-a", "choices": [{"index": 0, "finish_reason": "tool_calls",
        "message": {"content": "", "tool_calls": [{"id": "call-synthetic", "type": "function",
                     "function": {"name": "workspace.inspect", "arguments": "{}"}}]}}]}
    responses.extend([httpx.Response(200, json=raw), httpx.Response(200, json=raw)])
    with pytest.raises(ResponseContractError, match="Native"):
        await ask(role, "outcome_harness", {}, {}, "Design only", {"tests": []}, max_contract_retries=1)
    assert len(calls) == 2
    assert len(list(role.output.glob("*/failure.json"))) == 2
    assert not list((role.output / "completed").glob("*.json"))


@pytest.mark.asyncio
async def test_model_transport_override_uses_stream_and_distinct_cache(transport):
    role, responses, calls = transport
    role.config["role_stream"] = False
    responses.append(httpx.Response(200, json={"model": "test-model-a", "choices": [
        {"index": 0, "finish_reason": "stop", "message": {"content": '{"status":"first"}'}}]}))
    request = {"mode": "test"}
    assert await role(request) == {"status": "first"}
    role.config["role_stream_by_model"] = {"test-model-a": True}
    responses.append(httpx.Response(200, text=encode([
        event({"reasoning_content": "synthetic internal text"}),
        event({"content": '{"status":"second"}'}, "stop"),
    ])))
    assert await role(request) == {"status": "second"}
    assert await role(request) == {"status": "second"}
    assert [c["stream"] for c in calls] == [False, True]
    assert len(list((role.output / "completed").glob("*.json"))) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("overrides", [None, [], {"unknown": True}, {"test-model-a": 1}])
async def test_model_transport_overrides_are_validated_before_cache(transport, overrides):
    role, responses, calls = transport
    responses.append(httpx.Response(200, text=encode([event({"content": '{}'}, "stop")])))
    await role({"mode": "test"})
    role.config["role_stream_by_model"] = overrides
    with pytest.raises(ValueError, match="role_stream_by_model"):
        await role({"mode": "test"})
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_model_transport_override_leaves_other_models_on_default(transport):
    role, responses, calls = transport
    role.config["roles"]["jury_member_2"] = "synthetic-other"
    role.config["role_stream"] = False
    role.config["role_stream_by_model"] = {"synthetic-other": True}
    responses.append(httpx.Response(200, json={"model": "test-model-a", "choices": [
        {"index": 0, "finish_reason": "stop", "message": {"content": '{}'}}]}))
    assert await role({"mode": "test"}) == {}
    assert calls[0]["stream"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize('status', [408, 429, 500, 502, 503, 504])
async def test_primary_nonstream_recovers_transient_http_once(transport, monkeypatch, status):
    from reward_engine.adapters.workbuddy import models
    from reward_engine.core.outcome.call_budget import ACTIVE
    role, responses, calls = transport
    role.config.update(role_stream=False, role_transport_recovery=True)
    waits = []
    async def wait(delay):
        waits.append(delay)
    monkeypatch.setattr(models, 'transport_retry_wait', wait)
    responses.extend([httpx.Response(status, text='synthetic transient failure'),
        httpx.Response(200, json={'model':'test-model-a','choices':[{'finish_reason':'stop','message':{'content':'{"status":"unknown"}'}}]})])
    budget = {'requests':0,'max_requests':2}
    marker = ACTIVE.set(budget)
    try:
        assert await role({'mode':'test'}) == {'status':'unknown'}
    finally:
        ACTIVE.reset(marker)
    assert budget['requests'] == 2
    assert calls[0] == calls[1]
    assert len(waits) == 1
    assert len(list(role.output.glob('*/failure.json'))) == 1
    assert len(list(role.output.glob('*/response.json'))) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('status,recovery,count', [(502,True,2),(400,True,1),(502,False,1)])
async def test_primary_nonstream_http_recovery_is_bounded(transport, monkeypatch, status, recovery, count):
    from reward_engine.adapters.workbuddy import models
    role, responses, calls = transport
    role.config.update(role_stream=False, role_transport_recovery=recovery)
    async def wait(delay):
        pass
    monkeypatch.setattr(models, 'transport_retry_wait', wait)
    responses.extend([httpx.Response(status, text='synthetic failure') for _ in range(count)])
    with pytest.raises(httpx.HTTPStatusError):
        await role({'mode':'test'})
    assert len(calls) == count
    assert len(list(role.output.glob('*/failure.json'))) == count


@pytest.mark.asyncio
async def test_nonstream_recovery_respects_shared_request_ceiling(transport, monkeypatch):
    from reward_engine.adapters.workbuddy import models
    from reward_engine.core.outcome.call_budget import ACTIVE
    role, responses, calls = transport
    role.config.update(role_stream=False, role_transport_recovery=True)
    async def unexpected_wait(delay):
        pytest.fail('Exhausted operation must not back off or retry')
    monkeypatch.setattr(models, 'transport_retry_wait', unexpected_wait)
    responses.append(httpx.Response(502, text='synthetic failure'))
    budget = {'requests':0,'max_requests':1}
    marker = ACTIVE.set(budget)
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await role({'mode':'test'})
    finally:
        ACTIVE.reset(marker)
    assert budget['requests'] == len(calls) == 1


@pytest.mark.asyncio
async def test_nonstream_transport_recovery_does_not_repair_json(transport):
    role, responses, calls = transport
    role.config.update(role_stream=False, role_transport_recovery=True)
    responses.append(httpx.Response(200, json={'model':'test-model-a','choices':[{'finish_reason':'stop','message':{'content':'{"status": invalid}'}}]}))
    with pytest.raises(json.JSONDecodeError):
        await role({'mode':'test'})
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_empty_nonstream_reply_preserves_contract_repair_budget(transport, monkeypatch):
    from reward_engine.adapters.workbuddy import models
    from reward_engine.core.outcome.call_budget import ACTIVE
    role, responses, calls = transport
    role.config.update(role_stream=False, role_transport_recovery=True)
    async def wait(delay):
        pass
    monkeypatch.setattr(models, 'transport_retry_wait', wait)
    for content in ('', '{"status":"unknown"}'):
        responses.append(httpx.Response(200,json={'model':'test-model-a','choices':[{'finish_reason':'stop','message':{'content':content}}]}))
    budget={'requests':0,'max_requests':6,'contract_repairs':0,'syntax_repairs':0}
    marker=ACTIVE.set(budget)
    try:
        assert await role({'mode':'test'}) == {'status':'unknown'}
    finally:
        ACTIVE.reset(marker)
    assert budget['transport_recoveries'] == 1
    assert budget['contract_repairs'] == budget['syntax_repairs'] == 0
    assert budget['requests'] == len(calls) == 2


@pytest.mark.asyncio
async def test_empty_reply_then_bad_contract_still_gets_semantic_correction(transport, monkeypatch):
    from reward_engine.adapters.workbuddy import models
    from reward_engine.core.outcome.context import ask, ResponseContractError
    role, responses, calls = transport
    role.config.update(role_stream=False, role_transport_recovery=True)
    role.bounded_stage = True
    async def wait(delay):
        pass
    monkeypatch.setattr(models, 'transport_retry_wait', wait)
    for content in ('', '{"status":"bad"}', '{"status":"ok"}'):
        responses.append(httpx.Response(200,json={'model':'test-model-a','choices':[{'finish_reason':'stop','message':{'content':content}}]}))
    def validate(value):
        if value.get('status') != 'ok':
            raise ResponseContractError('status', 'Expected ok')
        return value
    assert await ask(role, 'test', {}, {}, 'Return status ok', {'status':'ok'}, validate=validate) == {'status':'ok'}
    assert len(calls) == 3
    assert calls[0] == calls[1]
    recovery = json.loads(calls[2]['messages'][1]['content'])['response_recovery']
    assert recovery['details']['field'] == 'status'


@pytest.mark.asyncio
@pytest.mark.parametrize("final_status", ["ok", "bad4"])
async def test_opt_in_contract_repairs_are_bounded(transport, final_status):
    from reward_engine.core.outcome.context import ask, ResponseContractError
    role, responses, calls = transport
    role.config.update(role_stream=False)
    role.bounded_stage = True
    role.contract_retries = 3
    for status in ('bad1','bad2','bad3',final_status):
        responses.append(httpx.Response(200,json={'model':'test-model-a','choices':[{'finish_reason':'stop','message':{'content':json.dumps({'status':status})}}]}))
    def validate(value):
        if value['status'] != 'ok':
            raise ResponseContractError('status','Expected ok')
        return value
    if final_status == 'ok':
        assert await ask(role,'test',{}, {},'Return status ok',{'status':'ok'},validate=validate)=={'status':'ok'}
    else:
        with pytest.raises(ResponseContractError) as failure:
            await ask(role,'test',{}, {},'Return status ok',{'status':'ok'},validate=validate)
        assert failure.value.operation_budget['requests'] == 4
        assert failure.value.operation_budget['contract_repairs'] == 3
        assert failure.value.operation_budget.get('max_requests', 6) == 6
    assert len(calls)==4


@pytest.mark.parametrize("value", [0, 4, True, "3", None])
def test_contract_repair_setting_rejects_unbounded_or_ambiguous_values(value):
    from reward_engine.adapters.workbuddy.models import role_contract_retries
    with pytest.raises(ValueError):
        role_contract_retries({"role_contract_retries":value})


@pytest.mark.asyncio
async def test_rejected_edit_envelope_keeps_original_draft_and_diagnostics(transport):
    from reward_engine.core.outcome.context import ask, ResponseContractError
    from reward_engine.core.outcome.construction import fingerprint
    role, responses, calls = transport
    role.config.update(role_stream=False)
    role.bounded_stage = role.protocol_edits = True
    role.contract_retries = 3
    original={'criteria':[{'id':'synthetic','standard':'bad'}]}
    repair={'base_sha256':fingerprint(original),'edits':[{'op':'replace','path':['criteria',0,'standard'],'value':'fixed','reason':'Synthetic correction'}]}
    for value in (original, {**repair,'type':'json_object'}, repair):
        responses.append(httpx.Response(200,json={'model':'test-model-a','choices':[{'finish_reason':'stop','message':{'content':json.dumps(value)}}]}))
    def validate(value):
        if value['criteria'][0]['standard']!='fixed':
            raise ResponseContractError('standard','Expected fixed')
        return value
    assert await ask(role,'outcome_jury_map',{}, {},'Keep complete draft',original,validate=validate)=={'criteria':[{'id':'synthetic','standard':'fixed'}]}
    first,second=[json.loads(c['messages'][1]['content']) for c in calls[1:]]
    assert first['protocol_repair']['base_sha256']==second['protocol_repair']['base_sha256']==fingerprint(original)
    assert second['response_recovery']['rejected_response']==original
    assert second['response_recovery']['details']['field']=='standard'
    assert second['response_recovery']['repair_error']['details']['extra_fields']==['type']


@pytest.mark.asyncio
@pytest.mark.parametrize("wrapper", ["draft", "fields", "both"])
async def test_jury_wrappers_preserve_configured_contract_recovery(transport, wrapper):
    from reward_engine.core.outcome.context import ask, ResponseContractError
    from reward_engine.core.outcome.jury import ask_draft, field_repair_call

    role, responses, calls = transport
    role.config['role_stream'] = False
    role.bounded_stage = True
    role.contract_retries = 3
    for status in ('bad1', 'bad2', 'bad3', 'ok'):
        responses.append(httpx.Response(200, json={
            'model': 'test-model-a', 'choices': [{'finish_reason': 'stop',
            'message': {'content': json.dumps({'status': status})}}]}))

    def validate(value):
        if value['status'] != 'ok':
            raise ResponseContractError('status', 'Expected ok')
        return value

    call = field_repair_call(role) if wrapper in ('fields', 'both') else role
    query = ask_draft if wrapper in ('draft', 'both') else ask
    assert await query(call, 'outcome_jury_map', {}, {}, 'Return status ok',
                       {'status': 'ok'}, validate=validate) == {'status': 'ok'}
    assert len(calls) == 4
