"""Exercise Role -> real ASGI gateway -> synthetic upstream, including restarts."""

import json
import time
from uuid import uuid4

import httpx
import pytest

from reward_engine.adapters.workbuddy import models
from reward_engine.adapters.workbuddy.gateway import application
from reward_engine.adapters.workbuddy.io import read
from reward_engine.core.outcome.context import ask
from tests.support.role_transport import encode, event


def config(tmp_path):
    (tmp_path / "auth").write_text("synthetic")
    return {
        "output": str(tmp_path),
        "identity": "synthetic-revision",
        "auth_file": str(tmp_path / "auth"),
        "gateway_local": "http://gateway",
        "backend_url": "http://upstream/v1",
        "roles": {"judge": "test-model-a"},
        "model_concurrency": 2,
        "role_stream": False,
        "role_bounded_stage": True,
        "role_output_protocol": "model_authored_json_v2",
        "role_recovery_policy": "strict_bounded_v2",
        "role_input_projection": "canonical_sources_v1",
    }


def reply(content, **message):
    return httpx.Response(
        200,
        json={
            "model": "test-model-a",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"content": content, **message},
                }
            ],
        },
    )


def connect(cfg, monkeypatch, responses):
    client = httpx.AsyncClient
    calls = []

    async def no_wait(seconds):
        pass

    monkeypatch.setattr(models, "transport_retry_wait", no_wait)

    def upstream(request):
        calls.append(request)
        assert not any("x-role-" in h.lower() for h in request.headers)
        body = json.loads(request.content)
        assert "logical_operation_id" not in json.dumps(body)
        assert "physical_request_id" not in json.dumps(body)
        assert responses, "Unexpected physical upstream request"
        value = responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    app = application(
        cfg,
        backend_key="synthetic",
        client_factory=lambda **kw: client(
            transport=httpx.MockTransport(upstream), **kw
        ),
    )
    monkeypatch.setattr(
        models.httpx,
        "AsyncClient",
        lambda **kw: client(transport=httpx.ASGITransport(app), **kw),
    )
    role = models.Role(cfg, "judge", cfg["output"] + "/role")
    return role, calls


@pytest.mark.asyncio
async def test_transport_retry_waits_without_restarting_operation(tmp_path, monkeypatch):
    cfg = {**config(tmp_path), "role_contract_policy": "canonical_recovery_v1"}
    role, calls = connect(cfg, monkeypatch, [
        httpx.Response(503, headers={"Retry-After": "23"}), reply('{"status":"ok"}')])
    waits = []

    async def wait(seconds):
        waits.append(seconds)
        assert len(calls) == 1

    monkeypatch.setattr(models, "transport_retry_wait", wait, raising=False)
    assert await ask(role, "synthetic", {}, {}, "Assess", {}) == {"status": "ok"}
    assert waits == [23]
    budget = read(next(role.output.glob('operation_budgets/*.json')))
    assert budget['requests'] == 2 and budget['transport_recoveries'] == 1
    assert len({r['logical_operation_id'] for r in budget['receipts']}) == 1
    assert len({r['deadline'] for r in budget['receipts']}) == 1


@pytest.mark.asyncio
async def test_construction_resume_places_current_feedback_after_rejected_answer(tmp_path, monkeypatch):
    cfg = {**config(tmp_path), "role_contract_policy": "canonical_recovery_v1"}
    role, calls = connect(cfg, monkeypatch, [reply('{"criteria":[]}')])
    rejected = {'criteria': [{'synthetic_bad': True}] * 21}
    request = {'mode': 'outcome_expectations', 'role_output_protocol': 'model_authored_json_v2',
               'role_input_projection': 'canonical_sources_v1', 'source_requirements': {},
               'source_catalog': {}, 'response': {'criteria': ['schema']},
               'resume_correction': {'previous_response': rejected,
                   'previous_rejection': {'reason': 'Too many criteria', 'details': {}},
                   'repair_worklist': {'criteria_count': 21}, 'instruction': 'Return complete corrected JSON'}}
    await role(request)
    messages = json.loads(calls[0].content)['messages']
    assert [m['role'] for m in messages] == ['system', 'user', 'assistant', 'user']
    assert json.loads(messages[2]['content']) == rejected
    base, feedback = json.loads(messages[1]['content']), json.loads(messages[3]['content'])
    assert 'resume_correction' not in base
    assert feedback['response'] == request['response']
    assert feedback['resume_correction']['repair_worklist']['criteria_count'] == 21
    assert 'previous_response' not in feedback['resume_correction']
    assert read(next(role.output.glob('*/request.json'))) == request


@pytest.mark.asyncio
@pytest.mark.parametrize('status', [400, 401, 403, 404, 422])
async def test_permanent_upstream_http_error_is_not_disguised_as_retryable_502(tmp_path, monkeypatch, status):
    role, calls = connect(config(tmp_path), monkeypatch, [httpx.Response(status)])
    with pytest.raises(httpx.HTTPStatusError) as caught:
        await role({'mode': 'synthetic'})
    assert caught.value.response.status_code == status
    assert len(calls) == 1
    budget = read(next(role.output.glob('operation_budgets/*.json')))
    assert budget['requests'] == 1 and budget['transport_recoveries'] == 0


@pytest.mark.parametrize('header,expected', [('30', 30), ('99999', 60), ('0', 1),
                                          ('nan', 15), ('-10', 15), ('invalid', 15)])
def test_retry_delay_is_bounded(header, expected):
    response = httpx.Response(503, headers={'Retry-After': header}, request=httpx.Request('POST', 'http://synthetic'))
    error = httpx.HTTPStatusError('Synthetic', request=response.request, response=response)
    assert models.transport_retry_delay(error) == expected


@pytest.mark.asyncio
async def test_backoff_cannot_extend_deadline_or_dispatch_after_expiry(tmp_path, monkeypatch):
    import asyncio

    from reward_engine.core.outcome.call_budget import ACTIVE

    role, calls = connect(config(tmp_path), monkeypatch, [httpx.Response(503)])
    entered = []

    async def wait(seconds):
        entered.append(seconds)
        await asyncio.sleep(10)

    monkeypatch.setattr(models, 'transport_retry_wait', wait)
    budget = {'requests': 0, 'max_requests': 8, 'transport_recoveries': 0,
              'started_at': time.time(), 'deadline': time.time() + 0.1}
    deadline = budget['deadline']
    token = ACTIVE.set(budget)
    try:
        with pytest.raises(TimeoutError):
            await role({'mode': 'synthetic'})
    finally:
        ACTIVE.reset(token)
    assert entered == [15] and len(calls) == 1
    assert budget['requests'] == 1 and budget['deadline'] == deadline


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("recover", [False, True])
async def test_provider_timeout_content_uses_bounded_transport_recovery(tmp_path, monkeypatch, stream, recover):
    notice = "[req_synthetic] [fixture-model]\n**Request exceeded 570s limit**\nSynthetic provider timeout."

    def response(text):
        if stream:
            # Detection must span arbitrary delta boundaries.
            return httpx.Response(200, text=encode([
                event({"content": text[:19]}), event({"content": text[19:]}, "stop")]))
        return reply(text)

    cfg = {**config(tmp_path), "role_stream": stream, "role_contract_policy": "canonical_recovery_v1"}
    role, calls = connect(cfg, monkeypatch, [response(notice), response('{"status":"ok"}' if recover else notice)])
    if recover:
        assert await ask(role, "synthetic", {}, {}, "Assess", {}) == {"status": "ok"}
    else:
        with pytest.raises((models.ProviderStreamError, httpx.HTTPStatusError)):
            await ask(role, "synthetic", {}, {}, "Assess", {})
        assert not list(role.output.glob("completed/*.json"))
    assert len(calls) == 2
    budget = read(next(role.output.glob("operation_budgets/*.json")))
    assert (budget["requests"], budget["transport_recoveries"], budget["syntax_repairs"],
            budget["contract_repairs"]) == (2, 1, 0, 0)
    receipts = [read(p) for p in tmp_path.glob("transport/roles/*/receipt.json")]
    failed = [r for r in receipts if r["status"] == "failed"]
    assert len(failed) == (1 if recover else 2)
    assert all(r["attempts"] == 1 for r in receipts)
    assert all(r["provider_errors"][0]["code"] == "provider_response_timeout" for r in failed)
    raw = list(tmp_path.glob("transport/roles/*/response-1.sse" if stream else "transport/roles/*/response.json"))
    assert any("Request exceeded 570s limit" in p.read_text() for p in raw)
    from reward_engine.adapters.workbuddy.role_audit import request_audit

    assert request_audit(tmp_path, config_identity=cfg["identity"])["all_current_contract_budgets_verified"]


@pytest.mark.parametrize("content", [
    '{"quote":"[req_synthetic] [fixture]\\n**Request exceeded 570s limit**"}',
    "The provider said: [req_synthetic] [fixture]\n**Request exceeded 570s limit**",
    "[req_synthetic] [fixture]\nA normal answer mentions a 570s limit.",
    "**Request exceeded 570s limit**",
])
def test_timeout_envelope_detection_does_not_match_quotes_or_generic_prose(content):
    from reward_engine.adapters.workbuddy.gateway import completion_timeout

    assert completion_timeout(content) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("recover", [False, True])
@pytest.mark.parametrize("markup", [
    '<｜DSML｜ calls>\n<｜DSML｜ invoke name="document.read">invalid</｜DSML｜ invoke>',
    '<tool_call>\n{"need_tools": []}\n</tool_call>',
])
async def test_native_markup_has_explicit_feedback_without_extra_repair_budget(
    tmp_path, monkeypatch, stream, recover, markup
):
    def response(content):
        if stream:
            return httpx.Response(200, text=encode([event({"content": content}, "stop")]))
        return reply(content)

    cfg = {**config(tmp_path), "role_stream": stream, "role_contract_policy": "canonical_recovery_v1"}
    role, calls = connect(cfg, monkeypatch, [
        response(markup), response('{"status":"ok"}' if recover else markup),
    ])
    if recover:
        assert await ask(role, "synthetic", {}, {}, "Assess", {}) == {"status": "ok"}
    else:
        with pytest.raises(json.JSONDecodeError, match="Native tool-call markup"):
            await ask(role, "synthetic", {}, {}, "Assess", {})
        assert not list(role.output.glob("completed/*.json"))
    assert len(calls) == 2
    request = json.loads(json.loads(calls[1].content)["messages"][-1]["content"])
    recovery = request["response_recovery"]
    assert recovery["error_type"] == "JSONDecodeError"
    assert "No requested tools were executed" in recovery["details"]["reason"]
    assert "tool requests as JSON data" in recovery["details"]["reason"]
    budget = read(next(role.output.glob("operation_budgets/*.json")))
    assert (budget["requests"], budget["syntax_repairs"], budget["contract_repairs"],
            budget["transport_recoveries"]) == (2, 1, 0, 0)
    failures = [read(path) for path in role.output.glob("*/failure.json")]
    assert all(row["error_type"] == "JSONDecodeError" and row["partial_output"] == markup
               for row in failures)
    assert len(failures) == (1 if recover else 2)


@pytest.mark.parametrize("marker", ["<｜DSML｜ calls>", "<tool_call>"])
def test_native_markup_inside_valid_json_is_literal_data(tmp_path, marker):
    role = models.Role(config(tmp_path), "judge", tmp_path / "role")
    data = {"quote": marker + ' is a quoted token, not a tool request'}
    assert role.decode_text(json.dumps(data, ensure_ascii=False)) == (data, [])


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_length_recovery_preserves_syntax_budget_and_explicit_output_limit(
    tmp_path, monkeypatch, stream
):
    """Provider truncation must not consume the one JSON syntax correction."""
    def response(content, finish):
        if stream:
            terminal = event({"content": content}, finish)
            terminal["usage"] = {"completion_tokens": 4096}
            return httpx.Response(200, text=encode([terminal]))
        return httpx.Response(200, json={
            "model": "test-model-a", "usage": {"completion_tokens": 4096},
            "choices": [{"index": 0, "finish_reason": finish,
                         "message": {"content": content}}],
        })

    cfg = {**config(tmp_path), "role_contract_policy": "canonical_recovery_v1",
           "role_stream": stream, "role_max_tokens": {"judge": 32768}}
    partial = '{"status":"incomplete'
    role, calls = connect(cfg, monkeypatch, [
        response(partial, "length"),
        response('{"status":0,"status":1}', "stop"),
        response('{"status":"ok"}', "stop"),
    ])
    args = (role, "synthetic", {}, {}, "Assess", {"status": "ok"})
    assert await ask(*args) == {"status": "ok"}
    assert await ask(*args) == {"status": "ok"}
    assert len(calls) == 3
    bodies = [json.loads(call.content) for call in calls]
    assert bodies[0] == bodies[1]
    assert all(body["max_tokens"] == 32768 for body in bodies)
    budget = read(next(role.output.glob("operation_budgets/*.json")))
    assert (budget["requests"], budget["transport_recoveries"],
            budget["syntax_repairs"], budget["contract_repairs"]) == (3, 1, 1, 0)
    failures = [read(p) for p in role.output.glob("*/failure.json")]
    assert {f["error_type"] for f in failures} == {"TruncatedFinalAnswer", "JSONDecodeError"}
    assert any(f["partial_output"] == partial for f in failures)
    from reward_engine.adapters.workbuddy.role_audit import request_audit

    assert request_audit(tmp_path)["all_current_contract_budgets_verified"]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_repeated_length_is_terminal_even_with_parseable_content(tmp_path, monkeypatch, stream):
    content = '{"status":"must-not-be-accepted"}'
    if stream:
        raw = httpx.Response(200, text=encode([event({"content": content}, "length")]))
    else:
        raw = httpx.Response(200, json={"model": "test-model-a", "choices": [
            {"index": 0, "finish_reason": "length", "message": {"content": content}}]})
    cfg = {**config(tmp_path), "role_stream": stream, "role_max_tokens": {"judge": 32768}}
    role, calls = connect(cfg, monkeypatch, [raw, raw])
    with pytest.raises(models.TruncatedFinalAnswer, match="requested_max_tokens=32768"):
        await ask(role, "synthetic", {}, {}, "Assess", {"status": "ok"})
    assert len(calls) == 2
    assert not list(role.output.glob("*/response.json"))
    assert not list(role.output.glob("completed/*.json"))
    failures = [read(p) for p in role.output.glob("*/failure.json")]
    assert len(failures) == 2
    assert all(f["details"]["requested_max_tokens"] == 32768 for f in failures)
    budget = read(next(role.output.glob("operation_budgets/*.json")))
    assert (budget["transport_recoveries"], budget["syntax_repairs"]) == (1, 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", [None, "canonical_recovery_v1"])
@pytest.mark.parametrize("first_failure", ["empty", "length"])
async def test_eight_real_requests_share_transport_syntax_content_evidence_budget(
    tmp_path, monkeypatch, policy, first_failure
):
    responses = [
        reply("", reasoning_content="This is not the answer"),
        reply('{"need_evidence":["a"]}'),
        reply('{"need_evidence":["b"]}'),
        reply('{"status":0,"status":1}'),
        reply('{"status":1}'),
        reply('{"status":2}'),
        reply('{"status":3}'),
        reply('{"status":"ok"}'),
    ]
    if first_failure == "length":
        responses[0] = httpx.Response(200, json={"model": "test-model-a", "choices": [
            {"index": 0, "finish_reason": "length", "message": {"content": '{"status":'}}]})
    cfg = config(tmp_path)
    if policy:
        cfg["role_contract_policy"] = policy
    role, calls = connect(cfg, monkeypatch, responses)

    def validate(row):
        if row.get("status") != "ok":
            raise ValueError("Expected ok")
        return row

    args = (
        role,
        "synthetic",
        {},
        {"a": "First source", "b": "Second source"},
        "Assess",
        {"status": "ok"},
    )
    assert await ask(*args, validate=validate) == {"status": "ok"}
    assert await ask(*args, validate=validate) == {"status": "ok"}
    assert len(calls) == 8 and not responses
    receipts = [read(p) for p in tmp_path.glob("transport/roles/*/receipt.json")]
    assert len(receipts) == 8 and sum(r["upstream_attempts"] for r in receipts) == 8
    assert len({r["logical_operation_id"] for r in receipts}) == 1
    assert len({r["physical_request_id"] for r in receipts}) == 8
    budgets = [read(p) for p in role.output.glob("operation_budgets/*.json")]
    assert len(budgets) == 1
    budget = budgets[0]
    assert (
        budget["requests"],
        budget["transport_recoveries"],
        budget["syntax_repairs"],
        budget["contract_repairs"],
    ) == (8, 1, 1, 3)
    failures = [read(p) for p in role.output.glob("*/failure.json")]
    assert {r["error_type"] for r in failures} == {
        "EmptyFinalAnswer" if first_failure == "empty" else "TruncatedFinalAnswer",
        "JSONDecodeError",
    }
    assert all(r["gateway_receipt"] for r in budget["receipts"])
    if policy:
        from reward_engine.adapters.workbuddy.role_audit import request_audit

        audit = request_audit(tmp_path)
        assert audit["all_current_contract_budgets_verified"], audit


@pytest.mark.asyncio
@pytest.mark.parametrize("tamper", ["physical_id", "deadline", "input_ledger", "missing_gateway", "budget_identity"])
async def test_budget_audit_requires_complete_correlated_proof(tmp_path, monkeypatch, tamper):
    from reward_engine.adapters.workbuddy.io import write
    from reward_engine.adapters.workbuddy.role_audit import request_audit

    cfg = {**config(tmp_path), "role_contract_policy": "canonical_recovery_v1"}
    role, calls = connect(cfg, monkeypatch, [reply('{"status":"ok"}')])
    await ask(role, "synthetic", {}, {}, "Assess", {"status": "ok"})
    assert len(calls) == 1 and request_audit(tmp_path)["all_current_contract_budgets_verified"]
    path = next(tmp_path.glob("transport/roles/*/receipt.json"))
    row = read(path)
    if tamper == "physical_id":
        row["physical_request_id"] = uuid4().hex
    elif tamper == "deadline":
        row["upstream_started_at"] = row["deadline"] + 1
    elif tamper == "input_ledger":
        ledger_path = next(tmp_path.glob("transport/role_operations/*/admissions.json"))
        ledger = read(ledger_path)
        ledger["requests"][uuid4().hex] = "missing"
        write(ledger_path, ledger)
    elif tamper == "budget_identity":
        budget_path = next(role.output.glob("operation_budgets/*.json"))
        budget = read(budget_path)
        budget["config_identity"] = "different-revision"
        write(budget_path, budget)
    else:
        path.unlink()
    if tamper != "missing_gateway":
        write(path, row)
    assert not request_audit(tmp_path, config_identity=cfg["identity"])["all_current_contract_budgets_verified"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["empty", "http", "connection", "stream"])
@pytest.mark.parametrize("policy", [None, "canonical_recovery_v1"])
async def test_transport_recovery_once_and_preserves_failed_receipts(
    tmp_path, monkeypatch, failure, policy
):
    cfg = config(tmp_path)
    if policy:
        cfg["role_contract_policy"] = policy
    if failure == "empty":
        bad = reply(None, reasoning_content="Not an answer")
    elif failure == "http":
        bad = httpx.Response(503)
    elif failure == "connection":
        bad = httpx.ConnectError("synthetic connection error")
    else:
        cfg["role_stream"] = True
        bad = httpx.Response(
            200, text='data: {"error":{"message":"synthetic upstream failure"}}\n\n'
        )
    role, calls = connect(cfg, monkeypatch, [bad, bad])
    with pytest.raises(
        (models.EmptyFinalAnswer, models.ProviderStreamError, httpx.HTTPStatusError)
    ):
        await ask(role, "synthetic", {}, {}, "Assess", {})
    assert len(calls) == 2
    assert len(list(role.output.glob("*/failure.json"))) == 2
    assert not list(role.output.glob("*/response.json"))
    budget = read(next(role.output.glob("operation_budgets/*.json")))
    assert budget["transport_recoveries"] == 1 and budget["syntax_repairs"] == 0
    assert all(
        read(p)["attempts"] == 1
        for p in tmp_path.glob("transport/roles/*/receipt.json")
    )
    if policy:
        from reward_engine.adapters.workbuddy.role_audit import request_audit

        audit = request_audit(tmp_path, config_identity=cfg["identity"])
        assert audit["current_contract_operations"] == 1
        assert audit["all_current_contract_budgets_verified"], audit


@pytest.mark.asyncio
async def test_failed_stream_recovers_as_fresh_complete_stream(tmp_path, monkeypatch):
    cfg = {**config(tmp_path), "role_stream": True}
    responses = [
        httpx.Response(200, text='data: {"error":{"message":"failure"}}\n\n'),
        httpx.Response(
            200, text=encode([event({"content": '{"status":"ok"}'}, "stop")])
        ),
    ]
    role, calls = connect(cfg, monkeypatch, responses)
    assert await ask(role, "synthetic", {}, {}, "Assess", {}) == {"status": "ok"}
    assert len(calls) == 2 and len(list(role.output.glob("*/failure.json"))) == 1
    assert all(json.loads(c.content)["stream"] is True for c in calls)


@pytest.mark.asyncio
async def test_gateway_rejects_ninth_and_duplicate_across_restart(tmp_path):
    cfg = config(tmp_path)
    calls = []

    def upstream(request):
        calls.append(request)
        return reply("{}")

    def app():
        return application(
            cfg,
            backend_key="synthetic",
            client_factory=lambda **kw: httpx.AsyncClient(
                transport=httpx.MockTransport(upstream), **kw
            ),
        )

    headers = {
        "Authorization": "Bearer synthetic",
        "X-Role-Operation-ID": uuid4().hex,
        "X-Role-Operation-Deadline": str(time.time() + 1700),
    }
    for i in range(9):
        headers["X-Role-Request-ID"] = uuid4().hex
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app()), base_url="http://gateway"
        ) as client:
            result = await client.post(
                "/roles/v1/chat/completions",
                json={"model": "test-model-a", "stream": False},
                headers=headers,
            )
            assert result.status_code == (200 if i < 8 else 409)
            if i == 0:
                duplicate = await client.post(
                    "/roles/v1/chat/completions",
                    json={"model": "test-model-a"},
                    headers=headers,
                )
                assert duplicate.status_code == 409
    assert len(calls) == 8
    receipts = [read(p) for p in tmp_path.glob("transport/roles/*/receipt.json")]
    rejected = [r for r in receipts if r["status"] == "rejected_before_execution"]
    assert len(rejected) == 2 and all(r["upstream_attempts"] == 0 for r in rejected)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw", ['{"status":1,"status":2}', "{} {}", "Explanation: {}", '{"status":']
)
async def test_ambiguous_json_requires_complete_resubmission(
    tmp_path, monkeypatch, raw
):
    role, calls = connect(
        config(tmp_path), monkeypatch, [reply(raw), reply('{"status":"ok"}')]
    )
    assert await ask(role, "synthetic", {}, {}, "Assess", {}) == {"status": "ok"}
    assert len(calls) == 2
    second = json.loads(json.loads(calls[1].content)["messages"][1]["content"])
    assert second["response_recovery"]["rejected_raw_response"] == raw
    assert "COMPLETE revised JSON" in second["response_recovery"]["instruction"]


@pytest.mark.asyncio
async def test_empty_truncation_does_not_spend_syntax_repair(tmp_path, monkeypatch):
    empty = httpx.Response(
        200,
        json={
            "model": "test-model-a",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "length",
                    "message": {
                        "content": "",
                        "reasoning_content": "Incomplete thought",
                    },
                }
            ],
        },
    )
    role, calls = connect(
        config(tmp_path), monkeypatch, [empty, reply('{"status":"ok"}')]
    )
    assert await ask(role, "synthetic", {}, {}, "Assess", {}) == {"status": "ok"}
    budget = read(next(role.output.glob("operation_budgets/*.json")))
    assert len(calls) == 2 and budget["syntax_repairs"] == 0
    assert (
        read(next(role.output.glob("*/failure.json")))["error_type"]
        == "EmptyFinalAnswer"
    )
