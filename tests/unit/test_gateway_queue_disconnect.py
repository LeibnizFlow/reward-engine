"""Abandoned clients must not consume admission or upstream execution."""
import asyncio
import json

import httpx
import pytest

from reward_engine.adapters.jiuwenswarm.gateway import register
from reward_engine.adapters.jiuwenswarm.policy import policy
from reward_engine.adapters.workbuddy.gateway import application


@pytest.mark.asyncio
@pytest.mark.parametrize('per_model', [1, 2])
@pytest.mark.parametrize('cancel_parent', [False, True])
async def test_abandoned_queue_releases_both_capacity_gates(tmp_path, per_model, cancel_parent):
    auth = tmp_path / 'auth'
    auth.write_text('synthetic-controller')
    config = {'output': str(tmp_path), 'auth_file': str(auth), 'identity': 'synthetic',
              'backend_url': 'http://synthetic/v1', 'model_concurrency': 1,
              'per_model_concurrency': per_model, 'roles': {'actor': 'synthetic-model'}}
    token = register(config, 'actor-synthetic', policy([]))
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def respond(request):
        calls.append(json.loads(request.content))
        entered.set()
        await release.wait()
        return httpx.Response(200, json={'model': 'synthetic-model',
                                        'choices': [{'message': {'content': 'ok'}, 'finish_reason': 'stop'}]})

    app = application(config, backend_key='synthetic', client_factory=lambda **kw:
                      httpx.AsyncClient(transport=httpx.MockTransport(respond), **kw))
    endpoint = next(r.endpoint for r in app.routes if r.path == '/{channel}/v1/chat/completions')

    class Request:
        headers = {'authorization': 'Bearer ' + token}
        disconnected = False

        async def json(self):
            return {'model': 'synthetic-model', 'messages': [], 'stream': False}

        async def is_disconnected(self):
            return self.disconnected

    first = asyncio.create_task(endpoint('actor-synthetic', Request()))
    await asyncio.wait_for(entered.wait(), 3)
    abandoned = Request()
    second = asyncio.create_task(endpoint('actor-synthetic', abandoned))
    for _ in range(100):
        if len(list((tmp_path / 'transport/actor-synthetic').glob('*/receipt.json'))) == 2:
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError('Second request was not queued')
    if cancel_parent:
        second.cancel()
    else:
        abandoned.disconnected = True
    # Do not cancel the request through wait_for: that would conceal a gateway
    # which ignores the actual disconnected peer.
    done, _ = await asyncio.wait({second}, timeout=3)
    try:
        assert done, 'Disconnected request remained queued'
        with pytest.raises(asyncio.CancelledError):
            await second
        rows = [json.loads(p.read_text()) for p in
                (tmp_path / 'transport/actor-synthetic').glob('*/receipt.json')]
        rejected = [r for r in rows if r['status'] == 'rejected_before_execution']
        assert len(rejected) == 1 and rejected[0]['error_type'] == 'CancelledError'
        assert len(calls) == 1
        release.set()
        assert (await asyncio.wait_for(first, 3)).status_code == 200
        assert (await asyncio.wait_for(endpoint('actor-synthetic', Request()), 3)).status_code == 200
        assert len(calls) == 2
    finally:
        release.set()
        second.cancel()
        await asyncio.gather(first, second, return_exceptions=True)


@pytest.mark.asyncio
async def test_already_disconnected_request_never_opens_upstream(tmp_path):
    auth = tmp_path / 'auth'
    auth.write_text('synthetic')
    config = {'output': str(tmp_path), 'auth_file': str(auth), 'identity': 'synthetic',
              'backend_url': 'http://synthetic/v1', 'model_concurrency': 1,
              'per_model_concurrency': 1, 'roles': {'actor': 'synthetic-model'}}
    calls = []

    async def respond(request):
        calls.append(request)
        return httpx.Response(200, json={'model': 'synthetic-model',
                                        'choices': [{'message': {'content': 'ok'}, 'finish_reason': 'stop'}]})

    app = application(config, backend_key='synthetic', client_factory=lambda **kw:
                      httpx.AsyncClient(transport=httpx.MockTransport(respond), **kw))
    endpoint = next(r.endpoint for r in app.routes if r.path == '/{channel}/v1/chat/completions')

    class Request:
        headers = {'authorization': 'Bearer synthetic'}
        disconnected = True

        async def json(self):
            return {'model': 'synthetic-model', 'messages': []}

        async def is_disconnected(self):
            return self.disconnected

    request = Request()
    with pytest.raises(asyncio.CancelledError):
        await endpoint('roles', request)
    assert calls == []
    request.disconnected = False
    assert (await asyncio.wait_for(endpoint('roles', request), 3)).status_code == 200
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_real_http_disconnect_is_observed_before_capacity_frees(tmp_path):
    from tests.support.jiuwenswarm_local import serving

    auth = tmp_path / 'auth'
    auth.write_text('synthetic')
    config = {'output': str(tmp_path), 'auth_file': str(auth), 'identity': 'synthetic',
              'backend_url': 'http://synthetic/v1', 'model_concurrency': 1,
              'roles': {'actor': 'synthetic-model'}}
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def respond(request):
        calls.append(request)
        entered.set()
        await release.wait()
        return httpx.Response(200, json={'model': 'synthetic-model',
                                        'choices': [{'message': {'content': 'ok'}, 'finish_reason': 'stop'}]})

    app = application(config, backend_key='synthetic', client_factory=lambda **kw:
                      httpx.AsyncClient(transport=httpx.MockTransport(respond), **kw))
    async with serving(app) as port, httpx.AsyncClient(trust_env=False, timeout=None) as client:
        url = f'http://127.0.0.1:{port}/roles/v1/chat/completions'
        args = {'headers': {'authorization': 'Bearer synthetic'},
                'json': {'model': 'synthetic-model', 'messages': []}}
        first = asyncio.create_task(client.post(url, **args))
        await asyncio.wait_for(entered.wait(), 3)
        second = asyncio.create_task(client.post(url, **args))
        try:
            for _ in range(150):
                if len(list((tmp_path / 'transport/roles').glob('*/receipt.json'))) == 2:
                    break
                await asyncio.sleep(0.02)
            else:
                raise AssertionError('Real HTTP request was not queued')
            second.cancel()
            await asyncio.gather(second, return_exceptions=True)
            for _ in range(150):
                rows = [json.loads(p.read_text()) for p in
                        (tmp_path / 'transport/roles').glob('*/receipt.json')]
                if any(r['status'] == 'rejected_before_execution' for r in rows):
                    break
                await asyncio.sleep(0.02)
            else:
                raise AssertionError('Gateway ignored HTTP peer disconnect')
            assert len(calls) == 1 and not first.done()
            release.set()
            assert (await asyncio.wait_for(first, 10)).status_code == 200
            assert (await asyncio.wait_for(client.post(url, **args), 10)).status_code == 200
            assert len(calls) == 2
        finally:
            release.set()
            second.cancel()
            await asyncio.gather(first, second, return_exceptions=True)
