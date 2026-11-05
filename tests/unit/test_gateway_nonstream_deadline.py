"""Complete responses use the configured deadline, not SSE idle timeouts."""

import asyncio
import json

import httpx
import pytest

from reward_engine.adapters.workbuddy.gateway import application


def config(tmp_path, deadline):
    token = tmp_path / "auth"
    token.write_text("synthetic")
    return {"output": str(tmp_path), "auth_file": str(token),
            "backend_url": "https://unused.invalid/v1", "roles": {"judge": "synthetic"},
            "model_concurrency": 1, "role_stream_timeout_s": deadline}


@pytest.mark.asyncio
async def test_complete_response_waits_for_configured_deadline(tmp_path):
    calls = []
    async def upstream(request):
        calls.append(request)
        # Model a complete reply that takes longer than the SSE idle allowance.
        if request.extensions["timeout"]["read"] < 300:
            raise httpx.ReadTimeout("Synthetic complete response needs 300 seconds")
        return httpx.Response(200, json={"model": "synthetic", "choices": []})
    async def no_wait(_):
        pass
    app = application(config(tmp_path, 900), backend_key="synthetic", retry_wait=no_wait,
                      client_factory=lambda **kw: httpx.AsyncClient(transport=httpx.MockTransport(upstream), **kw))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
        reply = await client.post("/roles/v1/chat/completions", json={"model": "synthetic", "stream": False},
                                  headers={"Authorization": "Bearer synthetic"})
        assert reply.status_code == 200
        assert (await client.get("/health")).json()["inflight"] == 0
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("delay_phase", ["headers", "body"])
async def test_nonstream_deadline_covers_headers_and_body_and_releases_capacity(tmp_path, delay_phase):
    closed = []
    class SlowBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            if delay_phase == "body":
                await asyncio.sleep(0.05)
            yield b'{"model":"synthetic","choices":[]}'
        async def aclose(self):
            closed.append(True)
    async def upstream(request):
        if delay_phase == "headers":
            await asyncio.sleep(0.05)
        return httpx.Response(200, stream=SlowBody())
    app = application(config(tmp_path, 0.01), backend_key="synthetic",
                      client_factory=lambda **kw: httpx.AsyncClient(transport=httpx.MockTransport(upstream), **kw))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
        reply = await client.post("/roles/v1/chat/completions", json={"model": "synthetic", "stream": False},
                                  headers={"Authorization": "Bearer synthetic"})
        assert reply.status_code == 502
        health = (await client.get("/health")).json()
        assert health["inflight"] == 0 and health["model_inflight"] == {"synthetic": 0}
    receipt = json.loads(next(tmp_path.glob("transport/roles/*/receipt.json")).read_text())
    assert receipt["status"] == "failed" and receipt["error_type"] == "TimeoutError"
    assert receipt["attempts"] == 1
    if delay_phase == "body":
        assert closed
