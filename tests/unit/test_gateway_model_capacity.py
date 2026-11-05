"""A saturated model must leave capacity available to other models."""

import asyncio
import json

import httpx
import pytest

from reward_engine.adapters.workbuddy.gateway import application


def gateway(tmp_path, upstream):
    auth = tmp_path / "auth"
    auth.write_text("synthetic")
    config = {
        "output": str(tmp_path), "auth_file": str(auth),
        "backend_url": "https://unused.invalid/v1",
        "model_concurrency": 2, "max_model_concurrency": 4,
        "per_model_concurrency": 4,
        "roles": {"actor": "slow", "judge": "fast"},
    }
    app = application(config, backend_key="synthetic", client_factory=lambda **kw:
                      httpx.AsyncClient(transport=httpx.MockTransport(upstream), **kw))
    return app, config


@pytest.mark.asyncio
@pytest.mark.parametrize("shrink_queued", [False, True])
async def test_model_control_keeps_other_model_moving(tmp_path, shrink_queued):
    release = asyncio.Event()
    entered = {model: asyncio.Event() for model in ("slow", "fast")}
    calls = []

    async def upstream(request):
        model = json.loads(request.content)["model"]
        calls.append(model)
        entered[model].set()
        if model == "slow":
            await release.wait()
        return httpx.Response(200, json={"model": model, "choices": []})

    app, _ = gateway(tmp_path, upstream)
    headers = {"Authorization": "Bearer synthetic"}
    tasks = []
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://local") as client:
        response = await client.post("/control/concurrency", headers=headers,
            json={"global_limit": 1 if shrink_queued else 2,
                  "model_limits": {"slow": 2 if shrink_queued else 1, "fast": 1}})
        assert response.status_code == 200

        async def submit(model):
            return await client.post("/roles/v1/chat/completions", headers=headers,
                json={"model": model, "messages": [], "stream": False})

        try:
            tasks.append(asyncio.create_task(submit("slow")))
            await asyncio.wait_for(entered["slow"].wait(), 3)
            tasks.append(asyncio.create_task(submit("slow")))
            for _ in range(100):
                if len(list(tmp_path.glob("transport/roles/*/receipt.json"))) == 2:
                    break
                await asyncio.sleep(0.01)
            else:
                raise AssertionError("Second slow request did not queue")
            if shrink_queued:
                # The second request was admitted to the old model semaphore,
                # but still awaits the global slot. It has not executed.
                await asyncio.sleep(0)
                response = await client.post("/control/concurrency", headers=headers,
                    json={"global_limit": 2, "model_limits": {"slow": 1}})
                assert response.status_code == 200
            tasks.append(asyncio.create_task(submit("fast")))
            waiting = asyncio.create_task(entered["fast"].wait())
            done, _ = await asyncio.wait({waiting}, timeout=0.3)
            waiting.cancel()
            await asyncio.gather(waiting, return_exceptions=True)
            assert done, "Slow-model queue consumed the other model's capacity"
            assert calls.count("slow") == 1
            assert (await tasks[-1]).status_code == 200
            health = (await client.get("/health")).json()
            assert health["model_limits"] == {"slow": 1, "fast": 1}
            assert health["inflight"] == 1
        finally:
            release.set()
            results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 3)
        assert all(isinstance(item, httpx.Response) and item.status_code == 200 for item in results)
        assert calls.count("slow") == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("limits", [None, [], {"unknown": 1}, {"slow": True},
                                    {"slow": 0}, {"slow": 5}, {"slow": 1.5}])
async def test_invalid_model_controls_leave_all_limits_unchanged(tmp_path, limits):
    async def upstream(_):
        raise AssertionError("Capacity controls must not call the provider")

    app, _ = gateway(tmp_path, upstream)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://local") as client:
        before = (await client.get("/health")).json()
        response = await client.post("/control/concurrency",
            headers={"Authorization": "Bearer synthetic"},
            json={"global_limit": 3, "model_limits": limits})
        assert response.status_code == 400
        assert (await client.get("/health")).json() == before
    assert not (tmp_path / "transport_limits.json").exists()
    assert not list(tmp_path.glob("transport_limit_events/*"))


@pytest.mark.asyncio
async def test_controls_preserve_model_limits_across_global_change_and_restart(tmp_path):
    async def upstream(_):
        raise AssertionError("Capacity controls must not call the provider")

    app, _ = gateway(tmp_path, upstream)
    headers = {"Authorization": "Bearer synthetic"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://local") as client:
        denied = await client.post("/control/concurrency",
            json={"global_limit": 2, "model_limits": {"slow": 1}})
        assert denied.status_code == 401
        first = await client.post("/control/concurrency", headers=headers,
            json={"global_limit": 2, "model_limits": {"slow": 1}})
        assert first.status_code == 200
        second = await client.post("/control/concurrency", headers=headers,
            json={"global_limit": 3})
        assert second.status_code == 200
        assert second.json()["model_limits"] == {"slow": 1, "fast": 4}
    restarted, _ = gateway(tmp_path, upstream)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(restarted), base_url="http://local") as client:
        health = (await client.get("/health")).json()
        assert health["limit"] == 3
        assert health["model_limits"] == {"slow": 1, "fast": 4}
        changed = await client.post("/control/concurrency", headers=headers,
            json={"global_limit": 3, "model_limits": {"fast": 2}})
        assert changed.status_code == 200
        assert changed.json()["model_limits"] == {"slow": 1, "fast": 2}
    events = [json.loads(p.read_text()) for p in tmp_path.glob("transport_limit_events/*")]
    assert any(row["previous_model_limits"] == {"slow": 1, "fast": 4}
               and row["model_limits"] == {"slow": 1, "fast": 2} for row in events)


@pytest.mark.asyncio
async def test_shrink_keeps_running_requests_and_defers_new_execution(tmp_path):
    releases = [asyncio.Event() for _ in range(3)]
    entered = [asyncio.Event() for _ in range(3)]
    slow_calls = []

    async def upstream(request):
        model = json.loads(request.content)["model"]
        if model == "slow":
            index = len(slow_calls)
            slow_calls.append(index)
            entered[index].set()
            await releases[index].wait()
        return httpx.Response(200, json={"model": model, "choices": []})

    app, _ = gateway(tmp_path, upstream)
    headers = {"Authorization": "Bearer synthetic"}
    tasks = []
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://local") as client:
        await client.post("/control/concurrency", headers=headers,
            json={"global_limit": 3, "model_limits": {"slow": 2, "fast": 1}})

        async def submit(model):
            return await client.post("/roles/v1/chat/completions", headers=headers,
                json={"model": model, "messages": [], "stream": False})

        try:
            for index in range(2):
                tasks.append(asyncio.create_task(submit("slow")))
                await asyncio.wait_for(entered[index].wait(), 3)
            response = await client.post("/control/concurrency", headers=headers,
                json={"global_limit": 3, "model_limits": {"slow": 1}})
            assert response.status_code == 200
            assert response.json()["model_inflight"]["slow"] == 2
            tasks.append(asyncio.create_task(submit("slow")))
            fast = asyncio.create_task(submit("fast"))
            tasks.append(fast)
            assert (await asyncio.wait_for(fast, 3)).status_code == 200
            assert len(slow_calls) == 2
            releases[0].set()
            assert (await asyncio.wait_for(tasks[0], 3)).status_code == 200
            await asyncio.sleep(0.03)
            assert len(slow_calls) == 2
            releases[1].set()
            await asyncio.wait_for(entered[2].wait(), 3)
            releases[2].set()
        finally:
            for release in releases:
                release.set()
            results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 3)
        assert all(isinstance(item, httpx.Response) and item.status_code == 200 for item in results)
        health = (await client.get("/health")).json()
        assert health["inflight"] == 0 and health["model_inflight"] == {"slow": 0, "fast": 0}


@pytest.mark.asyncio
async def test_roles_do_not_starve_actor_or_grader_on_same_model(tmp_path):
    """Exercise the shared HTTP gate with a role flood ahead of native work."""
    labels = ["first", *[f"role-{i}" for i in range(4)],
              *[f"actor-{i}" for i in range(3)], *[f"grade-{i}" for i in range(2)]]
    releases = {name: asyncio.Event() for name in labels}
    entered = {name: asyncio.Event() for name in labels}
    calls = []

    async def upstream(request):
        name = json.loads(request.content)["messages"][0]["content"]
        calls.append(name)
        entered[name].set()
        await releases[name].wait()
        return httpx.Response(200, json={"model": "slow", "choices": []})

    app, _ = gateway(tmp_path, upstream)
    tasks = []
    headers = {"Authorization": "Bearer synthetic"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://local") as client:
        await client.post("/control/concurrency", headers=headers,
                          json={"global_limit": 1, "model_limits": {"slow": 1}})
        try:
            for index, name in enumerate(labels):
                channel = "wb-synthetic" if name.startswith("actor") else (
                    "grade-wb-synthetic" if name.startswith("grade") else "roles")
                tasks.append(asyncio.create_task(client.post(
                    f"/{channel}/v1/chat/completions", headers=headers,
                    json={"model": "slow", "messages": [{"role": "user", "content": name}]})))
                for _ in range(100):
                    if len(list(tmp_path.glob("transport/*/*/receipt.json"))) + len(
                        list(tmp_path.glob("official.private/*/*/receipt.json"))) >= index + 1:
                        break
                    await asyncio.sleep(0.001)
                else:
                    raise AssertionError("HTTP request did not reach the gateway queue")
            await asyncio.wait_for(entered["first"].wait(), 2)
            expected = ["first", "actor-0", "grade-0", "role-0", "actor-1",
                        "grade-1", "role-1", "actor-2", "role-2", "role-3"]
            for index, name in enumerate(expected):
                for _ in range(100):
                    if len(calls) > index:
                        break
                    await asyncio.sleep(0.001)
                assert calls[index:index + 1] == [name], "Role backlog starved a different request class"
                releases[name].set()
        finally:
            for release in releases.values():
                release.set()
            results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 3)
        assert all(isinstance(r, httpx.Response) and r.status_code == 200 for r in results)
