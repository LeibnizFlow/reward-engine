"""A role amendment changes the real request and cache identity, not old samples."""

import json
from copy import deepcopy

import httpx
import pytest

from reward_engine.adapters.workbuddy.io import digest, read, role_model, write
from reward_engine.adapters.workbuddy.models import Role


@pytest.mark.asyncio
async def test_role_amendment_uses_gemini_and_keeps_old_model_cache(
    tmp_path, monkeypatch
):
    import reward_engine.adapters.workbuddy.models as models

    config = {
        "output": str(tmp_path),
        "identity": "original-samples",
        "roles": {
            "hypothesis_member_3": "glm-5.3",
            "jury_member_3": "gemini-3.8-flash",
        },
        "auth_file": str(tmp_path / "auth"),
        "gateway_local": "http://gateway",
    }
    (tmp_path / "auth").write_text("fixture")
    before = deepcopy(config)
    request = {"mode": "test", "response": {"source": "text"}}
    folder = tmp_path / "role"
    old_cache = (
        folder
        / "completed"
        / (digest({"request": request, "model": "glm-5.3"}) + ".json")
    )
    write(old_cache, {"answer": {"source": "old-glm"}})
    write(
        tmp_path / "role_models.json",
        {"roles": {"hypothesis_member_3": "gemini-3.8-flash"}},
    )
    requests = []

    def upstream(req):
        body = json.loads(req.content)
        requests.append(body)
        assert body["model"] == "gemini-3.8-flash"
        event = {
            "model": body["model"],
            "choices": [
                {
                    "delta": {"content": '{"source":"new-gemini"}'},
                    "finish_reason": "stop",
                }
            ],
        }
        return httpx.Response(
            200, text="data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n"
        )

    client = httpx.AsyncClient
    monkeypatch.setattr(
        models.httpx,
        "AsyncClient",
        lambda **kwargs: client(transport=httpx.MockTransport(upstream), **kwargs),
    )
    role = Role(config, "hypothesis_member_3", folder)
    assert await role(request) == {"source": "new-gemini"}
    assert await role(request) == {"source": "new-gemini"}
    assert len(requests) == 1
    assert read(old_cache)["answer"] == {"source": "old-glm"}
    assert config == before
    (tmp_path / "role_models.json").unlink()
    assert await role(request) == {"source": "old-glm"}
    assert len(requests) == 1


@pytest.mark.parametrize(
    "overrides",
    [
        {"hypothesis_member_3": "unconfigured-model"},
        {"unknown_role": "gemini-3.8-flash"},
        {"hypothesis_member_3": None},
        [],
    ],
)
def test_role_amendment_rejects_unconfigured_roles_and_models(tmp_path, overrides):
    config = {
        "output": str(tmp_path),
        "roles": {"hypothesis_member_3": "gemini-3.8-flash"},
    }
    write(tmp_path / "role_models.json", {"roles": overrides})
    with pytest.raises(ValueError, match="configured roles and gateway models"):
        role_model(config, "hypothesis_member_3")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "finish,actual",
    [
        ("stop", "gemini-3.8-flash"),
        (None, "gemini-3.8-flash"),
        ("length", "gemini-3.8-flash"),
        ("stop", "wrong-model"),
    ],
)
async def test_opt_in_nonstream_recovery_preserves_request_and_requires_complete_reply(
    tmp_path, monkeypatch, finish, actual
):
    import reward_engine.adapters.workbuddy.models as models

    config = {
        "output": str(tmp_path),
        "roles": {"jury_member_3": "gemini-3.8-flash"},
        "auth_file": str(tmp_path / "auth"),
        "gateway_local": "http://gateway",
    }
    (tmp_path / "auth").write_text("fixture")
    calls = []

    def upstream(request):
        body = json.loads(request.content)
        calls.append(body)
        if body["stream"]:
            return httpx.Response(
                200,
                text='data: {"error":{"message":"Campaign upstream stream failed"}}\n\n',
            )
        return httpx.Response(
            200,
            json={
                "model": actual,
                "choices": [
                    {
                        "finish_reason": finish,
                        "message": {"content": '{"source":"recovered"}'},
                    }
                ],
            },
        )

    client = httpx.AsyncClient
    monkeypatch.setattr(
        models.httpx,
        "AsyncClient",
        lambda **kwargs: client(transport=httpx.MockTransport(upstream), **kwargs),
    )
    role = Role(config, "jury_member_3", tmp_path / "role")
    role.nonstream_recovery_models = {"gemini-3.8-flash"}
    request = {"mode": "test"}
    if finish == "stop" and actual == "gemini-3.8-flash":
        assert await role(request) == {"source": "recovered"}
        assert await role(request) == {"source": "recovered"}
    else:
        with pytest.raises(ValueError, match="Incomplete or mismatched"):
            await role(request)
        assert not list((tmp_path / "role/completed").glob("*.json"))
    assert len(calls) == 2
    assert {
        k: v for k, v in calls[0].items() if k not in {"stream", "stream_options"}
    } == {k: v for k, v in calls[1].items() if k != "stream"}
    assert len(list((tmp_path / "role").glob("*/transport.json"))) == 1


@pytest.mark.asyncio
async def test_stream_recovery_is_disabled_by_default(tmp_path, monkeypatch):
    import reward_engine.adapters.workbuddy.models as models

    config = {
        "output": str(tmp_path),
        "roles": {"jury_member_3": "gemini-3.8-flash"},
        "auth_file": str(tmp_path / "auth"),
        "gateway_local": "http://gateway",
    }
    (tmp_path / "auth").write_text("fixture")
    calls = []

    def upstream(request):
        calls.append(request)
        return httpx.Response(200, text='data: {"error":{"message":"failed"}}\n\n')

    client = httpx.AsyncClient
    monkeypatch.setattr(
        models.httpx,
        "AsyncClient",
        lambda **kwargs: client(transport=httpx.MockTransport(upstream), **kwargs),
    )
    with pytest.raises(RuntimeError):
        await Role(config, "jury_member_3", tmp_path / "role")({"mode": "test"})
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_role_output_budget_applies_to_uncached_calls_and_keeps_completed_work(
    tmp_path, monkeypatch
):
    import reward_engine.adapters.workbuddy.models as models

    config = {
        "output": str(tmp_path),
        "roles": {"jury_reducer": "deepseek-v4-flash"},
        "auth_file": str(tmp_path / "auth"),
        "gateway_local": "http://gateway",
    }
    (tmp_path / "auth").write_text("fixture")
    write(tmp_path / "role_limits.json", {"role_max_tokens": {"jury_reducer": 65536}})
    calls = []

    def upstream(request):
        body = json.loads(request.content)
        calls.append(body)
        assert body["max_tokens"] == 65536
        event = {
            "model": body["model"],
            "choices": [
                {"delta": {"content": '{"valid":true}'}, "finish_reason": "stop"}
            ],
        }
        return httpx.Response(
            200, text="data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n"
        )

    client = httpx.AsyncClient
    monkeypatch.setattr(
        models.httpx,
        "AsyncClient",
        lambda **kwargs: client(transport=httpx.MockTransport(upstream), **kwargs),
    )
    role = Role(config, "jury_reducer", tmp_path / "role")
    assert await role({"mode": "test"}) == {"valid": True}
    write(tmp_path / "role_limits.json", {"role_max_tokens": {"jury_reducer": 32768}})
    assert await role({"mode": "test"}) == {"valid": True}
    assert len(calls) == 1
    assert len(list((tmp_path / "role").glob("*/request_parameters.json"))) == 1
    write(tmp_path / "role_limits.json", {"role_max_tokens": {"jury_reducer": True}})
    with pytest.raises(ValueError, match="Role output budgets"):
        await role({"mode": "uncached"})
    assert len(calls) == 1
