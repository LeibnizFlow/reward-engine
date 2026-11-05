"""Actor filtering is scoped and happens before transport admission."""

import json

import httpx
import pytest

from reward_engine.adapters.jiuwenswarm.gateway import register
from reward_engine.adapters.jiuwenswarm.policy import (
    filter_request,
    permissions,
    policy,
)
from reward_engine.adapters.workbuddy.gateway import application


def tools(*names):
    return [
        {"type": "function", "function": {"name": n, "parameters": {"type": "object"}}}
        for n in names
    ]


@pytest.mark.parametrize(
    "choice",
    [
        "invalid",
        {"type": "function", "function": {"name": "bash"}},
        {"function": {"name": "mcp_claw_read"}},
        True,
    ],
)
def test_invalid_tool_choices(choice):
    with pytest.raises(ValueError):
        filter_request(
            {"tools": tools("mcp_claw_read", "bash"), "tool_choice": choice},
            policy(["mcp_claw_read"]),
        )


@pytest.mark.asyncio
async def test_actor_filter_and_credentials_do_not_apply_to_review_or_grading(tmp_path):
    auth = tmp_path / "auth"
    auth.write_text("controller")
    config = {
        "output": str(tmp_path),
        "auth_file": str(auth),
        "identity": "frozen",
        "backend_url": "http://local.invalid/v1",
        "model_concurrency": 1,
        "roles": {"actor": "fixture", "jury_reducer": "fixture"},
    }
    contract = policy(["mcp_claw_read"])
    actor_token = register(config, "actor-synthetic", contract)
    sent = []

    async def respond(request):
        sent.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "model": "fixture",
            },
        )

    app = application(
        config,
        backend_key="local",
        client_factory=lambda **kw: httpx.AsyncClient(
            transport=httpx.MockTransport(respond), **kw
        ),
    )
    body = {
        "model": "fixture",
        "messages": [],
        "tools": tools("bash", "mcp_claw_read", "switch_mode"),
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/actor-synthetic/v1/chat/completions",
            json=body,
            headers={"authorization": "Bearer " + actor_token},
        )
        assert response.status_code == 200
        assert [t["function"]["name"] for t in sent[-1]["tools"]] == contract["allowed"]
        response = await client.post(
            "/actor-synthetic/v1/chat/completions",
            json={**body, "tool_choice": "bad"},
            headers={"authorization": "Bearer " + actor_token},
        )
        assert response.status_code == 400 and len(sent) == 1
        for channel in ("roles", "grade-wb-test"):
            denied = await client.post(
                f"/{channel}/v1/chat/completions",
                json=body,
                headers={"authorization": "Bearer " + actor_token},
            )
            assert denied.status_code == 401
            response = await client.post(
                f"/{channel}/v1/chat/completions",
                json=body,
                headers={"authorization": "Bearer controller"},
            )
            assert response.status_code == 200 and sent[-1]["tools"] == body["tools"]
    receipt = json.loads(
        next(
            (tmp_path / "transport/actor-synthetic").glob("*/tool_policy.json")
        ).read_text()
    )
    assert receipt["original_tools"] == ["bash", "mcp_claw_read", "switch_mode"]
    assert receipt["sent_tools"] == ["mcp_claw_read"] and receipt["policy_sha256"]
    permission = permissions(contract)
    assert (
        permission["tools"]["mcp_claw_read"] == "allow"
        and permission["tools"]["bash"] == "deny"
    )
