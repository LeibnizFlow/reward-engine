"""Providers that mandate thinking must not inherit a global disable request."""

from copy import deepcopy

import pytest

from reward_engine.adapters.workbuddy.io import read
from reward_engine.adapters.workbuddy.models import _role_request_options
from tests.unit.test_workbuddy_role_models import thinking_role as _thinking_role

thinking_role = _thinking_role


@pytest.mark.asyncio
async def test_per_model_provider_default_is_applied_recorded_and_not_global(
    thinking_role,
):
    role, calls, _ = thinking_role
    disabled = {"thinking": {"type": "disabled"}, "enable_thinking": False}
    role.config["role_request_options"] = disabled
    role.config["roles"]["other_role"] = "other-model"
    role.config["role_request_options_by_model"] = {"gemini-3.8-flash": {}}
    before = deepcopy(role.config)
    assert await role({"mode": "test"}) == {"status": "pass"}
    assert "enable_thinking" not in calls[0] and "thinking" not in calls[0]
    receipt = next(role.output.glob("*/thinking_parameters.json"))
    assert read(receipt)["options"] == {}
    assert read(receipt)["source"] == "config.role_request_options_by_model"
    assert _role_request_options(role.config, "other-model") == disabled
    assert role.config == before


@pytest.mark.parametrize(
    "overrides",
    [None, [], {"not-configured": {}}, {"gemini-3.8-flash": {"model": "other"}}],
)
@pytest.mark.asyncio
async def test_invalid_model_options_fail_before_network(thinking_role, overrides):
    role, calls, _ = thinking_role
    role.config["role_request_options_by_model"] = overrides
    with pytest.raises(ValueError, match="role_request_options"):
        await role({"mode": "test"})
    assert calls == []
