"""Fresh profiles bind precedence, inheritance, and immutable evidence offline."""

import json
from copy import deepcopy

import pytest

from reward_engine.adapters.fresh_config import freeze_settings, fresh_roles
from reward_engine.adapters.model_config import ROLES, verify_profile
from reward_engine.adapters.provider_config import resolve_provider
from reward_engine.adapters.workbuddy.io import write

PROTOCOL = "three_b_fixed_baselines_v1"


def test_profile_cli_derived_and_frozen_bytes(tmp_path):
    profile = {
        "schema": "reward_engine.model_config.v1",
        "roles": dict.fromkeys(ROLES, "fixture-model"),
        "role_max_tokens": {"jury_member_1": 4096},
    }
    path = tmp_path / "source.json"
    path.write_text(json.dumps(profile))
    overrides = ["jury_member_1=override-model"]
    roles = fresh_roles(path, overrides, PROTOCOL)
    assert roles["attack_member_1"] == roles["attack_reference_1"] == "override-model"
    assert not any(r.startswith("hypothesis") for r in roles)
    config = {
        "output": str(tmp_path),
        "roles": roles,
        "validation_benchmark": "workbuddy",
        "model_concurrency": 2,
        "per_model_concurrency": 2,
    }
    freeze_settings(config, path, overrides, PROTOCOL)
    assert config["role_max_tokens"]["attack_member_1"] == 4096
    assert config["roles"]["attack_member_1"] == "override-model"
    write(tmp_path / "manifest.json", {"model_config": config["model_config"]})
    verify_profile(config)
    altered = deepcopy(config)
    altered["roles"]["actor"] = "tampered"
    with pytest.raises(ValueError, match="effective"):
        verify_profile(altered)
    (tmp_path / "model_config.json").write_text("{}")
    with pytest.raises(ValueError):
        verify_profile(config)


def test_no_implicit_models_or_provider(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="Explicit models"):
        fresh_roles(None, [], PROTOCOL)
    for name in (
        "REWARD_ENGINE_BACKEND_URL",
        "OPENAI_BASE_URL",
        "REWARD_ENGINE_KEY_ENV",
        "REWARD_ENGINE_CREDENTIALS_FILE",
    ):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="explicit API"):
        resolve_provider(default_credentials=tmp_path / "missing")


def test_derived_role_override_rejected(tmp_path):
    path = tmp_path / "models.json"
    path.write_text(
        json.dumps(
            {
                "schema": "reward_engine.model_config.v1",
                "roles": dict.fromkeys(ROLES, "model"),
            }
        )
    )
    with pytest.raises(ValueError, match="derived"):
        fresh_roles(path, ["attack_member_1=other"], PROTOCOL)
