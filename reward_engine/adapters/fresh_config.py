"""Explicit model settings for new campaigns; frozen historical configs are untouched."""

from copy import deepcopy

from experiments.claw_eval.opencode.model_settings import configure_roles
from reward_engine.adapters.model_config import (
    ROLES,
    apply_profile,
    freeze_profile,
    load_profile,
)


def fresh_roles(path, entries, protocol, include_final=False):
    profile = load_profile(path)[1] if path else {}
    roles = dict.fromkeys(ROLES, "")
    if include_final:
        roles["final_selector"] = ""
    roles.update(profile.get("roles", {}))
    roles = configure_roles(roles, protocol, entries)
    missing = sorted(k for k, v in roles.items() if not v or v == "YOUR_MODEL_ID")
    if missing:
        raise ValueError(
            "Explicit models required via --model-config / --role-model: "
            + ", ".join(missing)
        )
    return roles


def freeze_settings(config, path, entries, protocol):
    if not path:
        return
    raw, profile = load_profile(path)
    apply_settings(config, profile, entries, protocol)
    config["model_config_precedence"] = "file_cli_derived_v1"
    config["model_config_role_overrides"] = list(entries or [])
    config["generation_protocol"] = protocol
    freeze_profile(config, path, raw)


def apply_settings(config, profile, entries, protocol):
    # Apply the file to base roles first. CLI choices and derived seats come last.
    final_roles = config["roles"]
    base = dict.fromkeys(ROLES, final_roles["actor"])
    base.update({k: v for k, v in final_roles.items() if k in ROLES})
    if "final_selector" in final_roles:
        base["final_selector"] = final_roles["final_selector"]
    config["roles"] = base
    config["validation_role_limits"] = {
        "role_max_tokens": {role: 32768 for role in base if role != "actor"}
    }
    apply_profile(config, profile)
    config["roles"] = configure_roles(config["roles"], protocol, entries)
    budgets = config["validation_role_limits"]["role_max_tokens"]
    for role in config["roles"]:
        if role.startswith("attack_"):
            source = (
                "jury_reducer"
                if role == "attack_reducer"
                else "jury_member_" + role.rsplit("_", 1)[1]
            )
            budgets[role] = budgets[source]
    config["validation_role_limits"]["role_max_tokens"] = {
        k: v for k, v in budgets.items() if k in config["roles"]
    }
    config["role_max_tokens"] = deepcopy(
        config["validation_role_limits"]["role_max_tokens"]
    )
    # Revalidate model-indexed options against the final overridden model set.
    for field in ("model_aliases", "role_request_options_by_model"):
        allowed = set(config["roles"].values())
        if field == "model_aliases" and "native_grader" in config:
            allowed.add(config["native_grader"]["model"])
        if not set(config.get(field, {})) <= allowed:
            raise ValueError(field + " references a model replaced by --role-model")
    from reward_engine.adapters.model_config import delivery_review_schema_models

    delivery_review_schema_models(config)
    if config.get("validation_benchmark") == "workbuddy":
        from reward_engine.adapters.model_config import workbuddy_official_protocol

        config["official_protocol"] = workbuddy_official_protocol(config)
