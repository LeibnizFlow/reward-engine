"""Strict, explicitly authored model profiles for fresh native validation runs.

Schema: reward_engine.model_config.v1. Optional maps: roles, role_max_tokens,
model_aliases, concurrency, role_request_options_by_model; optional native_grader.
Roles/tokens/concurrency override only named entries;
aliases replace the entire alias map (omission means exact model IDs only).
Internal-role token budgets are 1..131072; actor budgets stay 32768.
WorkBuddy's optional native_grader independently binds model, max_tokens and
request_options. Absence retains the historical actor-shared native route.
Concurrency keys are model_concurrency and per_model_concurrency, bounded by the
existing benchmark transport ceiling. No retries, model substitutions or jury rules.
Per-model request options are {}, the complete disabled-thinking object, or
{"reasoning_effort": "low" | "high" | "max"}; these forms cannot be combined.
"""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

from reward_engine.adapters import delivery_review_schema
from reward_engine.adapters.workbuddy.io import read

SCHEMA = "reward_engine.model_config.v1"
FILENAME = "model_config.json"
ROLES = frozenset(
    {
        "actor",
        "jury_reducer",
        "hypothesis_chooser",
        "harness_designer",
        "reward_calibration_judge",
    }
    | {f"{family}_member_{i}" for family in ("jury", "hypothesis") for i in range(1, 4)}
)
CONCURRENCY = frozenset({"model_concurrency", "per_model_concurrency"})


def _object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Model profile contains duplicate JSON keys")
        value[key] = item
    return value


def _nonfinite(_value):
    raise ValueError("Model profile must contain finite JSON values")


def _name(value):
    return (
        isinstance(value, str)
        and bool(value)
        and len(value) <= 256
        and not any(c.isspace() or ord(c) < 32 for c in value)
    )


def _validate_request_options(options, *, per_model=False):
    """Share the exact allowlist between frozen profiles and Role requests."""
    if isinstance(options, dict):
        if not options:
            return {}
        if (
            set(options) == {"thinking", "enable_thinking"}
            and options.get("thinking") == {"type": "disabled"}
            and options.get("enable_thinking") is False
        ):
            return {"thinking": {"type": "disabled"}, "enable_thinking": False}
        if (
            per_model
            and set(options) == {"reasoning_effort"}
            and isinstance(options["reasoning_effort"], str)
            and options["reasoning_effort"] in {"low", "high", "max"}
        ):
            return {"reasoning_effort": options["reasoning_effort"]}
    raise ValueError(
        "role_request_options must be {} or the complete disabled-thinking object"
        + (
            " or {'reasoning_effort': 'low'|'high'|'max'} for a configured model"
            if per_model
            else ""
        )
    )


def _review_schema_models(value):
    if (
        not isinstance(value, list)
        or not all(_name(model) for model in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(
            "delivery_review_json_schema_models must be unique exact model IDs"
        )
    return list(value)


def delivery_review_schema_models(config):
    """Share frozen-profile and runtime checks; never match aliases or infer support."""
    key = delivery_review_schema.CONFIG_KEY
    if key not in config:
        return []
    selected = _review_schema_models(config[key])
    if config.get("validation_benchmark") != "workbuddy":
        raise ValueError("Delivery review JSON schema is WorkBuddy-only")
    if not set(selected) <= set(config["roles"].values()):
        raise ValueError("Delivery review JSON schema must name actual role models")
    return selected


def validate_native_grader(value):
    """Validate the entire opt-in native route, without inheriting role options."""
    if not isinstance(value, dict) or set(value) != {
        "model",
        "max_tokens",
        "request_options",
    }:
        raise ValueError("native_grader requires model, max_tokens and request_options")
    if not _name(value["model"]):
        raise ValueError("native_grader model must be an exact model ID")
    if type(value["max_tokens"]) is not int or not 1 <= value["max_tokens"] <= 131072:
        raise ValueError("native_grader max_tokens must be an integer in 1..131072")
    return {
        "model": value["model"],
        "max_tokens": value["max_tokens"],
        "request_options": _validate_request_options(
            value["request_options"], per_model=True
        ),
    }


def workbuddy_official_protocol(config):
    grader = (
        validate_native_grader(config["native_grader"])
        if "native_grader" in config
        else None
    )
    model = grader["model"] if grader else config["roles"]["actor"]
    description = "pinned WorkBuddy CompositeVerifier with " + model
    if grader:
        description += (
            f"; max_tokens={grader['max_tokens']}; request_options="
            + json.dumps(
                grader["request_options"], sort_keys=True, separators=(",", ":")
            )
        )
    return description


def parse_profile(raw):
    if len(raw) > 65536:
        raise ValueError("Model profile exceeds 64 KiB")
    try:
        profile = json.loads(raw, object_pairs_hook=_object, parse_constant=_nonfinite)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Model profile must be UTF-8 JSON") from error
    if not isinstance(profile, dict) or profile.get("schema") != SCHEMA:
        raise ValueError("Unsupported model profile schema")
    if set(profile) - {
        "schema",
        "roles",
        "role_max_tokens",
        "model_aliases",
        "concurrency",
        "role_request_options_by_model",
        "native_grader",
        delivery_review_schema.CONFIG_KEY,
    }:
        raise ValueError("Unknown model profile fields")
    for field in (
        "roles",
        "role_max_tokens",
        "model_aliases",
        "concurrency",
        "role_request_options_by_model",
    ):
        if field in profile and not isinstance(profile[field], dict):
            raise ValueError(f"Model profile {field} must be an object")
    roles = profile.get("roles", {})
    if not set(roles) <= ROLES or not all(_name(value) for value in roles.values()):
        raise ValueError(
            "Model profile roles must name supported roles and exact model IDs"
        )
    tokens = profile.get("role_max_tokens", {})
    if not set(tokens) <= ROLES - {"actor"} or any(
        type(value) is not int or not 1 <= value <= 131072 for value in tokens.values()
    ):
        raise ValueError(
            "role_max_tokens must name internal roles with 1..131072 tokens; actor is unsupported"
        )
    concurrency = profile.get("concurrency", {})
    if not set(concurrency) <= CONCURRENCY or any(
        type(value) is not int or not 1 <= value <= 128
        for value in concurrency.values()
    ):
        raise ValueError("Unsupported concurrency field or value outside 1..128")
    for model, options in profile.get("role_request_options_by_model", {}).items():
        if not _name(model):
            raise ValueError("Per-model request options must name exact model IDs")
        _validate_request_options(options, per_model=True)
    for model, aliases in profile.get("model_aliases", {}).items():
        if (
            not _name(model)
            or not isinstance(aliases, list)
            or not all(_name(alias) for alias in aliases)
            or len(set(aliases)) != len(aliases)
            or model in aliases
        ):
            raise ValueError("Model aliases must be explicit, unique model ID lists")
    if "native_grader" in profile:
        validate_native_grader(profile["native_grader"])
    if delivery_review_schema.CONFIG_KEY in profile:
        _review_schema_models(profile[delivery_review_schema.CONFIG_KEY])
    return profile


def load_profile(path):
    raw = Path(path).read_bytes()
    return raw, parse_profile(raw)


def apply_profile(config, profile):
    """Validate against the prepared role set, then apply atomically in memory."""
    result = deepcopy(config)
    if not set(profile.get("roles", {})) <= set(result["roles"]) or not set(
        profile.get("role_max_tokens", {})
    ) <= set(result["validation_role_limits"]["role_max_tokens"]):
        raise ValueError("Model profile names a role unavailable in this run")
    result["roles"].update(profile.get("roles", {}))
    schema_key = delivery_review_schema.CONFIG_KEY
    result.pop(schema_key, None)
    if schema_key in profile:
        result[schema_key] = deepcopy(profile[schema_key])
        result[schema_key] = delivery_review_schema_models(result)
    models = set(result["roles"].values())
    result.pop("native_grader", None)
    if "native_grader" in profile:
        if result.get("validation_benchmark") != "workbuddy":
            raise ValueError("native_grader is supported only for WorkBuddy validation")
        result["native_grader"] = validate_native_grader(profile["native_grader"])
        models.add(result["native_grader"]["model"])
    aliases = profile.get("model_aliases", {})
    if not set(aliases) <= models:
        raise ValueError("Aliases must name a model actually requested by this run")
    seen = set()
    for values in aliases.values():
        if set(values) & (models | seen):
            raise ValueError(
                "Model aliases must not conflate configured models or alias owners"
            )
        seen.update(values)
    result["model_aliases"] = deepcopy(aliases)
    options = profile.get("role_request_options_by_model", {})
    if not set(options) <= set(result["roles"].values()):
        raise ValueError(
            "Request options must name a model actually requested by this run"
        )
    result["role_request_options_by_model"] = {
        model: _validate_request_options(value, per_model=True)
        for model, value in options.items()
    }
    result["validation_role_limits"]["role_max_tokens"].update(
        profile.get("role_max_tokens", {})
    )
    for key, value in profile.get("concurrency", {}).items():
        result[key] = value
    # Keep the frozen launcher's existing transport ceiling, including WorkBuddy's 8.
    if any(
        result[key] > result.get("max_model_concurrency", 8)
        for key in ("model_concurrency", "per_model_concurrency")
    ):
        raise ValueError("Model profile exceeds the frozen transport ceiling")
    if result.get("validation_benchmark") == "workbuddy":
        result["official_protocol"] = workbuddy_official_protocol(result)
    config.pop("native_grader", None)
    config.pop(schema_key, None)
    config.update(result)


def effective_settings(config):
    return deepcopy(
        {
            "roles": config["roles"],
            "role_max_tokens": config["validation_role_limits"]["role_max_tokens"],
            "model_aliases": config.get("model_aliases", {}),
            "role_request_options": config.get("role_request_options", {}),
            "role_request_options_by_model": config.get(
                "role_request_options_by_model", {}
            ),
            "concurrency": {
                "model_concurrency": config["model_concurrency"],
                "per_model_concurrency": config["per_model_concurrency"],
            },
            **(
                {
                    "native_grader": config["native_grader"],
                    "official_protocol": config["official_protocol"],
                }
                if "native_grader" in config
                else {}
            ),
            **(
                {
                    delivery_review_schema.CONFIG_KEY: delivery_review_schema_models(
                        config
                    ),
                    "delivery_review_json_schema": delivery_review_schema.binding(),
                }
                if delivery_review_schema.CONFIG_KEY in config
                else {}
            ),
        }
    )


def freeze_profile(config, source, raw):
    root = Path(config["output"])
    path = root / FILENAME
    with path.open("xb") as stream:
        stream.write(raw)
    config["model_config"] = {
        "schema": SCHEMA,
        "path": FILENAME,
        "source_path": str(Path(source).resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "effective": effective_settings(config),
    }


def verify_profile(config):
    """Require the frozen bytes and effective settings before starting a gateway."""
    root = Path(config["output"])
    manifest = read(root / "manifest.json")
    if "model_config" not in config:
        if (
            "model_config" in manifest
            or (root / FILENAME).exists()
            or "native_grader" in config
            or delivery_review_schema.CONFIG_KEY in config
        ):
            raise ValueError("Frozen model profile binding is missing")
        return
    binding = config["model_config"]
    if (
        not isinstance(binding, dict)
        or set(binding) != {"schema", "path", "source_path", "sha256", "effective"}
        or binding["schema"] != SCHEMA
        or binding["path"] != FILENAME
        or binding != manifest.get("model_config")
    ):
        raise ValueError("Frozen model profile manifest binding changed")
    path = root / FILENAME
    if path.is_symlink() or not path.is_file():
        raise ValueError("Frozen model profile must be a regular file")
    raw, profile = load_profile(path)
    if hashlib.sha256(raw).hexdigest() != binding["sha256"]:
        raise ValueError("Frozen model profile bytes changed")
    if effective_settings(config) != binding["effective"]:
        raise ValueError("Frozen model profile effective settings changed")
    checked = deepcopy(config)
    apply_profile(checked, profile)
    if effective_settings(checked) != effective_settings(config):
        raise ValueError("Frozen model profile does not match effective settings")
    if any(
        (root / name).exists() for name in ("role_models.json", "transport_limits.json")
    ):
        raise ValueError(
            "Model-profile runs cannot apply unfrozen runtime model/concurrency overrides"
        )
