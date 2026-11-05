"""Fresh campaign role overrides, applied before derived attack seats."""

from reward_engine.adapters.jiuwenswarm.config import role_mapping


def validate_model(model):
    if not isinstance(model, str) or not model or model.strip() != model or any(c.isspace() for c in model) or "/" in model or "\\" in model:
        raise ValueError("Model must be an exact model ID")


def configure_roles(defaults, protocol, entries=None):
    roles = dict(defaults)
    allowed = set(role_mapping(defaults, protocol)) & set(defaults)
    seen = set()
    for entry in entries or ():
        role, separator, model = entry.partition("=")
        if not separator or role not in allowed or role in seen:
            raise ValueError("Unknown, duplicate or derived role override: " + role)
        validate_model(model)
        roles[role] = model
        seen.add(role)
    return role_mapping(roles, protocol)
