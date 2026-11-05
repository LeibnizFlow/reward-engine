"""Per-actor capability registration for the existing model gateway."""

import secrets

from reward_engine.adapters.workbuddy.io import digest, read, write


def register(config, channel, contract):
    from pathlib import Path

    path = Path(config["output"]) / "actor_policies" / (channel + ".json")
    if path.exists():
        raise ValueError("Actor gateway channel already registered")
    token = secrets.token_urlsafe(32)
    write(
        path,
        {
            "policy": contract,
            "policy_sha256": digest(contract),
            "token_sha256": digest(token),
            "model": config["roles"]["actor"],
            "campaign": config["identity"],
        },
    )
    return token


def authorize(config, channel, token):
    from pathlib import Path

    path = Path(config["output"]) / "actor_policies" / (channel + ".json")
    row = read(path)
    if (
        row["token_sha256"] != digest(token)
        or row["campaign"] != config["identity"]
        or row["policy_sha256"] != digest(row["policy"])
    ):
        raise ValueError("Invalid actor gateway capability")
    return row
