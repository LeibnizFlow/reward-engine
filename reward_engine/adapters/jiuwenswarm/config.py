"""Explicit defaults for new experiments; absent fields retain legacy meaning."""

from pathlib import Path

from reward_engine.adapters.workbuddy.io import file_hash, read
from reward_engine.core.outcome.three_b import PROTOCOL, validate_config

from .container import backend, fingerprints
from .policy import MODES


def options(
    *,
    actor_backend="jiuwenswarm",
    jiuwenswarm_mode=MODES[0],
    jiuwenswarm_bundle=None,
    generation_protocol=PROTOCOL,
    rounds=3,
    task_images_manifest=None,
    **constraints,
):
    config = dict(
        actor_backend=actor_backend,
        jiuwenswarm_mode=jiuwenswarm_mode,
        generation_protocol=generation_protocol,
        rounds=rounds,
        **constraints,
    )
    validate_config(config)
    if type(rounds) is not int or not 1 <= rounds <= 3:
        raise ValueError("rounds must be an integer in 1..3")
    if backend(config) == "jiuwenswarm":
        if jiuwenswarm_bundle is None:
            raise ValueError(
                "JiuwenSwarm requires an explicit immutable jiuwenswarm_bundle"
            )
        config["jiuwenswarm_bundle"] = str(
            Path(jiuwenswarm_bundle).expanduser().resolve()
        )
        config["runtime_identity"] = fingerprints(config)
        config["runtime_integrity_check"] = "manifest_only"
    if task_images_manifest is not None:
        path = Path(task_images_manifest).expanduser().resolve()
        document = read(path)
        records = document.get("records", [])
        if not records or len({r["task_id"] for r in records}) != len(records):
            raise ValueError("Task images must have unique task IDs")
        for row in records:
            image = row.get("image_id", "")
            if not image.startswith("sha256:") or len(image) != 71:
                raise ValueError("Task images must use immutable sha256 IDs")
            if not isinstance(row.get("input_sha256"), dict):
                raise ValueError("Task images require input hashes")
        config["task_images_manifest"] = str(path)
        config["task_images_sha256"] = file_hash(path)
    if generation_protocol == PROTOCOL:
        config.update(
            arm_protocol=PROTOCOL,
            hypothesis_enabled=False,
            full_audit=True,
            construction_strategy="counterexample_validated",
            adversarial_review_enabled=True,
            role_transport_recovery=True,
            role_stream=False,
            role_protocol_edits=True,
            role_bounded_stage=True,
            role_stage_timeout_s=1800,
            role_response_format="json_object",
            role_request_options={"thinking": {"type": "disabled"}, "enable_thinking": False},
        )
    return {k: v for k, v in config.items() if k not in constraints}


def verify(config):
    validate_config(config)
    if backend(config) == "jiuwenswarm" and fingerprints(config) != config.get(
        "runtime_identity"
    ):
        raise ValueError("JiuwenSwarm runtime identity changed")
    if (
        config.get("task_images_manifest")
        and file_hash(config["task_images_manifest"]) != config["task_images_sha256"]
    ):
        raise ValueError("Task images manifest changed")


def add_arguments(parser):
    runtime = parser.add_mutually_exclusive_group()
    runtime.add_argument(
        "--runtime", dest="actor_backend", type=runtime_name,
        choices=("jiuwenswarm", "opencode"),
        help="Actor runtime (jiuwen-swarm is also accepted); frozen at preparation",
    )
    runtime.add_argument("--actor-backend", choices=("jiuwenswarm", "opencode"),
                         help="Compatibility spelling of --runtime")
    parser.add_argument("--jiuwenswarm-mode", choices=MODES)
    parser.add_argument("--jiuwenswarm-bundle", type=Path)
    parser.add_argument("--generation-protocol", choices=(PROTOCOL, "legacy_abcd"))
    parser.add_argument("--rounds", type=int, choices=range(1, 4))
    parser.add_argument("--task-images-manifest", type=Path)


def runtime_name(value):
    """Normalize CLI spelling without changing persisted runtime identities."""
    return "jiuwenswarm" if value == "jiuwen-swarm" else value


def arguments(args):
    return {
        key: getattr(args, key)
        for key in (
            "actor_backend",
            "jiuwenswarm_mode",
            "jiuwenswarm_bundle",
            "generation_protocol",
            "rounds",
            "task_images_manifest",
        )
        if getattr(args, key, None) is not None
    }


def role_mapping(roles, protocol):
    if protocol != PROTOCOL:
        return roles
    result = {k: v for k, v in roles.items() if not k.startswith("hypothesis")}
    result["attack_reducer"] = result["jury_reducer"]
    for i in range(1, 4):
        for family in ("attack_member", "attack_reference"):
            result[f"{family}_{i}"] = result[f"jury_member_{i}"]
    return result
