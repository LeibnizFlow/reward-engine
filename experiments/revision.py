"""Explicit implementation revisions of a frozen experiment, preserving its evidence.

The experiment identity and completed stages stay fixed. Each revision freezes
new code and internal role transport settings, archives prior task records, and
binds its parent configuration. Actor, task, runtime and scoring settings cannot
change. An explicit storage reserve amendment is recorded separately. No actor
stage is reset or reconstructed by this operation.
"""

import argparse
import fcntl
import shutil
from pathlib import Path
from uuid import uuid4

from reward_engine.adapters.evidence_snapshot import EvidenceSnapshots
from reward_engine.adapters.integrity import tree_hashes as stream_tree_hashes
from reward_engine.adapters.integrity import (
    verification_cache,
    verification_detail,
    verification_progress,
)
from reward_engine.adapters.workbuddy.io import digest, file_hash, read, write
from reward_engine.adapters.workbuddy.models import (
    _role_request_options,
    role_contract_retries,
    role_stream,
    validate_role_budgets,
)
from reward_engine.core.outcome.authored_json import LIMITS

REPO = Path(__file__).resolve().parents[1]
ROLE_TRANSPORT_SETTINGS = {
    "role_transport_recovery": True,
    "role_response_format": "json_object",
    "role_request_options": {"thinking": {"type": "disabled"}, "enable_thinking": False},
}
ROLE_BOUNDED_SETTINGS = {**ROLE_TRANSPORT_SETTINGS, "role_bounded_stage": True,
                         "role_stage_timeout_s": 1800}
ROLE_SETTINGS = {**ROLE_BOUNDED_SETTINGS, "role_stream": False, "role_protocol_edits": True}
ROLE_PROTOCOL_SETTINGS = {"role_output_protocol": "model_authored_json_v2",
                          "role_recovery_policy": "strict_bounded_v2",
                          "failure_policy": "pause_repeated_v1"}
ROLE_OVERRIDE_FIELDS = {"role_request_options_by_model", "role_max_tokens", "role_stream_by_model", "role_input_projection", "role_contract_policy", "role_contract_retries"}
REVISION_FIELDS = {"code", "identity", "implementation_revision", "checkpoint_config_identity",
                   "minimum_free_gib", "minimum_free_bytes",
                   *ROLE_SETTINGS, *ROLE_OVERRIDE_FIELDS, *ROLE_PROTOCOL_SETTINGS}


def storage_settings(minimum_free_gib):
    if type(minimum_free_gib) is not int or not 5 <= minimum_free_gib <= 1024:
        raise ValueError("minimum_free_gib must be an integer in 5..1024")
    return {"minimum_free_gib": minimum_free_gib,
            "minimum_free_bytes": minimum_free_gib * 1024**3}


def validate_role_settings(settings, config):
    if not isinstance(settings, dict) or set(settings) - ROLE_SETTINGS.keys() - ROLE_OVERRIDE_FIELDS - ROLE_PROTOCOL_SETTINGS.keys():
        raise ValueError("Unsupported revision role settings")
    if any(settings.get(k) != v for k, v in ROLE_SETTINGS.items()):
        raise ValueError("Revision role settings changed")
    protocol = {k: settings[k] for k in ROLE_PROTOCOL_SETTINGS if k in settings}
    expected_protocol = {
        **ROLE_PROTOCOL_SETTINGS,
        "failure_policy": protocol.get("failure_policy", ROLE_PROTOCOL_SETTINGS["failure_policy"]),
    }
    if protocol and (
        protocol != expected_protocol
        or protocol.get("failure_policy") not in ("pause_repeated_v1", "isolate_tasks_v1")
    ):
        raise ValueError("Authored JSON revision requires all fixed protocol settings")
    if "role_input_projection" in settings and (
        settings["role_input_projection"] != "canonical_sources_v1" or protocol != expected_protocol
    ):
        raise ValueError("Canonical sources require the fixed authored JSON revision protocol")
    effective = {**config, **settings}
    if effective.get("role_contract_policy") is not None and (
        effective["role_contract_policy"] != "canonical_recovery_v1"
        or effective.get("role_input_projection") != "canonical_sources_v1"
        or effective.get("role_output_protocol") != "model_authored_json_v2"
    ):
        raise ValueError("Canonical recovery requires canonical sources and authored JSON")
    _role_request_options(effective)
    role_stream(effective)
    role_contract_retries(effective)
    validate_role_budgets(effective.get("role_max_tokens", {}), effective["roles"])
    return settings


def checkpoint_identity(config):
    return config.get("checkpoint_config_identity", config["identity"])


def config_path(config):
    return Path(config.get("implementation_revision", config["output"])) / "config.json"


def tree_hashes(root, *, ignore_bytecode=True):
    return stream_tree_hashes(root, file_hash, ignore_bytecode=ignore_bytecode)


def implementation_hashes(root):
    """Bind both maintained packages, separately from runtime _local outputs."""
    return {f"{package}/{name}": value
            for package in ("reward_engine", "experiments")
            for name, value in tree_hashes(Path(root) / package).items()}


def verifier(config):
    if "benchmark_root" in config:
        from experiments.claw_eval.opencode.campaign import verify_freeze
        return verify_freeze
    from experiments.workbuddy.experiment import verify
    return verify


def verify_revision(config, verify_parent):
    with verification_cache(Path(config['output']) / 'verification_cache.sqlite3'):
        return _verify_revision(config, verify_parent)


def verify_execution(config, verify_base):
    """Check executable inputs without walking archival copies of prior task state.

    Live checkpoint and native artifact checks still run at their reuse boundary.
    Preparation and final acceptance use verify_revision for the full audit.
    """
    verification_detail(archived_evidence="deferred_to_full_audit")
    if not config.get("implementation_revision"):
        return verify_base(config)
    with verification_cache(Path(config['output']) / 'verification_cache.sqlite3'):
        return _verify_revision(
            config, lambda parent: verify_execution(parent, verify_base),
            verify_archives=False,
        )


def _verify_revision(config, verify_parent, *, verify_archives=True):
    directory = Path(config["implementation_revision"])
    if directory.resolve().parent != Path(config["output"]).resolve() / "implementation_revisions":
        raise ValueError("Revision must belong to the experiment")
    manifest = read(directory / "manifest.json")
    parent_path = Path(manifest["parent_config"])
    if parent_path.resolve() == config_path(config).resolve():
        raise ValueError("Revision parent cycle")
    if file_hash(parent_path) != manifest["parent_config_sha256"]:
        raise ValueError("Revision parent configuration changed")
    parent = read(parent_path)
    resource_settings = manifest.get("resource_settings", {})
    if resource_settings and resource_settings != storage_settings(resource_settings.get("minimum_free_gib")):
        raise ValueError("Revision storage settings changed")
    if any(config.get(key) != resource_settings.get(key, parent.get(key))
           for key in ("minimum_free_gib", "minimum_free_bytes")):
        raise ValueError("Revision storage settings changed")
    if manifest["depth"] != (read(Path(parent["implementation_revision"]) / "manifest.json")["depth"] + 1
                             if parent.get("implementation_revision") else 1) or manifest["depth"] > 32:
        raise ValueError("Invalid revision chain depth")
    verify_parent(parent)
    verification_detail(revision=str(directory), revision_depth=manifest["depth"])
    if {k: v for k, v in config.items() if k not in REVISION_FIELDS} != {
        k: v for k, v in parent.items() if k not in REVISION_FIELDS
    } or checkpoint_identity(config) != checkpoint_identity(parent):
        raise ValueError("Implementation revision changed task, actor, runtime or scoring settings")
    settings = manifest.get("role_settings", ROLE_TRANSPORT_SETTINGS)
    if manifest.get("schema") in {"reward_engine.implementation_revision.v2", "reward_engine.implementation_revision.v3"}:
        validate_role_settings(settings, config)
        if manifest["schema"].endswith(".v3") and manifest.get("role_recovery_limits") != LIMITS:
            raise ValueError("Revision recovery limits changed")
    elif settings not in (ROLE_TRANSPORT_SETTINGS, ROLE_BOUNDED_SETTINGS, ROLE_SETTINGS):
        raise ValueError("Revision role settings changed")
    if any(
        config.get(k) != v for k, v in settings.items()
    ) or any(config.get(k) != parent.get(k)
             for k in (ROLE_SETTINGS.keys() | ROLE_OVERRIDE_FIELDS | ROLE_PROTOCOL_SETTINGS.keys()) - settings.keys()):
        raise ValueError("Revision role settings changed")
    if Path(config["code"]) != directory / "code":
        raise ValueError("Revision code location changed")
    if digest({k: v for k, v in config.items() if k != "identity"}) != config["identity"]:
        raise ValueError("Revision configuration identity changed")
    if config["identity"] != manifest["config_identity"]:
        raise ValueError("Revision manifest identity changed")
    if implementation_hashes(directory / "code") != manifest["code_hashes"]:
        raise ValueError("Revision implementation bytes changed")
    if verify_archives and stream_tree_hashes(directory / "previous_state", file_hash,
                          expected_files=len(manifest["previous_state_hashes"])) != manifest["previous_state_hashes"]:
        raise ValueError("Archived task evidence changed")


def prepare(output, *, parent_revision=None, role_settings=None, minimum_free_gib=None):
    with verification_progress(Path(output) / "verification-progress.json", "prepare_revision"):
        with verification_cache(Path(output) / 'verification_cache.sqlite3') as cache:
            return _prepare(output, parent_revision=parent_revision, role_settings=role_settings,
                            cache=cache, minimum_free_gib=minimum_free_gib)


def _prepare(output, *, parent_revision=None, role_settings=None, cache=None, minimum_free_gib=None):
    resources = storage_settings(minimum_free_gib) if minimum_free_gib is not None else {}
    output = Path(output).resolve()
    parent_path = (Path(parent_revision).resolve() if parent_revision else output) / "config.json"
    parent = read(parent_path)
    if Path(parent["output"]) != output or parent.get("generation_protocol") != "three_b_fixed_baselines_v1":
        raise ValueError("Explicit revisions require the same three B experiment")
    with (output / "campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # The final recursive verification checks the complete parent chain once,
        # after the snapshot has been built and before this revision is returned.
        # No earlier successful check is cached or trusted across mutation.
        overrides = {} if role_settings is None else role_settings
        if not isinstance(overrides, dict) or set(overrides) - ROLE_OVERRIDE_FIELDS - ROLE_PROTOCOL_SETTINGS.keys():
            raise ValueError("Only internal role output budgets, model options and transport may be overridden")
        settings = validate_role_settings({
            **ROLE_SETTINGS,
            **{k: parent[k] for k in ROLE_OVERRIDE_FIELDS | ROLE_PROTOCOL_SETTINGS.keys() if k in parent},
            **overrides,
        }, parent)
        directory = output / "implementation_revisions" / uuid4().hex
        directory.mkdir(parents=True)
        verification_detail(stage="copy_implementation", revision=str(directory))
        for name in ("reward_engine", "experiments"):
            shutil.copytree(REPO / name, directory / "code" / name,
                            ignore=shutil.ignore_patterns("__pycache__"))
        archive = directory / "previous_state"
        archive.mkdir()
        verification_detail(stage="copy_previous_state", revision=str(directory))
        copied = 0
        archive_hashes = {}
        snapshots = EvidenceSnapshots(output / 'evidence_objects', cache)

        def copy_evidence(source, destination):
            nonlocal copied
            archive_hashes[str(Path(destination).relative_to(archive))] = snapshots.copy(
                source, destination)
            copied += 1
            verification_detail(copied_files=copied, current_path=str(source))
            return destination

        if (output / "tasks").exists():
            if (output / 'tasks').is_symlink():
                raise ValueError('Frozen tree cannot contain symlinks')
            shutil.copytree(output / "tasks", archive / "tasks", copy_function=copy_evidence,
                            symlinks=True)
        for name in ("status.json", "process.json", "role_models.json", "role_limits.json"):
            if (output / name).exists():
                copy_evidence(output / name, archive / name)
        parent_window = Path(parent.get("implementation_revision", output)) / "failure_window.json"
        if parent_window.exists():
            copy_evidence(parent_window, archive / "failure_window.json")
        config = {**parent, **settings, **resources, "code": str(directory / "code"),
                  "implementation_revision": str(directory),
                  "checkpoint_config_identity": checkpoint_identity(parent)}
        config["identity"] = digest({k: v for k, v in config.items() if k != "identity"})
        verification_detail(stage="write_revision_manifest")
        write(directory / "config.json", config)
        write(directory / "manifest.json", {
            "schema": ("reward_engine.implementation_revision.v3" if set(settings) & ROLE_PROTOCOL_SETTINGS.keys() else "reward_engine.implementation_revision.v2"
                       if set(settings) & ROLE_OVERRIDE_FIELDS else "reward_engine.implementation_revision.v1"),
            "parent_config": str(parent_path), "parent_config_sha256": file_hash(parent_path),
            "depth": read(Path(parent["implementation_revision"]) / "manifest.json")["depth"] + 1
                     if parent.get("implementation_revision") else 1,
            "config_identity": config["identity"], "code_hashes": implementation_hashes(directory / "code"),
            "role_settings": settings,
            **({"resource_settings": resources} if resources else {}),
            **({"role_recovery_limits": LIMITS} if set(settings) & ROLE_PROTOCOL_SETTINGS.keys() else {}),
            "previous_state_hashes": dict(sorted(archive_hashes.items())),
        })
        verification_detail(stage="verify_revision_chain")
        verify_revision(config, verifier(parent))
        return directory


def load(output, revision=None):
    config = read((Path(revision).resolve() if revision else Path(output)) / "config.json")
    if Path(config["output"]).resolve() != Path(output).resolve():
        raise ValueError("Revision belongs to another experiment")
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent-revision", type=Path)
    parser.add_argument("--minimum-free-gib", type=int, help="Explicit operational storage reserve, recorded separately from model/task settings")
    parser.add_argument("--model-authored-json", action="store_true", help="Explicitly enable compact complete drafts, strict recovery and persistent failure pause")
    parser.add_argument("--role-settings", type=Path,
                        help="JSON object of internal role budgets, model transport options, role_input_projection and role_contract_policy overrides")
    args = parser.parse_args()
    print(prepare(args.output, parent_revision=args.parent_revision,
                  minimum_free_gib=args.minimum_free_gib,
                  role_settings={**(read(args.role_settings) if args.role_settings else {}),
                                 **(ROLE_PROTOCOL_SETTINGS if args.model_authored_json else {})}))


if __name__ == "__main__":
    main()
