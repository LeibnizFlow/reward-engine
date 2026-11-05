"""Explicit code upgrades keep completed stages and reject changed experiment inputs."""

from copy import deepcopy

import pytest

from experiments import revision
from reward_engine.adapters.workbuddy.io import digest, read, write
from reward_engine.core.outcome.three_b import Stages


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    for name in ("reward_engine", "experiments"):
        path = repo / name
        path.mkdir(parents=True)
        (path / "__init__.py").write_text("# synthetic code\n")
    monkeypatch.setattr(revision, "REPO", repo)
    run = tmp_path / "run"
    cfg = {"output": str(run), "code": str(run / "code"),
           "generation_protocol": "three_b_fixed_baselines_v1", "roles": {"actor": "same"},
           "jiuwenswarm_mode": "agent.code.normal", "runtime_identity": {"bundle": "fixed"},
           "image": "fixed-image", "task_ids": ["synthetic"], "rounds": 1}
    cfg["identity"] = digest(cfg)
    write(run / "config.json", cfg)

    def verify(config):
        if config.get("implementation_revision"):
            return revision.verify_revision(config, verify)
        assert config == cfg

    monkeypatch.setattr(revision, "verifier", lambda _: verify)
    return run, cfg, verify


@pytest.mark.asyncio
async def test_revision_preserves_old_evidence_and_reuses_completed_stage(frozen):
    run, cfg, verify = frozen
    records = {}
    async def actor():
        return {"candidate": "synthetic frozen delivery"}
    stages = Stages({"config": cfg["identity"]}, lambda k, v: records.__setitem__(k, v), records.get)
    await stages.run("seed_0", {"task": "synthetic"}, actor, actor=True)
    write(run / "tasks/synthetic/seed_0/stage.json", records["seed_0/stage"])
    before = (run / "config.json").read_bytes()
    directory = revision.prepare(run)
    updated = read(directory / "config.json")
    verify(updated)
    assert (run / "config.json").read_bytes() == before
    assert read(directory / "previous_state/tasks/synthetic/seed_0/stage.json") == records["seed_0/stage"]
    async def forbidden():
        pytest.fail("Completed actor must not be redrawn after a code revision")
    resumed = Stages({"config": revision.checkpoint_identity(updated)},
                     lambda *_: None, records.get, updated["identity"])
    assert await resumed.run("seed_0", {"task": "synthetic"}, forbidden, actor=True) == {"candidate": "synthetic frozen delivery"}


@pytest.mark.parametrize("field,value", [
    ("jiuwenswarm_mode", "agent.work.normal"), ("runtime_identity", {"bundle": "different"}),
    ("image", "changed-image"), ("roles", {"actor": "different"}), ("task_ids", ["other"]),
])
def test_revision_refuses_runtime_or_task_changes_even_with_recomputed_identity(frozen, field, value):
    run, _, verify = frozen
    directory = revision.prepare(run)
    cfg = read(directory / "config.json")
    cfg[field] = value
    cfg["identity"] = digest({k: v for k, v in cfg.items() if k != "identity"})
    manifest = read(directory / "manifest.json")
    manifest["config_identity"] = cfg["identity"]
    write(directory / "manifest.json", manifest)
    with pytest.raises(ValueError, match="changed task, actor, runtime"):
        verify(cfg)


def test_revision_archives_cannot_be_edited(frozen):
    run, _, verify = frozen
    write(run / "tasks/synthetic/failure.json", {"reason": "retained"})
    directory = revision.prepare(run)
    write(directory / "previous_state/tasks/synthetic/failure.json", {"reason": "erased"})
    with pytest.raises(ValueError, match="Archived task evidence"):
        verify(read(directory / "config.json"))


def test_storage_amendment_is_explicit_bound_and_preserves_checkpoint(frozen):
    run, original, verify = frozen
    directory = revision.prepare(run, minimum_free_gib=16)
    updated = read(directory / "config.json")
    verify(updated)
    assert updated["minimum_free_bytes"] == 16 * 1024**3
    assert read(run / "config.json") == original
    assert revision.checkpoint_identity(updated) == original["identity"]
    updated["minimum_free_gib"] = 15
    with pytest.raises(ValueError, match="storage settings"):
        verify(updated)


def test_storage_change_without_explicit_manifest_is_rejected(frozen):
    run, _, verify = frozen
    directory = revision.prepare(run)
    updated = read(directory / "config.json")
    updated.update(minimum_free_gib=16, minimum_free_bytes=16 * 1024**3)
    with pytest.raises(ValueError, match="storage settings"):
        verify(updated)


def test_revisions_share_read_only_objects_but_never_live_task_inodes(frozen):
    run, _, verify = frozen
    source = run / 'tasks/synthetic/receipt.json'
    write(source, {'value': 'original'})
    first = revision.prepare(run)
    second = revision.prepare(run, parent_revision=first)
    relative = 'previous_state/tasks/synthetic/receipt.json'
    a, b = first / relative, second / relative
    assert a.stat().st_ino == b.stat().st_ino
    assert a.stat().st_ino != source.stat().st_ino
    assert not a.stat().st_mode & 0o222
    source.write_text('mutable task changed in place')
    assert read(a) == {'value': 'original'}
    verify(read(second / 'config.json'))


def test_revision_verifies_when_archive_object_link_limit_is_reached(frozen, monkeypatch):
    import errno
    import os
    from pathlib import Path

    run, _, verify = frozen
    source = run / 'tasks/synthetic/receipt.json'
    write(source, {'value': 'retained evidence'})
    first = revision.prepare(run)
    relative = 'previous_state/tasks/synthetic/receipt.json'
    before = (first / relative).read_bytes()
    link = os.link

    def saturated(source, destination, **kwargs):
        if 'previous_state' in Path(destination).parts:
            raise OSError(errno.EMLINK, 'Synthetic object link limit')
        return link(source, destination, **kwargs)

    monkeypatch.setattr(os, 'link', saturated)
    second = revision.prepare(run, parent_revision=first)
    verify(read(second / 'config.json'))
    assert (first / relative).read_bytes() == (second / relative).read_bytes() == before
    assert (first / relative).stat().st_ino != (second / relative).stat().st_ino
    assert not (second / relative).stat().st_mode & 0o222


def test_snapshot_does_not_follow_task_directory_symlinks(frozen):
    run, _, _ = frozen
    write(run / 'outside/receipt.json', {'value': 'outside'})
    (run / 'tasks').mkdir()
    (run / 'tasks/alias').symlink_to(run / 'outside', target_is_directory=True)
    with pytest.raises(ValueError, match='symlinks'):
        revision.prepare(run)


def test_warm_parent_verification_rejects_corruption_with_restored_timestamp(frozen):
    import os

    run, _, verify = frozen
    write(run / 'tasks/synthetic/receipt.json', {'value': 'old'})
    first = revision.prepare(run)
    second = revision.prepare(run, parent_revision=first)
    config = read(second / 'config.json')
    verify(config)
    path = first / 'previous_state/tasks/synthetic/receipt.json'
    stamp = path.stat()
    path.chmod(0o644)
    path.write_text(path.read_text().replace('old', 'new'))
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    with pytest.raises(ValueError, match='Archived task evidence'):
        verify(config)


def test_preparation_verifies_parent_chain_once(frozen, monkeypatch):
    run, _, verify = frozen
    parent = revision.prepare(run)
    calls = []

    def counted(config):
        calls.append(config["identity"])
        return verify(config)

    monkeypatch.setattr(revision, "verifier", lambda _: counted)
    revision.prepare(run, parent_revision=parent)
    assert calls == [read(parent / "config.json")["identity"]]


def test_claw_startup_defers_archive_walk_but_full_audit_still_detects_changes(frozen, monkeypatch):
    from experiments.claw_eval.opencode import campaign

    run, _, verify = frozen
    write(run / "tasks/synthetic/receipt.json", {"evidence": "original"})
    first = revision.prepare(run)
    second = revision.prepare(run, parent_revision=first)
    config = read(second / "config.json")
    hashing = revision.stream_tree_hashes
    visited, served = [], []

    def counted(root, *args, **kwargs):
        visited.append(root)
        return hashing(root, *args, **kwargs)

    async def serve(actual, *args, **kwargs):
        served.append(actual)

    monkeypatch.setattr(revision, "stream_tree_hashes", counted)
    monkeypatch.setattr(campaign, "verify_freeze", verify)
    monkeypatch.setattr(campaign, "__file__", str(second / "code/experiments/claw_eval/opencode/campaign.py"))
    monkeypatch.setattr(campaign, "serve", serve)
    monkeypatch.setattr(campaign.sys, "argv", ["campaign", "run", "--output", str(run), "--revision", str(second)])
    campaign.main()
    assert served == [config]
    assert not any(path.name == "previous_state" for path in visited)
    assert first / "code/reward_engine" in visited
    assert second / "code/reward_engine" in visited
    progress = read(run / "verification-progress.json")
    assert progress["archived_evidence"] == "deferred_to_full_audit"
    write(first / "previous_state/tasks/synthetic/receipt.json", {"evidence": "changed"})
    with pytest.raises(ValueError, match="Archived task evidence changed"):
        verify(config)


@pytest.mark.parametrize("changed", ["current_code", "parent_code", "parent_config", "task"])
def test_execution_verification_retains_code_and_lineage_guards(frozen, changed):
    run, _, verify = frozen
    first = revision.prepare(run)
    second = revision.prepare(run, parent_revision=first)
    config = read(second / "config.json")
    revision.verify_execution(config, verify)
    if changed.endswith("code"):
        directory = second if changed == "current_code" else first
        (directory / "code/reward_engine/__init__.py").write_text("# altered code\n")
        message = "implementation bytes"
    elif changed == "parent_config":
        write(first / "config.json", {**read(first / "config.json"), "rounds": 2})
        message = "parent configuration"
    else:
        config["task_ids"] = ["altered task"]
        config["identity"] = digest({k: v for k, v in config.items() if k != "identity"})
        manifest = read(second / "manifest.json")
        write(second / "manifest.json", {**manifest, "config_identity": config["identity"]})
        message = "changed task, actor, runtime"
    with pytest.raises(ValueError, match=message):
        revision.verify_execution(config, verify)


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", ["inputs", "content"])
async def test_execution_verification_keeps_live_checkpoint_reuse_checks(frozen, changed):
    run, config, verify = frozen
    records = {}
    stages = Stages({"config": config["identity"]}, lambda k, v: records.__setitem__(k, v), records.get)

    async def execute():
        return {"delivery": "original"}

    await stages.run("seed_0", {"task": "synthetic"}, execute, actor=True)
    directory = revision.prepare(run)
    updated = read(directory / "config.json")
    revision.verify_execution(updated, verify)
    resumed = Stages({"config": revision.checkpoint_identity(updated)},
                     lambda *_: None, records.get, updated["identity"])
    inputs = {"task": "synthetic"}
    if changed == "inputs":
        inputs["task"] = "altered"
    else:
        records["seed_0/stage"]["value"]["delivery"] = "altered"

    async def forbidden():
        pytest.fail("Invalid checkpoint must not cause another Actor execution")

    with pytest.raises(ValueError, match=f"Checkpoint {changed} changed"):
        await resumed.run("seed_0", inputs, forbidden, actor=True)


def test_hashing_does_not_materialize_the_entire_tree(tmp_path, monkeypatch):
    (tmp_path / "evidence").write_text("synthetic receipt")

    def forbidden(*args, **kwargs):
        pytest.fail("Full recursive glob delays hashing and repeats metadata lookups")

    monkeypatch.setattr(type(tmp_path), "rglob", forbidden)
    assert revision.tree_hashes(tmp_path) == {
        "evidence": revision.file_hash(tmp_path / "evidence")
    }


def test_archive_is_read_once_after_copy_and_copy_corruption_is_detected(frozen, monkeypatch):
    run, _, _ = frozen
    write(run / "tasks/synthetic/receipt.json", {"synthetic": "original"})
    calls = []
    hashing = revision.stream_tree_hashes

    def counted(root, *args, **kwargs):
        if root.name == "previous_state":
            calls.append(root)
        return hashing(root, *args, **kwargs)

    monkeypatch.setattr(revision, "stream_tree_hashes", counted)
    revision.prepare(run)
    assert len(calls) == 1
    snapshot = revision.EvidenceSnapshots.copy

    def corrupt(self, source, destination):
        digest = snapshot(self, source, destination)
        if str(destination).endswith("previous_state/tasks/synthetic/receipt.json"):
            write(destination, {"synthetic": "corrupted copy"})
        return digest

    monkeypatch.setattr(revision.EvidenceSnapshots, "copy", corrupt)
    with pytest.raises(ValueError, match="Archived task evidence changed"):
        revision.prepare(run)


def test_parent_mutation_during_preparation_is_not_hidden_by_earlier_check(frozen, monkeypatch):
    run, _, _ = frozen
    write(run / "tasks/synthetic/receipt.json", {"synthetic": "original"})
    parent = revision.prepare(run)
    snapshot = revision.EvidenceSnapshots.copy

    def mutate(self, source, destination):
        digest = snapshot(self, source, destination)
        if str(destination).endswith("previous_state/tasks/synthetic/receipt.json"):
            write(parent / "previous_state/tasks/synthetic/receipt.json", {"synthetic": "changed parent"})
        return digest

    monkeypatch.setattr(revision.EvidenceSnapshots, "copy", mutate)
    with pytest.raises(ValueError, match="Archived task evidence changed"):
        revision.prepare(run, parent_revision=parent)


def test_revision_verifies_code_separately_from_generated_local_reports(frozen):
    run, _, verify = frozen
    directory = revision.prepare(run)
    config = read(directory / "config.json")
    write(directory / "code/_local/reports/result.json", {"status": "failed"})
    verify(config)
    (directory / "code/reward_engine/changed.py").write_text("changed = True\n")
    with pytest.raises(ValueError, match="implementation bytes"):
        verify(config)


def test_revision_chain_preserves_prior_transport_only_policy(frozen):
    run, _, verify = frozen
    first = revision.prepare(run)
    config = read(first / "config.json")
    for key in revision.ROLE_SETTINGS.keys() - revision.ROLE_TRANSPORT_SETTINGS.keys():
        config.pop(key)
    config["identity"] = digest({k: v for k, v in config.items() if k != "identity"})
    write(first / "config.json", config)
    manifest = read(first / "manifest.json")
    manifest.pop("role_settings")
    manifest["config_identity"] = config["identity"]
    write(first / "manifest.json", manifest)
    verify(config)
    second = revision.prepare(run, parent_revision=first)
    updated = read(second / "config.json")
    assert updated["role_bounded_stage"] is True
    assert revision.checkpoint_identity(updated) == revision.checkpoint_identity(config)
    verify(updated)


@pytest.mark.asyncio
async def test_revision_does_not_reset_uncertain_actor(frozen):
    run, cfg, _ = frozen
    records = {}
    async def failed():
        raise RuntimeError("ambiguous termination")
    stages = Stages({"config": cfg["identity"]}, lambda k, v: records.__setitem__(k, deepcopy(v)), records.get)
    with pytest.raises(RuntimeError, match="ambiguous"):
        await stages.run("B1", {}, failed, actor=True)
    updated = read(revision.prepare(run) / "config.json")
    resumed = Stages({"config": revision.checkpoint_identity(updated)}, lambda *_: None, records.get)
    with pytest.raises(RuntimeError, match="Uncertain actor"):
        await resumed.run("B1", {}, failed, actor=True)


def test_revision_chain_preserves_prior_bounded_policy(frozen):
    run, _, verify = frozen
    first = revision.prepare(run)
    config = read(first / "config.json")
    for key in revision.ROLE_SETTINGS.keys() - revision.ROLE_BOUNDED_SETTINGS.keys():
        config.pop(key)
    config["identity"] = digest({k: v for k, v in config.items() if k != "identity"})
    write(first / "config.json", config)
    manifest = read(first / "manifest.json")
    manifest["role_settings"] = revision.ROLE_BOUNDED_SETTINGS
    manifest["config_identity"] = config["identity"]
    write(first / "manifest.json", manifest)
    verify(config)
    second = revision.prepare(run, parent_revision=first)
    updated = read(second / "config.json")
    assert updated["role_stream"] is False and updated["role_protocol_edits"] is True
    assert revision.checkpoint_identity(updated) == revision.checkpoint_identity(config)
    verify(updated)


def test_explicit_internal_role_settings_are_frozen_and_inherited(frozen):
    run, _, verify = frozen
    settings = {"role_request_options_by_model": {"same": {}}, "role_max_tokens": {},
                "role_stream_by_model": {"same": True}}
    first = revision.prepare(run, role_settings=settings)
    cfg = read(first / "config.json")
    assert cfg["role_request_options_by_model"] == {"same": {}}
    verify(cfg)
    second = revision.prepare(run, parent_revision=first)
    assert read(second / "config.json")["role_request_options_by_model"] == {"same": {}}
    assert read(second / "config.json")["role_stream_by_model"] == {"same": True}
    verify(read(second / "config.json"))
    cfg["role_request_options_by_model"] = {}
    with pytest.raises(ValueError, match="role settings changed"):
        verify(cfg)


@pytest.mark.parametrize("settings", [
    {"roles": {"actor": "changed"}}, {"role_max_tokens": {"actor": 100}},
    {"role_max_tokens": {"unknown": 100}}, {"role_max_tokens": {"actor": True}},
    {"role_request_options_by_model": {"unknown": {}}},
    {"role_request_options_by_model": {"same": {"model": "other"}}},
    {"role_stream_by_model": {"unknown": True}},
    {"role_stream_by_model": {"same": 1}},
])
def test_revision_refuses_invalid_internal_role_overrides_before_archiving(frozen, settings):
    run, _, _ = frozen
    with pytest.raises(ValueError):
        revision.prepare(run, role_settings=settings)
    assert not (run / "implementation_revisions").exists()


def test_authored_json_requires_explicit_complete_revision_settings(frozen):
    run, _, verify = frozen
    with pytest.raises(ValueError, match="all fixed protocol"):
        revision.prepare(run, role_settings={"role_output_protocol": "model_authored_json_v2"})
    directory = revision.prepare(run, role_settings=revision.ROLE_PROTOCOL_SETTINGS)
    config = read(directory / "config.json")
    verify(config)
    assert read(directory / "manifest.json")["schema"] == "reward_engine.implementation_revision.v3"
    assert {k: config[k] for k in revision.ROLE_PROTOCOL_SETTINGS} == revision.ROLE_PROTOCOL_SETTINGS
    assert config["checkpoint_config_identity"] == read(run / "config.json")["identity"]


def test_isolated_admission_is_explicit_frozen_and_preserves_parent_failures(frozen):
    from reward_engine.adapters.failure_policy import FailureWindow
    run, _, verify = frozen
    parent = revision.prepare(run, role_settings=revision.ROLE_PROTOCOL_SETTINGS)
    parent_config = read(parent / "config.json")
    window = FailureWindow(parent_config)
    for n in range(5):
        window.terminal(str(n), {"status": "failed", "stage": "construction"})
    assert window.paused
    original = read(parent / "failure_window.json")
    updated = revision.prepare(run, parent_revision=parent, role_settings={
        **revision.ROLE_PROTOCOL_SETTINGS, "failure_policy": "isolate_tasks_v1",
    })
    cfg = read(updated / "config.json")
    verify(cfg)
    assert cfg["failure_policy"] == "isolate_tasks_v1"
    assert revision.checkpoint_identity(cfg) == revision.checkpoint_identity(parent_config)
    assert read(parent / "failure_window.json") == original
    assert not FailureWindow(cfg).paused
    inherited = revision.prepare(run, parent_revision=updated)
    verify(read(inherited / "config.json"))
    assert read(inherited / "config.json")["failure_policy"] == "isolate_tasks_v1"


@pytest.mark.parametrize("policy", [None, "canonical_recovery_v1"])
def test_canonical_projection_is_explicit_and_archives_parent_latch(frozen, policy):
    run, cfg, _ = frozen
    from reward_engine.adapters.workbuddy.io import write

    latch = {'paused': True, 'reason': 'Synthetic quorum failure', 'terminal': []}
    write(run / 'failure_window.json', latch)
    directory = revision.prepare(run, role_settings={**revision.ROLE_PROTOCOL_SETTINGS,
                                                    'role_input_projection': 'canonical_sources_v1',
                                                    **({'role_contract_policy': policy} if policy else {})})
    manifest = read(directory / 'manifest.json')
    assert manifest['role_settings']['role_input_projection'] == 'canonical_sources_v1'
    assert read(directory / 'previous_state/failure_window.json') == latch
    assert read(run / 'failure_window.json') == latch
    assert 'role_input_projection' not in cfg
    assert manifest['role_settings'].get('role_contract_policy') == policy


def test_recovery_policy_requires_source_and_output_contracts(frozen):
    run, _, _ = frozen
    with pytest.raises(ValueError, match="requires canonical"):
        revision.prepare(run, role_settings={'role_contract_policy': 'canonical_recovery_v1'})
