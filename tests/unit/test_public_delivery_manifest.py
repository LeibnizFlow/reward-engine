"""Prospective complete freezes: real prepare/validate, offline filesystem controls."""

import os
import shutil
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from experiments import validate_pipeline
from experiments.workbuddy import experiment
from reward_engine.adapters.workbuddy import public_delivery_audit as auditor
from reward_engine.adapters.workbuddy.io import digest, file_hash, read, write

SCOPE = "all-regular-files-except-__pycache__-v1"
PLUGIN = "reward_engine/legacy/adapters/opencode_plugin.mjs"
RESOURCES = [
    PLUGIN,
    "workbuddy_swarm/transport.ts",
    "workbuddy_swarm/runtime.json",
    "workbuddy_swarm/resource",
    "experiments/validate_pipeline.py",
]


def prepare_public_run(tmp_path, monkeypatch, task, contract):
    """Use the real producer and package; replace only external provisioning."""
    upstream = tmp_path / "local/workbuddy/runner/upstream"
    tasks = upstream / "datasets/wb-bench-office-v1.0/tasks"
    for tid in [task["task_id"], *[f"unused-{i}" for i in range(49)]]:
        path = tasks / tid
        path.mkdir(parents=True)
        (path / "task.toml").write_text("[agent]\ntimeout_sec = 10\n")
        (path / "instruction.md").write_text(task["prompt"])
    (upstream / ".env").write_text(
        "YIBU_API_KEY=offline-fixture\nYIBU_BASE_URL=http://offline.invalid/v1\n"
    )
    binary = upstream / "configs/harnesses/opencode-swarm/docker/rootfs/bin/opencode"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"offline binary fixture")
    external = upstream.parent / "workbuddy_swarm"
    external.mkdir()
    for name, value in {
        "__init__.py": b"# offline external fixture\n",
        "transport.ts": b"export const fixture: boolean = true;\n",
        "runtime.json": b'{"fixture":true}',
        "resource": b"\x00resource\xff",
    }.items():
        (external / name).write_bytes(value)
    for name in ("workbuddy_bench", "workbuddy_bench.runner"):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    provision = ModuleType("workbuddy_bench.runner.prepare_tasks")
    for name in (
        "ensure_apt_mirror",
        "ensure_composite_verifier_contract",
        "ensure_host_gateway_compose",
        "ensure_task_user",
    ):
        setattr(provision, name, Mock())
    monkeypatch.setitem(sys.modules, provision.__name__, provision)
    listener = MagicMock()
    listener.__enter__.return_value.getsockname.return_value = ("127.0.0.1", 12345)
    monkeypatch.setattr(
        experiment, "socket", SimpleNamespace(socket=Mock(return_value=listener))
    )
    monkeypatch.setattr(
        experiment,
        "subprocess",
        SimpleNamespace(check_output=Mock(return_value="offline-git-fixture\n")),
    )
    monkeypatch.setattr(experiment.importlib.metadata, "distributions", lambda: [])
    monkeypatch.setattr(validate_pipeline, "REPO", tmp_path)
    supplied = tmp_path / "supplied.json"
    write(supplied, contract)
    config = validate_pipeline.prepare(
        SimpleNamespace(
            output=tmp_path / "run",
            benchmark="workbuddy",
            task_id=task["task_id"],
            public_delivery_contract=supplied,
            credentials=upstream / ".env",
        )
    )
    experiment.verify(config)
    return config


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    from tests.unit.test_public_delivery_contract import task_and_contract

    task, contract = task_and_contract()
    config = prepare_public_run(tmp_path, monkeypatch, task, contract)
    root = Path(config["output"])
    manifest = read(root / "manifest.json")
    result = auditor.audit_code_binding(auditor.Reader(root), config, manifest)
    assert result["complete_file_set"] is True
    return SimpleNamespace(
        root=root, code=Path(config["code"]), config=config, manifest=manifest
    )


def test_actual_freeze_binding_includes_resources_and_launcher(prepared):
    reader = auditor.Reader(prepared.root)
    result = auditor.audit_code_binding(reader, prepared.config, prepared.manifest)
    assert result["scope"] == SCOPE
    assert result["complete_file_set"] is True
    assert result["frozen_file_count"] == len(prepared.manifest["code_hashes"])
    assert result["installed_package_file_count"] == len(
        [n for n in prepared.manifest["code_hashes"] if n.startswith("reward_engine/")]
    )
    for name in RESOURCES:
        assert reader.hashes["code/" + name] == prepared.manifest["code_hashes"][name]
    cache = prepared.code / "reward_engine/__pycache__/new.pyc"
    cache.parent.mkdir(exist_ok=True)
    cache.write_bytes(b"transient runtime cache")
    auditor.audit_code_binding(
        auditor.Reader(prepared.root), prepared.config, prepared.manifest
    )


@pytest.mark.parametrize("relative", RESOURCES)
@pytest.mark.parametrize("change", ["modify", "delete", "add"])
def test_additional_gate_rejects_resource_changes(prepared, relative, change):
    path = prepared.code / relative
    if change == "modify":
        path.write_bytes(b"changed runtime input")
        reason = "Frozen source changed: " + relative
    elif change == "delete":
        path.unlink()
        reason = "Frozen code manifest omitted or added files"
    else:
        path.with_name("extra_" + path.name).write_bytes(b"unbound runtime input")
        reason = "Frozen code manifest omitted or added files"
    report = auditor.audit_run(prepared.root)
    assert report["status"] == "fail"
    assert reason in report["reason"]


@pytest.mark.parametrize("target", ["config", "manifest"])
@pytest.mark.parametrize("marker", ["absent", None, "legacy", "unknown-version"])
def test_scope_mismatch_rejects_even_with_consistent_config_identity(
    prepared, target, marker
):
    row = getattr(prepared, target)
    if marker == "absent":
        del row["code_hash_scope"]
    else:
        row["code_hash_scope"] = marker
    prepared.config["identity"] = digest(
        {k: v for k, v in prepared.config.items() if k != "identity"}
    )
    prepared.manifest["config_identity"] = prepared.config["identity"]
    write(prepared.root / "config.json", prepared.config)
    write(prepared.root / "manifest.json", prepared.manifest)
    report = auditor.audit_run(prepared.root)
    assert report["status"] == "fail"
    assert "hash scope missing or unsupported" in report["reason"]


@pytest.mark.parametrize("rebind_identity", [False, True])
def test_removing_both_scope_markers_cannot_admit_complete_map(
    prepared, rebind_identity
):
    del prepared.config["code_hash_scope"]
    del prepared.manifest["code_hash_scope"]
    if rebind_identity:
        prepared.config["identity"] = digest(
            {k: v for k, v in prepared.config.items() if k != "identity"}
        )
        prepared.manifest["config_identity"] = prepared.config["identity"]
    write(prepared.root / "config.json", prepared.config)
    write(prepared.root / "manifest.json", prepared.manifest)
    report = auditor.audit_run(prepared.root)
    assert report["status"] == "fail"
    expected = (
        "omitted or added modules" if rebind_identity else "Config identity mismatch"
    )
    assert expected in report["reason"]


@pytest.mark.parametrize("change", ["omit", "invent", "not_mapping"])
def test_incomplete_manifest_rejected(prepared, change):
    hashes = prepared.manifest["code_hashes"]
    if change == "omit":
        del hashes[PLUGIN]
    elif change == "invent":
        hashes["workbuddy_swarm/missing.ts"] = "0" * 64
    else:
        prepared.manifest["code_hashes"] = list(hashes)
    write(prepared.root / "manifest.json", prepared.manifest)
    report = auditor.audit_run(prepared.root)
    assert report["status"] == "fail"
    assert (
        "Missing frozen source hashes"
        if change == "not_mapping"
        else "omitted or added files"
    ) in report["reason"]


@pytest.mark.parametrize("kind", ["file_symlink", "directory_symlink", "fifo"])
def test_nonregular_frozen_source_rejected_before_read(prepared, tmp_path, kind):
    path = prepared.code / PLUGIN
    if kind == "file_symlink":
        target = tmp_path / "same_bytes.mjs"
        target.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(target)
    elif kind == "directory_symlink":
        target = tmp_path / "outside"
        target.mkdir()
        (prepared.code / "linked").symlink_to(target, target_is_directory=True)
    else:
        path.unlink()
        os.mkfifo(path)
    report = auditor.audit_run(prepared.root)
    assert report["status"] == "fail"
    assert "Nonregular code source" in report["reason"]


@pytest.mark.parametrize("change", ["modify", "delete", "add", "symlink"])
def test_installed_package_resources_must_match(
    prepared, tmp_path, monkeypatch, change
):
    # Simulate another installed package using a temp copy, never mutate the real one.
    package = tmp_path / "installed/reward_engine"
    shutil.copytree(prepared.code / "reward_engine", package)
    monkeypatch.setattr(
        auditor,
        "__file__",
        str(package / "adapters/workbuddy/public_delivery_audit.py"),
    )
    path = package / PLUGIN.removeprefix("reward_engine/")
    if change == "modify":
        path.write_bytes(b"different installed MJS")
        reason = "installed package differs from frozen source"
    elif change == "delete":
        path.unlink()
        reason = "Installed package file set differs"
    elif change == "add":
        path.with_name("extra.mjs").write_bytes(b"unbound installed MJS")
        reason = "Installed package file set differs"
    else:
        path.unlink()
        path.symlink_to(prepared.code / PLUGIN)
        reason = "Nonregular code source"
    report = auditor.audit_run(prepared.root)
    assert report["status"] == "fail"
    assert reason in report["reason"]


def test_legacy_binding_reports_only_historical_python_package_scope(prepared):
    del prepared.config["code_hash_scope"]
    del prepared.manifest["code_hash_scope"]
    prepared.manifest["code_hashes"] = {
        name: sha
        for name, sha in prepared.manifest["code_hashes"].items()
        if name.endswith(".py")
    }
    (prepared.code / PLUGIN).write_bytes(b"historically unbound resource")
    report = auditor.audit_code_binding(
        auditor.Reader(prepared.root), prepared.config, prepared.manifest
    )
    assert report["complete_file_set"] is False
    assert report["scope"] == "legacy-listed-files-python-package"
    assert "no complete-resource" in report["limitation"]
    source = prepared.code / "reward_engine/core/outcome/context.py"
    source.write_bytes(b"changed bound Python")
    with pytest.raises(ValueError, match="Frozen source changed"):
        auditor.audit_code_binding(
            auditor.Reader(prepared.root), prepared.config, prepared.manifest
        )


def test_consistently_rehashed_frozen_resource_still_requires_installed_origin(
    prepared,
):
    (prepared.code / PLUGIN).write_bytes(b"changed frozen resource")
    prepared.manifest["code_hashes"][PLUGIN] = file_hash(prepared.code / PLUGIN)
    write(prepared.root / "manifest.json", prepared.manifest)
    report = auditor.audit_run(prepared.root)
    assert report["status"] == "fail"
    assert "installed package differs from frozen source" in report["reason"]
