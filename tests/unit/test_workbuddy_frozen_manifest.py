"""Freeze real temporary trees; no providers, containers or network provisioning."""

import hashlib
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from experiments import validate_pipeline as validation
from experiments.workbuddy import experiment
from reward_engine.adapters import provider_config
from reward_engine.adapters.workbuddy.io import digest, file_hash, read, write

SCOPE = "all-regular-files-except-__pycache__-v1"
FILES = {
    "reward_engine/__init__.py": b"# temporary package\n",
    "reward_engine/adapters/provider_config.py": Path(
        provider_config.__file__
    ).read_bytes(),
    "reward_engine/adapters/opencode_plugin.mjs": b"export default {};\n",
    "reward_engine/configs/.runtime.json": b'{"mode":"offline"}\n',
    "reward_engine/data/resource": b"\x00\x01runtime resource\xff",
    "experiments/__init__.py": b"# temporary experiments\n",
    "experiments/workbuddy/worker.py": b"# temporary worker\n",
    "workbuddy_swarm/__init__.py": b"# temporary external transport\n",
    "workbuddy_swarm/v4/plugin.ts": b"export const enabled: boolean = true;\n",
    "workbuddy_swarm/v4/README.md": b"Transport fixture resources\n",
}
RESOURCES = [
    "reward_engine/adapters/opencode_plugin.mjs",
    "workbuddy_swarm/v4/plugin.ts",
    "reward_engine/configs/.runtime.json",
    "reward_engine/data/resource",
]


@pytest.fixture
def prepare_tree(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    upstream = repo / "_local/native/workbuddy/runner/upstream"
    for name, content in FILES.items():
        root = upstream.parent if name.startswith("workbuddy_swarm/") else repo
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    cache = repo / "reward_engine/__pycache__/ignored.pyc"
    cache.parent.mkdir()
    cache.write_bytes(b"transient bytecode")
    for index in range(50):
        task = upstream / f"datasets/wb-bench-office-v1.0/tasks/task_{index:02}"
        task.mkdir(parents=True)
        (task / "task.toml").write_text("[agent]\ntimeout_sec = 10\n")
    (upstream / ".env").write_text(
        "YIBU_API_KEY=offline-fixture\nYIBU_BASE_URL=http://offline.invalid/v1\n"
    )
    binary = upstream / "configs/harnesses/opencode-swarm/docker/rootfs/bin/opencode"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"offline binary fixture")

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
    monkeypatch.setattr(experiment, "REPO", repo)
    monkeypatch.setattr(validation, "REPO", repo)

    def prepare(*, with_launcher=False, **provider_options):
        output = tmp_path / "frozen"
        if with_launcher:
            config = validation.prepare(
                SimpleNamespace(
                    output=output,
                    benchmark="workbuddy",
                    task_id="task_00",
                    **{"credentials": upstream / ".env", **provider_options},
                )
            )
        else:
            config = experiment.prepare(
                output, upstream=upstream, task_ids=["task_00"], **provider_options
            )
        return SimpleNamespace(
            config=config,
            output=output,
            code=Path(config["code"]),
            manifest=read(output / "manifest.json"),
        )

    return prepare


def test_real_prepare_binds_every_copied_resource(prepare_tree):
    frozen = prepare_tree()
    assert frozen.config["code_hash_scope"] == SCOPE
    assert frozen.manifest["code_hash_scope"] == SCOPE
    assert frozen.manifest["code_hashes"] == {
        name: hashlib.sha256(content).hexdigest() for name, content in FILES.items()
    }
    assert not (frozen.code / "reward_engine/__pycache__").exists()
    assert frozen.manifest["config_identity"] == frozen.config["identity"]
    experiment.verify(frozen.config)


@pytest.mark.parametrize("relative", RESOURCES)
@pytest.mark.parametrize("change", ["modify", "delete", "add"])
def test_non_python_resource_changes_are_rejected(prepare_tree, relative, change):
    frozen = prepare_tree()
    path = frozen.code / relative
    if change == "modify":
        path.write_bytes(b"different source or resource bytes")
    elif change == "delete":
        path.unlink()
    else:
        path = path.with_name("extra_" + path.name)
        path.write_bytes(b"additional runtime input")
    with pytest.raises(ValueError, match="Frozen implementation") as error:
        experiment.verify(frozen.config)
    assert path.relative_to(frozen.code).as_posix() in str(error.value)


def test_bytecode_cache_is_transient_but_added_python_is_not(prepare_tree):
    frozen = prepare_tree()
    cache = frozen.code / "reward_engine/__pycache__/generated.cpython-311.pyc"
    cache.parent.mkdir()
    cache.write_bytes(b"runtime bytecode")
    experiment.verify(frozen.config)
    (frozen.code / "reward_engine/extra.py").write_text("# unbound module\n")
    with pytest.raises(ValueError, match="unexpected=.*reward_engine/extra.py"):
        experiment.verify(frozen.config)


@pytest.mark.parametrize("change", ["omit", "invent", "escape", "not_mapping"])
def test_new_manifest_requires_complete_exact_file_set(prepare_tree, change):
    frozen = prepare_tree()
    hashes = frozen.manifest["code_hashes"]
    if change == "omit":
        del hashes[RESOURCES[0]]
    elif change == "invent":
        hashes["missing/resource.ts"] = "0" * 64
    elif change == "escape":
        hashes["../outside.mjs"] = "0" * 64
    else:
        frozen.manifest["code_hashes"] = list(hashes)
    write(frozen.output / "manifest.json", frozen.manifest)
    with pytest.raises(ValueError, match="Frozen implementation"):
        experiment.verify(frozen.config)


@pytest.mark.parametrize("marker", ["absent", None, "unknown-scope"])
@pytest.mark.parametrize("target", ["config", "manifest"])
def test_scope_cannot_silently_downgrade_to_legacy(prepare_tree, marker, target):
    frozen = prepare_tree()
    value = getattr(frozen, target)
    if marker == "absent":
        del value["code_hash_scope"]
    else:
        value["code_hash_scope"] = marker
    if target == "config":
        frozen.config["identity"] = digest(
            {k: v for k, v in frozen.config.items() if k != "identity"}
        )
        frozen.manifest["config_identity"] = frozen.config["identity"]
    write(frozen.output / "manifest.json", frozen.manifest)
    with pytest.raises(ValueError, match="hash scope missing or unsupported"):
        experiment.verify(frozen.config)


def test_scope_is_part_of_config_identity(prepare_tree):
    frozen = prepare_tree()
    del frozen.config["code_hash_scope"]
    del frozen.manifest["code_hash_scope"]
    write(frozen.output / "manifest.json", frozen.manifest)
    with pytest.raises(ValueError, match="configuration changed"):
        experiment.verify(frozen.config)


def test_new_manifest_is_bound_to_config_identity(prepare_tree):
    frozen = prepare_tree()
    frozen.manifest["config_identity"] = "different-config"
    write(frozen.output / "manifest.json", frozen.manifest)
    with pytest.raises(ValueError, match="manifest configuration identity mismatch"):
        experiment.verify(frozen.config)


def test_legacy_verification_retains_only_its_original_listed_scope(prepare_tree):
    frozen = prepare_tree()
    del frozen.config["code_hash_scope"]
    frozen.config["identity"] = digest(
        {k: v for k, v in frozen.config.items() if k != "identity"}
    )
    del frozen.manifest["code_hash_scope"]
    frozen.manifest["config_identity"] = frozen.config["identity"]
    frozen.manifest["code_hashes"] = {
        name: value
        for name, value in frozen.manifest["code_hashes"].items()
        if name.endswith(".py")
    }
    write(frozen.output / "manifest.json", frozen.manifest)
    # Compatibility is not retrospective resource-completeness certification.
    (frozen.code / RESOURCES[0]).write_bytes(b"historically unbound MJS")
    (frozen.code / RESOURCES[1]).unlink()
    (frozen.code / "additional.ts").write_bytes(b"historically unbound TS")
    experiment.verify(frozen.config)
    (frozen.code / "reward_engine/__init__.py").write_bytes(b"changed bound Python")
    with pytest.raises(ValueError, match="Frozen implementation changed"):
        experiment.verify(frozen.config)


@pytest.mark.parametrize("kind", ["file_symlink", "directory_symlink", "fifo"])
def test_non_regular_frozen_entries_are_rejected(prepare_tree, tmp_path, kind):
    frozen = prepare_tree()
    path = frozen.code / RESOURCES[0]
    if kind == "file_symlink":
        target = tmp_path / "same_bytes.mjs"
        target.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(target)
    elif kind == "directory_symlink":
        target = tmp_path / "outside"
        target.mkdir()
        (frozen.code / "extra_directory").symlink_to(target, target_is_directory=True)
    else:
        path.unlink()
        os.mkfifo(path)
    with pytest.raises(ValueError, match="not a regular file"):
        experiment.verify(frozen.config)


def test_real_validation_launcher_extension_preserves_complete_manifest(prepare_tree):
    frozen = prepare_tree(with_launcher=True)
    relative = "experiments/validate_pipeline.py"
    assert frozen.manifest["code_hash_scope"] == SCOPE
    assert frozen.config["code_hash_scope"] == SCOPE
    assert set(frozen.manifest["code_hashes"]) == set(FILES) | {relative}
    assert frozen.manifest["code_hashes"][relative] == file_hash(
        Path(validation.__file__)
    )
    assert frozen.manifest["config_identity"] == frozen.config["identity"]
    experiment.verify(frozen.config)
    (frozen.code / relative).write_text("# changed launcher\n")
    with pytest.raises(ValueError, match="Frozen implementation changed"):
        experiment.verify(frozen.config)


@pytest.mark.parametrize("target", ["dataset", "binary"])
def test_existing_non_code_checks_remain_enforced(prepare_tree, target):
    frozen = prepare_tree()
    if target == "dataset":
        path = Path(frozen.config["tasks_root"]) / "task_00/task.toml"
        error = "Frozen dataset changed"
    else:
        path = Path(frozen.config["harness_mount"]) / "bin/opencode"
        error = "Pinned OpenCode binary changed"
    path.write_bytes(b"changed fixture")
    with pytest.raises(ValueError, match=error):
        experiment.verify(frozen.config)
