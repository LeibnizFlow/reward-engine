"""Public input overlays respect the native service working directory."""

import json

import pytest
import yaml

from experiments.claw_eval.opencode.campaign import install_public_fixtures
from reward_engine.adapters.workbuddy.io import file_hash


@pytest.mark.parametrize("service_input", [False, True])
def test_declared_overlay_is_readable_from_its_consumer_cwd(tmp_path, monkeypatch, service_input):
    output = tmp_path / "run"
    benchmark = output / "benchmark"
    folder = benchmark / "tasks/synthetic"
    folder.mkdir(parents=True)
    relative = "shared/input.json"
    spec = {"sandbox_files": [relative]}
    if service_input:
        spec["services"] = [{"env": {"DATA_FILE": relative}}]
    (folder / "task.yaml").write_text(yaml.safe_dump(spec))
    overlay = tmp_path / "overlay"
    source = overlay / "tasks/synthetic" / relative
    source.parent.mkdir(parents=True)
    source.write_text('{"value": "synthetic input"}')
    archive = overlay.parent / "fixtures.tar.gz"
    archive.write_bytes(b"synthetic archive identity")
    (overlay / "SOURCE.json").write_text(json.dumps({"archive_sha256": file_hash(archive)}))
    monkeypatch.setenv("CLAW_EVAL_FIXTURE_OVERLAY", str(overlay))

    install_public_fixtures(benchmark, ["synthetic"], output)

    destination = (benchmark if service_input else folder) / relative
    assert destination.read_bytes() == source.read_bytes()
    proof = json.loads((output / "fixture_overlay.json").read_text())
    assert proof["files"] == {str(destination.relative_to(output)): file_hash(source)}
    assert not ((folder if service_input else benchmark) / relative).exists()
