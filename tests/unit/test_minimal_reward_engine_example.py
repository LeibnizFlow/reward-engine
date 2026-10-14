"""Teaching example integration: real core, scripted actors; no native/provider calls."""

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest

from reward_engine.examples.minimal import run
from reward_engine.examples.offline import Offline

ROOT = Path(__file__).resolve().parents[2]


def read(root, name):
    return json.loads((root / (name + ".json")).read_text())


def test_three_round_controller_review_selection_and_memory(tmp_path):
    root = tmp_path / "offline"
    summary = asyncio.run(run(root))
    assert summary["completed_rounds"] == 3
    assert summary["seed_count"] == 3 and summary["actor_count"] == 9
    assert summary["mode"] == "offline_deterministic"
    assert summary["native_evaluation"] == "not_run"
    assert not read(root, "scope")["adversarial_review_enabled"]
    assert len(read(root, "seeds")) == 3
    for n in range(1, 4):
        prefix = f"round_{n}"
        for stage in ("jury_map", "hypothesis_map", "ranking_map"):
            batch = read(root, f"{prefix}/{stage}")
            assert batch["complete"] and batch["succeeded"] == 3
        jury = read(root, f"{prefix}/jury")
        assert jury["criteria"][0]["id"] == "total"
        pack = read(root, f"{prefix}/pack")
        assert pack["tests"][0]["checker"] == "example.total"
        assert read(root, f"{prefix}/construction_smoke")["receipt"]["status"] == "pass"
        assert read(root, f"{prefix}/hypothesis")["status"] == "propose"
        for arm, expected in (("A", "fail"), ("B", "pass"), ("C", "pass")):
            results = read(root, f"{prefix}/{arm}/outcome_estimate")["results"]
            assert len(results) == 1 and results[0]["status"] == expected
            assert not results[0].get("error")
        assert read(root, f"{prefix}/ranking")["status"] == "compared"
        assert read(root, f"{prefix}/selected_delivery")["arm"] == "B"
        delivery = read(root, f"{prefix}/B/delivery")
        assert delivery["revision_count"] == 1
        assert delivery["review_status"] == "satisfied_under_estimator"
        first = read(root, f"{prefix}/actors/B/checks/delivery_1")
        last = read(root, f"{prefix}/actors/B/checks/delivery_2")
        assert first["measurement"]["results"][0]["status"] == "fail"
        assert last["measurement"]["results"][0]["status"] == "pass"
        assert first["candidate"]["candidate_id"] == f"round-{n}-B-v0"
        assert last["candidate"]["candidate_id"] == f"round-{n}-B-v1"
        feedback = read(root, f"{prefix}/actors/B/checks/feedback_1")["message"]
        messages = read(root, f"actors/round-{n}-B/conversation")
        answers = [
            json.loads(m["content"]) for m in messages if m["role"] == "assistant"
        ]
        assert json.loads(answers[0]["final_response"]) == {"total": 5}
        assert json.loads(answers[1]["final_response"]) == {"total": 6}
        assert {"role": "user", "content": feedback} in messages
        assert len(read(root, f"{prefix}/memory")) == n
        checkpoint = read(root, f"{prefix}/checkpoint")
        assert len(checkpoint["memory"]) == n
        assert len(checkpoint["pool"]) == 3 + 3 * n
        if n > 1:
            assert read(root, f"{prefix}/learning_context")["memories"]
            assert read(root, f"{prefix}/builder_reuse")["reused_criteria"] == ["total"]
    assert len(read(root, "round_1/memory")) == 1  # no shared growth
    # Core UUIDs/timings differ, but task behavior and the summary are reproducible.
    assert asyncio.run(run(tmp_path / "repeat")) == summary


def test_offline_cli_without_any_site_packages_and_no_overwrite(tmp_path):
    output = tmp_path / "stdlib-only"
    command = [
        sys.executable,
        "-S",
        "-m",
        "reward_engine.examples.minimal",
        "--output",
        str(output),
    ]
    completed = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, timeout=30
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["mode"] == "offline_deterministic"
    before = (output / "summary.json").read_bytes()
    duplicate = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, timeout=30
    )
    assert duplicate.returncode != 0 and "FileExistsError" in duplicate.stderr
    assert before == (output / "summary.json").read_bytes()


def test_cli_rejects_provider_options_without_live(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "reward_engine.examples.minimal",
            "--output",
            str(tmp_path / "unused"),
            "--model",
            "simulation",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0 and "require --live" in result.stderr
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize("broken", ["role", "actor", "ranking"])
def test_failures_are_retained_not_reported_as_success(tmp_path, broken):
    class Broken(Offline):
        async def role(self, request):
            if broken == "role":
                raise RuntimeError("scripted provider outage")
            if broken == "ranking" and "ranking" in request["mode"]:
                return {}  # no host-authored missing required partition
            return await super().role(request)

        async def actor(self, messages, *, arm, revision):
            if broken == "actor":
                return {"text": "not the required envelope"}
            return await super().actor(messages, arm=arm, revision=revision)

    root = tmp_path / broken
    with pytest.raises((RuntimeError, ValueError)):
        asyncio.run(run(root, backend=Broken()))
    assert read(root, "failure")["status"] == "failed"
    assert not (root / "summary.json").exists()
    assert list((root / "calls").glob("*/request.json"))


def test_valid_initial_delivery_does_not_force_a_revision(tmp_path):
    class AlreadyCorrect(Offline):
        async def actor(self, messages, *, arm, revision):
            assert revision == 0
            return {"final_response": '{"total":6}'}

    summary = asyncio.run(
        run(tmp_path / "no-revision-needed", backend=AlreadyCorrect())
    )
    assert all(r["revision_count"] == 0 for r in summary["rounds"])
    assert all(
        r["review_status"] == "satisfied_under_estimator" for r in summary["rounds"]
    )
    assert summary["memory_entries"] == 0  # no invented revision learning
