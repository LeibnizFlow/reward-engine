"""Supplemental audit behavior on synthetic frozen runs and real local checks."""

import asyncio
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from experiments import audit_deliveries as audit
from experiments.pipeline_audit import Receipts, delivered_measurement, receipt_digest
from reward_engine.adapters.workbuddy.evidence import collect
from reward_engine.adapters.workbuddy.io import digest, file_hash, read, write
from reward_engine.core.outcome.context import build_context
from reward_engine.core.outcome.harness import (
    MEASUREMENT_INPUT_SCHEMA,
    Registry,
    measurement_input_fingerprint,
)
from reward_engine.core.outcome.task_sequence import TaskSequence
from tests.support.claw_outcome_feedback_arms import PACK


@pytest.fixture(params=["claw_eval", "workbuddy"])
def original(request, tmp_path):
    """Enough real frozen provenance for the production verifiers, not a mock."""
    root = tmp_path / "original"
    code = root / "code"
    implementation = code / "reward_engine/frozen.py"
    implementation.parent.mkdir(parents=True)
    implementation.write_text("# immutable original implementation\n")
    task_file = root / "dataset/tasks/task/instruction.md"
    task_file.parent.mkdir(parents=True)
    task_file.write_text("Return JSON response and payload")
    binary = root / "harness/bin/opencode"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"offline binary fixture")
    config = {
        "output": str(root),
        "code": str(code),
        "tasks_root": str(task_file.parent.parent),
        "harness_mount": str(binary.parent.parent),
        "validation_benchmark": request.param,
        "validation_task_id": "task",
        "gateway_port": 12345,
        "rounds": 3,
        "batch": {},
        "roles": {"reward_calibration_judge": "offline-only"},
    }
    config["identity"] = digest(config)
    write(root / "config.json", config)
    write(
        root / "manifest.json",
        {
            "config_identity": config["identity"],
            "files": {
                str(p.relative_to(root)): file_hash(p)
                for p in (implementation, task_file, binary)
            },
            "code_hashes": {
                str(implementation.relative_to(code)): file_hash(implementation)
            },
            "prepared_dataset_hashes": {
                str(task_file.relative_to(root / "dataset")): file_hash(task_file)
            },
            "opencode_sha256": file_hash(binary),
        },
    )
    write(
        root / "validation_status.json",
        {"status": "execution_completed", "stage_audit": "pending"},
    )
    expected = {}
    for number in range(1, 4):
        folder = root / "tasks/task" / f"round_{number}"
        task = {"task_id": "task", "prompt": task_file.read_text()}
        pack = deepcopy(PACK)
        pack.update(
            task_id="task", task_prompt=task["prompt"], pack_id=f"pack-{number}"
        )
        pack["tests"] = [
            {
                "test_id": field,
                "criterion_id": "syntax",
                "checker": "offline.json",
                "question": f"Parse delivered {field}",
                "arguments": {"ref": f"product_1:artifact:{field}"},
            }
            for field in ("final_response", "payload")
        ]
        write(folder / "pack.json", pack)
        write(folder / "checkpoint.json", {"task": task, "completed_round": number})
        for arm in "ABC":
            native = folder / "actors" / arm
            turn = "1" if request.param == "claw_eval" else "delivery-1"
            snapshot = native / "snapshots" / turn
            workspace = snapshot / "workspace"
            workspace.mkdir(parents=True)
            response = workspace / "response.json"
            response.write_text(json.dumps({"round": number, "arm": arm}))
            payload = workspace / "payload.json"
            payload.write_text(
                "invalid JSON" if arm == "B" else json.dumps([number, arm])
            )
            candidate, _ = collect(
                workspace,
                candidate_id=f"original-{number}-{arm}",
                final_response=response.read_text(),
                initial_hashes={},
                output=snapshot,
            )
            # Keep the auxiliary artifact checker bound to actual retained bytes.
            # All workspace metadata and parsed sources come from the collector.
            candidate["artifact"]["payload"] = payload.read_text()
            write(snapshot / "candidate.json", candidate)
            row = {
                "status": "completed",
                "candidate": candidate,
                "session": f"session-{number}-{arm}",
            }
            partial = {
                "pack_id": pack["pack_id"],
                "results": [],
                "review_diagnostics": {"reason": "targeted original review"},
            }
            write(folder / arm / "delivery.json", {**row, "final_measurement": partial})
            write(folder / arm / "outcome_estimate.json", partial)
            write(native / "RESULT.json", row)
            write(native / "actor_state.json", {"turn": 1, "session": row["session"]})
            write(
                native / "OFFICIAL.private.json",
                {
                    "status": "scored",
                    "score": number / 10,
                    "private": "PRIVATE_SCORE_SENTINEL",
                },
            )
            (native / "artifact.bin").write_bytes(
                f"original artifact {number}/{arm}".encode()
            )
            expected[number, arm] = candidate
    return SimpleNamespace(
        root=root, config=config, expected=expected, implementation=implementation
    )


@pytest.fixture
def local_execution(monkeypatch):
    calls, servers, failures = [], [], set()

    async def check(test, context, evidence):
        assert "PRIVATE_SCORE_SENTINEL" not in str((test, context, evidence))
        ref = test["arguments"]["ref"]
        calls.append(
            {
                "test": deepcopy(test),
                "context": deepcopy(context),
                "evidence": deepcopy(evidence),
            }
        )
        if test["test_id"] in failures:
            raise RuntimeError("Retained offline checker failure")
        try:
            json.loads(evidence[ref])
            status = "pass"
        except json.JSONDecodeError:
            status = "fail"
        return {
            "status": status,
            "reason": "Parsed exact retained artifact field",
            "gap": "Invalid delivered JSON" if status == "fail" else "",
            "evidence_refs": [ref],
        }

    registry = Registry()
    registry.register("offline.json", check, "Parse the actual retained text locally")
    monkeypatch.setattr(audit, "registry", lambda *_: registry)

    class Server:
        def __init__(self, config):
            self.started = True
            self.should_exit = False
            self.stopped = False
            servers.append(self)

        async def serve(self):
            while not self.should_exit:
                await asyncio.sleep(0)
            self.stopped = True

    gateway = Mock(return_value=object())
    monkeypatch.setattr(audit, "application", gateway)
    monkeypatch.setattr(audit.uvicorn, "Server", Server)
    return SimpleNamespace(
        calls=calls, servers=servers, gateway=gateway, failures=failures
    )


def retained_files(root):
    return {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def receipt_path(original, number=1, arm="A"):
    return (
        original.root
        / "tasks/task"
        / f"round_{number}"
        / arm
        / "outcome_estimate.full.supplemental.json"
    )


@pytest.mark.asyncio
async def test_all_nine_exact_deliveries_execute_every_check_without_rewriting_originals(
    original, local_execution
):
    before = retained_files(original.root)
    await audit.run(original.root)
    assert len(local_execution.calls) == 18
    assert Counter(
        (
            json.loads(c["evidence"]["product_1:artifact:final_response"])["round"],
            json.loads(c["evidence"]["product_1:artifact:final_response"])["arm"],
        )
        for c in local_execution.calls
    ) == Counter({slot: 2 for slot in original.expected})
    for (number, arm), candidate in original.expected.items():
        result = read(receipt_path(original, number, arm))
        measured = result["measurement"]
        assert measured["pack_id"] == f"pack-{number}"
        assert {r["test_id"]: r["status"] for r in measured["results"]} == {
            "final_response": "pass",
            "payload": "fail" if arm == "B" else "pass",
        }
        assert measured["execution_summary"]["submitted_checks"] == 2
        folder = original.root / "tasks/task" / f"round_{number}"
        assert measured["measurement_input_schema"] == MEASUREMENT_INPUT_SCHEMA
        assert measured["measurement_input_sha256"] == measurement_input_fingerprint(
            read(folder / "pack.json"),
            *build_context(read(folder / "checkpoint.json")["task"], [candidate]),
        )
        turn = (
            "1"
            if original.config["validation_benchmark"] == "claw_eval"
            else "delivery-1"
        )
        snapshot = folder / "actors" / arm / "snapshots" / turn / "candidate.json"
        assert result["schema"] == "pipeline-supplemental-measurement-v1"
        assert result["source"] == snapshot.relative_to(original.root).as_posix()
        assert (
            result["pack_source"]
            == (folder / "pack.json").relative_to(original.root).as_posix()
        )
        assert result["candidate_sha256"] == receipt_digest(candidate)
        assert result["artifact_sha256"] == receipt_digest(candidate["artifact"])
        assert result["pack_sha256"] == receipt_digest(read(folder / "pack.json"))
        assert result["provenance"]["kind"] == "supplemental_full_measurement"
        assert result["provenance"]["producer"] == "experiments.audit_deliveries"
        assert result["provenance"]["created_at"]
        assert result["input_binding"] == digest(
            {
                "task": read(folder / "checkpoint.json")["task"],
                "candidate": candidate,
                "pack": read(folder / "pack.json"),
            }
        )
        for name in (
            folder / arm / "delivery.json",
            folder / "pack.json",
            folder / "checkpoint.json",
            snapshot,
        ):
            assert result["source_hashes"][str(name)] == file_hash(name)
        assert result["code_hashes"]["experiments/audit_deliveries.py"] == file_hash(
            Path(audit.__file__)
        )
        receipts = Receipts(original.root, closed=True)
        delivered_measurement(
            receipts,
            folder.relative_to(original.root),
            arm,
            original.config["validation_benchmark"],
        )
        assert receipts.facts["provenance"] == "supplemental"
        assert receipts.facts["primary_full_pack_measurement"] is False
    assert all(
        (original.root / name).read_bytes() == contents
        for name, contents in before.items()
    )
    assert all(s.should_exit and s.stopped for s in local_execution.servers)


@pytest.mark.asyncio
@pytest.mark.parametrize("cached", [False, True])
async def test_live_original_is_refused_before_gateway_or_measurement(
    original, local_execution, cached
):
    if cached:
        await audit.run(original.root)
        local_execution.calls.clear()
        local_execution.gateway.reset_mock()
    write(original.root / "validation_status.json", {"status": "running"})
    before = retained_files(original.root)
    with pytest.raises(ValueError, match="original execution to stop"):
        await audit.run(original.root)
    local_execution.gateway.assert_not_called()
    assert not local_execution.calls
    assert retained_files(original.root) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", ["configuration", "frozen_code"])
@pytest.mark.parametrize("cached", [False, True])
async def test_original_freeze_is_verified_before_dispatch(
    original, local_execution, changed, cached
):
    if cached:
        await audit.run(original.root)
        local_execution.calls.clear()
        local_execution.gateway.reset_mock()
    if changed == "configuration":
        config = read(original.root / "config.json")
        config["rounds"] = 1
        write(original.root / "config.json", config)
    else:
        original.implementation.write_text("# changed frozen implementation\n")
    before = retained_files(original.root)
    with pytest.raises(ValueError, match="changed|differs"):
        await audit.run(original.root)
    local_execution.gateway.assert_not_called()
    assert not local_execution.calls
    assert retained_files(original.root) == before


@pytest.mark.asyncio
async def test_matching_complete_receipts_resume_without_remeasurement(
    original, local_execution
):
    await audit.run(original.root)
    before = retained_files(original.root)
    local_execution.calls.clear()
    await audit.run(original.root)
    assert not local_execution.calls
    assert retained_files(original.root) == before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changed",
    [
        "input_binding",
        "source_hashes",
        "code_hashes",
        "candidate",
        "pack",
        "checkpoint",
        "delivery_metadata",
    ],
)
async def test_cached_receipt_never_bypasses_changed_inputs_or_code(
    original, local_execution, changed
):
    await audit.run(original.root)
    path = receipt_path(original)
    if changed in {"input_binding", "source_hashes", "code_hashes"}:
        row = read(path)
        row[changed] = "wrong" if changed == "input_binding" else {}
    else:
        folder = original.root / "tasks/task/round_1"
        path = folder / (
            "pack.json"
            if changed == "pack"
            else "checkpoint.json"
            if changed == "checkpoint"
            else "A/delivery.json"
        )
        row = read(path)
        if changed == "candidate":
            row["candidate"]["artifact"]["payload"] = "changed"
        elif changed == "pack":
            row["tests"][0]["question"] = "Changed standard"
        else:
            row["retained_diagnostic"] = (
                "changed bytes without changing the measured input"
            )
    write(path, row)
    before = retained_files(original.root)
    local_execution.calls.clear()
    with pytest.raises(ValueError, match="changed|differs"):
        await audit.run(original.root)
    assert not local_execution.calls
    assert retained_files(original.root) == before
    assert all(s.should_exit and s.stopped for s in local_execution.servers)


@pytest.mark.asyncio
async def test_explicit_effective_task_and_pack_drive_actual_checks(
    original, local_execution
):
    path = original.root / "tasks/task/round_1/A/delivery.json"
    row = read(path)
    task = {"task_id": "task", "prompt": "Current public follow-up"}
    pack = read(path.parent.parent / "pack.json")
    pack.update(pack_id="effective-pack", task_prompt=task["prompt"])
    row.update(effective_task=task, effective_pack=pack)
    write(path, row)
    await audit.run(original.root)
    assert all(c["context"]["task"] == task for c in local_execution.calls[:2])
    receipt = read(receipt_path(original))
    assert receipt["measurement"]["pack_id"] == "effective-pack"
    assert receipt["measurement"][
        "measurement_input_sha256"
    ] == measurement_input_fingerprint(pack, *build_context(task, [row["candidate"]]))
    assert receipt["input_binding"] == digest(
        {"task": task, "pack": pack, "candidate": row["candidate"]}
    )


@pytest.mark.asyncio
async def test_effective_task_without_pack_rebases_like_core_loop(
    original, local_execution
):
    path = original.root / "tasks/task/round_1/A/delivery.json"
    row = read(path)
    task = {"task_id": "task", "prompt": "Current public follow-up"}
    row["effective_task"] = task
    write(path, row)
    rebased = TaskSequence(task).rebase(read(path.parent.parent / "pack.json"))
    await audit.run(original.root)
    assert all(c["context"]["task"] == task for c in local_execution.calls[:2])
    assert {
        c["test"]["test_id"]: c["test"]["question"] for c in local_execution.calls[:2]
    } == {test["test_id"]: test["question"] for test in rebased["tests"]}
    receipt = read(receipt_path(original))
    assert receipt["measurement"]["pack_id"] == rebased["pack_id"]
    assert receipt["measurement"][
        "measurement_input_sha256"
    ] == measurement_input_fingerprint(
        rebased, *build_context(task, [row["candidate"]])
    )
    assert receipt["pack_sha256"] == receipt_digest(rebased)
    receipts = Receipts(original.root, closed=True)
    delivered_measurement(
        receipts,
        Path("tasks/task/round_1"),
        "A",
        original.config["validation_benchmark"],
    )
    assert receipts.facts["provenance"] == "supplemental"


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", ["missing_test", "measurement_pack"])
async def test_invalid_cached_measurement_cannot_be_accepted(
    original, local_execution, changed
):
    await audit.run(original.root)
    path = receipt_path(original)
    row = read(path)
    if changed == "missing_test":
        row["measurement"]["results"].pop()
    else:
        row["measurement"]["pack_id"] = "unrelated-pack"
    write(path, row)
    before = retained_files(original.root)
    local_execution.calls.clear()
    with pytest.raises(ValueError):
        await audit.run(original.root)
    assert not local_execution.calls
    assert retained_files(original.root) == before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changed", ["delivery", "snapshot", "session", "native_status"]
)
@pytest.mark.parametrize("cached", [False, True])
async def test_modified_delivery_cannot_be_audited_as_the_original_native_artifact(
    original, local_execution, changed, cached
):
    if cached:
        await audit.run(original.root)
        local_execution.calls.clear()
    folder = original.root / "tasks/task/round_1"
    turn = (
        "1" if original.config["validation_benchmark"] == "claw_eval" else "delivery-1"
    )
    path = {
        "delivery": folder / "A/delivery.json",
        "snapshot": folder / "actors/A/snapshots" / turn / "candidate.json",
        "session": folder / "actors/A/actor_state.json",
        "native_status": folder / "actors/A/RESULT.json",
    }[changed]
    row = read(path)
    if changed == "delivery":
        row["candidate"]["artifact"]["payload"] = "substituted artifact"
    elif changed == "snapshot":
        row["artifact"]["payload"] = "substituted artifact"
    elif changed == "session":
        row["session"] = "unrelated-session"
    else:
        row["status"] = "failed"
    write(path, row)
    before = retained_files(original.root)
    with pytest.raises(ValueError):
        await audit.run(original.root)
    assert not local_execution.calls
    assert retained_files(original.root) == before


@pytest.mark.asyncio
async def test_failed_fresh_check_is_retained_and_never_silently_retried(
    original, local_execution
):
    before = retained_files(original.root)
    local_execution.failures.add("payload")
    with pytest.raises(ValueError, match="Checker execution failed"):
        await audit.run(original.root)
    assert len(local_execution.calls) == 2
    receipt = read(receipt_path(original))
    results = {r["test_id"]: r for r in receipt["measurement"]["results"]}
    assert results["final_response"]["status"] == "pass"
    assert results["payload"]["status"] == "unknown"
    assert results["payload"]["error"] is True
    assert results["payload"]["diagnostic"] == "Retained offline checker failure"
    assert not receipt_path(original, arm="B").exists()
    assert all(
        (original.root / name).read_bytes() == contents
        for name, contents in before.items()
    )
    retained = retained_files(original.root)
    local_execution.failures.clear()
    local_execution.calls.clear()
    with pytest.raises(ValueError, match="Checker execution failed"):
        await audit.run(original.root)
    assert not local_execution.calls
    assert retained_files(original.root) == retained
    assert all(s.should_exit and s.stopped for s in local_execution.servers)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changed",
    [
        "schema",
        "source",
        "pack_source",
        "artifact_sha256",
        "candidate_sha256",
        "pack_sha256",
        "provenance",
    ],
)
async def test_cached_receipt_cannot_bypass_native_provenance_validation(
    original, local_execution, changed
):
    await audit.run(original.root)
    path = receipt_path(original)
    row = read(path)
    row[changed] = {} if changed == "provenance" else "unrelated"
    write(path, row)
    before = retained_files(original.root)
    local_execution.calls.clear()
    with pytest.raises(ValueError):
        await audit.run(original.root)
    assert not local_execution.calls
    assert retained_files(original.root) == before
