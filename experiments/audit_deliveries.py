"""Measure every delivered artifact against every generated check after a run.

This supplements retained original measurements; it does not alter learning,
selection, actor artifacts or official grades. All calls use the recorded model
routes and the actual artifacts, with exact input hashes in each receipt.
"""

import argparse
import asyncio
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import uvicorn

from experiments.pipeline_audit import (
    Receipts,
    delivered_measurement,
    final_candidate_source,
    measurement,
    receipt_digest,
)
from reward_engine.adapters.workbuddy.gateway import application
from reward_engine.adapters.workbuddy.io import digest, file_hash, read, write
from reward_engine.adapters.workbuddy.models import registry
from reward_engine.core.batch import BatchPolicy
from reward_engine.core.outcome.context import build_context
from reward_engine.core.outcome.harness import measure


async def run(root):
    root = root.resolve()
    config = read(root / "config.json")
    if read(root / "validation_status.json")["status"] == "running":
        raise ValueError("Wait for the original execution to stop before auditing")
    if config["validation_benchmark"] == "claw_eval":
        from experiments.claw_eval.opencode.campaign import verify_freeze

        verify_freeze(config)
    else:
        from experiments.workbuddy.experiment import verify

        verify(config)
    source = Path(__file__).resolve().parents[1]
    code_hashes = {
        str(p.relative_to(source)): file_hash(p)
        for p in (source / "reward_engine").rglob("*.py")
    }
    code_hashes["experiments/audit_deliveries.py"] = file_hash(Path(__file__))
    server = uvicorn.Server(
        uvicorn.Config(
            application(config),
            host="0.0.0.0",
            port=config["gateway_port"],
            log_level="warning",
            timeout_graceful_shutdown=10,
        )
    )
    server.capture_signals = nullcontext
    pending = asyncio.create_task(server.serve())
    try:
        for _ in range(100):
            if server.started:
                break
            if pending.done():
                raise RuntimeError("Supplemental audit gateway failed")
            await asyncio.sleep(0.1)
        else:
            raise TimeoutError("Supplemental audit gateway timed out")
        task_root = root / "tasks" / config["validation_task_id"]
        for number in range(1, 4):
            folder = task_root / f"round_{number}"
            checkpoint_path, pack_path = (
                folder / "checkpoint.json",
                folder / "pack.json",
            )
            task, pack = read(checkpoint_path)["task"], read(pack_path)
            for arm in "ABC":
                delivery_path = folder / arm / "delivery.json"
                row = read(delivery_path)
                native_source = final_candidate_source(
                    root,
                    (folder / "actors" / arm).relative_to(root),
                    config["validation_benchmark"],
                )
                observed = read(root / native_source)
                if row["candidate"] != observed:
                    raise ValueError("Delivery differs from the native final artifact")
                effective_task = row.get("effective_task") or task
                effective_pack = row.get("effective_pack") or pack
                if effective_task != task and effective_pack is pack:
                    from reward_engine.core.outcome.task_sequence import TaskSequence

                    effective_pack = TaskSequence(effective_task).rebase(pack)
                binding = digest(
                    {
                        "task": effective_task,
                        "candidate": row["candidate"],
                        "pack": effective_pack,
                    }
                )
                slot = (
                    root
                    / "supplemental_full_audit"
                    / config["validation_task_id"]
                    / f"round_{number}"
                    / arm
                )
                receipt = folder / arm / "outcome_estimate.full.supplemental.json"
                hashes = {
                    str(p): file_hash(p)
                    for p in (
                        delivery_path,
                        pack_path,
                        checkpoint_path,
                        root / native_source,
                    )
                }
                if receipt.exists():
                    previous = read(receipt)
                    if (
                        previous["input_binding"] != binding
                        or previous["source_hashes"] != hashes
                        or previous["code_hashes"] != code_hashes
                    ):
                        raise ValueError(
                            "Supplemental audit inputs/code changed; retain original receipts"
                        )
                    measurement(previous["measurement"], effective_pack)
                    delivered_measurement(
                        Receipts(root, closed=True),
                        folder.relative_to(root),
                        arm,
                        config["validation_benchmark"],
                    )
                    continue
                context, evidence = build_context(effective_task, [row["candidate"]])
                measured = await measure(
                    effective_pack,
                    registry(config, slot / "measurement"),
                    context,
                    evidence,
                    policy=BatchPolicy(**config["batch"]),
                    test_ids=None,
                )
                write(
                    receipt,
                    {
                        "schema": "pipeline-supplemental-measurement-v1",
                        "source": native_source.as_posix(),
                        "pack_source": pack_path.relative_to(root).as_posix(),
                        "artifact_sha256": receipt_digest(observed["artifact"]),
                        "candidate_sha256": receipt_digest(observed),
                        "pack_sha256": receipt_digest(effective_pack),
                        "provenance": {
                            "kind": "supplemental_full_measurement",
                            "producer": "experiments.audit_deliveries",
                            "created_at": datetime.now(timezone.utc).isoformat(),
                        },
                        "input_binding": binding,
                        "source_hashes": hashes,
                        "code_hashes": code_hashes,
                        "measurement": measured,
                    },
                )
                measurement(measured, effective_pack)
                print(
                    f"Measured round {number} arm {arm}: {len(measured['results'])} check receipts",
                    flush=True,
                )
    finally:
        server.should_exit = True
        await pending


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(run(args.output))


if __name__ == "__main__":
    main()
