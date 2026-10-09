"""Explicit replay of complete first-round products, with immutable provenance."""

from pathlib import Path

from reward_engine.adapters.workbuddy.io import file_hash, read, write


def freeze_sources(root, task_ids):
    sources = {}
    for task_id in task_ids:
        for phase, labels in (("seed", ("1", "2", "3")), ("round_1", "ABC")):
            for label in labels:
                slot = root / "tasks" / task_id
                slot /= f"seeds/{label}" if phase == "seed" else f"round_1/actors/{label}"
                row, spec = read(slot / "RESULT.json"), read(slot / "spec.json")
                if row.get("status") != "completed" or not row.get("candidate"):
                    raise ValueError("Replay requires all three complete seeds and A/B/C products")
                if spec["task"]["task_id"] != task_id or spec["grade"] != (phase != "seed"):
                    raise ValueError("Replay source task or grading scope differs")
                if phase != "seed" and read(slot / "OFFICIAL.private.json").get("status") != "scored":
                    raise ValueError("Replay requires valid original native scores")
                sources[f"{task_id}/{phase}/{label}"] = {
                    "path": str(slot), "candidate_id": row["candidate"]["candidate_id"],
                    "hashes": {str(p.relative_to(slot)): file_hash(p)
                               for p in slot.rglob("*") if p.is_file()},
                }
    return sources


def replay(source, spec, output):
    original = Path(source["path"])
    for name, expected in source["hashes"].items():
        if file_hash(original / name) != expected:
            raise ValueError(f"Replay source changed: {name}")
    prior_spec = read(original / "spec.json")
    if prior_spec["task"] != spec["task"] or prior_spec["grade"] != spec["grade"]:
        raise ValueError("Replay source task or grading scope differs")
    row = read(original / "RESULT.json")
    if row["candidate"]["candidate_id"] != spec["candidate_id"]:
        raise ValueError("Replay candidate identity differs")
    provenance = {**source, "original_spec_identity": row["spec_identity"],
                  "original_resources": row.get("resources", {}),
                  "generation_protocol": "Original source prompt, pack and session; see source spec.json",
                  "current_protocol": "Reassess the existing product with the current Jury and measurements"}
    write(output / "REUSE.json", provenance)
    # Previous pack estimates and memory are not measurements of the current pack.
    row.update(spec_identity=spec["spec_identity"], reused_from=str(original),
               final_measurement=None, memory=[], task_progress=None, resources={})
    if spec["grade"]:
        score = read(original / "OFFICIAL.private.json")
        score["reused_from"] = str(original / "OFFICIAL.private.json")
        write(output / "OFFICIAL.private.json", score)
    write(output / "ACTOR_RESULT.json", row)
    write(output / "RESULT.json", row)
    return row
