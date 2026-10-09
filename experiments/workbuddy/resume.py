"""Restore complete rounds without regenerating memory identities or earlier work."""

from reward_engine.adapters.workbuddy.io import read
from reward_engine.core.outcome.arms import arm_spec, has_hypothesis


def retained_arm(phase, summary, arm, row):
    """Reconstruct only the treatment annotation added by the core after sampling."""
    saved = summary["arms"][arm]
    if "hypothesis_applied" not in saved and "hypothesis_applied" not in row:
        return row
    protocol = read(phase / "arm_protocol.json")
    method = read(phase / "hypothesis.json")
    spec = arm_spec(protocol["id"], arm)
    applied = spec["hypothesis"] and has_hypothesis(method)
    if (summary["arm_protocol"] != protocol["id"]
            or protocol["arms"][arm] != spec
            or protocol["hypothesis_applied"] is not has_hypothesis(method)
            or saved.get("hypothesis_applied") is not applied
            or ("hypothesis_applied" in row and row["hypothesis_applied"] is not applied)):
        raise ValueError("Checkpoint hypothesis treatment differs from retained protocol")
    return {**row, "hypothesis_applied": applied}


def completed_rounds(root, task, seeds):
    pool, history, checkpoint = list(seeds), [], None
    for number in range(1, 4):
        phase = root / f"round_{number}"
        if not (phase / "checkpoint.json").exists():
            break
        candidate = read(phase / "checkpoint.json")
        summary = read(phase / "summary.private.json")
        if summary.get("status") != "completed":
            break
        if candidate["task"] != task or candidate["completed_round"] != number:
            raise ValueError("Checkpoint task or round differs")
        for arm in "ABC":
            row = read(phase / f"actors/{arm}/RESULT.json")
            if (summary["arms"][arm] != retained_arm(phase, summary, arm, row)
                    or row.get("status") != "completed"):
                raise ValueError("Checkpoint product differs from retained actor result")
            if summary["official"][arm] != read(phase / f"actors/{arm}/OFFICIAL.private.json"):
                raise ValueError("Checkpoint official receipt differs")
            pool.append(row["candidate"])
        if candidate["pool"] != pool:
            raise ValueError("Checkpoint pool differs from observed products")
        for key, file in (("memory", "memory.json"), ("checks", "check_sources.json"),
                          ("previous_pack", "pack.json")):
            if candidate[key] != read(phase / file):
                raise ValueError(f"Checkpoint {key} differs from retained state")
        history.append(summary)
        checkpoint = candidate
    return history, checkpoint
