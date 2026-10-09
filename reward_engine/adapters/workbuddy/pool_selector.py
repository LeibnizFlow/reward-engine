"""Source binding and persistence for independent final pool selection."""

from pathlib import Path

from reward_engine.adapters.workbuddy.io import (
    digest,
    file_hash,
    read,
    role_model,
    role_timeout_s,
    write,
)
from reward_engine.adapters.workbuddy.models import Role
from reward_engine.core.outcome import pool_selector


def implementation_identity():
    """Frozen code identity for the final pool selector's runtime binding."""
    package = Path(__file__).resolve().parents[2]
    paths = [
        *sorted((package / "core/outcome").glob("*.py")),
        *sorted((package / "adapters/workbuddy").glob("*.py")),
        package / "adapters/outcome_measurement.py",
        package / "core/batch.py",
    ]
    return digest({str(p.relative_to(package)): file_hash(p) for p in paths})


def collect_candidates(task_root):
    """Read completed round-3 public snapshots only; no private labels or previous verdicts."""
    root = Path(task_root)
    slots = [("round", 3, a, root / "round_3" / "actors" / a) for a in "ABC"]
    candidates, sources = {}, {}
    for kind, number, slot, path in slots:
        saved = read(path / "RESULT.json")
        if saved.get("status") != "completed" or not saved.get("candidate"):
            raise ValueError(
                f"Final pool requires a completed delivery: {kind}/{number}/{slot}"
            )
        candidate = saved["candidate"]
        key = f"{kind}/{number}/{slot}"
        candidates[key] = candidate
        sources[key] = {
            "candidate_id": candidate["candidate_id"],
            "source_kind": kind,
            "source_round": number,
            "source_slot": slot,
            "arm": slot if kind == "round" else None,
            "source": str((path / "RESULT.json").relative_to(root)),
        }
    return candidates, sources


def bind_source(decision, sources):
    source = sources.get(decision["candidate_key"])
    return {k: v for k, v in decision.items() if k != "ranking"} | {
        "round": 3,
        "arm": None,
        "candidate_id": None,
        "source": None,
        **(source or {}),
        "evidence_source": "final_selection/decision.json",
    }


async def finalize(config, task_root, task):
    root = Path(task_root) / "final_selection"
    candidates, sources = collect_candidates(task_root)

    def restore(name):
        path = root / (name + ".json")
        return read(path) if path.exists() else None

    def record(name, value):
        write(root / (name + ".json"), value)

    # Bind once: a later amendment must not silently change a partial deliberation.
    model = role_model(config, "final_selector")
    bound = {
        **config,
        "roles": {**config["roles"], "final_selector": model},
        "output": str(root),
        "role_stream_timeout_s": role_timeout_s(config),
    }
    runtime = {
        "model": model,
        "implementation": implementation_identity(),
        "source_binding": digest(sources),
        "endpoint": config["gateway_local"],
        "timeout": bound["role_stream_timeout_s"],
        "max_bytes": Role.request_budget_bytes,
    }
    previous = restore("sources")
    if previous is not None and previous != sources:
        raise ValueError("Final pool provenance changed")
    record("sources", sources)
    result = await pool_selector.select(
        task,
        candidates,
        Role(bound, "final_selector", root / "models"),
        runtime=runtime,
        record=record,
        restore=restore,
        max_bytes=Role.request_budget_bytes,
    )
    selected = bind_source(result, sources)
    record("selected_delivery", selected)
    return selected
