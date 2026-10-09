"""Reporting-only join between an independent selection and its exact official receipt."""

from pathlib import Path

from experiments.workbuddy.scoring import scoring_errors
from reward_engine.adapters.workbuddy.io import read


def selected_score(task_root, selected):
    base = {
        "selection_status": selected.get("status"),
        "candidate_id": selected.get("candidate_id"),
        "source_kind": selected.get("source_kind"),
        "source_round": selected.get("source_round"),
        "source_slot": selected.get("source_slot"),
        "arm": selected.get("arm"),
        "source": selected.get("source"),
        "status": "not_run",
        "overall": None,
    }
    if not selected.get("candidate_key"):
        return base
    kind, number, slot = (
        selected["source_kind"],
        selected["source_round"],
        selected["source_slot"],
    )
    if (
        kind == "seed"
        and number == 0
        and slot in {"1", "2", "3"}
        and selected["arm"] is None
    ):
        relative = Path("seeds") / slot
    elif (
        kind == "round"
        and type(number) is int
        and number in {1, 2, 3}
        and slot in "ABC"
        and len(slot) == 1
        and selected["arm"] == slot
    ):
        relative = Path(f"round_{number}") / "actors" / slot
    else:
        raise ValueError("Invalid selected delivery origin")
    if selected["candidate_key"] != f"{kind}/{number}/{slot}" or selected[
        "source"
    ] != str(relative / "RESULT.json"):
        raise ValueError("Selected delivery path does not match origin")
    root = Path(task_root) / relative
    actual = read(root / "RESULT.json")
    if (
        actual.get("status") != "completed"
        or actual["candidate"]["candidate_id"] != selected["candidate_id"]
    ):
        raise ValueError("Selected candidate differs from source delivery")
    path = root / "OFFICIAL.private.json"
    receipt = read(path) if path.exists() else {"status": "not_run"}
    native = Path(receipt["native_score"]) if receipt.get("native_score") else None
    errors = (
        scoring_errors(read(native))
        if native and native.exists()
        else receipt.get("scoring_errors", [])
    )
    status = "judge_error" if errors else receipt["status"]
    return {
        **base,
        "status": status,
        "overall": receipt.get("overall") if status == "scored" else None,
        "judge_errors": len(errors),
    }
