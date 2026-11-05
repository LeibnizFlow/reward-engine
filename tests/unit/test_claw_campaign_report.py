"""Legacy reporting retains zero scores and distinguishes missing grading."""

from experiments.claw_eval.campaign import report
from reward_engine.adapters.workbuddy.io import read, write


def test_legacy_report_counts_zero_scores_and_missing_usage(tmp_path):
    task = tmp_path / "tasks/synthetic"
    write(task / "state.json", {"status": "completed"})
    for arm, score in (("A", 0), ("B", 1)):
        actor = task / "round_1/actors" / arm
        write(actor / "RESULT.json", {"status": "completed"})
        write(actor / "OFFICIAL.private.json", {"status": "graded", "task_score": score})
    write(tmp_path / "transport/model/call/receipt.json", {
        "model": "synthetic", "attempts": 2,
        "stream_attempts": [{"usage": {"prompt_tokens": 10, "completion_tokens": 5}}],
    })
    report({"output": str(tmp_path), "task_ids": ["synthetic"], "seeds": 3})
    result = read(tmp_path / "report.json")
    assert result["completed_actors"] == 2
    assert result["valid_scores"] == 2
    assert result["tasks"]["synthetic"]["scores"] == {"A": 0, "B": 1, "C": None}
    assert result["paired_differences"]["B-A"] == {"n": 1, "mean": 1}
    assert result["usage"]["transport/synthetic"]["missing_usage"] == 1
    assert (tmp_path / "scores.csv").exists()
    assert (tmp_path / "REPORT.md").exists()
