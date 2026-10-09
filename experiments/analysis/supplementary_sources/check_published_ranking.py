"""Check current positive-tie manuscript presentation without regenerating files."""

import csv
from pathlib import Path

import numpy as np
from publish_tie_credit_ranking import CONDITIONS, load, table_bodies

ROOT = Path(__file__).resolve().parents[1]


def check():
    summary, _ = load()
    for path, body in table_bodies().items():
        assert (ROOT / path).read_text() == body, f"Stale ranking table: {path}"
        dest = ROOT / ("main.tex" if "order_agreement" in path else "appendix.tex")
        assert body.strip() in dest.read_text(), f"Stale embedded table: {path}"
    with (ROOT / "data/frozen_reward_tie_credit/task_statistics.csv").open() as f:
        rows = list(csv.DictReader(f))
    for b, conditions in summary["benchmarks"].items():
        for condition in CONDITIONS:
            s = conditions[condition]
            eligible = [
                r
                for r in rows
                if r["benchmark"] == b
                and r["threshold"] == condition
                and r["positive_tie_score"] != ""
            ]
            assert len(eligible) == s["defined_tasks"]
            values = []
            for r in eligible:
                c, d, t = [
                    int(r[k])
                    for k in ("correct_pairs", "reversed_pairs", "reward_tie_pairs")
                ]
                assert c + d + t == int(r["eligible_pairs"]) > 0
                v = (c + t - d) / (c + d + t)
                values.append(v)
                assert np.isclose(v, float(r["positive_tie_score"]))
            assert np.isclose(np.mean(values), s["positive_tie_score"]["mean"])
            assert np.isclose(
                s["positive_tie_score"]["mean"], 2 * s["order_agreement"]["mean"] - 1
            )
    main = (ROOT / "main.tex").read_text()
    appendix = (ROOT / "appendix.tex").read_text()
    assert r"\tau_+=2a-1" in main and r"not Kendall" in main
    assert "chosen after inspecting several thresholds" in appendix
    assert r"not standard Kendall" in appendix
    for text in (main, appendix):
        for stale in (
            "0.171",
            "0.226",
            "0.233",
            "0.216",
            "0.110",
            "0.243",
            r"Mean $\tau_b$",
            r"mean $\tau_b$",
        ):
            assert stale not in text, f"Stale Kendall result in manuscript: {stale}"
    for phrase in ("non-reversal", "Entirely reward-tied", "undefined"):
        assert phrase in main + appendix
    assert "Unfiltered" in main and "Filtered" in main and r"$\geq3$" in appendix
    assert (ROOT / "figures/rewardpack_joint_repair_tau.pdf").is_file()
    print(
        "Current agreement / positive-tie definitions, counts, tables and manuscript consistency: passed"
    )


if __name__ == "__main__":
    check()
