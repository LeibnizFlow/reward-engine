"""Independent metric ledger, test, historical-preservation and table checks."""

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, wilcoxon

ROOT = Path(__file__).resolve().parents[1]


def check_metrics():
    out = json.loads((ROOT / "data/main_statistics.json").read_text())
    expected = {
        "workbuddy": [(98, 43, 21), (106, 42, 28), (101, 44, 20), (107, 43, 27)],
        "claw_eval": [(285, 102, 85), (292, 102, 91), (271, 103, 73), (292, 104, 87)],
    }
    tests = {
        "workbuddy": [
            (30, 1, 19, 0.1524, 0.4067),
            (26, 1, 23, 0.7754, 0.8206),
            (29, 2, 19, 0.1934, 0.0700),
        ],
        "claw_eval": [
            (58, 11, 35, 0.0220, 0.0327),
            (33, 8, 63, 0.0029, 0.00025),
            (53, 19, 32, 0.0295, 0.0309),
        ],
    }
    for benchmark, threshold in [("workbuddy", 80), ("claw_eval", 75)]:
        summary = out[benchmark + "_three_round"]
        if benchmark == "workbuddy":
            raw = json.loads((ROOT / "data/office50.json").read_text())["rounds"]
            rows = []
            flags = []
            for t in raw:
                for r in t["rounds"]:
                    row = dict(r, task=t["task"])
                    for k in filter(None, r["invalid_scoring_arms"].split(",")):
                        flags.append((t["task"], r["round"], k, r[k]))
                        row[k] = 0
                    # Keep the archived selected outcome; the imputed arm was not selected.
                    rows.append(row)
            assert flags == [("analyst-forecast-extract-L3-018", 1, "A", 72.8)]
            assert summary["imputations"][0]["analysis_score"] == 0
            assert len(summary["imputations"]) == 1
        else:
            rows = json.loads((ROOT / "data/claw_eval_continuation.json").read_text())[
                "tasks"
            ]
        tasks = sorted({r["task"] for r in rows})
        index = {(r["task"], r["round"]): r for r in rows}
        assert len(index) == 3 * len(tasks) == len(rows)
        arrays = {
            k: np.array([[index[t, n][k] for n in (1, 2, 3)] for t in tasks])
            for k in ("A", "B", "C", "selected_score")
        }
        assert (
            summary["threshold"] == threshold
            and summary["bootstrap"]["iterations"] == 200000
        )
        assert summary["bootstrap"]["unit"] == (
            "task" if benchmark == "workbuddy" else "scenario"
        )
        for (k, a), counts in zip(arrays.items(), expected[benchmark], strict=False):
            v = summary["arms"][k]
            passed = a >= threshold
            actual = (
                int(passed.sum()),
                int(passed.any(1).sum()),
                int(passed.all(1).sum()),
            )
            assert (
                actual
                == counts
                == (v["passed_runs"], v["pass_at_3_tasks"], v["pass_power_3_tasks"])
            )
            assert np.isclose(v["mean_score"], a.mean())
            assert np.isclose(v["average_pass_rate"], passed.mean())
            assert np.isclose(v["pass_at_3"], counts[1] / len(tasks))
            assert np.isclose(v["pass_power_3"], counts[2] / len(tasks))
        for k, target in zip(
            ("B", "C", "selected_score"), tests[benchmark], strict=False
        ):
            diff = arrays[k].mean(1) - arrays["A"].mean(1)
            w, t, left = (
                int((diff > 1e-8).sum()),
                int((abs(diff) <= 1e-8).sum()),
                int((diff < -1e-8).sum()),
            )
            c = summary["comparisons"][k + "_vs_A"]["mean_score"]
            assert (w, t, left) == target[:3] == (c["wins"], c["ties"], c["losses"])
            assert np.isclose(
                c["sign_test_two_sided_p"], binomtest(w, w + left, 0.5).pvalue
            )
            assert np.isclose(
                c["wilcoxon_two_sided_p"], wilcoxon(diff[abs(diff) > 1e-8]).pvalue
            )
            assert abs(c["sign_test_two_sided_p"] - target[3]) < 0.00005
            assert abs(c["wilcoxon_two_sided_p"] - target[4]) < 0.00005
            assert np.isclose(c["median_difference"], np.median(diff))
            assert np.isclose(c["mean_difference"], diff.mean())
        ledger = list(
            csv.DictReader(
                (ROOT / "data" / f"{benchmark}_three_round_summary.csv").open()
            )
        )
        assert len(ledger) == len(tasks)
        for i, row in enumerate(ledger):
            assert row["task"] == tasks[i]
            for k, a in arrays.items():
                assert np.isclose(float(row[k + "_mean"]), a[i].mean())
                assert int(row[k + "_pass_count"]) == int((a[i] >= threshold).sum())
        oracle = np.maximum.reduce([arrays[k] for k in "ABC"])
        assert np.isclose(summary["utility"]["oracle_mean"], oracle.mean())
        for k, a in arrays.items():
            assert np.isclose(summary["utility"]["mean_regret"][k], (oracle - a).mean())
            assert summary["utility"]["best_hits"][k] == int(
                np.isclose(a, oracle, atol=1e-8, rtol=0).sum()
            )
    historical = json.loads(
        (ROOT / "data/historical_endpoint_statistics.json").read_text()
    )
    assert historical == out["historical_endpoints"]
    assert historical["workbuddy_round3"] == out["workbuddy"]
    assert historical["claw_eval_round3"] == out["claw_eval_final"]
    for name, dest in [
        ("main_results", "main.tex"),
        ("three_round_success", "main.tex"),
        ("three_round_diagnostics", "appendix.tex"),
        ("three_round_utility", "appendix.tex"),
    ]:
        assert (ROOT / "tables" / f"{name}.tex").read_text().strip() in (
            ROOT / dest
        ).read_text()
    ranking = (ROOT / "tables/known_reward_order_agreement.tex").read_text()
    assert "82.33 [74.67, 89.33]" in ranking and "80.53 [74.75, 86.08]" in ranking
    kendall = json.loads((ROOT / "data/known_reward_kendall.json").read_text())
    assert kendall["bootstrap"]["iterations"] == 20000
    for b, n, total, value in [
        ("workbuddy", 32, 50, 0.17134885148732476),
        ("claw_eval", 65, 104, 0.22636577244776426),
    ]:
        k = kendall["benchmarks"][b]
        assert (k["defined_tasks"], k["total_tasks"]) == (n, total)
        assert np.isclose(k["tau_b"]["mean"], value)
    from check_published_ranking import check as check_published_ranking

    check_published_ranking()
    assert ranking.strip() in (ROOT / "main.tex").read_text()
    print(
        "Three-round counts, paired tests, imputation, ledgers, utility and table integrity: passed"
    )


if __name__ == "__main__":
    check_metrics()
