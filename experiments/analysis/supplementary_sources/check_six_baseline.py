"""Independent audit of six-run estimands, endpoints and preserved pool metrics."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, wilcoxon

ROOT = Path(__file__).resolve().parents[1]


def check():
    main = json.loads((ROOT / "data/main_statistics.json").read_text())
    for b, threshold, n in [("workbuddy", 80, 50), ("claw_eval", 75, 104)]:
        s = json.loads((ROOT / f"data/{b}_six_baseline_statistics.json").read_text())
        rows = list(csv.DictReader((ROOT / f"data/{b}_six_baseline_ledger.csv").open()))
        assert len(rows) == n and len({r["task"] for r in rows}) == n
        arrays = {
            k: np.array([[float(r[f"{k}_{j}"]) for j in (1, 2, 3)] for r in rows])
            for k in ("A", "B", "C", "selected_score", "seed")
        }
        seed_array = arrays.pop("seed")
        if b == "workbuddy":
            source = json.loads((ROOT / "data/workbuddy_seed_scores.json").read_text())[
                "scores"
            ]
            lookup = {(r["task"], r["seed"]): r["score"] for r in source}
        else:
            source = json.loads((ROOT / "data/claw_eval.json").read_text())["tasks"]
            lookup = {
                (r["task"], j): r["baseline_scores"][f"seed_{j}"]
                for r in source
                for j in (1, 2, 3)
            }
            original = {r["task"]: r for r in source}
            np.testing.assert_allclose(
                arrays["A"][:, 0],
                [original[r["task"]]["baseline_scores"]["A_run"] for r in rows],
                atol=1e-8,
                rtol=0,
            )
        np.testing.assert_equal(
            seed_array, [[lookup[r["task"], j] for j in (1, 2, 3)] for r in rows]
        )
        old_rows = list(
            csv.DictReader((ROOT / f"data/{b}_three_round_summary.csv").open())
        )
        old_lookup = {r["task"]: r for r in old_rows}
        for k, a in arrays.items():
            np.testing.assert_allclose(
                a.mean(1), [float(old_lookup[r["task"]][k + "_mean"]) for r in rows]
            )
        baseline = np.concatenate([seed_array, arrays["A"]], 1)
        assert s == main[b + "_six_baseline"] and s["threshold"] == threshold
        for k, a in dict(baseline=baseline, **arrays).items():
            v = s["arms"][k]
            p = a >= threshold
            g = p.reshape(n, -1, 3)
            assert v["n_runs"] == p.size and v["passed_runs"] == int(p.sum())
            assert v["success_denominator"] == g.shape[0] * g.shape[1]
            assert v["pass_at_3_count"] == int(g.any(2).sum())
            assert v["pass_power_3_count"] == int(g.all(2).sum())
            assert np.isclose(v["mean_score"], a.mean())
            assert np.isclose(v["average_pass_rate"], p.mean())
            assert np.isclose(v["pass_at_3"], g.any(2).mean())
            assert np.isclose(v["pass_power_3"], g.all(2).mean())
        for scope, comparisons in [(None, s["comparisons"])] + [
            (int(r), ep["comparisons"]) for r, ep in s["rounds"].items()
        ]:
            for k, a in arrays.items():
                x = a.mean(1) if scope is None else a[:, scope - 1]
                d = x - baseline.mean(1)
                c = comparisons[k + "_vs_baseline"]["mean_score"]
                w = int((d > 1e-8).sum())
                left = int((d < -1e-8).sum())
                assert (c["wins"], c["ties"], c["losses"]) == (w, n - w - left, left)
                assert np.isclose(c["mean_difference"], d.mean()) and np.isclose(
                    c["median_difference"], np.median(d)
                )
                assert np.isclose(
                    c["sign_test_two_sided_p"],
                    binomtest(w, w + left).pvalue if w + left else 1,
                )
                assert np.isclose(
                    c["wilcoxon_two_sided_p"],
                    wilcoxon(d[abs(d) > 1e-8]).pvalue if w + left else 1,
                )
                rate = (
                    (a >= threshold).mean(1)
                    if scope is None
                    else (a[:, scope - 1] >= threshold).astype(float)
                )
                assert np.isclose(
                    comparisons[k + "_vs_baseline"]["average_pass_rate"][
                        "mean_difference"
                    ],
                    (rate - (baseline >= threshold).mean(1)).mean(),
                )
        # Neither the six-run mean nor historical seed candidates enter the runtime pool.
        assert s["utility"] == main[b + "_three_round"]["utility"]
        for r, ep in s["rounds"].items():
            oracle = np.maximum.reduce([arrays[k][:, int(r) - 1] for k in "ABC"])
            for k, a in arrays.items():
                assert np.isclose(
                    ep["utility"]["mean_regret"][k], (oracle - a[:, int(r) - 1]).mean()
                )
                assert ep["utility"]["best_hits"][k] == int(
                    np.isclose(a[:, int(r) - 1], oracle, atol=1e-8, rtol=0).sum()
                )
        for path, h in s["input_sha256"].items():
            assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == h
        if b == "workbuddy":
            i = [r["task"] for r in rows].index("analyst-forecast-extract-L3-018")
            assert arrays["A"][i, 0] == 0 and len(s["imputations"]) == 1
        else:
            assert (
                s["arms"]["baseline"]["passed_runs"],
                s["arms"]["baseline"]["pass_at_3_count"],
                s["arms"]["baseline"]["pass_power_3_count"],
            ) == (560, 204, 163)
            assert s["excluding_known_exposure"]["n_tasks"] == 103
    seeds = json.loads((ROOT / "data/workbuddy_seed_scores.json").read_text())
    for r in seeds["scores"]:
        p = ROOT / r["receipt"]
        assert hashlib.sha256(p.read_bytes()).hexdigest() == r["receipt_sha256"]
        detail = json.loads((p.parent / "composite_score.json").read_text())["score"]
        assert all(
            j.get("ok") is not False and not j.get("errors") for j in detail["judges"]
        )
        assert not any(
            v.get("metadata", {}).get("error_type") for v in detail["verdicts"]
        )
        assert np.isclose(100 * detail["overall"], r["score"])
    print(
        "Six-run scores, triplet counts, paired tests, endpoints, source hashes, imputation and pool utility: passed"
    )


if __name__ == "__main__":
    check()
