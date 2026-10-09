#!/usr/bin/env python3
"""Paired significance tests and selection-headroom statistics for the main text.

Reads frozen WorkBuddy rounds and newly frozen seed grades, historical claw-eval scores, selection-quality data,
and the verified claw-eval continuation summary. No actor, grader, or model calls.

Outputs data/main_statistics.json. Every number cited in Section 4 of the
ICLR revision that is not already produced by another prepare_* script comes
from this file. workbuddy is historical round three (threshold 85);
workbuddy_three_round is the zero-imputed three-round analysis (threshold 80).
The claw_eval key is historical round one; claw_eval_final is
round three. The original three-round keys remain historical diagnostics.
workbuddy_six_baseline and claw_eval_six_baseline supply the current main-text comparisons.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
TOL = 1e-8
ITERS = 200_000


def wtl(a: np.ndarray, b: np.ndarray) -> tuple[int, int, int]:
    d = a - b
    return int((d > TOL).sum()), int((np.abs(d) <= TOL).sum()), int((d < -TOL).sum())


def task_bootstrap(diff: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), (ITERS, len(diff)))
    return np.percentile(diff[idx].mean(1), [2.5, 97.5]).tolist()


def cluster_bootstrap(diff: np.ndarray, clusters: list[str], seed: int) -> list[float]:
    """Resample whole clusters; weight sampled task versions equally."""
    labels = sorted(set(clusters))
    members = [np.array([i for i, c in enumerate(clusters) if c == lab]) for lab in labels]
    sums = np.array([diff[m].sum() for m in members])
    sizes = np.array([len(m) for m in members])
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(labels), (ITERS, len(labels)))
    means = sums[draw].sum(1) / sizes[draw].sum(1)
    return np.percentile(means, [2.5, 97.5]).tolist()


def paired_tests(x: np.ndarray, y: np.ndarray) -> dict:
    w, t, l = wtl(x, y)
    d = x - y
    nonzero = d[np.abs(d) > TOL]
    return {
        "wins": w, "ties": t, "losses": l,
        "median_difference": float(np.median(d)),
        "sign_test_two_sided_p": float(binomtest(w, w + l, 0.5).pvalue) if w + l else 1.0,
        "wilcoxon_two_sided_p": float(wilcoxon(nonzero).pvalue) if len(nonzero) else 1.0,
    }


def mcnemar_exact(x_pass: np.ndarray, y_pass: np.ndarray) -> dict:
    b = int((x_pass & ~y_pass).sum())  # x passes, y fails
    c = int((~x_pass & y_pass).sum())
    return {"x_only": b, "y_only": c,
            "exact_two_sided_p": float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0}


def headroom(selected: float, uniform: float, oracle: float) -> float:
    return (selected - uniform) / (oracle - uniform)


def main() -> None:
    assert (ROOT / 'data/workbuddy_seed_scores.json').is_file(), 'Freeze all 150 seed grades before publishing the six-run baseline.'
    out: dict = {"iterations": ITERS, "tie_tolerance": TOL}

    wb = json.loads((ROOT / "data/office50.json").read_text())["tasks"]
    a = np.array([t["A"] for t in wb]); b = np.array([t["B"] for t in wb])
    c = np.array([t["C"] for t in wb]); s = np.array([t["selected_score"] for t in wb])
    oracle = np.maximum(np.maximum(a, b), c)
    uniform = (a + b + c) / 3
    out["workbuddy"] = {
        "n_tasks": len(wb),
        "means": {"task_only": a.mean(), "reward_revision": b.mean(),
                  "hypothesis": c.mean(), "reward_engine": s.mean(),
                  "uniform": uniform.mean(), "oracle": oracle.mean()},
        "reward_engine_vs_task_only": {
            "mean_difference": float((s - a).mean()),
            "task_bootstrap_ci95": task_bootstrap(s - a, 20260915),
            **paired_tests(s, a),
        },
        "candidates_vs_task_only": {
            name: {"mean_difference": float((x - a).mean()),
                   "task_bootstrap_ci95": task_bootstrap(x - a, 20260915),
                   **paired_tests(x, a)}
            for name, x in (("reward_revision", b), ("hypothesis", c))},
        "pass_at_85": {"task_only": int((a >= 85).sum()), "reward_engine": int((s >= 85).sum()),
                        "reward_revision": int((b >= 85).sum()), "hypothesis": int((c >= 85).sum()),
                        "mcnemar": mcnemar_exact(s >= 85, a >= 85)},
        "headroom_recovered_over_uniform": headroom(s.mean(), uniform.mean(), oracle.mean()),
        "selected_arm_counts": {k: sum(t["selected_arm"] == k for t in wb) for k in "ABC"},
        "largest_positive_difference": float((s - a).max()),
        "largest_negative_difference": float((s - a).min()),
    }

    claw = json.loads((ROOT / "data/claw_eval.json").read_text())["tasks"]
    rows = list(csv.DictReader((ROOT / "data/selection_quality.csv").open()))
    cluster_of = {r["task"]: r["cluster"] for r in rows if r["benchmark"] != "WorkBuddy"}
    clusters = [cluster_of[t["task"]] for t in claw]
    am = np.array([t["A_mean"] for t in claw]); ar = np.array([t["A_run"] for t in claw])
    cb = np.array([t["B"] for t in claw]); cc = np.array([t["C"] for t in claw])
    cs = np.array([t["selected_score"] for t in claw])
    runs = np.array([[t["baseline_scores"][k] for k in ("A_run", "seed_1", "seed_2", "seed_3")]
                     for t in claw])
    c_oracle = np.maximum(np.maximum(ar, cb), cc)
    c_uniform = (ar + cb + cc) / 3
    out["claw_eval"] = {
        "n_tasks": len(claw), "n_scenario_clusters": len(set(clusters)),
        "means": {"task_only_mean": am.mean(), "reward_revision": cb.mean(),
                  "hypothesis": cc.mean(), "reward_engine": cs.mean(),
                  "uniform_pool": c_uniform.mean(), "oracle_pool": c_oracle.mean(),
                  "best_of_four_task_only_runs": runs.max(1).mean()},
        "reward_engine_vs_task_only_mean": {
            "mean_difference": float((cs - am).mean()),
            "task_bootstrap_ci95": task_bootstrap(cs - am, 20260920),
            "scenario_cluster_bootstrap_ci95": cluster_bootstrap(cs - am, clusters, 20260920),
            **paired_tests(cs, am),
        },
        "candidates_vs_task_only_mean": {
            name: {"mean_difference": float((x - am).mean()),
                   "task_bootstrap_ci95": task_bootstrap(x - am, 20260920),
                   **paired_tests(x, am)}
            for name, x in (("reward_revision", cb), ("hypothesis", cc))},
        "pass_at_75": {"task_only_runs_passed": int((runs >= 75).sum()), "task_only_runs": int(runs.size),
                        "reward_engine": int((cs >= 75).sum()),
                        "reward_revision": int((cb >= 75).sum()), "hypothesis": int((cc >= 75).sum())},
        "headroom_recovered_over_uniform": headroom(cs.mean(), c_uniform.mean(), c_oracle.mean()),
        "selected_arm_counts": {k: sum(t["selected_arm"] == k for t in claw) for k in "ABC"},
    }
    # Preserve the frozen first-round diagnostics; publish the fixed continuation
    # endpoint as a separate result, never a best-of-rounds score.
    continuation = json.loads((ROOT / "data/claw_eval_continuation_statistics.json").read_text())
    assert continuation['primary_round'] == 3
    assert continuation['snapshot_sha256'] == hashlib.sha256((ROOT/'data/claw_eval_continuation.json').read_bytes()).hexdigest()
    out["claw_eval_final"] = continuation["rounds"]["3"]
    out["claw_eval_three_round"] = continuation["three_round_summary"]
    from prepare_three_round_statistics import workbuddy_summary
    out["workbuddy_three_round"] = workbuddy_summary()
    out["historical_endpoints"] = {"workbuddy_round3": out["workbuddy"],
                                    "claw_eval_round3": out["claw_eval_final"]}
    (ROOT / "data/historical_endpoint_statistics.json").write_text(
        json.dumps(out["historical_endpoints"], indent=2, default=float) + "\n")
    path = ROOT / "data/main_statistics.json"
    path.write_text(json.dumps(out, indent=2, default=float) + "\n")
    write_main_table(out)
    from prepare_six_baseline import main as prepare_six
    prepare_six()
    from sync_manuscript_tables import normalize_active_tables
    normalize_active_tables()
    print(json.dumps(out, indent=2, default=float))


def fmt_p(p: float) -> str:
    return r"$<\!0.001$" if p < 0.001 else f"{p:.3f}"


def fmt_num(v: float) -> str:
    return f"$-${abs(v):.2f}" if v < 0 else f"{v:.2f}"


def fmt_ci(ci: list[float]) -> str:
    return f"[{fmt_num(ci[0])}, {fmt_num(ci[1])}]"


def fmt_delta(v: float) -> str:
    return f"$-${abs(v):.2f}" if v < 0 else f"+{v:.2f}"


def comparison_cells(c: dict, interval: str = 'task_bootstrap_ci95') -> str:
    return (f"{fmt_delta(c['mean_difference'])} {fmt_ci(c[interval])} & "
            f"{c['wins']}/{c['ties']}/{c['losses']} & {fmt_p(c['sign_test_two_sided_p'])}")


def write_main_table(out: dict) -> None:
    """Keep historical endpoints in the appendix; summarize three rounds for both benchmarks in main."""
    wb, cl = out["workbuddy"], out["claw_eval_final"]
    wp, cp = wb["pass_at_85"], cl["pass_at_75"]
    wc, cc = wb["candidates_vs_task_only"], cl["comparisons"]
    cluster_interval = 'scenario_cluster_bootstrap_ci95'

    def rate(k: int, n: int, bold: bool = False) -> str:
        pct = f"{100*k/n:.2f}"
        return (f"\\textbf{{{pct}}}" if bold else pct) + f" ({k}/{n})"

    def mean(v: float, bold: bool = False) -> str:
        return f"\\textbf{{{v:.2f}}}" if bold else f"{v:.2f}"

    lines = [
        r"\begin{tabular}{lrrlrr}",
        r"\toprule",
        r"Method & Mean & Pass rate (\%) & $\Delta$ mean [95\% CI] & W/T/L & Sign test $p$ \\",
        r"\midrule",
        r"\multicolumn{6}{l}{\textit{(a) WorkBuddy Bench Office: 50 tasks, pass if score $\geq 85$}} \\",
        f"Baseline & {mean(wb['means']['task_only'])} & {rate(wp['task_only'], 50)} & --- & --- & --- \\\\",
        f"Reward-guided revision & {mean(wb['means']['reward_revision'])} & {rate(wp['reward_revision'], 50)} & "
        f"{comparison_cells(wc['reward_revision'])} \\\\",
        f"Hypothesis guidance & {mean(wb['means']['hypothesis'])} & {rate(wp['hypothesis'], 50)} & "
        f"{comparison_cells(wc['hypothesis'])} \\\\",
        f"Selected by reward & {mean(wb['means']['reward_engine'], True)} & {rate(wp['reward_engine'], 50, True)} & "
        f"{comparison_cells(wb['reward_engine_vs_task_only'])} \\\\",
        r"\midrule",
        r"\multicolumn{6}{l}{\textit{(b) claw-eval: round 3, 104 task versions, pass if score $\geq 75$}} \\",
        f"Baseline & {mean(cl['means']['A'])} & "
        f"{rate(cp['A'], 104)} & --- & --- & --- \\\\",
        f"Reward-guided revision & {mean(cl['means']['B'])} & {rate(cp['B'], 104)} & "
        f"{comparison_cells(cc['B_vs_A'], cluster_interval)} \\\\",
        f"Hypothesis guidance & {mean(cl['means']['C'])} & {rate(cp['C'], 104)} & "
        f"{comparison_cells(cc['C_vs_A'], cluster_interval)} \\\\",
        f"Selected by reward & {mean(cl['means']['selected_score'], True)} & {rate(cp['selected_score'], 104, True)} & "
        f"{comparison_cells(cc['selected_score_vs_A'], cluster_interval)} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    (ROOT / "tables/candidate_results.tex").write_text("\n".join(lines) + "\n")
    from prepare_three_round_statistics import write_tables
    write_tables(out)


if __name__ == "__main__":
    main()
