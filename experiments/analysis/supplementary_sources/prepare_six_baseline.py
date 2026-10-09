"""Six-run baseline reporting, separate from immutable historical analyses.

Only frozen scores are read here. Baseline success aggregates two triplets,
never best-of-six or an all-six success event. Runtime utility stays in A/B/C.
"""

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from prepare_claw_continuation import three_round_arrays
from prepare_main_statistics import fmt_ci, fmt_delta, paired_tests
from prepare_three_round_statistics import LABEL, workbuddy_rows

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("A", "B", "C", "selected_score")
METRICS = ("mean_score", "average_pass_rate", "pass_at_3", "pass_power_3")
LABELS = dict(LABEL, baseline="Baseline (six runs)", A="Round baseline")


def read(p):
    return json.loads(Path(p).read_text())


def write(p, value):
    Path(p).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def inputs(benchmark):
    if benchmark == "workbuddy":
        rows, imputed = workbuddy_rows()
        seeds = read(ROOT / "data/workbuddy_seed_scores.json")
        assert len(seeds["scores"]) == 150 and seeds["actor_calls"] == 0
        lookup = {(r["task"], r["seed"]): r["score"] for r in seeds["scores"]}
        assert len(lookup) == 150 and all(
            r["status"] == "scored" for r in seeds["scores"]
        )
    else:
        rows = read(ROOT / "data/claw_eval_continuation.json")["tasks"]
        lookup = {
            (t["task"], s): t["baseline_scores"][f"seed_{s}"]
            for t in read(ROOT / "data/claw_eval.json")["tasks"]
            for s in (1, 2, 3)
        }
    tasks, clusters, arrays = three_round_arrays(rows)
    seeds = np.array([[lookup[t, s] for s in (1, 2, 3)] for t in tasks], dtype=float)
    assert (
        seeds.shape == (len(tasks), 3)
        and np.isfinite(seeds).all()
        and ((0 <= seeds) & (seeds <= 100)).all()
    )
    return tasks, clusters, arrays, seeds


def measurements(scores, threshold, triplets=True):
    passed = scores >= threshold
    m = dict(mean_score=scores.mean(1), average_pass_rate=passed.mean(1))
    if triplets:
        groups = passed.reshape(len(scores), -1, 3)
        m.update(pass_at_3=groups.any(2).mean(1), pass_power_3=groups.all(2).mean(1))
    return m


def arm_summary(scores, threshold, triplets=True):
    m = measurements(scores, threshold, triplets)
    result = {k: float(v.mean()) for k, v in m.items()}
    result.update(passed_runs=int((scores >= threshold).sum()), n_runs=int(scores.size))
    if triplets:
        groups = (scores >= threshold).reshape(len(scores), -1, 3)
        result.update(
            success_denominator=int(groups.shape[0] * groups.shape[1]),
            success_unit="task-groups" if groups.shape[1] > 1 else "tasks",
            pass_at_3_count=int(groups.any(2).sum()),
            pass_power_3_count=int(groups.all(2).sum()),
        )
    return result


def paired_all(x, y, clusters, benchmark):
    """Same resample indices for every metric; retain each task's full block."""
    metrics = list(x)
    diff = np.stack([x[k] - y[k] for k in metrics], axis=1)
    labels = sorted(set(clusters))
    members = [np.flatnonzero(np.array(clusters) == s) for s in labels]
    if benchmark == "workbuddy":
        members = [np.array([i]) for i in range(len(clusters))]
    sums = np.array([diff[m].sum(0) for m in members])
    sizes = np.array([len(m) for m in members])
    rng = np.random.default_rng(20260915 if benchmark == "workbuddy" else 20261005)
    estimates = []
    for start in range(0, 200000, 5000):
        idx = rng.integers(0, len(members), (min(5000, 200000 - start), len(members)))
        estimates.append(sums[idx].sum(1) / sizes[idx].sum(1)[:, None])
    ci = np.percentile(np.concatenate(estimates), [2.5, 97.5], axis=0)
    interval = (
        "task_bootstrap_ci95"
        if benchmark == "workbuddy"
        else "scenario_cluster_bootstrap_ci95"
    )
    out = {
        k: dict(
            mean_difference=float(diff[:, i].mean()), **{interval: ci[:, i].tolist()}
        )
        for i, k in enumerate(metrics)
    }
    out["mean_score"].update(paired_tests(x["mean_score"], y["mean_score"]))
    return out


def summarize(benchmark, tasks, clusters, arrays, seeds):
    threshold = 80 if benchmark == "workbuddy" else 75
    baseline = np.concatenate([seeds, arrays["A"]], axis=1)
    bm = measurements(baseline, threshold)
    scores = dict(baseline=baseline, **arrays)
    result = dict(
        n_tasks=len(tasks),
        n_scenario_clusters=len(set(clusters)),
        threshold=threshold,
        baseline_runs_per_task=6,
        baseline_triplets=["historical_seeds_1_2_3", "round_A_1_2_3"],
        bootstrap=dict(
            iterations=200000,
            seed=20260915 if benchmark == "workbuddy" else 20261005,
            unit="task" if benchmark == "workbuddy" else "scenario",
        ),
        definitions=dict(
            baseline_mean="Equal task mean of seed1, seed2, seed3, A1, A2, A3.",
            pass_rate="Threshold every score before averaging; baseline denominator 6n, other arms 3n.",
            success="Baseline averages the any/all success indicators of the two triplets within each task; denominator 2n task-groups. Other arms use n tasks.",
            endpoints="Each endpoint compared with the same six-run baseline, including later rounds; descriptive, not a prospective control.",
            utility="Actual same-round A/B/C candidate pool only; six-run baseline is not selectable.",
        ),
        arms={k: arm_summary(a, threshold) for k, a in scores.items()},
        baseline_components={
            k: arm_summary(a, threshold)
            for k, a in [("historical_seeds", seeds), ("round_A", arrays["A"])]
        },
        comparisons={
            k + "_vs_baseline": paired_all(
                measurements(a, threshold), bm, clusters, benchmark
            )
            for k, a in arrays.items()
        },
    )
    result["rounds"] = {}
    for r in (1, 3):
        ra = {k: a[:, r - 1 : r] for k, a in arrays.items()}
        ms = {k: measurements(a, threshold, False) for k, a in ra.items()}
        rm = {k: bm[k] for k in ("mean_score", "average_pass_rate")}
        oracle = np.maximum.reduce([a for k, a in ra.items() if k in "ABC"])[:, 0]
        uniform = np.mean([ra[k][:, 0] for k in "ABC"], axis=0)
        selected = ra["selected_score"][:, 0]
        result["rounds"][str(r)] = dict(
            arms={k: arm_summary(a, threshold, False) for k, a in ra.items()},
            comparisons={
                k + "_vs_baseline": paired_all(m, rm, clusters, benchmark)
                for k, m in ms.items()
            },
            same_round_comparisons={
                k + "_vs_A": paired_all(ms[k], ms["A"], clusters, benchmark)
                for k in ARMS[1:]
            },
            utility=dict(
                uniform_mean=float(uniform.mean()),
                oracle_mean=float(oracle.mean()),
                headroom_recovered=float(
                    (selected - uniform).mean() / (oracle - uniform).mean()
                ),
                mean_regret={
                    k: float((oracle - a[:, 0]).mean()) for k, a in ra.items()
                },
                best_hits={
                    k: int(np.isclose(a[:, 0], oracle, atol=1e-8, rtol=0).sum())
                    for k, a in ra.items()
                },
                best_hit_rate={
                    k: float(np.isclose(a[:, 0], oracle, atol=1e-8, rtol=0).mean())
                    for k, a in ra.items()
                },
                selected_vs_uniform=paired_all(
                    ms["selected_score"],
                    dict(
                        mean_score=uniform,
                        average_pass_rate=np.mean(
                            [ra[k] >= threshold for k in "ABC"], axis=0
                        )[:, 0],
                    ),
                    clusters,
                    benchmark,
                ),
            ),
        )
    ledger = []
    for i, task in enumerate(tasks):
        row = dict(task=task, cluster=clusters[i])
        for j in range(3):
            row[f"seed_{j + 1}"] = float(seeds[i, j])
        for k, a in arrays.items():
            for j in range(3):
                row[f"{k}_{j + 1}"] = float(a[i, j])
        row.update({f"baseline_{k}": float(v[i]) for k, v in bm.items()})
        row["seed_pass_count"] = int((seeds[i] >= threshold).sum())
        row["A_pass_count"] = int((arrays["A"][i] >= threshold).sum())
        ledger.append(row)
    return result, ledger


def threshold_sweep(arrays, seeds):
    # Exhaustive empirical breakpoints and conventional integer thresholds.
    # Remove representation noise only in exploratory breakpoint enumeration.
    arrays = {k: np.round(v, 10) for k, v in arrays.items()}
    seeds = np.round(seeds, 10)
    baseline = np.concatenate([seeds, arrays["A"]], 1)
    thresholds = sorted(
        set(range(0, 101))
        | {float(x) for a in [baseline, *arrays.values()] for x in a.ravel()}
    )
    rows = []
    for t in thresholds:
        b = measurements(baseline, t)["pass_power_3"].mean()
        s = measurements(arrays["selected_score"], t)["pass_power_3"].mean()
        rows.append(
            dict(
                threshold=t,
                baseline_pass_power_3=float(b),
                selected_pass_power_3=float(s),
                difference_pp=float(100 * (s - b)),
            )
        )
    best = max(r["difference_pp"] for r in rows)
    integer_rows = [r for r in rows if r["threshold"] == int(r["threshold"])]
    integer_best = max(r["difference_pp"] for r in integer_rows)
    intervals = []
    for i, r in enumerate(rows):
        if abs(r["difference_pp"] - best) < 1e-8:
            lo = rows[i - 1]["threshold"] if i else 0
            if intervals and intervals[-1]["upper_inclusive"] == lo:
                intervals[-1]["upper_inclusive"] = r["threshold"]
            else:
                intervals.append(
                    dict(lower_exclusive=lo, upper_inclusive=r["threshold"])
                )
    return dict(
        interpretation="Exploratory observed absolute percentage-point advantage; does not change reporting thresholds. Breakpoint sweep removes only floating representation noise by rounding scores to 10 decimals; primary statistics use unrounded scores.",
        maximum_difference_pp=best,
        maximizing_intervals=intervals,
        maximum_integer_difference_pp=integer_best,
        maximizing_integer_thresholds=[
            r["threshold"]
            for r in integer_rows
            if abs(r["difference_pp"] - integer_best) < 1e-8
        ],
        maximizing_tested_thresholds=[
            r["threshold"] for r in rows if abs(r["difference_pp"] - best) < 1e-8
        ],
        rows=rows,
    )


def fmt_test(p, digits=4):
    return (
        r"$<10^{-" + str(digits) + r"}$" if p < 10 ** (-digits) else f"{p:.{digits}f}"
    )


def table(name, columns, header, rows):
    (ROOT / "tables" / name).write_text(
        "\n".join(
            [
                r"\begin{tabular}{" + columns + "}",
                r"\toprule",
                header + r" \\",
                r"\midrule",
                *rows,
                r"\bottomrule",
                r"\end{tabular}",
            ]
        )
        + "\n"
    )


def write_tables(out):
    main = []
    diag = []
    success = []
    components = []
    hyp = []
    paired = []
    utilities = []
    shown = ("baseline", "B", "C", "selected_score")
    for benchmark, s in out.items():
        name = "WorkBuddy" if benchmark == "workbuddy" else "claw-eval"
        ci = (
            "task_bootstrap_ci95"
            if benchmark == "workbuddy"
            else "scenario_cluster_bootstrap_ci95"
        )
        main.append(
            r"\multicolumn{6}{l}{\textit{"
            + f"{name}: three-round summary, threshold {s['threshold']}"
            + r"}} \\"
        )
        diag.append(r"\multicolumn{6}{l}{\textit{" + name + r"}} \\")
        best = max(s["arms"][k]["mean_score"] for k in shown)
        bp = max(s["arms"][k]["average_pass_rate"] for k in shown)
        for k in shown:
            v = s["arms"][k]
            m = f"{v['mean_score']:.2f}"
            p = f"{100 * v['average_pass_rate']:.2f}"
            if abs(v["mean_score"] - best) < 1e-8:
                m = r"\textbf{" + m + "}"
            if abs(v["average_pass_rate"] - bp) < 1e-8:
                p = r"\textbf{" + p + "}"
            cells = [LABELS[k], m, p]
            if k == "baseline":
                cells += ["---"] * 3
            else:
                comp = s["comparisons"][k + "_vs_baseline"]
                c = comp["mean_score"]
                rate = comp["average_pass_rate"]
                cells += [
                    fmt_delta(c["mean_difference"]) + " " + fmt_ci(c[ci]),
                    f"{c['wins']}/{c['ties']}/{c['losses']}",
                    fmt_test(c["sign_test_two_sided_p"]),
                ]
                diag.append(
                    " & ".join(
                        [
                            LABELS[k],
                            f"{c['median_difference']:+.2f}",
                            fmt_test(c["sign_test_two_sided_p"]),
                            fmt_test(c["wilcoxon_two_sided_p"], 5),
                            fmt_delta(100 * rate["mean_difference"])
                            + " "
                            + fmt_ci([100 * x for x in rate[ci]]),
                            f"{v['passed_runs']}/{v['n_runs']}",
                        ]
                    )
                    + r" \\"
                )
                vals = [name + " / " + LABELS[k]]
                for metric in METRICS:
                    c = comp[metric]
                    scale = 1 if metric == "mean_score" else 100
                    vals.append(
                        fmt_delta(scale * c["mean_difference"])
                        + " "
                        + fmt_ci([scale * x for x in c[ci]])
                    )
                paired.append(" & ".join(vals) + r" \\")
            main.append(" & ".join(cells) + r" \\")
        main.append(r"\midrule")
        diag.append(r"\midrule")
        for k, v in s["baseline_components"].items():
            components.append(
                " & ".join(
                    [
                        name,
                        "Historical seeds"
                        if k == "historical_seeds"
                        else "Round baseline",
                        f"{v['mean_score']:.2f}",
                        f"{100 * v['average_pass_rate']:.2f} ({v['passed_runs']}/{v['n_runs']})",
                        f"{100 * v['pass_at_3']:.2f} ({v['pass_at_3_count']}/{v['success_denominator']})",
                        f"{100 * v['pass_power_3']:.2f} ({v['pass_power_3_count']}/{v['success_denominator']})",
                    ]
                )
                + r" \\"
            )
        for r, ep in s["rounds"].items():
            rows = []
            ds = []
            same = []
            same_rates = []
            for k in ("baseline", *ARMS):
                v = s["arms"]["baseline"] if k == "baseline" else ep["arms"][k]
                vals = [
                    LABELS[k],
                    f"{v['mean_score']:.2f}",
                    f"{100 * v['average_pass_rate']:.2f} ({v['passed_runs']}/{v['n_runs']})",
                ]
                if k == "baseline":
                    vals += ["---"] * 3
                else:
                    c = ep["comparisons"][k + "_vs_baseline"]["mean_score"]
                    p = ep["comparisons"][k + "_vs_baseline"]["average_pass_rate"]
                    vals += [
                        fmt_delta(c["mean_difference"]) + " " + fmt_ci(c[ci]),
                        f"{c['wins']}/{c['ties']}/{c['losses']}",
                        fmt_test(c["sign_test_two_sided_p"]),
                    ]
                    ds.append(
                        " & ".join(
                            [
                                LABELS[k],
                                f"{c['median_difference']:+.2f}",
                                fmt_test(c["wilcoxon_two_sided_p"], 5),
                                fmt_delta(100 * p["mean_difference"])
                                + " "
                                + fmt_ci([100 * x for x in p[ci]]),
                            ]
                        )
                        + r" \\"
                    )
                rows.append(" & ".join(vals) + r" \\")
            for k in ARMS[1:]:
                comp = ep["same_round_comparisons"][k + "_vs_A"]
                c = comp["mean_score"]
                p = comp["average_pass_rate"]
                same.append(
                    " & ".join(
                        [
                            LABELS[k],
                            fmt_delta(c["mean_difference"]) + " " + fmt_ci(c[ci]),
                            f"{c['wins']}/{c['ties']}/{c['losses']}",
                            f"{c['median_difference']:+.2f}",
                            fmt_test(c["sign_test_two_sided_p"]),
                            fmt_test(c["wilcoxon_two_sided_p"], 5),
                        ]
                    )
                    + r" \\"
                )
                same_rates.append(
                    " & ".join(
                        [
                            LABELS[k],
                            fmt_delta(100 * p["mean_difference"])
                            + " "
                            + fmt_ci([100 * x for x in p[ci]]),
                        ]
                    )
                    + r" \\"
                )
            prefix = f"{benchmark}_round{r}"
            table(
                prefix + "_six_baseline.tex",
                "lrrlrr",
                r"Method & Mean & Pass (\%) & $\Delta$ mean [95\% CI] & W/T/L & Sign $p$",
                rows,
            )
            table(
                prefix + "_six_diagnostics.tex",
                "lrrl",
                r"Versus six-run baseline & Median $\Delta$ & Wilcoxon $p$ & $\Delta$ pass (pp) [95\% CI]",
                ds,
            )
            table(
                prefix + "_same_round.tex",
                "llrrrr",
                r"Versus round baseline & $\Delta$ mean [95\% CI] & W/T/L & Median $\Delta$ & Sign $p$ & Wilcoxon $p$",
                same,
            )
            table(
                prefix + "_same_round_pass.tex",
                "ll",
                r"Versus round baseline & $\Delta$ pass (pp) [95\% CI]",
                same_rates,
            )
            u = ep["utility"]
            for k in ARMS:
                utilities.append(
                    " & ".join(
                        [
                            name,
                            f"{r}",
                            LABELS[k],
                            f"{u['uniform_mean']:.2f}",
                            f"{u['oracle_mean']:.2f}",
                            f"{100 * (ep['arms'][k]['mean_score'] - u['uniform_mean']) / (u['oracle_mean'] - u['uniform_mean']):.2f}",
                            f"{u['mean_regret'][k]:.2f}",
                            f"{100 * u['best_hit_rate'][k]:.2f} ({u['best_hits'][k]}/{s['n_tasks']})",
                        ]
                    )
                    + r" \\"
                )
    # Preserve the hypothesis-specific appendix view with the current baseline.
    for benchmark, s in out.items():
        name = "WorkBuddy" if benchmark == "workbuddy" else "claw-eval"
        ci = (
            "task_bootstrap_ci95"
            if benchmark == "workbuddy"
            else "scenario_cluster_bootstrap_ci95"
        )
        hyp.append(r"\multicolumn{6}{l}{\textit{" + name + r"}} \\")
        for k in ("baseline", "C"):
            v = s["arms"][k]
            cells = [
                LABELS[k],
                f"{v['mean_score']:.2f}",
                f"{100 * v['average_pass_rate']:.2f} ({v['passed_runs']}/{v['n_runs']})",
            ]
            cells += [
                f"{100 * v[m]:.2f} ({v[m + '_count']}/{v['success_denominator']})"
                for m in ("pass_at_3", "pass_power_3")
            ]
            if k == "baseline":
                cells += ["---"]
            else:
                c = s["comparisons"]["C_vs_baseline"]["mean_score"]
                cells += [fmt_delta(c["mean_difference"]) + " " + fmt_ci(c[ci])]
            hyp.append(" & ".join(cells) + r" \\")
    table(
        "hypothesis_arm.tex",
        "lrrrrl",
        r"Method & Mean & Avg. pass (\%) & $\mathrm{pass@3}$ (\%) & $\mathrm{Pass}^{3}$ (\%) & $\Delta$ mean [95\% CI]",
        hyp,
    )
    s = out["claw_eval"]["excluding_known_exposure"]
    rows = []
    for k in shown:
        v = s["arms"][k]
        cells = [
            LABELS[k],
            f"{v['mean_score']:.2f}",
            f"{100 * v['average_pass_rate']:.2f} ({v['passed_runs']}/{v['n_runs']})",
        ]
        cells += [
            f"{100 * v[m]:.2f} ({v[m + '_count']}/{v['success_denominator']})"
            for m in ("pass_at_3", "pass_power_3")
        ]
        rows.append(" & ".join(cells) + r" \\")
    table(
        "claw_eval_six_baseline_sensitivity.tex",
        "lrrrr",
        r"Method & Mean & Avg. pass (\%) & $\mathrm{pass@3}$ (\%) & $\mathrm{Pass}^{3}$ (\%)",
        rows,
    )
    for k in shown:
        cells = [LABELS[k]]
        for s in out.values():
            v = s["arms"][k]
            for metric in ("pass_at_3", "pass_power_3"):
                cells.append(
                    f"{100 * v[metric]:.2f} ({v[metric + '_count']}/{v['success_denominator']})"
                )
        success.append(" & ".join(cells) + r" \\")
    table(
        "main_results.tex",
        "lrrlrr",
        r"Method & Mean & Avg. pass (\%) & $\Delta$ mean [95\% CI] & W/T/L & Sign $p$",
        main[:-1],
    )
    table(
        "three_round_success.tex",
        "lrrrr",
        r" & \multicolumn{2}{c}{WorkBuddy} & \multicolumn{2}{c}{claw-eval} \\"
        + "\n"
        + r"Method & $\mathrm{pass@3}$ (\%) & $\mathrm{Pass}^{3}$ (\%) & $\mathrm{pass@3}$ (\%) & $\mathrm{Pass}^{3}$ (\%)",
        success,
    )
    table(
        "three_round_diagnostics.tex",
        "lrrrlr",
        r"Versus six-run baseline & Median $\Delta$ & Sign $p$ & Wilcoxon $p$ & $\Delta$ avg. pass (pp) [95\% CI] & Pass runs",
        diag[:-1],
    )
    table(
        "six_baseline_components.tex",
        "llrrrr",
        r"Benchmark & Triplet & Mean & Avg. pass (\%) & $\mathrm{pass@3}$ (\%) & $\mathrm{Pass}^{3}$ (\%)",
        components,
    )
    table(
        "six_baseline_comparisons.tex",
        "lllll",
        r"Benchmark / method & $\Delta$ mean & $\Delta$ avg. pass (pp) & $\Delta\mathrm{pass@3}$ (pp) & $\Delta\mathrm{Pass}^{3}$ (pp)",
        paired,
    )
    rows = []
    for label, group in out["workbuddy"]["sensitivities"].items():
        name = {
            "omit_grader_exception": "Omit grader exception",
            "omit_ambiguous_checker_mentions": "Omit seven flagged tasks",
            "omit_mentions_and_grader_exception": "Omit both",
        }[label]
        for scope, title in [
            ("three_round", "Three-round mean"),
            ("final_round", "Final round"),
        ]:
            for arm in ARMS[1:]:
                c = group[scope][arm]["mean_score"]
                rows.append(
                    " & ".join(
                        [
                            name,
                            str(group["n_tasks"]),
                            title,
                            LABELS[arm],
                            fmt_delta(c["mean_difference"])
                            + " "
                            + fmt_ci(c["task_bootstrap_ci95"]),
                            f"{c['wins']}/{c['ties']}/{c['losses']}",
                        ]
                    )
                    + r" \\"
                )
    table(
        "workbuddy_six_baseline_sensitivity.tex",
        "lrllll",
        r"Subset & $n$ & Summary & Method & $\Delta$ mean [95\% CI] & W/T/L",
        rows,
    )
    table(
        "endpoint_utility.tex",
        "lrlrrrrr",
        r"Benchmark & Round & Method & Uniform & Oracle & Headroom (\%) & Regret & Best hit (\%)",
        utilities,
    )


def main(benchmarks=("workbuddy", "claw_eval")):
    output = {}
    for b in benchmarks:
        tasks, clusters, arrays, seeds = inputs(b)
        result, ledger = summarize(b, tasks, clusters, arrays, seeds)
        old = read(ROOT / "data/main_statistics.json")[b + "_three_round"]
        result["utility"] = copy.deepcopy(old["utility"])
        result["selected_arm_counts"] = old["selected_arm_counts"]
        if b == "workbuddy":
            result["imputations"] = old["imputations"]
            exceptions = set(
                read(ROOT / "data/workbuddy_office_analysis.json")["score_sensitivity"][
                    "excluded_tasks"
                ]
            )
            flags = {
                r["task"]
                for r in read(ROOT / "data/workbuddy_office_evidence.json")[
                    "selected_measurements"
                ]
                if r["ranking_mentions_grader_or_checker"]
            }
            result["sensitivities"] = {}
            for label, excluded in [
                ("omit_grader_exception", exceptions),
                ("omit_ambiguous_checker_mentions", flags),
                ("omit_mentions_and_grader_exception", flags | exceptions),
            ]:
                keep = np.array([t not in excluded for t in tasks])
                cs = [c for c, k in zip(clusters, keep, strict=False) if k]
                bm = measurements(
                    np.concatenate([seeds[keep], arrays["A"][keep]], 1), 80
                )
                result["sensitivities"][label] = dict(
                    n_tasks=int(keep.sum()),
                    excluded_tasks=sorted(excluded),
                    three_round={
                        k: paired_all(measurements(arrays[k][keep], 80), bm, cs, b)
                        for k in ARMS[1:]
                    },
                    final_round={
                        k: paired_all(
                            measurements(arrays[k][keep, 2:3], 80, False),
                            {m: bm[m] for m in ("mean_score", "average_pass_rate")},
                            cs,
                            b,
                        )
                        for k in ARMS[1:]
                    },
                )
        if b == "claw_eval":
            keep = np.array([t != "T101_wal_recovery" for t in tasks])
            sub, _ = summarize(
                b,
                [t for t, k in zip(tasks, keep, strict=False) if k],
                [c for c, k in zip(clusters, keep, strict=False) if k],
                {k: v[keep] for k, v in arrays.items()},
                seeds[keep],
            )
            result["excluding_known_exposure"] = sub
        result["input_sha256"] = {
            str(p.relative_to(ROOT)): sha(p)
            for p in (
                [ROOT / "data/office50.json", ROOT / "data/workbuddy_seed_scores.json"]
                if b == "workbuddy"
                else [
                    ROOT / "data/claw_eval.json",
                    ROOT / "data/claw_eval_continuation.json",
                ]
            )
        }
        write(ROOT / f"data/{b}_six_baseline_statistics.json", result)
        write(
            ROOT / f"data/{b}_six_baseline_threshold_sensitivity.json",
            threshold_sweep(arrays, seeds),
        )
        with (ROOT / f"data/{b}_six_baseline_ledger.csv").open("w") as f:
            w = csv.DictWriter(f, fieldnames=list(ledger[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(ledger)
        output[b] = result
        print(
            b,
            result["arms"]["baseline"],
            result["comparisons"]["selected_score_vs_baseline"]["mean_score"],
            flush=True,
        )
    if len(output) == 2:
        stats = read(ROOT / "data/main_statistics.json")
        for b, s in output.items():
            stats[b + "_six_baseline"] = s
        write(ROOT / "data/main_statistics.json", stats)
        write_tables(output)
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", choices=["workbuddy", "claw_eval"])
    a = p.parse_args()
    main((a.benchmark,) if a.benchmark else ("workbuddy", "claw_eval"))
