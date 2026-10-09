"""Additional prespecified-for-this-audit summaries of frozen archive scores.

No model calls; these retrospective sensitivities do not replace primary results.
"""

import collections
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
N, SEED = 200_000, 20260920


def comparison(values, clusters=None):
    values = np.asarray(values, dtype=float)
    if clusters is None:
        clusters = list(range(len(values)))
    grouped = collections.defaultdict(list)
    for value, key in zip(values, clusters, strict=True):
        grouped[key].append(value)
    sums = np.array([sum(v) for v in grouped.values()])
    sizes = np.array([len(v) for v in grouped.values()])
    rng = np.random.default_rng(SEED)
    samples = []
    for _ in range(N // 10_000):
        ix = rng.integers(len(sums), size=(10_000, len(sums)))
        samples.append(sums[ix].sum(axis=1) / sizes[ix].sum(axis=1))
    return dict(
        n_tasks=len(values),
        n_clusters=len(sums),
        delta=float(values.mean()),
        ci95=np.quantile(np.concatenate(samples), [0.025, 0.975]).tolist(),
        wins=int((values > 1e-8).sum()),
        ties=int((abs(values) <= 1e-8).sum()),
        losses=int((values < -1e-8).sum()),
    )


def read(name):
    return json.loads((ROOT / "data" / name).read_text())


def table(name, spec, header, rows):
    text = "\n".join(
        [
            r"\begin{tabular}{" + spec + "}",
            r"\toprule",
            header + r" \\",
            r"\midrule",
            *[r + r" \\" for r in rows],
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    (ROOT / "tables" / f"{name}.tex").write_text(text + "\n")


def interval(item):
    return f"{item['delta']:+.2f} & [{item['ci95'][0]:.2f}, {item['ci95'][1]:.2f}]"


def write_tables(result):
    for benchmark, suffix in [("claw-eval", "claw"), ("WorkBuddy Office", "workbuddy")]:
        rows = []
        for arm, s in result["final_measurements"][benchmark].items():
            count = s["statuses"]
            label = {
                "A": "Baseline",
                "B": "Reward-guided revision",
                "C": "Hypothesis guidance",
                "Selected": "Selected by reward",
            }[arm]
            rows.append(
                f"{label} & {s['tasks_with_rows']}/{s['n_tasks']} & {s['n_rows']} & "
                f"{count.get('pass', 0)}/{count.get('fail', 0)}/{count.get('unknown', 0)} & {s['error_rows']}"
            )
        table(
            f"{suffix}_final_coverage",
            "lrrrr",
            "Path & Tasks with rows & Rows & Pass/Fail/Unknown & Errors",
            rows,
        )
    labels = {
        "B_vs_A_mean": r"Reward-guided revision $-$ historical baseline mean",
        "C_vs_A_mean": r"Hypothesis guidance $-$ historical baseline mean",
        "selected_score_vs_A_mean": r"Selected by reward $-$ historical baseline mean",
    }
    rows = [
        labels[key] + " & " + interval(value)
        for key, value in result["claw_clustered"].items()
    ]
    rows.extend(
        [
            "Selected by reward $-$ reward-guided revision & "
            + interval(result["claw_selector_references"]["always_B"]),
            "Selected by reward $-$ uniform expectation & "
            + interval(result["claw_selector_references"]["uniform"]),
        ]
    )
    table(
        "claw_cluster_sensitivity",
        "lrr",
        r"Comparison & Difference & Scenario 95\% interval",
        rows,
    )
    rows = []
    for key, label in [
        ("omit_ambiguous_checker_mentions", "Omit seven flagged tasks"),
        ("omit_mentions_and_grader_exception", "Also omit grader exception"),
    ]:
        s = result["workbuddy_selector_boundary"][key]
        n = s["comparisons"]["B"]["n_tasks"]
        rows.append(
            f"{label} & {n} & Selected by reward $-$ baseline & "
            + interval(s["comparisons"]["selected_score"])
        )
        rows.append(
            f"{label} & {n} & Selected by reward $-$ reward-guided revision & "
            + interval(s["selection_vs_B"])
        )
    table(
        "workbuddy_boundary_sensitivity",
        "lr lrr",
        r"Subset & $n$ & Comparison & Difference & Task 95\% interval",
        rows,
    )


def main():
    claw = read("claw_eval.json")["tasks"]
    scenario = [re.sub(r"^T\d+(?:zh)?_", "", r["task"]) for r in claw]
    groups = collections.defaultdict(list)
    for r, key in zip(claw, scenario, strict=True):
        groups[key].append(r)
    assert len(groups) == 68
    assert collections.Counter(map(len, groups.values())) == {1: 32, 2: 36}
    assert all(
        {r["language"] for r in group} == {"zh", "en"}
        for group in groups.values()
        if len(group) == 2
    )
    result = dict(
        bootstrap=dict(
            iterations=N,
            seed=SEED,
            interval="percentile",
            cluster_estimator="resampled task-score sums divided by resampled task counts",
        ),
        interpretation="Exploratory archive sensitivity; no causal or new-run inference.",
    )
    result["scenario_membership"] = {
        k: [r["task"] for r in g] for k, g in groups.items()
    }
    result["claw_clustered"] = {
        f"{arm}_vs_{base}": comparison([r[arm] - r[base] for r in claw], scenario)
        for arm, base in [
            ("B", "A_mean"),
            ("C", "A_mean"),
            ("selected_score", "A_mean"),
        ]
    }
    result["claw_selector_references"] = {
        "always_B": comparison([r["selected_score"] - r["B"] for r in claw], scenario),
        "uniform": comparison(
            [
                r["selected_score"] - np.mean([r[k] for k in ["A_run", "B", "C"]])
                for r in claw
            ],
            scenario,
        ),
    }
    office = read("office50.json")["tasks"]
    evidence = read("workbuddy_office_evidence.json")
    flags = {
        r["task"]
        for r in evidence["selected_measurements"]
        if r["ranking_mentions_grader_or_checker"]
    }
    exceptions = set(
        read("workbuddy_office_analysis.json")["score_sensitivity"]["excluded_tasks"]
    )
    result["workbuddy_selector_boundary"] = {}
    for label, exclude in [
        ("omit_ambiguous_checker_mentions", flags),
        ("omit_mentions_and_grader_exception", flags | exceptions),
    ]:
        rows = [r for r in office if r["task"] not in exclude]
        result["workbuddy_selector_boundary"][label] = dict(
            excluded_tasks=sorted(exclude),
            comparisons={
                arm: comparison([r[arm] - r["A"] for r in rows])
                for arm in ["B", "C", "selected_score"]
            },
            selection_vs_B=comparison([r["selected_score"] - r["B"] for r in rows]),
        )
    evidence = read("archive_review_evidence.json")
    result["final_measurements"] = {}
    for benchmark in ["WorkBuddy Office", "claw-eval"]:
        entries = [r for r in evidence["measurements"] if r["benchmark"] == benchmark]
        summaries = {}
        for arm in ["A", "B", "C", "Selected"]:
            subset = (
                [r for r in entries if r["selected"]]
                if arm == "Selected"
                else [r for r in entries if r["arm"] == arm]
            )
            items = [item for r in subset for item in r["results"]]
            summaries[arm] = dict(
                n_tasks=len(subset),
                tasks_with_rows=sum(bool(r["results"]) for r in subset),
                n_rows=len(items),
                statuses=dict(collections.Counter(item["status"] for item in items)),
                error_rows=sum(bool(item["error"]) for item in items),
                review_statuses=dict(
                    collections.Counter(r["review_status"] for r in subset)
                ),
            )
        result["final_measurements"][benchmark] = summaries
    errors = collections.Counter()
    for entry in evidence["measurements"]:
        if entry["benchmark"] != "claw-eval":
            continue
        for row in entry["results"]:
            if not row["error"]:
                continue
            reason, diagnostic = row.get("reason") or "", row.get("diagnostic") or ""
            if "JSONDecodeError" in reason:
                category = "JSON parsing"
            elif "Evidence expansion allowance exhausted" in diagnostic:
                category = "Evidence read allowance"
            elif "Claim audit validation failed" in reason:
                category = "Claim audit validation"
            else:
                category = "Other execution or budget errors"
            errors[category] += 1
    result["claw_final_error_categories"] = dict(errors)
    worst = min(claw, key=lambda r: r["C"] - r["A_mean"])
    subset = [r for r in claw if r["task"] != worst["task"]]
    result["claw_influence"] = dict(
        excluded_task=worst["task"],
        rule="largest C minus A_mean loss",
        comparison=comparison(
            [r["C"] - r["A_mean"] for r in subset],
            [re.sub(r"^T\d+(?:zh)?_", "", r["task"]) for r in subset],
        ),
    )
    write_tables(result)
    output = ROOT / "data/archive_sensitivity.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(output)
    print("Final coverage:", result["final_measurements"])


if __name__ == "__main__":
    main()
