"""Freeze published WorkBuddy Bench Office summaries and derive manuscript tables/figures.

Run with --source ROOT to refresh the snapshot, or without it to reproduce
all paper outputs offline from data/office50.json. Requires numpy/matplotlib.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The existing archive includes the original reporting environment.
try:
    import matplotlib
except ImportError:
    vendor = Path("machine_path_b294ba70281e71ee")
    if vendor.exists():
        sys.path.insert(0, str(vendor))
    import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 -- select headless backend first
import numpy as np  # noqa: E402 -- optional plotting environment initialized above

p = argparse.ArgumentParser()
p.add_argument("--source", type=Path)
args = p.parse_args()
if args.source:
    report = args.source / "reports/workbuddy/office50"
    detail = json.loads((report / "report_detail.json").read_text())
    result = json.loads((report / "results.json").read_text())
    data = {
        "tasks": [
            {k: v for k, v in t.items() if k not in ("cohort", "source")}
            for t in result["tasks"]
        ],
        "rounds": [
            {k: t[k] for k in ["task", "rounds"]}
            for t in json.loads((report / "report_data.json").read_text())["tasks"]
        ],
        "metrics": {k: v for k, v in detail["metrics"].items() if k != "cohorts"},
        "protocol_note": "All 50 WorkBuddy Office tasks are reported together. Accepted scores, valid zeros, and recorded runtime selections are retained; final benchmark scores are not selection inputs.",
        "valid_score_records": detail["valid_score_records"],
        "clean_round_comparison_tasks": detail["clean_round_comparison_tasks"],
        "review_counts": detail["review_counts"],
    }
    for t in data["rounds"]:
        t["rounds"] = [
            {
                k: r[k]
                for k in [
                    "round",
                    "A",
                    "B",
                    "C",
                    "selected_arm",
                    "selected_score",
                    "invalid_scoring_arms",
                ]
            }
            for r in t["rounds"]
        ]
    (ROOT / "data/office50.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    )
    experiment = args.source / "local/workbuddy/outcome/office50/experiment.json"
    e = json.loads(experiment.read_text())
    files = [
        report / "results.json",
        report / "report_data.json",
        report / "report_detail.json",
        report / "rounds.csv",
        experiment,
    ]
    provenance = {
        "source_root": str(args.source),
        "files": {
            str(f.relative_to(args.source)): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in files
        },
        "scoring": e["scoring"],
        "snapshot_sha256": hashlib.sha256(
            (ROOT / "data/office50.json").read_bytes()
        ).hexdigest(),
    }
    (ROOT / "data/office50_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n"
    )
d = json.loads((ROOT / "data/office50.json").read_text())
tasks = d["tasks"]
rounds = d["rounds"]
metrics = d["metrics"]
assert len(tasks) == len({t["task"] for t in tasks}) == 50
assert len(rounds) == len({t["task"] for t in rounds}) == 50
assert sum(len(t["rounds"]) for t in rounds) == 150
for t, r in zip(tasks, rounds, strict=False):
    assert t["task"] == r["task"]
    last = next(x for x in r["rounds"] if x["round"] == 3)
    assert all(abs(t[a] - last[a]) < 1e-7 for a in ["A", "B", "C", "selected_score"])
    assert t["selected_score"] == t[t["selected_arm"]]
base = np.array([t["A"] for t in tasks])
stats = {}
for arm in ["B", "C", "selected"]:
    col = "selected_score" if arm == "selected" else arm
    values = np.array([t[col] for t in tasks])
    delta = values - base
    rng = np.random.default_rng(20260915)
    ci = np.quantile(
        delta[rng.integers(0, 50, size=(200000, 50))].mean(axis=1), [0.025, 0.975]
    )
    ref = metrics["comparisons_vs_A"][arm]
    assert np.allclose(ci, ref["paired_task_bootstrap_ci95"], atol=1e-10)
    stats[arm] = {
        "mean": float(values.mean()),
        "delta": float(delta.mean()),
        "ci": ci.tolist(),
        "wins": int((delta > 1e-8).sum()),
        "ties": int((abs(delta) <= 1e-8).sum()),
        "losses": int((delta < -1e-8).sum()),
    }
(ROOT / "data/office50_statistics.json").write_text(json.dumps(stats, indent=2) + "\n")
labels = {
    "A": "Baseline",
    "B": "Reward-guided revision",
    "C": "Hypothesis guidance",
    "selected": "Selected by reward",
    "All": "All choices",
}
lines = [
    r"\begin{tabular}{lrrrr}",
    r"\toprule",
    r"Path & Score & $\Delta$ vs. baseline & 95\% interval & W/T/L \\",
    r"\midrule",
    f"Baseline & {base.mean():.2f} & --- & --- & --- \\\\",
]
for arm in stats:
    st = stats[arm]
    lo, hi = st["ci"]
    lines.append(
        f"{labels[arm]} & {st['mean']:.2f} & {st['delta']:+.2f} & [{lo:.2f}, {hi:.2f}] & {st['wins']}/{st['ties']}/{st['losses']} \\\\"
    )
lines += [r"\bottomrule", r"\end{tabular}"]
(ROOT / "tables/office50_main.tex").write_text("\n".join(lines) + "\n")
lines = []
for part in range(2):
    lines.extend(
        [
            r"\begin{table}[htbp]",
            r"\centering\footnotesize\setlength{\tabcolsep}{2.5pt}",
            r"\begin{tabular}{>{\raggedright\arraybackslash}p{0.35\linewidth}rrrrrr}",
            r"\toprule Task & \shortstack{Task\\only} & \shortstack{Reward\\revision} & \shortstack{Hypothesis\\guidance} & Choice & Score & Gap \\ \midrule",
        ]
    )
    for t in tasks[part * 25 : (part + 1) * 25]:
        tid = t["task"].replace("-", r"-\allowbreak ")
        lines.append(
            tid
            + " & "
            + " & ".join(f"{t[a]:.2f}" for a in "ABC")
            + f" & {dict(A='Task', B='Revision', C='Guidance')[t['selected_arm']]} & {t['selected_score']:.2f} & {max(t[a] for a in 'ABC') - t['selected_score']:.2f}"
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Task scores, part "
            + str(part + 1)
            + r" of 2. Choice abbreviates the task-only, reward-guided revision, or Hypothesis-guided candidate. Selected scores follow recorded runtime decisions; Gap is the highest candidate score minus the selected score.}",
            r"\label{tab:alltasks" + str(part + 1) + "}",
            r"\end{table}",
        ]
    )
(ROOT / "tables/office50_tasks.tex").write_text("\n".join(lines) + "\n")
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    }
)
colors = {"B": "#256aa3", "C": "#c57b27", "selected": "#3d8054"}
fig, axes = plt.subplots(
    1, 2, figsize=(6.35, 2.10), gridspec_kw={"width_ratios": [1.3, 1]}
)
order = np.argsort(np.array([t["B"] for t in tasks]) - base, kind="stable")
for a, mark in [("B", "o"), ("C", "x")]:
    delta = np.array([t[a] for t in tasks]) - base
    axes[0].scatter(
        np.arange(1, 51),
        delta[order],
        s=12,
        marker=mark,
        label=labels[a] + " minus baseline",
        color=colors[a],
        alpha=0.85,
    )
axes[0].axhline(0, color=".45", lw=0.7)
axes[0].set(
    xlabel="Task index (sorted by revision gain)",
    ylabel="Score difference",
    ylim=(-105, 105),
)
axes[0].legend(frameon=False, fontsize=7, loc="upper left")
axes[0].grid(axis="y", alpha=0.15)
for i, a in enumerate(stats):
    st = stats[a]
    lo, hi = st["ci"]
    axes[1].errorbar(
        st["delta"],
        i,
        xerr=[[st["delta"] - lo], [hi - st["delta"]]],
        fmt="o",
        color=colors[a],
        capsize=3,
        markersize=4,
    )
axes[1].axvline(0, color=".45", lw=0.7)
axes[1].set(
    yticks=[0, 1, 2],
    yticklabels=[labels["B"], labels["C"], labels["selected"]],
    xlabel="Mean difference vs. task only (95% interval)",
    ylim=(2.5, -0.5),
)
axes[1].grid(axis="x", alpha=0.15)
fig.tight_layout(pad=0.6)
fig.savefig(ROOT / "figures/office50_differences.pdf", bbox_inches="tight")
fig.savefig(ROOT / "figures/office50_differences.png", dpi=180, bbox_inches="tight")
plt.close(fig)


# Supplementary analysis is derived offline, without changing the score archive.
def paired_interval(differences):
    values = np.asarray(differences)
    rng = np.random.default_rng(20260915)
    return np.quantile(
        values[rng.integers(0, len(values), size=(200000, len(values)))].mean(axis=1),
        [0.025, 0.975],
    ).tolist()


def table(name, columns, heading, rows):
    lines = [
        r"\begin{tabular}{" + columns + "}",
        r"\toprule",
        heading + r" \\",
        r"\midrule",
    ]
    lines += [" & ".join(str(x) for x in row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "tables" / name).write_text("\n".join(lines) + "\n")


analysis = {
    "median_differences": {},
    "selection": {},
    "selector_references": {},
    "delivery_review": {},
}
for arm in ["B", "C", "selected_score"]:
    analysis["median_differences"][arm] = float(
        np.median([t[arm] - t["A"] for t in tasks])
    )
selection_rows = []
for arm in ["A", "B", "C", "All"]:
    sub = [t for t in tasks if arm == "All" or t["selected_arm"] == arm]
    gaps = np.array([max(t[a] for a in "ABC") - t["selected_score"] for t in sub])
    st = {
        "count": len(sub),
        "share_percent": 100 * len(sub) / 50,
        "best_hits": int((np.abs(gaps) <= 1e-8).sum()),
        "mean_gap": float(gaps.mean()),
    }
    analysis["selection"][arm] = st
    selection_rows.append(
        [
            labels[arm],
            st["count"],
            f"{st['share_percent']:.0f}\\%",
            f"{st['best_hits']}/{st['count']}",
            f"{st['mean_gap']:.2f}",
        ]
    )
assert sum(analysis["selection"][a]["count"] for a in "ABC") == 50
assert analysis["selection"]["All"]["best_hits"] == metrics["selection_hits"]
assert np.isclose(
    analysis["selection"]["All"]["mean_gap"], metrics["selection_mean_regret"]
)
table(
    "workbuddy_selection.tex",
    "lrrrr",
    "Selected path & Tasks & Share & Best hits & Mean gap",
    selection_rows,
)

# Retrospective references use only the already frozen A/B/C candidate pool.
selected_values = np.array([t["selected_score"] for t in tasks])
reference_rows = []
for name, label, values in [
    ("always_B", "Always reward-guided revision", np.array([t["B"] for t in tasks])),
    (
        "uniform",
        "Uniform candidates (expected)",
        np.array([(t["A"] + t["B"] + t["C"]) / 3 for t in tasks]),
    ),
]:
    delta = selected_values - values
    hits = (
        sum(abs(t["B"] - max(t[a] for a in "ABC")) <= 1e-8 for t in tasks)
        if name == "always_B"
        else sum(
            sum(abs(t[a] - max(t[q] for q in "ABC")) <= 1e-8 for a in "ABC") / 3
            for t in tasks
        )
    )
    st = {
        "mean": float(values.mean()),
        "selected_delta": float(delta.mean()),
        "ci95": paired_interval(delta),
        "best_hits": float(hits),
    }
    analysis["selector_references"][name] = st
    reference_rows.append(
        [
            label,
            f"{st['mean']:.2f}",
            f"{st['selected_delta']:+.2f}",
            f"[{st['ci95'][0]:.2f}, {st['ci95'][1]:.2f}]",
            f"{st['best_hits']:.2f}/50"
            if name == "uniform"
            else f"{int(st['best_hits'])}/50",
        ]
    )
table(
    "workbuddy_selector_references.tex",
    "lrrrr",
    r"Reference & Score & Selected by reward $-$ reference & 95\% interval & Best hits",
    reference_rows,
)
review_rows = []
review_records = [
    m
    for m in json.loads((ROOT / "data/archive_review_evidence.json").read_text())[
        "measurements"
    ]
    if m["benchmark"] == "WorkBuddy Office" and m["arm"] == "B"
]
assert len(review_records) == len({m["task"] for m in review_records}) == 50
for name, label in [
    ("budget_exhausted_unresolved", "Unresolved at budget exhaustion"),
    ("check_unavailable", "Check unavailable"),
    ("full_score_target_satisfied", "Internal check target met"),
    ("no_material_gap_found", "No material gap found"),
]:
    count = sum(m["review_status"] == name for m in review_records)
    analysis["delivery_review"][name] = {"count": count, "percent": 100 * count / 50}
    review_rows.append([label, count, f"{100 * count / 50:.2f}\\%"])
assert sum(s["count"] for s in analysis["delivery_review"].values()) == 50
table(
    "workbuddy_review.tex",
    "lrr",
    "Reward-guided delivery review status & Revisions & Share",
    review_rows,
)

# Diagnose a documented grader exception without silently revising accepted scores.
evidence = json.loads((ROOT / "data/workbuddy_office_evidence.json").read_text())
assert (
    evidence["score_snapshot_sha256"]
    == hashlib.sha256((ROOT / "data/office50.json").read_bytes()).hexdigest()
)
lookup = {t["task"]: t for t in tasks}
assert (
    len(evidence["receipt_audit"])
    == len({(r["task"], r["arm"]) for r in evidence["receipt_audit"]})
    == 150
)
for r in evidence["receipt_audit"]:
    assert r["score"] == lookup[r["task"]][r["arm"]]
coverage = evidence["selected_measurements"]
assert len(coverage) == len({r["task"] for r in coverage}) == 50
coverage_rows = []
analysis["selected_measurements"] = {}
for arm in ["A", "B", "C", "All"]:
    rows = [r for r in coverage if arm == "All" or r["selected_arm"] == arm]
    st = {
        "tasks": len(rows),
        "tasks_with_measurement_rows": sum(r["measurement_rows"] > 0 for r in rows),
        "measurement_rows": sum(r["measurement_rows"] for r in rows),
        "pass": sum(r["status_counts"]["pass"] for r in rows),
        "fail": sum(r["status_counts"]["fail"] for r in rows),
        "unknown": sum(r["status_counts"]["unknown"] for r in rows),
        "execution_error_rows": sum(r["execution_error_rows"] for r in rows),
    }
    analysis["selected_measurements"][arm] = st
    coverage_rows.append(
        [
            labels[arm],
            st["tasks"],
            st["tasks_with_measurement_rows"],
            st["measurement_rows"],
            st["pass"],
            st["fail"],
            st["unknown"],
            st["execution_error_rows"],
        ]
    )
assert not any(r["official_verifier_observed"] for r in coverage)
analysis["selector_input_audit"] = {
    "official_verifier_observed_true": sum(
        r["official_verifier_observed"] is True for r in coverage
    ),
    "official_verifier_observed_false": sum(
        r["official_verifier_observed"] is False for r in coverage
    ),
    "ranking_mentions_grader_or_checker": sum(
        r["ranking_mentions_grader_or_checker"] for r in coverage
    ),
}
table(
    "workbuddy_measurement_coverage.tex",
    "lrrrrrrr",
    "Choice & Tasks & With rows & Rows & Pass & Fail & Unknown & Errors",
    coverage_rows,
)
flagged = sorted(
    {
        r["task"]
        for r in evidence["receipt_audit"]
        if any(j["evaluator_exception"] for j in r["judges"])
    }
)
assert flagged == ["priority-sync-notification-pipeline"]
sensitivity = [t for t in tasks if t["task"] not in flagged]
analysis["score_sensitivity"] = {
    "excluded_tasks": flagged,
    "n_tasks": len(sensitivity),
    "paths": {},
}
sens_rows = []
for arm, label in [
    ("A", labels["A"]),
    ("B", labels["B"]),
    ("C", labels["C"]),
    ("selected_score", labels["selected"]),
]:
    values = [t[arm] for t in sensitivity]
    delta = [t[arm] - t["A"] for t in sensitivity]
    st = {
        "mean": float(np.mean(values)),
        "delta": float(np.mean(delta)),
        "ci95": paired_interval(delta),
    }
    analysis["score_sensitivity"]["paths"][arm] = st
    sens_rows.append(
        [
            label,
            f"{st['mean']:.2f}",
            "---" if arm == "A" else f"{st['delta']:+.2f}",
            "---" if arm == "A" else f"[{st['ci95'][0]:.2f}, {st['ci95'][1]:.2f}]",
        ]
    )
table(
    "workbuddy_sensitivity.tex",
    "lrrr",
    "Path & Score & $\\Delta$ vs. baseline & 95\\% interval",
    sens_rows,
)
case_rows = []
for key, label in [
    ("priority-sync-notification-pipeline", "Priority synchronization"),
    ("channel-period-compare-L4-017", "Channel comparison"),
]:
    t = lookup[key]
    case_rows.append(
        [label]
        + [f"{t[a]:.2f}" for a in "ABC"]
        + [
            dict(A="Task", B="Revision", C="Guidance")[t["selected_arm"]],
            f"{t['selected_score'] - t['A']:+.2f}",
            f"{max(t[a] for a in 'ABC') - t['selected_score']:.2f}",
        ]
    )
table(
    "workbuddy_cases.tex",
    "lrrrrrr",
    r"Task & \shortstack{Task\\only} & \shortstack{Reward\\revision} & \shortstack{Hypothesis\\guidance} & Choice & $\Delta$ vs. task only & Gap",
    case_rows,
)
(ROOT / "data/workbuddy_office_analysis.json").write_text(
    json.dumps(analysis, indent=2) + "\n"
)
print(
    "Verified: 50 task summaries, 150 accepted final receipts; generated results and supplementary analyses."
)
