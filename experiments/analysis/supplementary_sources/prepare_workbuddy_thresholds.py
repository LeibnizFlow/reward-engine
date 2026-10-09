"""WorkBuddy score-threshold sensitivity with the official pass threshold of 85.

Uses frozen scores without new grading calls.
"""

import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("A", "B", "C", "selected_score")
OFFICIAL_THRESHOLD = 85


def counts_at(rows, threshold):
    return {arm: sum(row[arm] >= threshold for row in rows) for arm in ARMS}


def analyze(rows):
    assert rows and len({r["task"] for r in rows}) == len(rows)
    for row in rows:
        assert all(
            isinstance(row[a], (int, float))
            and not isinstance(row[a], bool)
            and math.isfinite(row[a])
            and 0 <= row[a] <= 100
            for a in ARMS
        )
        assert row["selected_score"] == row[row["selected_arm"]]
    n = len(rows)
    grid = []
    for threshold in range(101):
        counts = counts_at(rows, threshold)
        grid.append(
            {
                "threshold": threshold,
                "counts": counts,
                "rates_percent": {a: 100 * counts[a] / n for a in ARMS},
                "difference_pp": {
                    a: 100 * (counts[a] - counts["A"]) / n for a in ARMS[1:]
                },
            }
        )
    cuts = sorted({0.0, 100.0} | {r[a] for r in rows for a in ARMS})
    intervals = []
    for lower, upper in zip(cuts, cuts[1:], strict=False):
        counts = counts_at(rows, upper)
        intervals.append(
            {"lower_exclusive": lower, "upper_inclusive": upper, "counts": counts}
        )
    return {
        "n_tasks": n,
        "official_pass_threshold": OFFICIAL_THRESHOLD,
        "grid": grid,
        "exact_intervals": intervals,
    }


def generate():
    source = ROOT / "data/office50.json"
    rows = json.loads(source.read_text())["tasks"]
    assert len(rows) == 50
    result = analyze(rows)
    result["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    (ROOT / "data/workbuddy_threshold_sensitivity.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    with (ROOT / "data/workbuddy_threshold_sensitivity.csv").open(
        "w", newline=""
    ) as stream:
        names = (
            ["threshold"]
            + [f"{a}_passes" for a in ARMS]
            + [f"{a}_rate_percent" for a in ARMS]
            + [f"{a}_minus_A_pp" for a in ARMS[1:]]
        )
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        for point in result["grid"]:
            writer.writerow(
                {
                    "threshold": point["threshold"],
                    **{f"{a}_passes": point["counts"][a] for a in ARMS},
                    **{f"{a}_rate_percent": point["rates_percent"][a] for a in ARMS},
                    **{f"{a}_minus_A_pp": point["difference_pp"][a] for a in ARMS[1:]},
                }
            )
    thresholds = sorted(
        {50, 60, 70, 75, 80, 85, 90, 95, result["official_pass_threshold"]}
    )
    lines = [
        r"\begin{tabular}{rrrrrrr}",
        r"\toprule",
        r"Threshold & \shortstack{Task only\\(\%)} & \shortstack{Reward revision\\(\%)} & \shortstack{Hypothesis guidance\\(\%)} & \shortstack{Selected by\\reward (\%)} & \shortstack{Revision gain\\(pp)} & \shortstack{Selection gain\\(pp)} \\",
        r"\midrule",
    ]
    for threshold in thresholds:
        point = result["grid"][threshold]
        cells = (
            [str(threshold)]
            + [f"{point['rates_percent'][a]:.2f}" for a in ARMS]
            + [f"{point['difference_pp'][a]:+.2f}" for a in ("B", "selected_score")]
        )
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "tables/workbuddy_threshold_sensitivity.tex").write_text(
        "\n".join(lines) + "\n"
    )
    return result


def draw(result):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(
        2, 1, figsize=(6.6, 4.3), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]}
    )
    points = [
        {"upper_inclusive": 0.0, "counts": {a: result["n_tasks"] for a in ARMS}}
    ] + result["exact_intervals"]
    x = [p["upper_inclusive"] for p in points]
    names = {
        "A": "Task only",
        "B": "Reward-guided revision",
        "C": "Hypothesis guidance",
        "selected_score": "Selected by reward",
    }
    colors = {
        "A": "#555555",
        "B": "#256aa3",
        "C": "#c57b27",
        "selected_score": "#3d8054",
    }
    styles = {"A": ":", "B": "-", "C": "--", "selected_score": "-."}
    for arm in ARMS:
        axes[0].step(
            x,
            [100 * p["counts"][arm] / result["n_tasks"] for p in points],
            where="pre",
            label=names[arm],
            color=colors[arm],
            linestyle=styles[arm],
            lw=1.4,
        )
        if arm != "A":
            axes[1].step(
                x,
                [
                    100 * (p["counts"][arm] - p["counts"]["A"]) / result["n_tasks"]
                    for p in points
                ],
                where="pre",
                color=colors[arm],
                linestyle=styles[arm],
                lw=1.4,
            )
    t = result["official_pass_threshold"]
    for ax in axes:
        ax.axvline(t, color="#7e548b", linestyle=":", lw=1)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set(ylabel="Score ≥ threshold (%)", ylim=(-3, 103))
    axes[0].legend(loc="lower left", frameon=False, ncol=2, fontsize=8)
    axes[0].text(
        t - 1,
        99,
        f"{t}: official pass threshold",
        ha="right",
        va="top",
        color="#7e548b",
        fontsize=8,
    )
    axes[1].axhline(0, color=".45", lw=0.7)
    axes[1].set(
        xlabel="Threshold on recorded composite score / 100",
        ylabel="Difference from task only (pp)",
        xlim=(0, 100),
    )
    fig.tight_layout(pad=0.7)
    fig.savefig(
        ROOT / "figures/workbuddy_threshold_sensitivity.pdf", bbox_inches="tight"
    )
    fig.savefig(
        ROOT / "figures/workbuddy_threshold_sensitivity.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


if __name__ == "__main__":
    result = generate()
    draw(result)
    print(
        json.dumps(
            {
                "official_pass_threshold": result["official_pass_threshold"],
                "n_tasks": result["n_tasks"],
            },
            indent=2,
        )
    )
