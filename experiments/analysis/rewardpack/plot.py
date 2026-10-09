"""Plot the distribution of defined task-local Kendall tau-b values."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .inputs import OUTPUT, read


def plot(output=OUTPUT):
    with (output / "task_statistics.csv").open(newline="", encoding="utf-8") as handle:
        tasks = list(csv.DictReader(handle))
    summary = read(output / "summary.json")
    groups = ("workbuddy", "claw_eval")
    values = {group: [float(row["tau_b"]) for row in tasks
                      if row["benchmark"] == group and row["tau_b"]] for group in groups}
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.25), sharey=True, constrained_layout=True)
    bins = np.linspace(-1, 1, 11)
    for axis, group, title, color in zip(axes, groups,
                                         ("WorkBuddy", "claw-eval"),
                                         ("#207a6e", "#bf6639"), strict=True):
        axis.hist(values[group], bins=bins, color=color, edgecolor="white", linewidth=.8)
        axis.axvline(0, color="#30363b", linewidth=.9)
        if values[group]:
            axis.axvline(np.mean(values[group]), color="#30363b", linewidth=1.5, linestyle="--")
        else:
            axis.text(.5, .5, "No defined correlations", ha="center", transform=axis.transAxes)
        axis.set_title(f"{title}  (n={len(values[group])}/{summary[group]['tasks']})")
        axis.set_xlim(-1, 1)
        axis.set_xticks(np.linspace(-1, 1, 5))
        axis.set_xlabel(r"Task Kendall $\tau_b$")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Tasks")
    target = output / "task_tau_distribution.pdf"
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)
    print(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    plot(parser.parse_args().output)


if __name__ == "__main__":
    main()
