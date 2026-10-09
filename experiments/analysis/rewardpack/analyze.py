"""Task-local rank agreement, coverage, and clustered uncertainty."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, rankdata, spearmanr

from .inputs import OUTPUT, REPO, canonical, read, sha

ITERATIONS = 20_000
SEED = 20260929


def task_statistics(rows):
    paired = [r for r in rows if r.get("reward") is not None and r.get("official_score") is not None]
    result = {"trajectories": len(rows), "paired": len(paired), "eligible_for_correlation": len(paired) >= 3,
              "tau_b": None, "spearman_rho": None, "undefined_reason": None,
              "official_strict_pairs": 0, "official_tie_pairs": 0,
              "correct_strict_pairs": 0, "reward_tie_on_official_strict": 0,
              "certain_interval_pairs": 0, "correct_certain_interval_pairs": 0,
              "overlapping_interval_pairs": 0}
    if len(paired) >= 3:
        reward = [r["reward"] for r in paired]
        official = [r["official_score"] for r in paired]
        if len(set(reward)) == 1 or len(set(official)) == 1:
            result["undefined_reason"] = ("both_constant" if len(set(reward)) == len(set(official)) == 1
                                           else "reward_constant" if len(set(reward)) == 1 else "official_constant")
        else:
            result["tau_b"] = float(kendalltau(reward, official, variant="b").statistic)
            result["spearman_rho"] = float(spearmanr(reward, official).statistic)
    for a, b in itertools.combinations(rows, 2):
        x, y = a.get("official_score"), b.get("official_score")
        if x is None or y is None:
            continue
        if x == y:
            result["official_tie_pairs"] += 1
            continue
        ra, rb = a.get("reward"), b.get("reward")
        if ra is not None and rb is not None:
            result["official_strict_pairs"] += 1
            if ra == rb:
                result["reward_tie_on_official_strict"] += 1
            elif (ra - rb) * (x - y) > 0:
                result["correct_strict_pairs"] += 1
        elif a.get("r_min") is not None and b.get("r_min") is not None:
            if a["r_min"] > b["r_max"] or b["r_min"] > a["r_max"]:
                result["certain_interval_pairs"] += 1
                result["correct_certain_interval_pairs"] += int(
                    (a["r_min"] - b["r_min"]) * (x - y) > 0)
            else:
                result["overlapping_interval_pairs"] += 1
    return result


def bootstrap(task_rows, benchmark, metric="tau_b"):
    if benchmark == "workbuddy":
        clusters = {r["task"]: [r] for r in task_rows}
    else:
        clusters = defaultdict(list)
        for row in task_rows:
            clusters[row["scenario"]].append(row)
    keys = sorted(clusters)
    rng = np.random.default_rng(SEED)
    estimates = []
    for _ in range(ITERATIONS):
        sampled = [r for index in rng.integers(len(keys), size=len(keys))
                   for r in clusters[keys[index]] if r[metric] is not None]
        if sampled:
            estimates.append(float(np.mean([r[metric] for r in sampled])))
    return {"iterations": ITERATIONS, "seed": SEED, "clusters": len(keys),
            "defined_replicates": len(estimates),
            "ci95": np.quantile(estimates, [.025, .975]).tolist() if estimates else None}


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list))
                             else value for key, value in row.items()})


