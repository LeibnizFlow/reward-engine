"""Recompute task-local statistics and clustered intervals from frozen candidate scores."""
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import kendalltau, spearmanr
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

def main():
    root = Path(__file__).resolve().parents[1] / 'data/rewardpack_joint'
    with (root / 'trajectory_scores.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    expected = json.loads((root / 'summary.json').read_text())
    groups = defaultdict(list)
    for row in rows:
        for field in ('reward', 'official_score', 'r_min', 'r_max'):
            row[field] = float(row[field]) if row[field] else None
        groups[(row['benchmark'], row['task'])].append(row)
    output = {}
    for benchmark in ('workbuddy', 'claw_eval'):
        tasks = [{'task': task, 'scenario': records[0]['scenario'], **task_statistics(records)}
                 for (b, task), records in sorted(groups.items()) if b == benchmark]
        target = expected[benchmark]
        report = {}
        for metric, mean_key, ci_key in (
                ('tau_b', 'mean_tau_b', 'bootstrap'),
                ('spearman_rho', 'mean_spearman_rho', 'spearman_bootstrap')):
            values = [r[metric] for r in tasks if r[metric] is not None]
            mean = float(np.mean(values)) if values else None
            assert mean is target[mean_key] if mean is None else abs(mean - target[mean_key]) < 1e-12
            actual = bootstrap(tasks, benchmark, metric)
            if actual['ci95'] is None:
                assert target[ci_key]['ci95'] is None
            else:
                assert np.allclose(actual['ci95'], target[ci_key]['ci95'], atol=1e-12, rtol=0)
            assert actual['defined_replicates'] == target[ci_key]['defined_replicates']
            report[metric] = {'mean': mean, 'ci95': actual['ci95']}
        assert sum(t['paired'] for t in tasks) == target['complete_paired_trajectories']
        assert sum(t['tau_b'] is not None for t in tasks) == target['correlations_defined']
        output[benchmark] = report
    print(json.dumps({'status': 'reproduced', 'results': output}, indent=2))

if __name__ == '__main__':
    main()
