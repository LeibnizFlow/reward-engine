"""Verify and expose Kendall tau-b for the frozen main-text Ranking pool.

No judgments are generated. Recompute from frozen criterion counts and grades;
retain the original 20,000-draw, task/scenario bootstrap and undefined values.
"""

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from compute_joint_offline_metrics import BOOTSTRAPS, SEED, bootstrap_mean
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[1]


def prepare():
    summary = json.loads((ROOT / "data/known_reward_metrics/summary.json").read_text())
    for name, sha in summary["protocol"]["sources"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha
    receipts = defaultdict(Counter)
    with (ROOT / "data/rewardpack_joint_repair/criterion_receipts.csv").open() as f:
        for row in csv.DictReader(f):
            receipts[row["identity"]][row["status"]] += 1
    groups = defaultdict(list)
    with (ROOT / "data/rewardpack_joint_repair/trajectory_scores.csv").open() as f:
        for row in csv.DictReader(f):
            groups[row["benchmark"], row["task"]].append(row)
    records = defaultdict(list)
    for (benchmark, task), rows in sorted(groups.items()):
        assert len(rows) == 3
        rewards, grades = [], []
        for row in rows:
            counts = receipts[row["identity"]]
            known = counts["pass"] + counts["fail"]
            assert known > 0
            rewards.append(counts["pass"] / known)
            grades.append(float(row["official_score"]))
        constant_reward, constant_grade = len(set(rewards)) == 1, len(set(grades)) == 1
        reason = (
            "both_constant"
            if constant_reward and constant_grade
            else "reward_constant"
            if constant_reward
            else "official_constant"
            if constant_grade
            else None
        )
        tau = (
            None
            if reason
            else float(kendalltau(rewards, grades, variant="b").statistic)
        )
        records[benchmark].append(
            dict(
                task=task,
                scenario=rows[0]["scenario"],
                tau_b=tau,
                undefined_reason=reason,
            )
        )
    result = dict(
        definition="Per-task Kendall tau-b of P/(P+F) and frozen native grades; mean over defined tasks.",
        bootstrap=dict(
            iterations=BOOTSTRAPS,
            seed=SEED,
            units="WorkBuddy tasks; claw-eval scenarios; all original clusters retained.",
        ),
        sources=summary["protocol"]["sources"],
        llm_calls=0,
        benchmarks={},
    )
    for benchmark, rows in records.items():
        estimate = bootstrap_mean(rows, "tau_b", benchmark)
        expected = summary[benchmark]["known_all"]
        assert estimate == expected["tau_b"]
        defined = sum(r["tau_b"] is not None for r in rows)
        assert defined == expected["defined_correlation_tasks"]
        result["benchmarks"][benchmark] = dict(
            tau_b=estimate,
            defined_tasks=defined,
            total_tasks=len(rows),
            undefined_reasons=dict(
                Counter(r["undefined_reason"] for r in rows if r["undefined_reason"])
            ),
            order_agreement=expected["full_credit"],
        )
    (ROOT / "data/known_reward_kendall.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    # Historical Kendall output stays separate from the current positive-tie presentation.
    print(json.dumps(result["benchmarks"], indent=2))
    return result


if __name__ == "__main__":
    prepare()
