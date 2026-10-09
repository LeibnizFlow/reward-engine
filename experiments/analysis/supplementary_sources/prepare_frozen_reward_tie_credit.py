"""User-defined positive tie credit on the original frozen RewardPack pairs.

This is NOT standard Kendall tau-b: (C+T-D)/(C+T+D) = 2*order_agreement-1.
Original tau-b and gap-restricted tau-b-style values remain separate references.
"""

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from compute_joint_offline_metrics import BOOTSTRAPS, SEED, bootstrap_mean

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/frozen_reward_gap_analysis"
OUT = ROOT / "data/frozen_reward_tie_credit"
THRESHOLDS = (None, 2, 3, 5, 10)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def positive_tie_score(c, d, t):
    assert min(c, d, t) >= 0
    n = c + d + t
    return (c + t - d) / n if n else None


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def display(m):
    return (
        "未定义"
        if m["mean"] is None
        else f"{m['mean']:.3f} [{m['ci95'][0]:.3f}, {m['ci95'][1]:.3f}]"
    )


def prepare():
    source = json.loads((SOURCE / "summary.json").read_text())
    assert source["validation"]["original_point_estimates_and_intervals_reproduced"]
    # Historical snapshot hashes also record the then-current paper. Later manuscript
    # revisions are expected; only scientific inputs must remain immutable.
    for p, h in source["validation"]["preserved_file_hashes"].items():
        if p.startswith("data/"):
            assert sha(ROOT / p) == h, p
    dependencies = [
        "data/frozen_reward_gap_analysis/summary.json",
        "data/frozen_reward_gap_analysis/pairs.csv",
        "data/frozen_reward_gap_analysis/REPORT.md",
        "scripts/prepare_frozen_reward_tie_credit.py",
        "scripts/compute_joint_offline_metrics.py",
    ]
    hashes = {p: sha(ROOT / p) for p in dependencies}
    pairs = read_csv(SOURCE / "pairs.csv")
    groups = defaultdict(list)
    for p in pairs:
        groups[p["benchmark"], p["task"]].append(p)
    assert len(pairs) == 462 and len(groups) == 154
    tasks = []
    for (b, task), ps in sorted(groups.items()):
        assert len(ps) == 3
        for threshold in THRESHOLDS:
            selected = [
                p
                for p in ps
                if p["outcome"] != "official_tie"
                and (threshold is None or p[f"eligible_gap_{threshold}"] == "True")
            ]
            counts = Counter(p["outcome"] for p in selected)
            c, d, t = counts["correct"], counts["reversed"], counts["reward_tie"]
            n = c + d + t
            assert n == len(selected)
            tasks.append(
                dict(
                    benchmark=b,
                    task=task,
                    scenario=ps[0]["scenario"],
                    threshold="unfiltered" if threshold is None else str(threshold),
                    eligible_pairs=n,
                    correct_pairs=c,
                    reversed_pairs=d,
                    reward_tie_pairs=t,
                    positive_tie_score=positive_tie_score(c, d, t),
                    order_agreement=(c + t) / n if n else None,
                    all_reward_tied=bool(n and t == n),
                    undefined_reason=None
                    if n
                    else "no_eligible_officially_ordered_pairs",
                )
            )
    results = {}
    flat = []
    for b in ("workbuddy", "claw_eval"):
        results[b] = {}
        for threshold in THRESHOLDS:
            key = "unfiltered" if threshold is None else str(threshold)
            rows = [r for r in tasks if r["benchmark"] == b and r["threshold"] == key]
            prior = (
                source["benchmarks"][b]["unfiltered_control"]
                if threshold is None
                else source["benchmarks"][b]["thresholds"][key]
            )
            agreement = bootstrap_mean(rows, "order_agreement", b)
            reference = prior["metrics"]["full_credit"]
            for k in ("mean", "ci95"):
                np.testing.assert_allclose(
                    agreement[k], reference[k], atol=1e-14, rtol=0
                )
            score = bootstrap_mean(rows, "positive_tie_score", b)
            np.testing.assert_allclose(
                score["mean"], 2 * agreement["mean"] - 1, atol=1e-14, rtol=0
            )
            np.testing.assert_allclose(
                score["ci95"],
                [2 * v - 1 for v in agreement["ci95"]],
                atol=1e-14,
                rtol=0,
            )
            assert score["defined_replicates"] == agreement["defined_replicates"]
            counts = {
                k: sum(r[k] for r in rows)
                for k in (
                    "eligible_pairs",
                    "correct_pairs",
                    "reversed_pairs",
                    "reward_tie_pairs",
                )
            }
            old_tau = prior["metrics"]["tau_b" if threshold is None else "tau_gap"]
            result = dict(
                positive_tie_score=score,
                order_agreement=agreement,
                reference_tau=old_tau,
                reference_tau_definition="standard Kendall tau-b"
                if threshold is None
                else "gap-restricted tau-b-style",
                total_tasks=len(rows),
                defined_tasks=sum(r["positive_tie_score"] is not None for r in rows),
                all_reward_tied_tasks_included=sum(r["all_reward_tied"] for r in rows),
                pair_counts=counts,
            )
            assert result["defined_tasks"] == reference["defined_tasks"]
            results[b][key] = result
            flat.append(
                dict(
                    benchmark=b,
                    threshold=key,
                    positive_tie_score=score["mean"],
                    ci_low=score["ci95"][0],
                    ci_high=score["ci95"][1],
                    defined_tasks=result["defined_tasks"],
                    total_tasks=len(rows),
                    all_reward_tied_tasks_included=result[
                        "all_reward_tied_tasks_included"
                    ],
                    order_agreement=agreement["mean"],
                    reference_tau=old_tau["mean"],
                    reference_tau_defined_tasks=old_tau["defined_tasks"],
                    **counts,
                )
            )
    assert all(sha(ROOT / p) == h for p, h in hashes.items())
    summary = dict(
        protocol=dict(
            name="positive_tie_order_score",
            definition="(C+T-D)/(C+T+D)",
            relation_to_agreement="Exactly 2 * task-mean order agreement - 1; not independent evidence.",
            not_standard_kendall_tau=True,
            scores=dict(correct=1, reward_tie=1, reversed=-1),
            official_ties="Excluded as in original order agreement; all-official-tied tasks remain undefined.",
            all_reward_ties="Score 1 when at least one eligible official-ordered pair exists.",
            aggregation="Per-task score then equal mean over defined tasks; original frozen 3-candidate pools only.",
            thresholds=["unfiltered", 2, 3, 5, 10],
            bootstrap=dict(
                iterations=BOOTSTRAPS,
                seed=SEED,
                units="WorkBuddy tasks; claw-eval scenario clusters; all original units retained",
            ),
            sources=hashes,
            model_calls=0,
        ),
        benchmarks=results,
        validation=dict(
            original_order_agreement_reproduced=True,
            affine_point_and_ci_identity_verified=True,
            inputs_unchanged=True,
        ),
    )
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "task_statistics.csv", tasks)
    write_csv(OUT / "summary.csv", flat)
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    lines = [
        "# 平局计正分的排序一致性：原冻结 RewardPack 候选池",
        "",
        "按用户指定：同序与 RewardPack 平局均记 +1，逆序记 −1。全 reward 平局且有合格 official 严格 pair 的任务得 1，纳入任务均值。Official 平局仍按原口径排除；没有合格 pair 的任务仍未定义，不填零或一。",
        "",
        "**这不是标准 Kendall τ-b。** 将其标记为平局计正分指标 τ₊=(C+T−D)/(C+T+D)，恰好等于 2×order agreement−1；数值、覆盖率及区间都是同一 agreement 信息的线性变换，不构成第二项独立证据。保留原标准 τ-b / 分差限定 τ-b 型指标作对照。",
        "",
        "候选、官方分数、RewardPack 的 P/(P+F)、分差筛选、任务等权平均与原冻结池分析一致。区间使用 20,000 次 bootstrap、seed 20260929：WorkBuddy 按全部任务，claw-eval 按全部场景簇保留双语关联。没有新增模型调用或修改论文。",
        "",
        "| Benchmark | Official 分差 | 平局计正分 τ₊ [95% CI] | 有定义任务 | 纳入的全 reward 平局任务 | 原 τ 对照 [95% CI] | 原 τ 有定义任务 |",
        "|---|---|---|---|---:|---|---|",
    ]
    for b, thresholds in results.items():
        for key, r in thresholds.items():
            name = "WorkBuddy" if b == "workbuddy" else "claw-eval"
            label = "未筛选" if key == "unfiltered" else "≥" + key
            lines.append(
                f"| {name} | {label} | {display(r['positive_tie_score'])} | {r['defined_tasks']}/{r['total_tasks']} | {r['all_reward_tied_tasks_included']} | {display(r['reference_tau'])} | {r['reference_tau']['defined_tasks']}/{r['total_tasks']} |"
            )
    lines += [
        "",
        "复现：`python scripts/prepare_frozen_reward_tie_credit.py`。summary.json 包含原 agreement、原 τ 对照、pair 计数及覆盖率；task_statistics.csv 保留每任务的 C/D/T 和计算结果。旧报告保持原样。",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[8:]))
    return summary


if __name__ == "__main__":
    prepare()
