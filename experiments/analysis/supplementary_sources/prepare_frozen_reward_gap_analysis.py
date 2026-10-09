"""Gap-restricted agreement on the exact original frozen RewardPack triads.

Reproduce the published unfiltered metrics first. No candidate reselection,
new judgments, new baseline scores, Jury ranks, or manuscript edits.
"""

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from fractions import Fraction
from itertools import combinations
from pathlib import Path

import numpy as np
from compute_joint_offline_metrics import BOOTSTRAPS, SEED, bootstrap_mean
from compute_known_reward_metrics import score_counts, task_metrics

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/rewardpack_joint_repair"
OUT = ROOT / "data/frozen_reward_gap_analysis"
THRESHOLDS = (2, 3, 5, 10)
TOLERANCE = 1e-8
FIELDS = ("tau_gap", "full_credit", "strict_accuracy", "half_accuracy", "tie_rate")
NAMES = {"workbuddy": "WorkBuddy", "claw_eval": "claw-eval"}


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pair_outcome(a, b):
    if a["official_score"] == b["official_score"]:
        return "official_tie"
    x, y = (
        Fraction(a["passed"], a["known_count"]),
        Fraction(b["passed"], b["known_count"]),
    )
    if x == y:
        return "reward_tie"
    return (
        "correct"
        if (x > y) == (a["official_score"] > b["official_score"])
        else "reversed"
    )


def filtered_metrics(pairs, threshold):
    eligible = [p for p in pairs if p["official_gap"] >= threshold - TOLERANCE]
    counts = Counter(p["outcome"] for p in eligible)
    assert not counts["official_tie"]
    c, d, t = counts["correct"], counts["reversed"], counts["reward_tie"]
    n = len(eligible)
    assert c + d + t == n
    return dict(
        eligible_pairs=n,
        correct_pairs=c,
        reversed_pairs=d,
        reward_tie_pairs=t,
        tau_gap=(c - d) / math.sqrt(n * (c + d)) if c + d else None,
        full_credit=(c + t) / n if n else None,
        strict_accuracy=c / n if n else None,
        half_accuracy=(c + 0.5 * t) / n if n else None,
        tie_rate=t / n if n else None,
        tau_undefined_reason=None
        if c + d
        else "all_reward_ties"
        if n
        else "no_eligible_pairs",
    )


def summarize(rows, benchmark, fields=FIELDS):
    result = dict(
        total_tasks=len(rows),
        sampling_units=len(
            {r["task"] if benchmark == "workbuddy" else r["scenario"] for r in rows}
        ),
        metrics={
            f: dict(
                **bootstrap_mean(rows, f, benchmark),
                defined_tasks=sum(r[f] is not None for r in rows),
            )
            for f in fields
        },
    )
    if fields == FIELDS:
        result["pair_counts"] = {
            k: sum(r[k] for r in rows)
            for k in (
                "eligible_pairs",
                "correct_pairs",
                "reversed_pairs",
                "reward_tie_pairs",
            )
        }
        result["tau_undefined_reasons"] = dict(
            Counter(
                r["tau_undefined_reason"] for r in rows if r["tau_undefined_reason"]
            )
        )
    return result


def display(metric, percentage=False):
    if metric["mean"] is None:
        return "未定义"
    scale, decimals = (100, 2) if percentage else (1, 3)
    lo, hi = metric["ci95"]
    return f"{scale * metric['mean']:.{decimals}f} [{scale * lo:.{decimals}f}, {scale * hi:.{decimals}f}]"


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def prepare():
    original = json.loads((ROOT / "data/known_reward_metrics/summary.json").read_text())
    source_hashes = original["protocol"]["sources"]
    for p, digest in source_hashes.items():
        assert sha(ROOT / p) == digest, p
    preserved_paths = list(source_hashes) + [
        "data/known_reward_metrics/summary.json",
        "data/known_reward_kendall.json",
        "data/jury_gap_analysis/summary.json",
        "main.tex",
        "appendix.tex",
        "main.pdf",
        "tables/known_reward_order_agreement.tex",
    ]
    preserved = {p: sha(ROOT / p) for p in preserved_paths}
    receipts = defaultdict(list)
    for row in read_csv(SOURCE / "criterion_receipts.csv"):
        receipts[row["identity"]].append(row)
    groups = defaultdict(list)
    candidates = []
    for row in read_csv(SOURCE / "trajectory_scores.csv"):
        cs = receipts[row["identity"]]
        assert len({c["criterion_id"] for c in cs}) == len(cs)
        assert all(
            c["benchmark"] == row["benchmark"]
            and c["task"] == row["task"]
            and c["slot"] == row["slot"]
            for c in cs
        )
        candidate = {
            k: row[k]
            for k in ("identity", "benchmark", "task", "scenario", "slot", "pack_id")
        }
        candidate.update(
            official_score=float(row["official_score"]), **score_counts(cs)
        )
        assert candidate["known_count"] > 0
        candidate["reward_fraction"] = str(
            Fraction(candidate["passed"], candidate["known_count"])
        )
        candidates.append(candidate)
        groups[row["benchmark"], row["task"]].append(candidate)
    assert (
        len(candidates) == 462
        and len(groups) == 154
        and len({r["identity"] for r in candidates}) == 462
    )
    assert Counter(r["benchmark"] for r in candidates) == {
        "workbuddy": 150,
        "claw_eval": 312,
    }
    control_rows = []
    pair_rows = []
    task_rows = []
    for (b, task), rows in sorted(groups.items()):
        assert len(rows) == 3 and len({r["pack_id"] for r in rows}) == 1
        control = task_metrics(rows, "known")
        control_rows.append(
            dict(benchmark=b, task=task, scenario=rows[0]["scenario"], **control)
        )
        pairs = []
        for a, z in combinations(sorted(rows, key=lambda r: r["identity"]), 2):
            gap = abs(a["official_score"] - z["official_score"])
            pair = dict(
                benchmark=b,
                task=task,
                scenario=a["scenario"],
                identity_1=a["identity"],
                slot_1=a["slot"],
                official_1=a["official_score"],
                reward_1=a["reward_fraction"],
                identity_2=z["identity"],
                slot_2=z["slot"],
                official_2=z["official_score"],
                reward_2=z["reward_fraction"],
                official_gap=gap,
                outcome=pair_outcome(a, z),
            )
            pair.update({f"eligible_gap_{t}": gap >= t - TOLERANCE for t in THRESHOLDS})
            flags = [pair[f"eligible_gap_{t}"] for t in THRESHOLDS]
            assert flags == sorted(flags, reverse=True)
            pairs.append(pair)
            pair_rows.append(pair)
        for gap in THRESHOLDS:
            task_rows.append(
                dict(
                    benchmark=b,
                    task=task,
                    scenario=rows[0]["scenario"],
                    threshold=gap,
                    **filtered_metrics(pairs, gap),
                )
            )
    results = {}
    summary_rows = []
    for b in NAMES:
        controls = [r for r in control_rows if r["benchmark"] == b]
        control = summarize(
            controls, b, ("tau_b", "full_credit", "strict_accuracy", "half_accuracy")
        )
        expected = original[b]["known_all"]
        for field, estimate in control["metrics"].items():
            for key in ("mean", "ci95"):
                np.testing.assert_allclose(
                    estimate[key], expected[field][key], rtol=0, atol=1e-14
                )
            assert (
                estimate["defined_replicates"] == expected[field]["defined_replicates"]
            )
        assert (
            control["metrics"]["tau_b"]["defined_tasks"]
            == expected["defined_correlation_tasks"]
        )
        assert (
            control["metrics"]["full_credit"]["defined_tasks"]
            == expected["pair_evaluable_tasks"]
        )
        control["pair_counts"] = expected["pair_counts"]
        for key, value in expected["pair_counts"].items():
            assert sum(r[key] for r in controls) == value
        results[b] = dict(unfiltered_control=control, thresholds={})
        for gap in THRESHOLDS:
            rows = [
                r for r in task_rows if r["benchmark"] == b and r["threshold"] == gap
            ]
            s = summarize(rows, b)
            results[b]["thresholds"][str(gap)] = s
            flat = dict(
                benchmark=b,
                threshold=gap,
                total_tasks=s["total_tasks"],
                **s["pair_counts"],
            )
            for field, m in s["metrics"].items():
                flat.update(
                    {
                        field: m["mean"],
                        field + "_ci_low": m["ci95"][0] if m["ci95"] else None,
                        field + "_ci_high": m["ci95"][1] if m["ci95"] else None,
                        field + "_defined_tasks": m["defined_tasks"],
                    }
                )
            summary_rows.append(flat)
    assert all(sha(ROOT / p) == digest for p, digest in preserved.items())
    result = dict(
        protocol=dict(
            candidate_selection="Exactly the original frozen 3 candidates per task; no reselection or substitutions.",
            reward="Exact Fraction(P, P+F) from selectively repaired original criterion receipts; unknown excluded.",
            official_scores="Original frozen trajectory_scores.csv labels; no later baseline grading or imputation substituted.",
            thresholds=list(THRESHOLDS),
            threshold_rule="absolute official difference >= threshold - 1e-8",
            order_agreement="(C+T)/(C+D+T); reward ties full credit, official ties excluded.",
            tau_gap="(C-D)/sqrt((C+D+T)*(C+D)); all reward ties undefined. Restricted tau-b-style statistic, not ordinary full-pool Kendall tau-b.",
            unfiltered_tau="Original standard Kendall tau-b, including native ties in its normalization.",
            aggregation="Within-task metric then mean over defined tasks; no cross-round averaging or pair-count weighting.",
            bootstrap=dict(
                iterations=BOOTSTRAPS,
                seed=SEED,
                units="WorkBuddy tasks; claw-eval scenarios retaining bilingual versions; all original units retained",
            ),
            model_calls=0,
            sources=source_hashes,
            code_sha256=sha(Path(__file__)),
        ),
        validation=dict(
            original_point_estimates_and_intervals_reproduced=True,
            preserved_file_hashes=preserved,
            candidates=462,
            tasks=154,
            pair_slots=len(pair_rows),
        ),
        benchmarks=results,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in [
        ("candidate_scores.csv", candidates),
        ("pairs.csv", pair_rows),
        ("task_statistics.csv", task_rows),
        ("summary.csv", summary_rows),
    ]:
        write_csv(OUT / name, rows)
    (OUT / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    lines = [
        "# 原冻结 RewardPack 候选池：official 分差分析",
        "",
        "严格复用原正文 82.33% / 80.53% 对应的 154 个任务、462 个候选及其原生标签，未重新挑选候选。仅在同一任务的原三个候选内部，筛选 official 分差 ≥2、≥3、≥5、≥10 的 pairs。没有混入 Jury 排名、新增补评回执、后续六轨迹 baseline 或其零值填补。",
        "",
        "RewardPack 分数为修复后的原 criterion 回执 P/(P+F)，使用精确分数判定平局。Order agreement 将 reward 平局计 1、同序计 1、逆序计 0；先算每任务比例，再对有合格 pairs 的任务等权平均。它是不逆序率，不是严格区分正确率。",
        "",
        "筛选后的限定 τ=(C−D)/√[(C+D+T)(C+D)]；C/D/T 为同序/逆序/reward 平局数。没有合格 pair 或全 reward 平局时 τ 未定义，不填零。该指标是分差限定的 τ-b 型指标，区别于未筛选对照的标准 Kendall τ-b（后者对 native ties 的归一化仍保留）。",
        "",
        "区间沿用 20,000 次 bootstrap、seed=20260929：WorkBuddy 按全部 50 个任务；claw-eval 按全部 68 个场景簇，保留双语关联。无合格 pair 的任务留在抽样框中，指标只平均有定义任务。阈值边界容差 1e-8。",
        "",
        "| Benchmark | Official 分差 | τ [95% CI] | Order agreement % [95% CI] | 严格同序 % | 平局半分 % | τ 有定义任务 | Agreement 有定义任务 | 合格 pairs |",
        "|---|---|---|---|---:|---:|---|---|---:|",
    ]
    for b, res in results.items():
        for label, s, tf in [
            ("未筛选（原值复现）", res["unfiltered_control"], "tau_b")
        ] + [(f"≥{g}", v, "tau_gap") for g, v in res["thresholds"].items()]:
            m = s["metrics"]
            n = s["pair_counts"].get(
                "eligible_pairs", s["pair_counts"].get("strict_pairs")
            )
            lines.append(
                f"| {NAMES[b]} | {label} | {display(m[tf])} | {display(m['full_credit'], True)} | {100 * m['strict_accuracy']['mean']:.2f} | {100 * m['half_accuracy']['mean']:.2f} | {m[tf]['defined_tasks']}/{s['total_tasks']} | {m['full_credit']['defined_tasks']}/{s['total_tasks']} | {n} |"
            )
    lines += [
        "",
        "各阈值使用嵌套 pair 集合，但有定义任务集合可变，特别是 τ 与 agreement 的任务集合不一定相同。不能把平均指标变化直接解释为同一批任务上的改进。全部 reward 平局可得到 100% agreement，同时 τ 未定义。",
        "",
        "文件：summary.json 保存置信区间、有效 bootstrap 次数、覆盖率、未定义原因和校验信息；pairs.csv 保留全部 462 个原始 pair 槽位及四个阈值标记；candidate_scores.csv 与 task_statistics.csv 可追溯候选和任务指标。复现命令：`python scripts/prepare_frozen_reward_gap_analysis.py`。原始数据、Jury 分析、正文表和 PDF 均保持原样。",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[9:]))
    return result


if __name__ == "__main__":
    prepare()
