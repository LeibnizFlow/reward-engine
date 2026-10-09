"""Read-only reanalysis of fixed triads with separated official score gaps.

Only the NEW output directory is written. No scorer, model, actor or original
analysis entrypoint is invoked. Original and repaired judgments stay separate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from copy import deepcopy
from itertools import combinations
from pathlib import Path

import numpy as np

from .analyze import task_statistics, write_csv
from .inputs import OUTPUT, canonical, read, sha

SEED = 20260929
ITERATIONS = 20_000
GAP_EPSILON = 1e-8
VERSIONS = ("original", "repaired")
IDENTITY_FIELDS = (
    "benchmark",
    "task",
    "scenario",
    "slot",
    "pack_id",
    "pack_sha256",
    "trace_sha256",
    "snapshot_sha256",
    "official_receipt_sha256",
    "official_status",
)


def load_scores(path):
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    seen = set()
    for row in rows:
        if not row.get("identity") or row["identity"] in seen:
            raise ValueError("Missing or duplicate candidate identity")
        seen.add(row["identity"])
        if row["benchmark"] not in {"workbuddy", "claw_eval"} or not row["scenario"]:
            raise ValueError("Benchmark and clustering scenario must be known")
        for field in ("reward", "r_min", "r_max", "official_score"):
            raw = row[field]
            value = float(raw) if raw != "" else None
            if value is not None and (
                not math.isfinite(value)
                or not 0 <= value <= (100 if field == "official_score" else 1)
            ):
                raise ValueError(f"Invalid {field} in {row['identity']}")
            row[field] = value
        low, high, reward = (row[k] for k in ("r_min", "r_max", "reward"))
        if (low is None) != (high is None) or low is not None and low > high:
            raise ValueError("Invalid reward interval")
        if reward is not None and low is not None and not low <= reward <= high:
            raise ValueError("Reward is outside its recorded interval")
    return rows


def group_scores(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["benchmark"], row["task"])].append(row)
    for records in groups.values():
        if len(records) != 3 or len({r["scenario"] for r in records}) != 1:
            raise ValueError(
                "Each fixed task must have exactly three candidates and one scenario"
            )
        records.sort(key=lambda r: r["identity"])
    return dict(groups)


def verify_versions(original, repaired):
    a, b = ({r["identity"]: r for r in rows} for rows in (original, repaired))
    if set(a) != set(b):
        raise ValueError("Versions must contain exactly the same candidate identities")
    for key, row in a.items():
        for field in (*IDENTITY_FIELDS, "official_score"):
            if row.get(field) != b[key].get(field):
                raise ValueError(f"Candidate binding mismatch: {key} / {field}")


def confirmed_runtime_tasks(audit, rows):
    """Only an already confirmed runtime failure qualifies, never disagreement."""
    groups = group_scores(rows)
    flags = {}
    for case in audit.get("cases", []):
        if (
            case.get("diagnosis") != "native_runtime_failure"
            or case.get("confidence") != "confirmed"
        ):
            continue
        key = case["benchmark"], case["task"]
        ids = {r["identity"] for r in groups.get(key, [])}
        cited = {r["identity"] for r in case["candidates"]}
        if not ids or not cited <= ids:
            raise ValueError(
                "Runtime-failure audit is not bound to the fixed candidates"
            )
        flags[key] = {
            "case": case["case"],
            "diagnosis": case["diagnosis"],
            "finding": case["finding_zh"],
            "candidate_ids": sorted(cited),
        }
    return flags


def select_tasks(groups, min_gap, runtime_flags):
    selected, audit = set(), []
    for key, rows in sorted(groups.items()):
        valid = all(
            r["official_score"] is not None and r["official_status"] == "valid"
            for r in rows
        )
        ordered = sorted(
            rows,
            key=lambda r: (
                r["official_score"] if r["official_score"] is not None else -1,
                r["identity"],
            ),
        )
        scores = [r["official_score"] for r in ordered]
        gaps = [scores[1] - scores[0], scores[2] - scores[1]] if valid else None
        minimum = min(gaps) if gaps else None
        keep = valid and (min_gap is None or minimum + GAP_EPSILON >= min_gap)
        if keep:
            selected.add(key)
        audit.append(
            {
                "benchmark": key[0],
                "task": key[1],
                "scenario": rows[0]["scenario"],
                "selected": keep,
                "reason": "selected"
                if keep
                else "invalid_official_score"
                if not valid
                else "adjacent_gap_below_threshold",
                "official_scores_ascending": scores,
                "candidate_ids_ascending": [r["identity"] for r in ordered],
                "adjacent_gaps": gaps,
                "min_adjacent_gap": minimum,
                "threshold": min_gap,
                "confirmed_runtime_failure": key in runtime_flags,
            }
        )
    return selected, audit


def pair_statistics(rows):
    pairs = []
    for a, b in combinations(sorted(rows, key=lambda r: r["identity"]), 2):
        official = a["official_score"] - b["official_score"]
        ra, rb = a["reward"], b["reward"]
        scalar = ra is not None and rb is not None
        if official == 0:
            status = "official_tie"
        elif not scalar:
            status = "unresolved"
        elif ra == rb:
            status = "reward_tie"
        else:
            status = "correct" if (ra - rb) * official > 0 else "reversed"
        interval = "not_applicable"
        if status == "unresolved":
            bounds = [
                (r["reward"], r["reward"])
                if r["reward"] is not None
                else (r["r_min"], r["r_max"])
                for r in (a, b)
            ]
            (al, ah), (bl, bh) = bounds
            if None in (al, ah, bl, bh):
                interval = "missing_bounds"
            elif al > bh or bl > ah:
                interval = "correct" if (al - bl) * official > 0 else "reversed"
            else:
                interval = (
                    "overlapping"  # Touching intervals do not establish strict order.
                )
        pairs.append(
            {
                "candidate_a": a["identity"],
                "candidate_b": b["identity"],
                "official_gap": abs(official),
                "status": status,
                "interval_status": interval,
            }
        )
    return pairs


def describe_task(rows):
    base = task_statistics(rows)
    complete = all(r["reward"] is not None for r in rows)
    if not complete:
        base["undefined_reason"] = "incomplete_reward"
    pairs = pair_statistics(rows)
    counts = Counter(p["status"] for p in pairs)
    intervals = Counter(p["interval_status"] for p in pairs)
    result = {
        "benchmark": rows[0]["benchmark"],
        "task": rows[0]["task"],
        "scenario": rows[0]["scenario"],
        "complete": complete,
        "complete_candidates": sum(r["reward"] is not None for r in rows),
        "tau_b": base["tau_b"],
        "spearman_rho": base["spearman_rho"],
        "undefined_reason": base["undefined_reason"],
        "all_pairs": len(pairs),
        **{
            key: counts[key]
            for key in (
                "correct",
                "reversed",
                "reward_tie",
                "unresolved",
                "official_tie",
            )
        },
        "interval_correct": intervals["correct"],
        "interval_reversed": intervals["reversed"],
        "interval_overlapping": intervals["overlapping"],
        "interval_missing": intervals["missing_bounds"],
        "top1_hit": None,
        "regret": None,
        "random_top1_hit": None,
        "random_regret": None,
    }
    if complete:
        best_reward = max(r["reward"] for r in rows)
        best_official = max(r["official_score"] for r in rows)
        winners = [r for r in rows if r["reward"] == best_reward]
        result.update(
            top1_hit=sum(r["official_score"] == best_official for r in winners)
            / len(winners),
            regret=best_official
            - sum(r["official_score"] for r in winners) / len(winners),
            random_top1_hit=sum(r["official_score"] == best_official for r in rows) / 3,
            random_regret=best_official - sum(r["official_score"] for r in rows) / 3,
        )
    return result, pairs


def metric_parts(row):
    known = row["correct"] + row["reversed"] + row["reward_tie"]
    strict = known + row["unresolved"]
    certain = row["interval_correct"] + row["interval_reversed"]
    parts = {
        key: (row[key] or 0, int(row[key] is not None))
        for key in (
            "tau_b",
            "spearman_rho",
            "top1_hit",
            "regret",
            "random_top1_hit",
            "random_regret",
        )
    }
    parts.update(
        strict_pair_accuracy=(row["correct"], known),
        non_tie_pair_accuracy=(row["correct"], row["correct"] + row["reversed"]),
        pair_coverage=(known, strict),
        complete_task_coverage=(int(row["complete"]), 1),
        reward_tie_rate_on_complete_pairs=(row["reward_tie"], known),
        interval_accuracy=(row["interval_correct"], certain),
        interval_coverage=(certain, row["unresolved"]),
    )
    parts.update(
        {
            key + "_fraction": (row[key], strict)
            for key in ("correct", "reversed", "reward_tie", "unresolved")
        }
    )
    return parts


METRICS = (
    "tau_b",
    "spearman_rho",
    "top1_hit",
    "regret",
    "random_top1_hit",
    "random_regret",
    "strict_pair_accuracy",
    "non_tie_pair_accuracy",
    "pair_coverage",
    "complete_task_coverage",
    "reward_tie_rate_on_complete_pairs",
    "interval_accuracy",
    "interval_coverage",
    "correct_fraction",
    "reversed_fraction",
    "reward_tie_fraction",
    "unresolved_fraction",
)


def cluster_arrays(task_rows, benchmark, keys=None):
    def cluster(row):
        return row["scenario"] if benchmark == "claw_eval" else row["task"]

    keys = sorted({cluster(r) for r in task_rows}) if keys is None else keys
    index = {key: i for i, key in enumerate(keys)}
    numerators = np.zeros((len(keys), len(METRICS)))
    denominators = np.zeros_like(numerators)
    for row in task_rows:
        parts = metric_parts(row)
        for j, name in enumerate(METRICS):
            num, den = parts[name]
            numerators[index[cluster(row)], j] += num
            denominators[index[cluster(row)], j] += den
    return keys, numerators, denominators


def ratio(num, den):
    return np.divide(
        num, den, out=np.full_like(num, np.nan, dtype=float), where=den != 0
    )


def resampled_estimates(nums, dens, *, iterations, seed):
    """Resample whole tasks/scenarios, retaining undefined metrics in the pool."""
    if not len(nums):
        return np.full((iterations, len(METRICS)), np.nan)
    rng = np.random.default_rng(seed)
    result = []
    for start in range(0, iterations, 1000):
        weights = rng.multinomial(
            len(nums),
            np.full(len(nums), 1 / len(nums)),
            size=min(1000, iterations - start),
        )
        result.append(ratio(weights @ nums, weights @ dens))
    return np.concatenate(result)


def interval(values):
    valid = values[np.isfinite(values)]
    return {
        "defined_replicates": len(valid),
        "ci95": np.quantile(valid, [0.025, 0.975]).tolist() if len(valid) else None,
    }


def summarize(task_rows, benchmark, *, iterations=ITERATIONS, seed=SEED):
    keys, nums, dens = cluster_arrays(task_rows, benchmark)
    points = ratio(nums.sum(axis=0), dens.sum(axis=0))
    samples = resampled_estimates(nums, dens, iterations=iterations, seed=seed)
    metrics = {}
    for j, name in enumerate(METRICS):
        metrics[name] = {
            "value": float(points[j]) if np.isfinite(points[j]) else None,
            "denominator": int(dens[:, j].sum()),
            "supporting_clusters": int(np.count_nonzero(dens[:, j])),
            **interval(samples[:, j]),
        }
    return {
        "tasks": len(task_rows),
        "clusters": len(keys),
        "complete_tasks": sum(r["complete"] for r in task_rows),
        "complete_candidates": sum(r["complete_candidates"] for r in task_rows),
        "correlations_defined": sum(r["tau_b"] is not None for r in task_rows),
        "undefined_reasons": dict(
            Counter(r["undefined_reason"] for r in task_rows if r["undefined_reason"])
        ),
        "pair_counts": {
            key: sum(r[key] for r in task_rows)
            for key in (
                "all_pairs",
                "correct",
                "reversed",
                "reward_tie",
                "unresolved",
                "official_tie",
                "interval_correct",
                "interval_reversed",
                "interval_overlapping",
                "interval_missing",
            )
        },
        "bootstrap": {
            "iterations": iterations,
            "seed": seed,
            "unit": "scenario" if benchmark == "claw_eval" else "task",
        },
        "metrics": metrics,
    }


def common_support(groups_a, groups_b, selected):
    """Paired metrics have identical candidate-pair/task support in both versions."""
    outputs = {v: {} for v in VERSIONS}
    for key in sorted(selected):
        source = {"original": groups_a[key], "repaired": groups_b[key]}
        available = set.intersection(
            *[
                {r["identity"] for r in rows if r["reward"] is not None}
                for rows in source.values()
            ]
        )
        for version, rows in source.items():
            copied = deepcopy(rows)
            for row in copied:
                if row["identity"] not in available:
                    row.update(reward=None, r_min=None, r_max=None)
            outputs[version][key] = describe_task(copied)[0]
            # This matched analysis is scalar-only. Interval ordering belongs
            # to each protocol's independent evidence/coverage analysis.
            for field in (
                "interval_correct",
                "interval_reversed",
                "interval_overlapping",
            ):
                outputs[version][key][field] = 0
            outputs[version][key]["interval_missing"] = outputs[version][key][
                "unresolved"
            ]
        if any(outputs[v][key]["tau_b"] is None for v in VERSIONS):
            for version in VERSIONS:
                row = outputs[version][key]
                if row["tau_b"] is not None:
                    row.update(
                        tau_b=None,
                        spearman_rho=None,
                        undefined_reason="outside_common_correlation_support",
                    )
    return outputs


def paired_delta(original, repaired, benchmark, *, iterations=ITERATIONS, seed=SEED):
    keys, an, ad = cluster_arrays(original, benchmark)
    other_keys, bn, bd = cluster_arrays(repaired, benchmark, keys)
    comparable = [
        j
        for j, name in enumerate(METRICS)
        if not name.startswith("interval_") and name != "non_tie_pair_accuracy"
    ]
    if keys != other_keys or not np.array_equal(ad[:, comparable], bd[:, comparable]):
        raise ValueError("Paired deltas require identical support for every metric")
    a = resampled_estimates(an, ad, iterations=iterations, seed=seed)
    b = resampled_estimates(bn, bd, iterations=iterations, seed=seed)
    point = ratio(bn.sum(axis=0), bd.sum(axis=0)) - ratio(
        an.sum(axis=0), ad.sum(axis=0)
    )
    return {
        name: {
            "repaired_minus_original": float(point[j])
            if np.isfinite(point[j])
            else None,
            "denominator": int(ad[:, j].sum()),
            **interval(b[:, j] - a[:, j]),
        }
        for j, name in enumerate(METRICS)
        if j in comparable
    }


def render_report(summary):
    def value(s, metric, percent=False):
        v = s["metrics"][metric]["value"]
        return "—" if v is None else f"{v * 100:.1f}%" if percent else f"{v:.3f}"

    def fmt(x):
        return "—" if x is None else f"[{x[0]:.3f}, {x[1]:.3f}]"

    lines = [
        "# 官方分差明确子集：固定三轨迹排序诊断",
        "",
        f"主分析为 {summary['main_label']}，所有预定阈值和完整任务集均保留。官方分数尺度为 0–100。",
        "仅重算既有分数：无模型评分、无 actor rollout；不是新 Reward 协议的效果验证。",
        "筛选不使用 reward；原始版与技术修复版分别报告。以下主表保留全部已入选任务，包括已知 oracle 异常。",
        "",
        "| 分差 | Benchmark | 版本 | 任务/完整/相关性有定义 | τ-b | ρ | 正确/逆序/平局/未知配对 | 严格配对准确率 | 配对覆盖率 |",
        "|---|---|---|---:|---:|---:|---|---:|---:|",
    ]
    for label, block in summary["analyses"].items():
        for benchmark, variants in block.items():
            for version in ("repaired", "original"):
                s = variants["all_selected"]["independent"][version]
                c = s["pair_counts"]
                counts = "/".join(
                    str(c[k])
                    for k in ("correct", "reversed", "reward_tie", "unresolved")
                )
                lines.append(
                    f"| {label} | {benchmark} | {version} | {s['tasks']}/{s['complete_tasks']}/{s['correlations_defined']} | "
                    f"{value(s, 'tau_b')} | {value(s, 'spearman_rho')} | {counts} | "
                    f"{value(s, 'strict_pair_accuracy', True)} | {value(s, 'pair_coverage', True)} |"
                )
    lines.extend(
        [
            "",
            "四类配对只统计官方严格有序的配对；全任务基线中的官方平局另列于 summary.json。",
            "准确率分母是 reward 双方完整的官方严格配对，reward 平局算未排对；unknown 保留在覆盖率和四类分解中，不填零。",
            "",
            "## 主阈值选优与区间排序",
            "",
            "reward 并列最高时在并列集合中均匀选择，表中为期望命中率/regret。选优仅在三条标量 reward 完整的任务上计算。",
            "",
            "| Benchmark | 版本 | 可选优任务 | Top-1 | 随机 Top-1 | Regret | 随机 Regret | 区间确定配对/未知配对 | 区间准确率 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    main = summary["analyses"][summary["main_label"]]
    for benchmark, variants in main.items():
        for version in ("repaired", "original"):
            s = variants["all_selected"]["independent"][version]
            c = s["pair_counts"]
            lines.append(
                f"| {benchmark} | {version} | {s['complete_tasks']} | {value(s, 'top1_hit', True)} | "
                f"{value(s, 'random_top1_hit', True)} | {value(s, 'regret')} | {value(s, 'random_regret')} | "
                f"{c['interval_correct'] + c['interval_reversed']}/{c['unresolved']} | {value(s, 'interval_accuracy', True)} |"
            )
    lines.extend(
        [
            "",
            "## 相关性区间与共同覆盖比较",
            "",
            "| Benchmark | 版本 | τ-b 95% CI | ρ 95% CI | τ 有效 bootstrap / 总次数 |",
            "|---|---|---|---|---:|",
        ]
    )
    for benchmark, variants in main.items():
        for version in ("repaired", "original"):
            s = variants["all_selected"]["independent"][version]
            tau, rho = (s["metrics"][k] for k in ("tau_b", "spearman_rho"))
            lines.append(
                f"| {benchmark} | {version} | {fmt(tau['ci95'])} | {fmt(rho['ci95'])} | "
                f"{tau['defined_replicates']}/{s['bootstrap']['iterations']} |"
            )
    lines.extend(
        [
            "",
            "| Benchmark | 共同相关性任务 | 原始 τ-b | 修复 τ-b | 配对 Δτ-b（95% CI） |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for benchmark, variants in main.items():
        common = variants["all_selected"]["common"]
        delta = common["paired_delta"]["tau_b"]
        delta_text = (
            "—"
            if delta["repaired_minus_original"] is None
            else f"{delta['repaired_minus_original']:.3f} {fmt(delta['ci95'])}"
        )
        lines.append(
            f"| {benchmark} | {common['original']['correlations_defined']} | {value(common['original'], 'tau_b')} | "
            f"{value(common['repaired'], 'tau_b')} | {delta_text} |"
        )
    lines.extend(
        [
            "",
            "共同覆盖比较分别匹配可用候选配对、三条完整任务及两版均有定义的相关性任务。",
            "若共同覆盖差值为零而各自完整覆盖的均值不同，差异来自可评估样本变化，不能解释为这些共同任务的排序改善。",
        ]
    )
    lines.extend(
        [
            "",
            "## 已确认 oracle 执行异常敏感性",
            "",
            "只排除审查中 diagnosis=native_runtime_failure 且 confidence=confirmed 的任务；不把其他分歧或零分当执行错误。",
            "",
            "| Benchmark | 剔除任务 | 版本 | 剩余任务/相关性有定义 | τ-b | 严格配对准确率 |",
            "|---|---:|---|---:|---:|---:|",
        ]
    )
    for benchmark, variants in main.items():
        for version in ("repaired", "original"):
            old = variants["all_selected"]["independent"][version]
            new = variants["without_confirmed_runtime_errors"]["independent"][version]
            lines.append(
                f"| {benchmark} | {old['tasks'] - new['tasks']} | {version} | {new['tasks']}/{new['correlations_defined']} | "
                f"{value(new, 'tau_b')} | {value(new, 'strict_pair_accuracy', True)} |"
            )
    lines.extend(["", "## 解释边界", ""])
    for benchmark, variants in main.items():
        for version in VERSIONS:
            n = variants["all_selected"]["independent"][version]["correlations_defined"]
            if n <= 2:
                lines.append(
                    f"- {benchmark}/{version} 主阈值只有 {n} 个任务相关性有定义；零个时不可估计，一个或两个时仅作个案描述。"
                )
    lines.extend(
        [
            "- 相关性只对有定义任务取均值；全同分与不完整任务仍保留在清单和覆盖统计中。",
            "- WorkBuddy 按 task、claw-eval 按 scenario 聚类 bootstrap，保留入选双语版本的依赖。每个指标记录有效重复次数和支持簇数。",
            "- 原始与修复版改变了部分评估配置；共同覆盖差值是配对描述，不证明某个修复机制的因果收益。",
            "- 分差阈值是操作性定义，不是已验证的 grader 噪声界限。历史候选按可用性及官方分数选取，结论不外推到随机候选或未见任务。",
            "- 完整 JSON 含各阈值、异常剔除、共同覆盖、区间排序、选优与全部置信区间；不得挑选相关性最高的阈值作为结论。",
            "",
        ]
    )
    lines.extend(
        [
            "## 复现",
            "",
            "从仓库根目录运行，输出目录必须是新目录：",
            "",
            "```sh",
            "python -m experiments.analysis --analysis score_gap --data-root data --output /tmp/reward-gap-analysis",
            "```",
            "",
            "protocol.json 保存阈值、输入及实现哈希；selection.csv 保存全部任务的入选/排除理由。",
            "",
        ]
    )
    return "\n".join(lines)


def run(
    original_dir,
    repaired_dir,
    output,
    *,
    min_gap=5.0,
    sensitivity=(2.0, 10.0),
    iterations=ITERATIONS,
    seed=SEED,
):
    thresholds = sorted(set([min_gap, *sensitivity]))
    if (
        any(not math.isfinite(t) or t <= GAP_EPSILON for t in thresholds)
        or iterations < 1
    ):
        raise ValueError(
            "Thresholds must be positive; bootstrap iterations must be nonzero"
        )
    if output.exists():
        raise FileExistsError("Output directory already exists; use a new directory")
    dirs = {"original": original_dir, "repaired": repaired_dir}
    paths = {
        str(root / name): sha(root / name)
        for root in dirs.values()
        for name in ("trajectory_scores.csv", "protocol.json", "manifest.public.json")
    }
    audit_path = repaired_dir / "case_audit.json"
    paths[str(audit_path)] = sha(audit_path)
    if any(v is None for v in paths.values()):
        raise ValueError(
            "Required frozen score/protocol/manifest/audit input is missing"
        )
    sources = {
        v: load_scores(root / "trajectory_scores.csv") for v, root in dirs.items()
    }
    verify_versions(sources["original"], sources["repaired"])
    groups = {v: group_scores(rows) for v, rows in sources.items()}
    for version, root in dirs.items():
        manifest = read(root / "manifest.public.json")
        if len(manifest) != len(sources[version]) or {
            r["identity"] for r in manifest
        } != {r["identity"] for r in sources[version]}:
            raise ValueError("Score rows do not match the frozen candidate manifest")
        lookup = {r["identity"]: r for r in manifest}
        for row in sources[version]:
            source = lookup[row["identity"]]
            if any(
                row[k] != source[k] for k in ("benchmark", "task", "scenario", "slot")
            ) or any(
                row[k + "_sha256"] != source["hashes"][k]
                for k in ("pack", "trace", "snapshot")
            ):
                raise ValueError("Score identity hashes differ from frozen manifest")
    flags = confirmed_runtime_tasks(read(audit_path), sources["original"])
    selections, audits = {}, []
    for threshold in [None, *thresholds]:
        label = "all" if threshold is None else f"gap_{threshold:g}"
        selections[label], rows = select_tasks(groups["original"], threshold, flags)
        audits.extend({"subset": label, **row} for row in rows)
    protocol = {
        "name": "fixed_triad_official_gap_subset_v1",
        "main_gap": min_gap,
        "sensitivity_gaps": [t for t in thresholds if t != min_gap],
        "official_scale": [0, 100],
        "selection": "min(sorted_official[1]-sorted_official[0], sorted_official[2]-sorted_official[1]) >= threshold",
        "gap_epsilon": GAP_EPSILON,
        "reward_ties": "exact stored scalar equality",
        "input_hashes": paths,
        "implementation_sha256": {
            str(Path(__file__)): sha(Path(__file__)),
            str(Path(__file__).with_name("analyze.py")): sha(
                Path(__file__).with_name("analyze.py")
            ),
        },
        "iterations": iterations,
        "seed": seed,
        "main_version": "repaired",
        "model_calls": 0,
        "prior_results_seen": True,
        "runtime_exclusions": [
            {"benchmark": k[0], "task": k[1], **v} for k, v in flags.items()
        ],
    }
    output.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        (output / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    save("protocol.json", protocol)
    save("selection.json", audits)
    write_csv(output / "selection.csv", audits)
    summary = {
        "protocol_sha256": canonical(protocol),
        "main_label": f"gap_{min_gap:g}",
        "analyses": {},
    }
    all_tasks, all_pairs, summary_rows = [], [], []
    for label, selected in selections.items():
        summary["analyses"][label] = {}
        common = common_support(groups["original"], groups["repaired"], selected)
        independent = {v: {} for v in VERSIONS}
        for version in VERSIONS:
            for key in sorted(selected):
                task, pairs = describe_task(groups[version][key])
                independent[version][key] = task
                annotation = {
                    "subset": label,
                    "version": version,
                    "confirmed_runtime_failure": key in flags,
                }
                all_tasks.append({**annotation, "support": "independent", **task})
                all_tasks.append(
                    {**annotation, "support": "common", **common[version][key]}
                )
                all_pairs.extend(
                    {**annotation, "benchmark": key[0], "task": key[1], **pair}
                    for pair in pairs
                )
        for benchmark in ("workbuddy", "claw_eval"):
            block = summary["analyses"][label][benchmark] = {}
            for variant, exclude in (
                ("all_selected", False),
                ("without_confirmed_runtime_errors", True),
            ):
                keys = sorted(
                    k
                    for k in selected
                    if k[0] == benchmark and (not exclude or k not in flags)
                )
                block[variant] = {}
                for support, data in (("independent", independent), ("common", common)):
                    block[variant][support] = {}
                    for version in VERSIONS:
                        report = summarize(
                            [data[version][k] for k in keys],
                            benchmark,
                            iterations=iterations,
                            seed=seed,
                        )
                        block[variant][support][version] = report
                        summary_rows.append(
                            {
                                "subset": label,
                                "benchmark": benchmark,
                                "variant": variant,
                                "support": support,
                                "version": version,
                                "tasks": report["tasks"],
                                "complete_tasks": report["complete_tasks"],
                                "correlations_defined": report["correlations_defined"],
                                **{
                                    name: m["value"]
                                    for name, m in report["metrics"].items()
                                },
                            }
                        )
                block[variant]["common"]["paired_delta"] = paired_delta(
                    [common["original"][k] for k in keys],
                    [common["repaired"][k] for k in keys],
                    benchmark,
                    iterations=iterations,
                    seed=seed,
                )
    if any(sha(Path(p)) != expected for p, expected in paths.items()):
        raise RuntimeError(
            "Frozen inputs changed during analysis; output is not publishable"
        )
    write_csv(output / "task_statistics.csv", all_tasks)
    write_csv(output / "pair_statistics.csv", all_pairs)
    write_csv(output / "summary.csv", summary_rows)
    save("summary.json", summary)
    (output / "RESULTS.md").write_text(render_report(summary), encoding="utf-8")
    save(
        "status.json",
        {"status": "complete", "input_hashes_verified": True, "model_calls": 0},
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=OUTPUT / "joint_original")
    parser.add_argument("--repaired", type=Path, default=OUTPUT / "joint_repaired")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-gap", type=float, default=5)
    parser.add_argument("--sensitivity-gaps", type=float, nargs="*", default=[2, 10])
    parser.add_argument("--bootstrap-iterations", type=int, default=ITERATIONS)
    args = parser.parse_args()
    run(
        args.original,
        args.repaired,
        args.output,
        min_gap=args.min_gap,
        sensitivity=args.sensitivity_gaps,
        iterations=args.bootstrap_iterations,
    )
    print(args.output / "RESULTS.md")


if __name__ == "__main__":
    main()
