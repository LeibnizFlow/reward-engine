"""Recompute historical ranking metrics with criterion unknowns worth 0.5."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from experiments.analysis.rewardpack import analyze as historical_analysis
from experiments.analysis.rewardpack import subset_analysis as subset
from experiments.analysis.rewardpack.analyze import write_csv
from experiments.analysis.rewardpack.inputs import (
    OUTPUT,
    REPO,
    canonical,
    read,
    sha,
)

VERSIONS = ("original", "repaired")
POLICIES = ("original", "midpoint")
THRESHOLDS = (("all", None), ("gap_2", 2), ("gap_5", 5), ("gap_10", 10))


def track(path, hashes):
    path = path.resolve()
    digest = sha(path)
    if digest is None:
        raise ValueError(f"Required input is missing: {path}")
    if str(path) in hashes and hashes[str(path)] != digest:
        raise ValueError(f"Input changed during analysis: {path}")
    hashes[str(path)] = digest
    return digest


def score_counts(score, criterion_ids):
    """Validate complete criterion judgments; missing receipts are not unknowns."""
    criteria = score.get("criteria", [])
    observed = [c.get("criterion_id") for c in criteria]
    if (
        not criterion_ids
        or len(set(criterion_ids)) != len(criterion_ids)
        or len(observed) != len(criterion_ids)
        or set(observed) != set(criterion_ids)
    ):
        raise ValueError("Duplicate, missing, or unexpected criterion")
    counts = Counter(c.get("status") for c in criteria)
    if set(counts) - {"pass", "fail", "unknown"}:
        raise ValueError("Invalid criterion status")
    k, p, u = len(criteria), counts["pass"], counts["unknown"]
    if any(
        type(score.get(key)) is not int or score[key] != value
        for key, value in (("criterion_count", k), ("passed", p), ("unknown", u))
    ):
        raise ValueError("P/U/K disagree with criterion judgments")
    expected = {
        "r_min": p / k,
        "r_max": (p + u) / k,
        "reward": p / k if u == 0 else None,
    }
    for key, value in expected.items():
        actual = score.get(key)
        if value is None:
            valid = actual is None
        else:
            valid = (
                type(actual) in (int, float)
                and math.isfinite(actual)
                and math.isclose(actual, value, rel_tol=0, abs_tol=1e-12)
            )
        if not valid:
            raise ValueError(f"Stored {key} disagrees with P/U/K")
    mid = score["reward"] if u == 0 else (p + 0.5 * u) / k
    if not math.isclose(
        mid, (score["r_min"] + score["r_max"]) / 2, rel_tol=0, abs_tol=1e-12
    ):
        raise ValueError("Midpoint disagrees with uncertainty bounds")
    technical = sum(
        c["status"] == "unknown"
        and bool(
            c.get("protocol_error")
            or c.get("error")
            or c.get("unknown_kind") == "technical"
        )
        for c in criteria
    )
    return {
        "passed_criteria": p,
        "failed_criteria": counts["fail"],
        "unknown_criteria": u,
        "criterion_count": k,
        "unknown_fraction": u / k,
        "technical_unknown_criteria": technical,
        "original_reward": score["reward"],
        "reward_midpoint": mid,
    }


def load_version(root, hashes):
    names = (
        "trajectory_scores.csv",
        "manifest.public.json",
        "protocol.json",
        "replay_config.json",
    )
    for name in names:
        track(root / name, hashes)
    rows = subset.load_scores(root / "trajectory_scores.csv")
    manifests = read(root / "manifest.public.json")
    manifest = {r["identity"]: r for r in manifests}
    if len(manifest) != len(manifests) or set(manifest) != {
        r["identity"] for r in rows
    }:
        raise ValueError("Manifest/score identities differ or repeat")
    protocol_hash = canonical(read(root / "protocol.json"))
    config_hash = canonical(read(root / "replay_config.json"))
    output = []
    for row in rows:
        item = manifest[row["identity"]]
        for key in ("benchmark", "task", "scenario", "slot", "pack_id"):
            if row[key] != item[key]:
                raise ValueError(f"Manifest binding mismatch: {key}")
        if any(
            row[key + "_sha256"] != item["hashes"][key]
            for key in ("pack", "trace", "snapshot")
        ):
            raise ValueError("Score/manifest input hashes differ")
        path = root / "evaluations" / row["identity"] / "receipt.json"
        if track(path, hashes) != row["reward_receipt_sha256"]:
            raise ValueError("Receipt hash mismatch")
        receipt = read(path)
        if (
            any(
                receipt.get(k) != row[k]
                for k in (
                    "identity",
                    "benchmark",
                    "task",
                    "slot",
                    "pack_id",
                    "pack_sha256",
                )
            )
            or receipt.get("input_hashes") != item["hashes"]
            or receipt.get("protocol_sha256") != protocol_hash
            or receipt.get("config_sha256") != config_hash
            or receipt.get("evaluation_method") != "joint_criterion"
            or receipt.get("harness_executed") is not False
        ):
            raise ValueError(
                "Receipt identity/protocol binding mismatch or non-joint receipt"
            )
        pack_path = REPO / item["paths"]["pack"]
        if track(pack_path, hashes) != row["pack_sha256"]:
            raise ValueError("Pack hash mismatch")
        pack = read(pack_path)
        if pack["pack_id"] != row["pack_id"] or pack["task_id"] != row["task"]:
            raise ValueError("Pack belongs to another task")
        counts = score_counts(receipt["score"], [c["id"] for c in pack["criteria"]])
        if any(row[k] != receipt["score"][k] for k in ("reward", "r_min", "r_max")):
            raise ValueError("CSV score differs from receipt")
        if int(row["unknown_criteria"]) != counts["unknown_criteria"]:
            raise ValueError("CSV unknown count differs from receipt")
        output.append({**row, **counts, "receipt_path": str(path.resolve())})
    subset.group_scores(output)
    return output


def classify(row):
    if not row["complete"]:
        return "incomplete"
    if row["official_tie"] == 3:
        return "official_constant"
    if row["reversed"]:
        return "has_inversion"
    if row["correct"] == 3:
        return "strict_correct"
    return "coarse_correct" if row["correct"] else "no_strict_separation"


def summarize(tasks, benchmark, iterations, seed):
    result = subset.summarize(tasks, benchmark, iterations=iterations, seed=seed)
    result["categories"] = dict(Counter(classify(t) for t in tasks))
    eligible = [t for t in tasks if t["complete"] and t["official_tie"] == 0]
    successes = sum(t["correct"] == 3 for t in eligible)
    result["strict_order_accuracy"] = {
        "correct": successes,
        "eligible_tasks": len(eligible),
        "value": successes / len(eligible) if eligible else None,
    }
    return result


def check_reference(actual, expected):
    for key in (
        "tasks",
        "clusters",
        "complete_tasks",
        "complete_candidates",
        "correlations_defined",
        "undefined_reasons",
        "pair_counts",
    ):
        if actual[key] != expected[key]:
            raise ValueError(f"Original statistics differ from reference: {key}")
    for key, metric in expected["metrics"].items():
        current = actual["metrics"][key]
        for field in ("value", "denominator", "supporting_clusters"):
            a, b = current[field], metric[field]
            if a != b and (
                a is None
                or b is None
                or not math.isclose(a, b, rel_tol=0, abs_tol=1e-12)
            ):
                raise ValueError(
                    f"Original metric differs from reference: {key}/{field}"
                )


def run(
    original_dir,
    repaired_dir,
    output,
    *,
    reference_summary=None,
    iterations=subset.ITERATIONS,
    seed=subset.SEED,
):
    if output.exists():
        raise FileExistsError("Use a new output directory")
    if iterations < 1:
        raise ValueError("Bootstrap iterations must be positive")
    hashes = {}
    sources = {
        "original": load_version(original_dir, hashes),
        "repaired": load_version(repaired_dir, hashes),
    }
    subset.verify_versions(sources["original"], sources["repaired"])
    audit_path = repaired_dir / "case_audit.json"
    track(audit_path, hashes)
    flags = subset.confirmed_runtime_tasks(read(audit_path), sources["original"])
    grouped = {
        version: {
            policy: subset.group_scores(
                [
                    {
                        **r,
                        "reward": r["reward_midpoint"]
                        if policy == "midpoint"
                        else r["original_reward"],
                    }
                    for r in rows
                ]
            )
            for policy in POLICIES
        }
        for version, rows in sources.items()
    }
    reference, reference_selection = None, None
    if reference_summary is not None:
        track(reference_summary, hashes)
        selection_path = reference_summary.with_name("selection.json")
        track(selection_path, hashes)
        reference, reference_selection = read(reference_summary), read(selection_path)
    summary = {
        "main_version": "repaired",
        "main_subsets": ["all", "gap_5"],
        "analyses": {},
    }
    task_output, pair_output, selections, flat_summary = [], [], [], []
    for label, threshold in THRESHOLDS:
        selected, audit = subset.select_tasks(
            grouped["original"]["original"], threshold, flags
        )
        selections.extend({"subset": label, **r} for r in audit)
        if reference_selection is not None:
            expected = {
                (r["benchmark"], r["task"])
                for r in reference_selection
                if r["subset"] == label and r["selected"]
            }
            if expected != selected:
                raise ValueError("Task selection differs from existing report")
        descriptions = {
            version: {policy: {} for policy in POLICIES} for version in VERSIONS
        }
        for version in VERSIONS:
            for policy in POLICIES:
                for key in sorted(selected):
                    task, pairs = subset.describe_task(grouped[version][policy][key])
                    descriptions[version][policy][key] = task
                    metadata = {
                        "subset": label,
                        "grading_version": version,
                        "scoring_policy": policy,
                        "confirmed_runtime_failure": key in flags,
                    }
                    task_output.append(
                        {
                            **metadata,
                            **task,
                            "category": classify(task),
                            "new_scalar_task": policy == "midpoint"
                            and not descriptions[version]["original"][key]["complete"],
                            "new_correlation_task": policy == "midpoint"
                            and task["tau_b"] is not None
                            and descriptions[version]["original"][key]["tau_b"] is None,
                        }
                    )
                    pair_output.extend(
                        {**metadata, "benchmark": key[0], "task": key[1], **pair}
                        for pair in pairs
                    )
        summary["analyses"][label] = {}
        for benchmark in ("workbuddy", "claw_eval"):
            bench = summary["analyses"][label][benchmark] = {}
            for variant in ("all_selected", "without_confirmed_runtime_errors"):
                keys = {
                    k
                    for k in selected
                    if k[0] == benchmark
                    and (variant == "all_selected" or k not in flags)
                }
                block = bench[variant] = {}
                for version in VERSIONS:
                    data = descriptions[version]
                    entry = block[version] = {"independent": {}}
                    for policy in POLICIES:
                        stats = summarize(
                            [data[policy][k] for k in sorted(keys)],
                            benchmark,
                            iterations,
                            seed,
                        )
                        entry["independent"][policy] = stats
                        if reference is not None and policy == "original":
                            check_reference(
                                stats,
                                reference["analyses"][label][benchmark][variant][
                                    "independent"
                                ][version],
                            )
                        flat_summary.append(
                            {
                                "subset": label,
                                "benchmark": benchmark,
                                "variant": variant,
                                "grading_version": version,
                                "scoring_policy": policy,
                                "tasks": stats["tasks"],
                                "complete_tasks": stats["complete_tasks"],
                                "correlations_defined": stats["correlations_defined"],
                                **{k: v["value"] for k, v in stats["metrics"].items()},
                            }
                        )
                    common = subset.common_support(
                        grouped[version]["original"], grouped[version]["midpoint"], keys
                    )
                    a, b = ([common[v][k] for k in sorted(keys)] for v in VERSIONS)
                    delta = subset.paired_delta(
                        a, b, benchmark, iterations=iterations, seed=seed
                    )
                    for metric in delta.values():
                        metric["midpoint_minus_original"] = metric.pop(
                            "repaired_minus_original"
                        )
                        if metric["midpoint_minus_original"] not in (None, 0.0):
                            raise ValueError(
                                "Midpoint changed an already known score on common support"
                            )
                    entry["common"] = {
                        "original": summarize(a, benchmark, iterations, seed),
                        "midpoint": summarize(b, benchmark, iterations, seed),
                        "paired_delta": delta,
                    }
                    entry["new_scalar_tasks"] = summarize(
                        [
                            data["midpoint"][k]
                            for k in sorted(keys)
                            if not data["original"][k]["complete"]
                        ],
                        benchmark,
                        iterations,
                        seed,
                    )
                    entry["new_correlation_tasks"] = summarize(
                        [
                            data["midpoint"][k]
                            for k in sorted(keys)
                            if data["original"][k]["tau_b"] is None
                            and data["midpoint"][k]["tau_b"] is not None
                        ],
                        benchmark,
                        iterations,
                        seed,
                    )
    source_paths = [
        Path(__file__),
        Path(subset.__file__),
        Path(historical_analysis.__file__),
    ]
    protocol = {
        "name": "historical_joint_midpoint_v1",
        "rule": "(P + 0.5 * U) / K; criterion-equal",
        "unknown_policy": "All recorded unknown criteria, including technical errors; missing receipts rejected",
        "input_hashes": hashes,
        "implementation_hashes": {str(p.resolve()): sha(p) for p in source_paths},
        "iterations": iterations,
        "seed": seed,
        "thresholds": dict(THRESHOLDS),
        "prior_results_seen": True,
        "post_hoc_imputation": True,
        "model_calls": 0,
        "reference_statistics_verified": reference is not None,
        "runtime_exclusions": [
            {"benchmark": k[0], "task": k[1], **v} for k, v in flags.items()
        ],
    }
    for path, digest in hashes.items():
        if sha(Path(path)) != digest:
            raise ValueError(f"Input changed before output: {path}")
    output.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        (output / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )

    summary["protocol_sha256"] = canonical(protocol)
    save("protocol.json", protocol)
    save("selection.json", selections)
    save("summary.json", summary)
    write_csv(output / "selection.csv", selections)
    # Stored rank columns belong to the old scalar and must not be mislabeled as midpoint ranks.
    write_csv(
        output / "trajectory_scores.csv",
        [
            {
                "grading_version": version,
                **{
                    k: v
                    for k, v in r.items()
                    if k not in ("reward_rank", "official_rank")
                },
            }
            for version, rows in sources.items()
            for r in rows
        ],
    )
    write_csv(output / "task_statistics.csv", task_output)
    write_csv(output / "pair_statistics.csv", pair_output)
    write_csv(output / "summary.csv", flat_summary)
    (output / "RESULTS.md").write_text(render(summary), encoding="utf-8")
    save(
        "status.json",
        {
            "status": "complete",
            "model_calls": 0,
            "input_hashes_verified": True,
            "reference_statistics_verified": reference is not None,
            "receipt_count": sum(len(r) for r in sources.values()),
            "common_deltas_zero": True,
        },
    )
    return summary


def render(summary):
    def value(stats, key):
        v = stats["metrics"][key]["value"]
        return "—" if v is None else f"{v:.3f}"

    lines = [
        "# 历史 joint reward：unknown=0.5 重计算",
        "",
        "本报告重算已完成的原始版与修复版历史 joint 判分，不是正在运行的真实 Harness 结果。",
        "每条轨迹的 midpoint=(pass+0.5×unknown)/criterion_count；所有 unknown 包括技术错误均参与。",
        "这是事后赋值假设，不是新增测量证据。原分数、区间、unknown 比例保留，未调用模型。",
        "",
        "## 修复版主结果",
        "",
        "| 子集 | Benchmark | 算法 | 总题数/完整标量/相关性有定义 | τ-b | ρ | Top-1 | Regret | 配对准确率 | 配对覆盖率 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in ("all", "gap_5", "gap_2", "gap_10"):
        for benchmark, variants in summary["analyses"][label].items():
            entry = variants["all_selected"]["repaired"]
            for policy, stats in entry["independent"].items():
                values = " | ".join(
                    value(stats, k)
                    for k in (
                        "tau_b",
                        "spearman_rho",
                        "top1_hit",
                        "regret",
                        "strict_pair_accuracy",
                        "pair_coverage",
                    )
                )
                lines.append(
                    f"| {label} | {benchmark} | {policy} | "
                    f"{stats['tasks']}/{stats['complete_tasks']}/{stats['correlations_defined']} | {values} |"
                )
    lines += [
        "",
        "相关性为逐题系数的宏平均，排除不可定义项但报告其数量；没有把所有轨迹混合计算。",
        "Top-1 在三条标量均完整的任务上计算，并列最高分按均匀选择；官方同分的随机基线见 JSON。",
        "严格配对准确率只针对官方严格有序配对，reward 同分算未排对；分母为标量均可用的配对。",
        "",
        "## 新增相关性样本与区间",
        "",
        "| 子集 | Benchmark | 原相关性任务 | 新增相关性任务 | 新增任务 τ-b | 新增任务 ρ | midpoint τ 95% CI | midpoint ρ 95% CI |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for label in ("all", "gap_5", "gap_2", "gap_10"):
        for benchmark, variants in summary["analyses"][label].items():
            entry = variants["all_selected"]["repaired"]
            a, b, new = (
                entry["independent"]["original"],
                entry["independent"]["midpoint"],
                entry["new_correlation_tasks"],
            )
            lines.append(
                f"| {label} | {benchmark} | {a['correlations_defined']} | {new['tasks']} | "
                f"{value(new, 'tau_b')} | {value(new, 'spearman_rho')} | "
                f"{b['metrics']['tau_b']['ci95']} | {b['metrics']['spearman_rho']['ci95']} |"
            )
    lines += [
        "",
        "原来有确定分数的候选没有 unknown，midpoint 与原分数逐条相同。因此同一历史版本在共同支持上的",
        "τ、ρ、Top-1、regret 和严格配对准确率差值全部为零（无法定义则保持空值）。总体均值变化来自新增样本，不能解释为共同样本排序改善。",
        "新增可计算标量任务与新增相关性任务分开统计：全部 unknown 会成为 0.5，同分任务依然没有相关性。",
        "",
        "## 两个历史版本与官方运行异常敏感性",
        "",
        "| 子集 | Benchmark | 异常处理 | 判分版本 | 算法 | 任务/相关性有定义 | τ-b | ρ |",
        "|---|---|---|---|---|---:|---:|---:|",
    ]
    for label, benchmarks in summary["analyses"].items():
        for benchmark, variants in benchmarks.items():
            for variant, versions in variants.items():
                for version, entry in versions.items():
                    for policy, stats in entry["independent"].items():
                        lines.append(
                            f"| {label} | {benchmark} | {variant} | {version} | {policy} | "
                            f"{stats['tasks']}/{stats['correlations_defined']} | {value(stats, 'tau_b')} | {value(stats, 'spearman_rho')} |"
                        )
    lines += [
        "",
        "主结果保留全部选定任务；敏感性只排除既有审查已确认的官方运行错误，不因评分分歧临时排除任务。",
        "所有标准指标使用固定种子、20,000 次聚类 bootstrap（默认）；WorkBuddy 按 task，claw-eval 按 scenario。",
        "完整区间、分母、有效 bootstrap 次数、同分率、完整排序成功率和逐题分类保存在 summary.json 与 CSV。",
        "输入凭据、Pack、manifest、原始评分及既有分析报告的哈希都在 protocol.json；原评分统计已与既有报告校验（若配置参考报告）。",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=OUTPUT / "joint_original")
    parser.add_argument("--repaired", type=Path, default=OUTPUT / "joint_repaired")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reference-summary", type=Path, default=OUTPUT / "score_gap/summary.json"
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=subset.ITERATIONS)
    args = parser.parse_args()
    run(
        args.original,
        args.repaired,
        args.output,
        reference_summary=args.reference_summary,
        iterations=args.bootstrap_iterations,
    )
    print(args.output / "RESULTS.md")


if __name__ == "__main__":
    main()
