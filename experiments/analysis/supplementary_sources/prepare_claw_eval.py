"""Freeze/rebuild the historical round-one claw-eval snapshot, not later rounds.

Use prepare_claw_continuation.py for the current three-round endpoint.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/claw_eval.json"
PROVENANCE = ROOT / "data/claw_eval_provenance.json"
ITERATIONS, SEED = 200000, 20260920


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(source):
    sys.path.insert(0, str(source))
    from experiments.claw_eval.opencode import build_seed_baseline_draft as collector
    report = source / "reports/claw_eval/office104_amean_draft_20260924"
    # The first-round report was archived when the continuation was published.
    # Resolve only byte-identical historical inputs, never the latest endpoint.
    archived = source / "reports/claw_eval/claw-eval-report-round1-20260927"
    old_sources = read(PROVENANCE)['source_files']
    if not (report / 'results.json').is_file():
        expected = old_sources.get(str((report / 'results.json').relative_to(source)),
                                   old_sources.get(str((archived / 'results.json').relative_to(source))))
        assert digest(archived / 'results.json') == expected
        report = archived
    if not (collector.SOURCE_REPORT / 'results.json').is_file():
        relocated = archived / 'historical_run'
        for name in ('results.json', 'source_map.json', 'infra_audit.json'):
            expected = old_sources.get(str((collector.SOURCE_REPORT / name).relative_to(source)),
                                       old_sources.get(str((relocated / name).relative_to(source))))
            assert digest(relocated / name) == expected
        collector.SOURCE_REPORT = relocated
    document = read(report / "results.json")
    checked, original_metrics = collector.collect(report, collector.SCORES)
    assert len(checked) == len(document["tasks"]) == 104
    rows, evidence = [], []
    for old, row in zip(checked, document["tasks"], strict=True):
        assert old["task"] == row["task"]
        for field in ("A_run", "B", "C", "selected_arm", "selected_score"):
            assert old[field] == row[field]
        scores = [c["score"] for c in old["baseline_candidates"]]
        assert np.isclose(row["A_mean"], sum(scores) / 4, atol=1e-8, rtol=0)
        exported = {k: row[k] for k in (
            "task", "category", "language", "difficulty", "A_mean",
            "A_run", "B", "C", "selected_arm", "selected_score"
        )}
        exported["baseline_scores"] = dict(zip(("A_run", "seed_1", "seed_2", "seed_3"), scores))
        rows.append(exported)
        from urllib.parse import unquote
        task_root = collector.CAMPAIGN / "tasks" / row["task"]
        task_evidence = {
            "task": row["task"],
            "task_yaml_sha256": digest(collector.CAMPAIGN / "benchmark/tasks" / row["task"] / "task.yaml"),
            "candidates": [],
        }
        for index, candidate in enumerate(row["baseline_candidates"]):
            receipt_path = (report / unquote(candidate["receipt"])).resolve()
            receipt = read(receipt_path)
            assert receipt["status"] == "graded"
            assert np.isclose(receipt["task_score"] * 100, scores[index], atol=1e-8, rtol=0)
            assert receipt["protocol"] == "native-claw-eval-custom-judge"
            assert receipt["judge_model"] == "deepseek-v4-flash"
            slot = task_root / ("round_1/actors/A" if index == 0 else f"seeds/{index}")
            task_evidence["candidates"].append({
                "id": candidate["id"], "score": candidate["score"],
                "receipt_sha256": digest(receipt_path),
                "trace_sha256": digest(slot / "trace.jsonl"),
                "snapshot_sha256": digest(slot / "environment-snapshot.json"),
            })
        for arm in "BC":
            slot = task_root / f"round_1/actors/{arm}"
            receipt_path = slot / "OFFICIAL.private.json"
            receipt = read(receipt_path)
            assert receipt["status"] == "graded"
            assert np.isclose(receipt["task_score"] * 100, row[arm], atol=1e-8, rtol=0)
            for field, filename in (("trace", "trace.jsonl"), ("snapshot", "environment-snapshot.json")):
                assert receipt["binding"][field] == digest(slot / filename)
            task_evidence["candidates"].append({
                "id": arm, "score": row[arm], "receipt_sha256": digest(receipt_path),
                "trace_sha256": digest(slot / "trace.jsonl"),
                "snapshot_sha256": digest(slot / "environment-snapshot.json"),
            })
        evidence.append(task_evidence)
    assert document["metrics"]["original_run_metrics"] == original_metrics
    metrics = {k: v for k, v in document['metrics'].items() if k != 'groups'}
    metrics['original_run_metrics'] = {k: v for k, v in metrics['original_run_metrics'].items() if k != 'groups'}
    write(DATA, {
        "benchmark": "claw-eval", "n_tasks": 104,
        "baseline_formula": "(A_run + seed_1 + seed_2 + seed_3) / 4",
        "tasks": rows, "reference_metrics": metrics,
    })
    paths = [
        report / "results.json",
        collector.SOURCE_REPORT / "results.json",
        collector.SOURCE_REPORT / "source_map.json",
        collector.SOURCE_REPORT / "infra_audit.json",
        collector.CAMPAIGN / "manifest.json",
        collector.CAMPAIGN / "benchmark/README.md",
        source / "experiments/claw_eval/opencode/build_seed_mean_draft.py",
    ]
    write(PROVENANCE, {
        "schema": "claw_eval_manuscript_provenance.v1",
        "snapshot_sha256": digest(DATA),
        "source_files": {str(p.relative_to(source)): digest(p) for p in paths},
        "native_protocol": "native-claw-eval-custom-judge",
        "judge_model": "deepseek-v4-flash",
        "bootstrap": {"iterations": ITERATIONS, "seed": SEED},
        "evidence": evidence,
    })


def paired(rows, arm, baseline="A_mean"):
    delta = np.array([r[arm] - r[baseline] for r in rows])
    rng = np.random.default_rng(SEED)
    samples = np.concatenate([
        delta[rng.integers(0, len(rows), size=(10000, len(rows)))].mean(axis=1)
        for _ in range(ITERATIONS // 10000)
    ])
    return {
        "mean": float(np.mean([r[arm] for r in rows])),
        "delta": float(delta.mean()), "median": float(np.median(delta)),
        "ci95": np.quantile(samples, [.025, .975]).tolist(),
        "wins": int((delta > 1e-8).sum()), "ties": int((abs(delta) <= 1e-8).sum()),
        "losses": int((delta < -1e-8).sum()),
    }


def assert_pair(computed, reference, baseline):
    for key, ref in (("mean", "mean"), ("delta", f"mean_delta_vs_{baseline}"),
                     ("median", "median_delta"), ("ci95", "paired_task_bootstrap_ci95"),
                     ("wins", "wins"), ("ties", "ties"), ("losses", "losses")):
        assert np.allclose(computed[key], reference[ref], atol=1e-8, rtol=0), (key, computed, reference)


def table(name, header, rows, spec):
    lines = [r"\begin{tabular}{" + spec + "}", r"\toprule",
             header + r" \\", r"\midrule", *[r + r" \\" for r in rows],
             r"\bottomrule", r"\end{tabular}"]
    (ROOT / f"tables/{name}.tex").write_text("\n".join(lines) + "\n")


def comparison_row(label, item):
    lo, hi = item["ci95"]
    return f"{label} & {item['mean']:.2f} & {item['delta']:+.2f} & [{lo:.2f}, {hi:.2f}] & {item['wins']}/{item['ties']}/{item['losses']}"


def build():
    data, provenance = read(DATA), read(PROVENANCE)
    assert provenance["snapshot_sha256"] == digest(DATA)
    rows, reference = data["tasks"], data["reference_metrics"]
    assert len(rows) == len({r["task"] for r in rows}) == 104
    assert len(provenance["evidence"]) == 104
    assert sum(len(r["candidates"]) for r in provenance["evidence"]) == 624
    for row in rows:
        assert len(row["baseline_scores"]) == 4
        assert np.isclose(row["A_mean"], sum(row["baseline_scores"].values()) / 4, atol=1e-8, rtol=0)
        assert row["selected_score"] == row["A_run" if row["selected_arm"] == "A" else row["selected_arm"]]
    result = {
        "means": {a: float(np.mean([r[a] for r in rows])) for a in ("A_mean", "B", "C")},
        "vs_A_mean": {a: paired(rows, a) for a in ("B", "C", "selected_score")},
        "bootstrap": {"iterations": ITERATIONS, "seed": SEED},
    }
    for arm in result["means"]:
        assert np.isclose(result["means"][arm], reference["means"][arm], atol=1e-8, rtol=0)
    for arm in "BC":
        assert_pair(result["vs_A_mean"][arm], reference["comparisons_vs_A_mean"][arm], "A_mean")
    original = reference["original_run_metrics"]
    selected = np.array([r["selected_score"] for r in rows])
    best = np.array([max(r[a] for a in ("A_run", "B", "C")) for r in rows])
    result["selection"] = {"mean": float(selected.mean()), "best_mean": float(best.mean()),
                           "best_hits": int((abs(best - selected) <= 1e-8).sum()),
                           "mean_gap": float((best - selected).mean())}
    for key, ref in (("mean", "selected_mean"), ("best_mean", "best_available_mean"),
                     ("best_hits", "selection_hits"), ("mean_gap", "selection_mean_regret")):
        assert np.isclose(result["selection"][key], original[ref], atol=1e-8, rtol=0)
    write(ROOT / "data/claw_eval_statistics.json", result)
    table("claw_eval_main", r"Method & Score & $\Delta$ vs.\ task-only mean & 95\% interval & W/T/L", [
        f"Task-only mean & {result['means']['A_mean']:.2f} & --- & --- & ---",
        comparison_row("Reward-guided revision", result["vs_A_mean"]["B"]),
        comparison_row("Hypothesis guidance", result["vs_A_mean"]["C"]),
        comparison_row("Selected by reward", result["vs_A_mean"]["selected_score"]),
    ], "lrrrr")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(6.35, 2.35), gridspec_kw={"width_ratios": [1.4, 1]})
    order = np.argsort([r["B"] - r["A_mean"] for r in rows], kind="stable")
    for i, (arm, marker, color, label) in enumerate((("B", "o", "#256aa3", "Reward-guided revision"), ("C", "x", "#c57b27", "Hypothesis guidance"))):
        delta = np.array([r[arm] - r["A_mean"] for r in rows])
        axes[0].scatter(np.arange(1, 105), delta[order], marker=marker, s=11, label=label, color=color, alpha=.8)
        st = result["vs_A_mean"][arm]
        axes[1].errorbar(st["delta"], 1-i, xerr=[[st["delta"]-st["ci95"][0]], [st["ci95"][1]-st["delta"]]],
                         fmt=marker, color=color, capsize=3)
    axes[0].axhline(0, color=".5", linewidth=.7)
    axes[0].set(xlabel="Task index (sorted by reward-guided difference)", ylabel="Score difference vs. task-only mean", ylim=(-105, 105))
    axes[0].legend(frameon=False)
    axes[1].axvline(0, color=".5", linewidth=.7)
    axes[1].set(yticks=[1, 0], yticklabels=["Reward-guided revision", "Hypothesis guidance"], xlabel="Mean difference and 95% interval", ylim=(-.6, 1.6))
    fig.tight_layout()
    fig.savefig(ROOT / "figures/claw_eval_differences.pdf")
    fig.savefig(ROOT / "figures/claw_eval_differences.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"status": "passed", "tasks": 104, "baseline_scores": 416,
                      "receipts": 624, "means": result["means"], "selection": result["selection"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    if args.source:
        freeze(args.source.resolve())
    build()
