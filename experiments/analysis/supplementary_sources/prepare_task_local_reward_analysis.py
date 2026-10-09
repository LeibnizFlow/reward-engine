"""Task-local ranking and reward-argmax selection on frozen artifacts; no model calls."""

import hashlib
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np
from prepare_claw_reward_agreement import clustered_correlations, reconstruct
from prepare_reward_agreement import fmt, interval, paired_rewards, save_csv, table

ROOT = Path(__file__).resolve().parents[1]
SEED, REPLICATES, TOL = 20260928, 20000, 1e-8
PAIRS = [(0, 1), (0, 2), (1, 2)]
POLICIES = {
    "reward": "Cached-reward argmax",
    "A": "Baseline",
    "B": "Reward-guided revision",
    "C": "Hypothesis guidance",
    "uniform": "Uniform expectation",
    "oracle": "Oracle upper bound",
}


def read(name):
    return json.loads((ROOT / "data" / name).read_text())


def resample_weights(clusters, replicates=REPLICATES):
    keys = list(dict.fromkeys(clusters))
    group = np.array([keys.index(k) for k in clusters])
    draws = np.random.default_rng(SEED).integers(
        len(keys), size=(replicates, len(keys))
    )
    counts = np.zeros((replicates, len(keys)), dtype=int)
    np.add.at(counts, (np.arange(replicates)[:, None], draws), 1)
    return counts[:, group]


def estimate(values, weights, denominator=None):
    values = np.asarray(values, float)
    den = (
        np.ones(len(values)) if denominator is None else np.asarray(denominator, float)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        boots = (weights @ values) / (weights @ den)
    return dict(
        value=float(values.sum() / den.sum()) if den.sum() else None, **interval(boots)
    )


def ranking(differences, y, ids, weights, benchmark, method):
    counts = Counter()
    correct = np.zeros(len(y))
    eligible = np.zeros(len(y))
    ledger = []
    for t, task in enumerate(ids):
        for k, (a, b) in enumerate(PAIRS):
            dx, dy = float(differences[t, k]), float(y[t, a] - y[t, b])
            outcome = (
                "external_tie"
                if abs(dy) <= TOL
                else "reward_tie"
                if abs(dx) <= TOL
                else "concordant"
                if dx * dy > 0
                else "discordant"
            )
            counts[outcome] += 1
            correct[t] += outcome == "concordant"
            eligible[t] += abs(dy) > TOL
            ledger.append(
                dict(
                    benchmark=benchmark,
                    task=task,
                    method=method,
                    pair="ABC"[a] + "ABC"[b],
                    reward_difference=dx,
                    external_difference=dy,
                    outcome=outcome,
                )
            )
    return dict(
        counts={
            k: counts[k]
            for k in ("concordant", "discordant", "reward_tie", "external_tie")
        },
        accuracy=estimate(correct, weights, eligible),
    ), ledger


def reward_weights(x):
    x = np.asarray(x, float)
    if x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all():
        raise ValueError("Expected finite three-candidate rewards")
    mask = np.abs(x - x.max(axis=1, keepdims=True)) <= TOL
    return mask / mask.sum(axis=1, keepdims=True)


def selection(x, y, ids, clusters, weights, benchmark):
    if (
        x.shape != y.shape
        or not np.isfinite(y).all()
        or (y < 0).any()
        or (y > 100).any()
    ):
        raise ValueError("Invalid paired external scores")
    chosen = reward_weights(x)
    best = y.max(axis=1)
    best_mask = np.abs(y - best[:, None]) <= TOL
    policy_weights = {
        "reward": chosen,
        **{a: np.tile(np.eye(3)[i], (len(y), 1)) for i, a in enumerate("ABC")},
        "uniform": np.full_like(y, 1 / 3),
        "oracle": best_mask / best_mask.sum(axis=1, keepdims=True),
    }
    scores = {p: (w * y).sum(axis=1) for p, w in policy_weights.items()}
    result, ledger = {}, []
    for p, w in policy_weights.items():
        hits = (w * best_mask).sum(axis=1)
        result[p] = dict(
            mean_score=estimate(scores[p], weights),
            regret=estimate(best - scores[p], weights),
            best_hit_rate=estimate(hits, weights),
            best_hit_count=float(hits.sum()),
        )
        for i, task in enumerate(ids):
            ledger.append(
                dict(
                    benchmark=benchmark,
                    task=task,
                    cluster=clusters[i],
                    policy=p,
                    weight_A=float(w[i, 0]),
                    weight_B=float(w[i, 1]),
                    weight_C=float(w[i, 2]),
                    score=float(scores[p][i]),
                    regret=float(best[i] - scores[p][i]),
                    best_hit=float(hits[i]),
                )
            )
    gains = {
        p: estimate(scores["reward"] - scores[p], weights)
        for p in ("uniform", "A", "B", "C")
    }
    return dict(
        policies=result,
        reward_gains=gains,
        reward_tied_tasks=int((chosen.max(axis=1) < 1).sum()),
    ), ledger


def correlation_summary(x, y, clusters, replicates=REPLICATES):
    result = clustered_correlations(x, y, clusters, replicates)
    for name in ("pearson", "spearman"):
        if not np.isfinite(result[name]["value"]):
            result[name]["value"] = None
    return result


def exact_center(rows):
    return np.array([[float(v - sum(row) / len(row)) for v in row] for row in rows])


def load_inputs():
    wb = read("office50.json")["tasks"]
    x, y, ledger = paired_rewards(wb, read("semantic_archive.json")["records"])
    wx = exact_center(
        [
            [Fraction(r["reward_fraction"]) for r in ledger[i : i + 3]]
            for i in range(0, len(ledger), 3)
        ]
    )
    wy = exact_center([[Fraction(str(t[a])) for a in "ABC"] for t in wb])
    archive = read("claw_verifier_archive.json")
    assert (
        archive["external_snapshot_sha256"]
        == hashlib.sha256((ROOT / "data/claw_eval.json").read_bytes()).hexdigest()
    )
    records = archive["records"]
    labels = {t["task"]: t for t in read("claw_eval.json")["tasks"]}
    assert len(records) == len({t["task"] for t in records}) == 104 and set(labels) == {
        r["task"] for r in records
    }
    cx = np.array([[r["scores"][a] for a in "ABC"] for r in records])
    cy = np.array(
        [
            [labels[r["task"]]["A_run" if a == "A" else a] for a in "ABC"]
            for r in records
        ]
    )
    direct = []
    for r in records:
        replay = reconstruct(r["calls"])
        assert replay is not None
        assert all(abs(replay["scores"][a] - r["scores"][a]) < 1e-12 for a in "ABC")
        direct.append(
            [
                replay["pairs"]["ABC"[a] + "ABC"[b]]["probability"] - 0.5
                for a, b in PAIRS
            ]
        )
    assert x.shape == (50, 3) and cx.shape == (104, 3)
    assert np.allclose(cx.mean(axis=1), 0.5, rtol=0, atol=1e-12)
    return {
        "WorkBuddy": (
            x,
            y,
            [t["task"] for t in wb],
            [t["task"] for t in wb],
            wx,
            wy,
            None,
        ),
        "claw-eval": (
            cx,
            cy,
            [r["task"] for r in records],
            [r["scenario"] for r in records],
            cx - 0.5,
            exact_center([[Fraction(str(v)) for v in row] for row in cy]),
            np.array(direct),
        ),
    }


def build():
    results = {}
    pairs = []
    selections = []
    candidates = []
    for name, (x, y, ids, clusters, xc, yc, direct) in load_inputs().items():
        weights = resample_weights(clusters)
        rank, rows = ranking(
            np.array([x[:, a] - x[:, b] for a, b in PAIRS]).T,
            y,
            ids,
            weights,
            name,
            "aggregate",
        )
        pairs.extend(rows)
        sel, rows = selection(x, y, ids, clusters, weights, name)
        selections.extend(rows)
        results[name] = dict(
            tasks=len(ids),
            artifacts=x.size,
            clusters=len(set(clusters)),
            pooled=correlation_summary(x, y, clusters),
            centered=correlation_summary(xc, yc, clusters),
            ranking=rank,
            selection=sel,
        )
        if direct is not None:
            r, rows = ranking(direct, y, ids, weights, name, "direct_pair")
            results[name]["direct_ranking"] = r
            pairs.extend(rows)
        for i, task in enumerate(ids):
            for j, a in enumerate("ABC"):
                candidates.append(
                    dict(
                        benchmark=name,
                        task=task,
                        cluster=clusters[i],
                        arm=a,
                        reward=float(x[i, j]),
                        external_score=float(y[i, j]),
                        centered_reward=float(xc[i, j]),
                        centered_external=float(yc[i, j]),
                    )
                )
    sources = [
        "office50.json",
        "semantic_archive.json",
        "claw_eval.json",
        "claw_verifier_archive.json",
    ]
    output = dict(
        definitions=dict(
            primary_ranking="Concordant / external-nontied pairs; reward ties count as incorrect.",
            selection="Reward argmax; uniform among candidates within tolerance of maximum; external labels never break reward ties.",
            primary_selection_contrast="Reward argmax minus uniform expectation; fixed-candidate contrasts are secondary.",
            inference="Exploratory percentile intervals conditional on frozen candidates and judgments; not multiplicity adjusted or causal.",
            tolerance=TOL,
            bootstrap=dict(
                replicates=REPLICATES,
                seed=SEED,
                unit="WorkBuddy task; claw-eval scenario; task-weighted estimates",
            ),
        ),
        sources={
            f"data/{f}": hashlib.sha256((ROOT / "data" / f).read_bytes()).hexdigest()
            for f in sources
        },
        benchmarks=results,
    )
    (ROOT / "data/task_local_reward_analysis.json").write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n"
    )
    for f, rows in [
        ("task_local_reward_pairs.csv", pairs),
        ("reward_argmax_selection.csv", selections),
        ("task_local_reward_candidates.csv", candidates),
    ]:
        save_csv(f, rows)
    metric_rows = []
    selection_rows = []
    gain_rows = []

    def percent(s):
        return (
            "Undefined"
            if s["value"] is None
            else fmt(dict(value=100 * s["value"], ci95=[100 * v for v in s["ci95"]]), 1)
        )

    for name, r in results.items():
        metric_rows.append(
            f"{name} & {r['tasks']} & {percent(r['ranking']['accuracy'])} & {fmt(r['centered']['pearson'])} & {fmt(r['centered']['spearman'])}"
        )
        for p, label in POLICIES.items():
            s = r["selection"]["policies"][p]
            selection_rows.append(
                f"{name} & {label} & {fmt(s['mean_score'], 2)} & {fmt(s['regret'], 2)} & {percent(s['best_hit_rate'])}"
            )
        gain_rows.append(
            name
            + " & "
            + " & ".join(
                fmt(r["selection"]["reward_gains"][p], 2)
                for p in ("uniform", "A", "B", "C")
            )
        )
    table(
        "task_local_reward_metrics",
        "lrlll",
        r"Benchmark & Tasks & Pair accuracy (\%) & Centered Pearson & Centered Spearman",
        metric_rows,
    )
    table(
        "reward_argmax_selection",
        "lllll",
        r"Benchmark & Policy & External score & Regret & Best hit (\%)",
        selection_rows,
    )
    table(
        "reward_argmax_gains",
        "lllll",
        r"Benchmark & vs. Uniform & vs. Baseline & vs. Revision & vs. Hypothesis",
        gain_rows,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    build()
