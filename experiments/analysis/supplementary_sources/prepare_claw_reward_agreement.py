"""Freeze cached claw verifier scores and reproduce their agreement offline.

--source RUNNER reads existing files only. No model or actor entrypoint is used.
The default uses the compact manuscript snapshot and NumPy/SciPy only.
"""

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from prepare_reward_agreement import checked, corr, fmt, interval, save_csv, table

ROOT = Path(__file__).resolve().parents[1]
SEED, REPLICATES = 20260928, 20000
DIMENSIONS = ("correctness", "requirements", "delivery")
ROLES = ("jury_member_1", "jury_member_2", "jury_member_3")
SNAPSHOT = ROOT / "data/claw_verifier_archive.json"


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def reconstruct(calls):
    """Replay archived orientation/model averaging; unavailable is not zero."""
    lookup = {(c["role"], c["orientation"]): c for c in calls}
    if len(lookup) != len(calls):
        raise ValueError("Duplicate pair call")
    scores, pairs = dict.fromkeys("ABC", 0.0), {}

    def preference(call, a, b):
        if call is None or call["status"] != "completed":
            return None
        levels = call["levels"]
        if any(v is None for arm in (a, b) for v in levels[arm]):
            return None
        for arm in (a, b):
            if len(levels[arm]) != 3 or any(
                type(v) is not int or not 0 <= v <= 19 for v in levels[arm]
            ):
                raise ValueError("Invalid ordinal level")
        delta = (sum(levels[a]) - sum(levels[b])) / 57
        return 1 / (1 + math.exp(-delta))

    for a, b in combinations("ABC", 2):
        values = []
        for role in ROLES:
            forward = preference(lookup.get((role, a + b)), a, b)
            backward = preference(lookup.get((role, b + a)), b, a)
            if forward is not None and backward is not None:
                values.append((forward + 1 - backward) / 2)
        if len(values) < 2:
            return None
        p = sum(values) / len(values)
        pairs[a + b] = dict(probability=p, models=len(values))
        scores[a] += p / 2
        scores[b] += (1 - p) / 2
    return dict(scores=scores, pairs=pairs)


def freeze(source):
    packs = source / "local/claw_eval/office104_web19_composite_20260924/tasks"
    cache = source / "data/claw_eval/verifier_analysis/original"
    labels = read(cache / "labels.private.json")
    tasks = read(ROOT / "data/claw_eval.json")["tasks"]
    assert {p.parent.parent.name for p in packs.glob("*/round_1/pack.json")} == {
        t["task"] for t in tasks
    }
    records = []
    for t in tasks:
        tid = t["task"]
        location = cache / "tasks" / tid
        decision_path = location / "decision.json"
        inp = location / "input.json"
        decision, inputs = read(decision_path), read(inp)
        assert decision["task_id"] == tid and decision["input_sha256"] == sha(inp)
        for a in "ABC":
            assert inputs["source_hashes"][a] == sha(
                packs / tid / "round_1" / a / "delivery.json"
            )
            assert (
                abs(100 * labels[tid]["scores"][a] - t["A_run" if a == "A" else a])
                < 1e-7
            )
        pack_path = packs / tid / "round_1/pack.json"
        pack = read(pack_path)
        assert pack["task_id"] == tid
        calls = []
        for a, b in [(a, b) for a in "ABC" for b in "ABC" if a != b]:
            for role in ROLES:
                path = location / "calls/pair" / role / (a + b) / "decision.json"
                call = read(path)
                calls.append(
                    dict(
                        role=role,
                        orientation=a + b,
                        status=call["status"],
                        model=call.get("model"),
                        sha256=sha(path),
                        levels={
                            arm: [
                                call["answer"]["scores"][arm][dim]["level"]
                                for dim in DIMENSIONS
                            ]
                            for arm in (a, b)
                        }
                        if call["status"] == "completed"
                        else None,
                    )
                )
        verifier = decision["verifier"]
        replay = reconstruct(calls)
        if verifier["status"] == "selected":
            assert replay is not None
            for a in "ABC":
                assert math.isclose(
                    replay["scores"][a], verifier["scores"][a], abs_tol=1e-12
                )
            for pair, item in replay["pairs"].items():
                assert item["models"] == verifier["pairs"][pair]["models"]
                assert math.isclose(
                    item["probability"],
                    verifier["pairs"][pair]["probability"],
                    abs_tol=1e-12,
                )
        else:
            assert replay is None and "scores" not in verifier
        records.append(
            dict(
                task=tid,
                scenario=re.sub(r"^T\d+(?:zh)?_", "", tid),
                pack=dict(
                    pack_id=pack["pack_id"],
                    status=pack["status"],
                    criteria=len(pack["criteria"]),
                    tests=len(pack["tests"]),
                    checkers=dict(Counter(t["checker"] for t in pack["tests"])),
                    sha256=sha(pack_path),
                ),
                status=verifier["status"],
                scores=verifier.get("scores"),
                calls=calls,
                input_sha256=sha(inp),
                decision_sha256=sha(decision_path),
            )
        )
    payload = dict(
        scope="Independent cached three-dimension verifier; not a RewardPack execution score.",
        dimensions=DIMENSIONS,
        external_snapshot_sha256=sha(ROOT / "data/claw_eval.json"),
        verification=dict(
            tasks=104,
            delivery_bindings=312,
            external_scores=312,
            pair_calls=sum(len(r["calls"]) for r in records),
            source_core_sha256=sha(
                source / "experiments/claw_eval/verifier_selection/core.py"
            ),
        ),
        records=records,
    )
    SNAPSHOT.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def apply_repair(source, directory, original=None, reference_directory=None):
    """Overlay validated repairs only on originally unavailable calls."""
    import sys

    sys.path.insert(0, str(source.resolve()))
    from experiments.claw_eval.verifier_selection.core import validate_pair

    archive = read(SNAPSHOT)
    assert "repair" not in archive, (
        "Import the original archive before applying a repair"
    )
    if original is None:
        original = read(ROOT / "data/claw_reward_agreement.json")
    assert "repair" not in original, "Original agreement analysis required"
    protocol = read(directory / "protocol.json")
    assert set(protocol["task_ids"]) == set(original["coverage"]["unavailable_tasks"])
    cache = source / "data/claw_eval/verifier_analysis/original/tasks"
    repaired_calls = 0
    outcomes = []
    original_valid = {
        r["task"]: r["scores"] for r in archive["records"] if r["scores"] is not None
    }
    for r in archive["records"]:
        if r["task"] not in protocol["task_ids"]:
            continue
        inputs = read(cache / r["task"] / "input.json")
        assert (
            sha(cache / r["task"] / "input.json")
            == r["input_sha256"]
            == protocol["inputs"][r["task"]]
        )
        for index, old in enumerate(r["calls"]):
            path = (
                directory
                / "tasks"
                / r["task"]
                / old["role"]
                / old["orientation"]
                / "decision.json"
            )
            if not path.exists():
                continue
            assert old["status"] != "completed", (
                "Successful original calls cannot be replaced"
            )
            new = read(path)
            prior_repair = None
            if reference_directory:
                extra = (
                    reference_directory
                    / "tasks"
                    / r["task"]
                    / old["role"]
                    / old["orientation"]
                    / "decision.json"
                )
                if extra.exists():
                    assert new["status"] != "completed"
                    prior_repair = dict(status=new["status"], sha256=sha(path))
                    path, new = extra, read(extra)
            assert new["original_sha256"] == old["sha256"]
            assert new["model"] == old["model"]
            assert len(set(new["read_refs"])) <= protocol["max_total_read_refs"]
            assert all(
                o["attempt"] <= protocol["max_attempts"] for o in new["observations"]
            )
            outcomes.append(
                dict(
                    task=r["task"],
                    role=old["role"],
                    orientation=old["orientation"],
                    status=new["status"],
                    sha256=sha(path),
                    observations=len(new["observations"]),
                    prior_repair=prior_repair,
                    cancelled_after_full_coverage=new.get(
                        "cancelled_after_full_coverage", False
                    ),
                )
            )
            if new["status"] != "completed":
                continue
            assert new["input_sha256"] == r["input_sha256"]
            arms = tuple(old["orientation"])
            validate_pair(
                new["answer"],
                inputs["public"],
                inputs["evidence"],
                new["read_refs"],
                arms,
            )
            r["calls"][index] = dict(
                role=old["role"],
                orientation=old["orientation"],
                status="completed",
                model=new["model"],
                sha256=sha(path),
                original_call=old,
                levels={
                    a: [new["answer"]["scores"][a][dim]["level"] for dim in DIMENSIONS]
                    for a in arms
                },
            )
            repaired_calls += 1
        replay = reconstruct(r["calls"])
        r["status"] = "selected" if replay is not None else "unavailable"
        r["scores"] = replay["scores"] if replay is not None else None
    assert all(
        r["scores"] == original_valid[r["task"]]
        for r in archive["records"]
        if r["task"] in original_valid
    )
    archive["repair"] = dict(
        protocol=protocol,
        protocol_sha256=sha(directory / "protocol.json"),
        original_analysis=original,
        repaired_calls=repaired_calls,
        outcomes=outcomes,
        unchanged_original_numeric_tasks=len(original_valid),
    )
    if (directory / "termination.json").exists():
        archive["repair"]["termination"] = read(directory / "termination.json")
        assert archive["repair"]["termination"]["original_protocol_sha256"] == sha(
            directory / "protocol.json"
        )
    if reference_directory:
        extra_protocol = read(reference_directory / "protocol.json")
        assert extra_protocol["reference_repair"]["parent_protocol_sha256"] == sha(
            directory / "protocol.json"
        )
        archive["repair"]["reference_repair_protocol"] = extra_protocol
        archive["repair"]["reference_repair_protocol_sha256"] = sha(
            reference_directory / "protocol.json"
        )
    SNAPSHOT.write_text(json.dumps(archive, indent=2, allow_nan=False) + "\n")


def cluster_indices(clusters, rng):
    groups = defaultdict(list)
    for i, key in enumerate(clusters):
        groups[key].append(i)
    units = list(groups.values())
    return np.array(
        [i for k in rng.integers(len(units), size=len(units)) for i in units[k]]
    )


def clustered_correlations(x, y, clusters, replicates=REPLICATES):
    if x.shape != y.shape or len(x) != len(clusters) or x.ndim != 2:
        raise ValueError("Mismatched task arrays")
    rng, boots = np.random.default_rng(SEED), {False: [], True: []}
    for start in range(0, replicates, 250):
        buckets = defaultdict(list)
        for _ in range(min(250, replicates - start)):
            ix = cluster_indices(clusters, rng)
            buckets[len(ix)].append(ix)
        for indices in buckets.values():
            ix = np.array(indices)
            bx = x[ix].reshape(len(ix), -1)
            by = y[ix].reshape(len(ix), -1)
            for ranked in boots:
                boots[ranked].extend(corr(bx, by, ranked).tolist())
    return dict(
        n_tasks=len(x),
        n_artifacts=x.size,
        n_scenarios=len(set(clusters)),
        **{
            name: dict(
                value=float(corr(x.ravel(), y.ravel(), ranked)),
                **interval(boots[ranked]),
            )
            for ranked, name in [(False, "pearson"), (True, "spearman")]
        },
    )


def build():
    archive = read(SNAPSHOT)
    tasks = read(ROOT / "data/claw_eval.json")["tasks"]
    by_id = {t["task"]: t for t in tasks}
    records = archive["records"]
    assert len(records) == len({r["task"] for r in records}) == 104
    assert {r["task"] for r in records} == set(by_id)
    assert archive["external_snapshot_sha256"] == sha(ROOT / "data/claw_eval.json")
    ledger = []
    valid = []
    for r in records:
        replay = reconstruct(r["calls"])
        if r["status"] == "selected":
            assert replay is not None
            checked(list(r["scores"].values()), 1)
            assert (
                set(r["scores"]) == set("ABC")
                and abs(sum(r["scores"].values()) - 1.5) < 1e-12
            )
            for a in "ABC":
                assert abs(replay["scores"][a] - r["scores"][a]) < 1e-12
            valid.append(r)
        else:
            assert replay is None and r["scores"] is None
        for a in "ABC":
            ledger.append(
                dict(
                    task=r["task"],
                    scenario=r["scenario"],
                    arm=a,
                    status=r["status"],
                    verifier_score=None if r["scores"] is None else r["scores"][a],
                    external_score=by_id[r["task"]]["A_run" if a == "A" else a],
                    pack_id=r["pack"]["pack_id"],
                )
            )
    x = np.array([[r["scores"][a] for a in "ABC"] for r in valid])
    y = np.array(
        [[by_id[r["task"]]["A_run" if a == "A" else a] for a in "ABC"] for r in valid]
    )
    clusters = [r["scenario"] for r in valid]
    correlations = {"All A/B/C": clustered_correlations(x, y, clusters)}
    for i, a in enumerate("ABC"):
        correlations[a] = clustered_correlations(
            x[:, i : i + 1], y[:, i : i + 1], clusters
        )
    coverage = dict(
        tasks=104,
        artifacts=312,
        available_tasks=len(valid),
        available_artifacts=3 * len(valid),
        unavailable_tasks=[r["task"] for r in records if r["scores"] is None],
        total_scenarios=len({r["scenario"] for r in records}),
        available_scenarios=len(set(clusters)),
        packs=len(records),
        criteria=sum(r["pack"]["criteria"] for r in records),
        tests=sum(r["pack"]["tests"] for r in records),
        pack_statuses=dict(Counter(r["pack"]["status"] for r in records)),
    )
    result = dict(
        scope=archive["scope"],
        coverage=coverage,
        correlations=correlations,
        bootstrap=dict(
            seed=SEED,
            replicates=REPLICATES,
            unit="scenario; retain every available task and all included arms",
            interval="percentile",
        ),
        interpretation="Conditional agreement on available verifier scores. Unknown tasks are missing, not baseline scores or zero. Each task score vector sums to 1.5 and is relative to its A/B/C pool.",
        sources={
            "data/claw_verifier_archive.json": sha(SNAPSHOT),
            "data/claw_eval.json": sha(ROOT / "data/claw_eval.json"),
        },
    )
    if "repair" in archive:
        result["repair"] = {
            k: v for k, v in archive["repair"].items() if k != "original_analysis"
        }
        result["repair"]["original_coverage"] = archive["repair"]["original_analysis"][
            "coverage"
        ]
        result["repair"]["original_correlations"] = archive["repair"][
            "original_analysis"
        ]["correlations"]
    (ROOT / "data/claw_reward_agreement.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    save_csv("claw_reward_agreement_pairs.csv", ledger)
    table(
        "claw_reward_agreement",
        "lrrll",
        r"Artifact pool & Tasks & Artifacts & Pearson $r$ [95\% CI] & Spearman $\rho$ [95\% CI]",
        [
            f"{dict(A='Baseline', B='Reward-guided revision', C='Hypothesis guidance').get(label, 'All candidates')} & {s['n_tasks']} & {s['n_artifacts']} & {fmt(s['pearson'])} & {fmt(s['spearman'])}"
            for label, s in correlations.items()
        ],
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--repair", type=Path)
    parser.add_argument("--reference-repair", type=Path)
    args = parser.parse_args()
    if args.reference_repair and not args.repair:
        parser.error("--reference-repair requires --repair")
    if args.repair and not args.source:
        parser.error("--repair requires --source")
    prior = (
        read(SNAPSHOT).get("repair", {}).get("original_analysis")
        if args.repair
        else None
    )
    if args.source:
        freeze(args.source)
    if args.repair:
        apply_repair(args.source, args.repair, prior, args.reference_repair)
    build()
