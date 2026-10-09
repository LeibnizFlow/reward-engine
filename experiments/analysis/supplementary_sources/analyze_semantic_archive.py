"""Reanalyse archived text-only RewardPack judgments; never call a model.

--source explicitly imports cached records after checking packs and task IDs.
The default path reproduces descriptive disagreement statistics offline.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "data/semantic_archive.json"


def read(path):
    return json.loads(path.read_text())


def freeze(source):
    location = (
        source / "experiments/workbuddy/studies/reward_pack_semantic_gap_20260918"
    )
    records = read(location / "output/semantic_records.json")
    office = read(ROOT / "data/office50.json")["tasks"]
    run_sources = {
        t["task"]: t["source"]
        for t in read(source / "reports/workbuddy/office50/results.json")["tasks"]
    }
    packs = {
        r["task"]: read(
            source
            / "local/workbuddy/outcome/office50"
            / run_sources[r["task"]]
            / "round_3/pack.json"
        )
        for r in office
    }
    assert len(records) == 1770
    assert len({(r["task"], r["arm"], r["test_id"]) for r in records}) == len(records)
    expected = {
        (task, arm, test["test_id"])
        for task, pack in packs.items()
        for arm in "ABC"
        for test in pack["tests"]
    }
    assert expected == {(r["task"], r["arm"], r["test_id"]) for r in records}
    compact = []
    for r in records:
        assert r["pack_id"] == packs[r["task"]]["pack_id"]
        test = next(
            t for t in packs[r["task"]]["tests"] if t["test_id"] == r["test_id"]
        )
        assert test["question"] == r["question"]
        assert r["result"]["status"] == "ok"
        assert 0 <= r["result"]["parsed"]["score"] <= 1
        compact.append(
            {
                **{
                    k: r[k]
                    for k in [
                        "task",
                        "arm",
                        "test_id",
                        "pack_id",
                        "model",
                        "include_inputs",
                        "artifact_chars",
                        "input_chars",
                    ]
                },
                "returned_model": r["result"]["returned_model"],
                "score": r["result"]["parsed"]["score"],
                "verdict": r["result"]["parsed"]["verdict"],
            }
        )
    SNAPSHOT.write_text(
        json.dumps(
            dict(
                source=str(location.relative_to(source)),
                scope="Cached final successful text-only judgments; not independent defect labels.",
                expected_tests=sum(len(p["tests"]) for p in packs.values()),
                records=compact,
            ),
            indent=2,
        )
        + "\n"
    )


def confusion(rows, scores):
    count = Counter(tp=0, fp=0, fn=0, tn=0)
    for row in rows:
        for arm in "BC":
            flagged = scores[row["task"], arm] < scores[row["task"], "A"] - 1e-8
            lower = row[arm] < row["A"] - 1e-8
            count[
                "tp"
                if flagged and lower
                else "fp"
                if flagged
                else "fn"
                if lower
                else "tn"
            ] += 1
    return dict(
        n_tasks=len(rows),
        comparisons=2 * len(rows),
        **count,
        precision=count["tp"] / (count["tp"] + count["fp"])
        if count["tp"] + count["fp"]
        else None,
        recall=count["tp"] / (count["tp"] + count["fn"])
        if count["tp"] + count["fn"]
        else None,
    )


def build():
    data = read(SNAPSHOT)
    records = data["records"]
    office = read(ROOT / "data/office50.json")["tasks"]
    per = defaultdict(list)
    for r in records:
        per[r["task"], r["arm"]].append(r["score"])
    assert len(per) == 150 and len(records) == 1770
    scores = {key: float(np.mean(values)) for key, values in per.items()}
    unavailable = {
        r["task"] for r in records if not r["artifact_chars"] or not r["include_inputs"]
    }
    retained = [r for r in office if r["task"] not in unavailable]
    discordant = []
    for r in office:
        for arm in "BC":
            if (
                r[arm] < r["A"] - 1e-8
                and scores[r["task"], arm] > scores[r["task"], "A"] + 1e-8
            ):
                discordant.append(
                    dict(
                        task=r["task"],
                        arm=arm,
                        semantic_A=scores[r["task"], "A"],
                        semantic_arm=scores[r["task"], arm],
                        external_A=r["A"],
                        external_arm=r[arm],
                    )
                )
    result = dict(
        n_judgments=len(records),
        n_tests=data["expected_tests"],
        n_artifacts=len(per),
        requested_models=dict(Counter(r["model"] for r in records)),
        returned_models=dict(Counter(r["returned_model"] for r in records)),
        means={
            a: float(
                np.mean([value for (task, arm), value in scores.items() if arm == a])
            )
            for a in "ABC"
        },
        no_extracted_artifact_count=len(
            {(r["task"], r["arm"]) for r in records if not r["artifact_chars"]}
        ),
        no_source_input_judgments=sum(not r["include_inputs"] for r in records),
        excluded_input_scope_tasks=sorted(unavailable),
        all=confusion(office, scores),
        input_scope_sensitivity=confusion(retained, scores),
        grader_exception_sensitivity=confusion(
            [r for r in office if r["task"] != "priority-sync-notification-pipeline"],
            scores,
        ),
        semantic_preference_with_external_loss=discordant,
        interpretation="Agreement with retained external score ordering, not defect detection accuracy; no inference that semantic preference caused external loss.",
    )
    (ROOT / "data/semantic_archive_statistics.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Subset & Comparisons & TP & FP & FN & TN & Precision/Recall \\",
        r"\midrule",
    ]
    for key, label in [
        ("all", "All 50 tasks"),
        ("input_scope_sensitivity", "42-task input sensitivity"),
        ("grader_exception_sensitivity", "49-task grader sensitivity"),
    ]:
        s = result[key]
        lines.append(
            f"{label} & {s['comparisons']} & {s['tp']} & {s['fp']} & {s['fn']} & {s['tn']} & {s['precision']:.3f}/{s['recall']:.3f}"
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (ROOT / "tables/workbuddy_semantic_disagreement.tex").write_text(
        "\n".join(lines) + "\n"
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k != "semantic_preference_with_external_loss"
            },
            indent=2,
        )
    )
    print("Higher semantic score despite external loss:", len(discordant))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    if args.source:
        freeze(args.source)
    build()
