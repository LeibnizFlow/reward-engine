"""Selection, denominators and uncertainty for fixed-triad gap diagnostics."""

import csv
import json
from copy import deepcopy

import numpy as np
import pytest

from experiments.analysis.rewardpack.subset_analysis import (
    METRICS,
    cluster_arrays,
    common_support,
    confirmed_runtime_tasks,
    describe_task,
    group_scores,
    pair_statistics,
    paired_delta,
    resampled_estimates,
    run,
    select_tasks,
    summarize,
    verify_versions,
)


def triad(
    rewards=(0, 0.5, 1),
    officials=(0, 50, 100),
    task="x",
    benchmark="workbuddy",
    scenario="x",
):
    return [
        {
            "identity": f"{task}-{i}",
            "benchmark": benchmark,
            "task": task,
            "scenario": scenario,
            "slot": f"slot-{i}",
            "pack_id": "pack",
            "pack_sha256": "packhash",
            "trace_sha256": f"trace-{i}",
            "snapshot_sha256": f"snapshot-{i}",
            "official_receipt_sha256": f"official-{i}",
            "official_status": "valid",
            "reward": reward,
            "r_min": reward,
            "r_max": reward,
            "official_score": official,
        }
        for i, (reward, official) in enumerate(zip(rewards, officials, strict=True))
    ]


@pytest.mark.parametrize(
    "scores,threshold,keep",
    [
        ((0, 95, 100), 10, False),
        ((0, 95, 100), 5, True),
        ((30, 35, 40), 5, True),
        ((30, 34.99999999999999, 40), 5, True),
        ((30, 34.999, 40), 5, False),
        ((0, 0, 100), 5, False),
        ((0, 0, 0), None, True),
    ],
)
def test_adjacent_gaps_not_total_span_and_inclusive_boundary(scores, threshold, keep):
    selected, audit = select_tasks(group_scores(triad(officials=scores)), threshold, {})
    assert bool(selected) is keep and audit[0]["selected"] is keep


def test_selection_ignores_reward_and_keeps_unknown_and_constant_tasks():
    rows = triad(rewards=(None, None, None)) + triad(rewards=(1, 1, 1), task="y")
    selected, _ = select_tasks(group_scores(rows), 5, {})
    assert selected == {("workbuddy", "x"), ("workbuddy", "y")}
    rows[0]["official_status"] = "invalid"
    selected, audit = select_tasks(group_scores(rows), 5, {})
    assert selected == {("workbuddy", "y")}
    assert audit[0]["reason"] == "invalid_official_score"


def test_reward_ties_are_failures_to_sort_not_dropped_and_top1_is_uniform():
    stats, pairs = describe_task(triad(rewards=(1, 1, 1)))
    assert stats["tau_b"] is None and stats["undefined_reason"] == "reward_constant"
    assert stats["reward_tie"] == 3 and {p["status"] for p in pairs} == {"reward_tie"}
    assert stats["top1_hit"] == stats["random_top1_hit"] == 1 / 3
    assert stats["regret"] == stats["random_regret"] == 50
    report = summarize([stats], "workbuddy", iterations=50)
    assert report["tasks"] == report["complete_tasks"] == 1
    assert report["metrics"]["strict_pair_accuracy"]["value"] == 0
    assert report["metrics"]["tau_b"]["defined_replicates"] == 0


def test_inverse_and_two_way_top_ties():
    stats, _ = describe_task(triad(rewards=(1, 0.5, 0)))
    assert stats["tau_b"] == stats["spearman_rho"] == -1
    assert stats["reversed"] == 3 and stats["regret"] == 100
    stats, _ = describe_task(triad(rewards=(0, 1, 1)))
    assert stats["top1_hit"] == 0.5 and stats["regret"] == 25


def test_partial_scores_keep_full_pair_denominator_and_correlations_undefined():
    rows = triad(rewards=(0, None, 1))
    rows[1].update(r_min=0.4, r_max=0.6)
    stats, _ = describe_task(rows)
    report = summarize([stats], "workbuddy", iterations=40)
    assert stats["correct"] == 1 and stats["unresolved"] == 2
    assert stats["tau_b"] is None and stats["top1_hit"] is None
    assert report["metrics"]["strict_pair_accuracy"]["value"] == 1
    assert report["metrics"]["pair_coverage"]["value"] == 1 / 3
    assert report["metrics"]["correct_fraction"]["value"] == 1 / 3
    assert report["metrics"]["interval_accuracy"]["value"] == 1
    assert report["metrics"]["interval_coverage"]["value"] == 1


def test_touching_intervals_and_official_ties_are_not_strict_order():
    rows = triad(rewards=(None, None, 1), officials=(0, 10, 10))
    rows[0].update(r_min=0, r_max=0.5)
    rows[1].update(r_min=0.5, r_max=1)
    pairs = pair_statistics(rows)
    assert pairs[0]["interval_status"] == "overlapping"
    assert pairs[2]["status"] == "official_tie"


def test_bilingual_versions_are_resampled_as_whole_scenarios_and_task_weighted():
    task_rows = [
        describe_task(triad(task=t, benchmark="claw_eval", scenario=s))[0]
        for t, s in [("en", "shared"), ("zh", "shared"), ("other", "other")]
    ]
    for row, value in zip(task_rows, [1, -1, 0.5], strict=True):
        row["tau_b"] = value
    keys, nums, dens = cluster_arrays(task_rows, "claw_eval")
    j = METRICS.index("tau_b")
    assert keys == ["other", "shared"]
    assert dens[:, j].tolist() == [1, 2] and nums[:, j].tolist() == [0.5, 0]
    actual = resampled_estimates(nums, dens, iterations=100, seed=3)[:, j]
    weights = np.random.default_rng(3).multinomial(2, [0.5, 0.5], size=100)
    expected = (weights @ nums[:, j]) / (weights @ dens[:, j])
    assert actual == pytest.approx(expected)
    assert summarize(task_rows, "claw_eval", iterations=20)["metrics"]["tau_b"][
        "value"
    ] == pytest.approx(1 / 6)


def test_undefined_tasks_remain_in_bootstrap_pool():
    rows = [
        describe_task(triad(task="a"))[0],
        describe_task(triad(rewards=(1, 1, 1), task="b"))[0],
    ]
    report = summarize(rows, "workbuddy", iterations=200)
    assert report["clusters"] == 2
    metric = report["metrics"]["tau_b"]
    assert 0 < metric["defined_replicates"] < 200
    assert metric["supporting_clusters"] == 1 and metric["ci95"] == [1, 1]
    empty = summarize([], "workbuddy", iterations=20)
    assert empty["tasks"] == 0 and empty["metrics"]["tau_b"]["value"] is None


def test_common_support_matches_pairs_top1_and_correlations_separately():
    a = group_scores(triad(rewards=(None, 0.5, 1)) + triad(rewards=(1, 1, 1), task="y"))
    b = group_scores(triad() + triad(task="y"))
    common = common_support(a, b, set(a))
    x, y = ("workbuddy", "x"), ("workbuddy", "y")
    for version in ("original", "repaired"):
        assert (
            common[version][x]["unresolved"] == 2
            and common[version][x]["top1_hit"] is None
        )
        assert common[version][y]["tau_b"] is None
    delta = paired_delta(
        list(common["original"].values()),
        list(common["repaired"].values()),
        "workbuddy",
        iterations=50,
    )
    assert delta["tau_b"]["repaired_minus_original"] is None
    assert delta["strict_pair_accuracy"]["repaired_minus_original"] == 0.75
    assert (
        "non_tie_pair_accuracy" not in delta
    )  # Its conditional denominator changes with reward ties.


def test_versions_must_preserve_candidate_and_official_bindings():
    a = triad()
    b = deepcopy(a)
    b[0]["reward"] = 0.2
    verify_versions(a, b)
    b[0]["snapshot_sha256"] = "different"
    with pytest.raises(ValueError, match="binding mismatch"):
        verify_versions(a, b)
    with pytest.raises(ValueError, match="same candidate"):
        verify_versions(a, b[:2])


def test_only_previously_confirmed_runtime_failures_qualify_for_exclusion():
    rows = triad()
    cases = [
        {
            "case": 1,
            "benchmark": "workbuddy",
            "task": "x",
            "diagnosis": "native_runtime_failure",
            "confidence": "confirmed",
            "finding_zh": "Existing verified execution error",
            "candidates": [{"identity": rows[0]["identity"]}],
        }
    ]
    assert set(confirmed_runtime_tasks({"cases": cases}, rows)) == {("workbuddy", "x")}
    cases[0]["diagnosis"] = "native_extraction_disagreement"
    assert confirmed_runtime_tasks({"cases": cases}, rows) == {}


def write_dataset(root, rows):
    root.mkdir()
    with (root / "trajectory_scores.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = [
        {
            **{k: r[k] for k in ("identity", "benchmark", "task", "scenario", "slot")},
            "hashes": {k: r[k + "_sha256"] for k in ("pack", "trace", "snapshot")},
        }
        for r in rows
    ]
    (root / "manifest.public.json").write_text(json.dumps(manifest))
    (root / "protocol.json").write_text('{"test_protocol": true}')
    (root / "case_audit.json").write_text('{"cases": []}')


def test_end_to_end_outputs_are_new_inputs_unchanged_and_empty_subsets_supported(
    tmp_path,
):
    a, b, output = (tmp_path / name for name in ("original", "repaired", "output"))
    write_dataset(a, triad(rewards=(1, 1, 1), officials=(0, 5, 10)))
    write_dataset(b, triad(officials=(0, 5, 10)))
    before = {p: p.read_bytes() for root in (a, b) for p in root.iterdir()}
    result = run(a, b, output, iterations=30)
    assert set(result["analyses"]) == {"all", "gap_2", "gap_5", "gap_10"}
    assert (
        result["analyses"]["gap_5"]["workbuddy"]["all_selected"]["independent"][
            "repaired"
        ]["tasks"]
        == 1
    )
    assert (
        result["analyses"]["gap_10"]["workbuddy"]["all_selected"]["independent"][
            "repaired"
        ]["tasks"]
        == 0
    )
    assert all(p.read_bytes() == content for p, content in before.items())
    assert (output / "RESULTS.md").is_file()
    assert json.loads((output / "status.json").read_text())["model_calls"] == 0
    with pytest.raises(FileExistsError, match="already exists"):
        run(a, b, output, iterations=30)
