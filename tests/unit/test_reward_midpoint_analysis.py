"""Receipt integrity, midpoint semantics and historical-analysis compatibility."""

import json

import pytest

from experiments.analysis.rewardpack.analyze import write_csv
from experiments.analysis.rewardpack.inputs import canonical, sha
from experiments.analysis.rewardpack.midpoint import (
    load_version,
    run,
    score_counts,
)
from experiments.analysis.rewardpack.subset_analysis import (
    run as reference_run,
)


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def score(statuses):
    k, p, u = len(statuses), statuses.count("pass"), statuses.count("unknown")
    return {
        "criteria": [
            {
                "criterion_id": f"c{i}",
                "status": status,
                "protocol_error": status == "unknown" and i == 0,
            }
            for i, status in enumerate(statuses)
        ],
        "criterion_count": k,
        "passed": p,
        "unknown": u,
        "r_min": p / k,
        "r_max": (p + u) / k,
        "reward": p / k if u == 0 else None,
    }


@pytest.mark.parametrize(
    "statuses,expected",
    [
        (["pass"] * 4, 1),
        (["fail"] * 4, 0),
        (["unknown"] * 4, 0.5),
        (["pass", "pass", "fail", "unknown"], 0.625),
        (["unknown", "fail"], 0.25),
    ],
)
def test_midpoint_and_bounds(statuses, expected):
    result = score_counts(score(statuses), [f"c{i}" for i in range(len(statuses))])
    assert result["reward_midpoint"] == expected
    assert result["unknown_fraction"] == statuses.count("unknown") / len(statuses)
    if statuses[0] == "unknown":
        assert result["technical_unknown_criteria"] == 1


@pytest.mark.parametrize(
    "change", ["missing", "duplicate", "unknown_status", "count", "bound", "scalar"]
)
def test_invalid_criterion_or_aggregate_rejected(change):
    result = score(["pass", "unknown"])
    if change == "missing":
        result["criteria"].pop()
    elif change == "duplicate":
        result["criteria"][1]["criterion_id"] = "c0"
    elif change == "unknown_status":
        result["criteria"][1]["status"] = "not_run"
    elif change == "count":
        result["passed"] = 2
    elif change == "bound":
        result["r_max"] = 0.9
    else:
        result["reward"] = 0.75
    with pytest.raises(ValueError):
        score_counts(result, ["c0", "c1"])


@pytest.fixture
def archives(tmp_path):
    known = [["fail"] * 4, ["pass"] * 2 + ["fail"] * 2, ["pass"] * 4]
    unknown = [
        ["pass"] * 3 + ["unknown"],
        ["pass"] * 2 + ["fail", "unknown"],
        ["pass", "fail", "fail", "unknown"],
    ]
    task_specs = [
        ("workbuddy", "known", [0, 10, 20], known),
        ("workbuddy", "unknown", [0, 5, 10], unknown),
        ("claw_eval", "official_tie", [0, 0, 100], known),
        ("claw_eval", "all_unknown", [0, 10, 20], [["unknown"] * 4] * 3),
    ]
    roots = [tmp_path / "original", tmp_path / "repaired"]
    for root in roots:
        protocol, config, manifest, rows = (
            {"name": root.name},
            {"model": "unused"},
            [],
            [],
        )
        save(root / "protocol.json", protocol)
        save(root / "replay_config.json", config)
        save(root / "case_audit.json", {"cases": []})
        for benchmark, task, officials, judgments in task_specs:
            pack_path = tmp_path / "packs" / f"{task}.json"
            pack = {
                "pack_id": task,
                "task_id": task,
                "criteria": [{"id": f"c{i}"} for i in range(4)],
            }
            save(pack_path, pack)
            for i in range(3):
                identity = f"{task}-{i}"
                hashes = {
                    "pack": sha(pack_path),
                    "trace": f"trace-{identity}",
                    "snapshot": f"snapshot-{identity}",
                }
                item = {
                    "identity": identity,
                    "benchmark": benchmark,
                    "task": task,
                    "scenario": task,
                    "slot": str(i),
                    "pack_id": task,
                    "hashes": hashes,
                    "paths": {"pack": str(pack_path)},
                }
                manifest.append(item)
                scored = score(judgments[i])
                receipt = {
                    k: item[k]
                    for k in ("identity", "benchmark", "task", "slot", "pack_id")
                }
                receipt.update(
                    pack_sha256=hashes["pack"],
                    input_hashes=hashes,
                    protocol_sha256=canonical(protocol),
                    config_sha256=canonical(config),
                    evaluation_method="joint_criterion",
                    harness_executed=False,
                    score=scored,
                )
                path = root / "evaluations" / identity / "receipt.json"
                save(path, receipt)
                rows.append(
                    {
                        "identity": identity,
                        "benchmark": benchmark,
                        "task": task,
                        "scenario": task,
                        "slot": str(i),
                        "pack_id": task,
                        "official_score": officials[i],
                        "official_status": "valid",
                        "reward": scored["reward"],
                        "r_min": scored["r_min"],
                        "r_max": scored["r_max"],
                        "unknown_criteria": scored["unknown"],
                        "reward_receipt_sha256": sha(path),
                        "official_receipt_sha256": f"official-{identity}",
                        **{k + "_sha256": v for k, v in hashes.items()},
                    }
                )
        save(root / "manifest.public.json", manifest)
        write_csv(root / "trajectory_scores.csv", rows)
    return roots


def test_receipts_and_pack_binding_validated(archives):
    rows = load_version(archives[0], {})
    assert len(rows) == 12
    for row in rows:
        if row["original_reward"] is not None:
            assert row["reward_midpoint"] == row["original_reward"]


@pytest.mark.parametrize("failure", ["missing", "hash", "identity", "criterion"])
def test_invalid_archive_rejected(archives, failure):
    import csv

    root = archives[0]
    rows = list(csv.DictReader((root / "trajectory_scores.csv").open()))
    path = root / "evaluations" / rows[0]["identity"] / "receipt.json"
    if failure == "missing":
        path.unlink()
    else:
        receipt = json.loads(path.read_text())
        if failure == "identity":
            receipt["identity"] = "another-candidate"
        elif failure == "criterion":
            receipt["score"]["criteria"].pop()
        else:
            receipt["extra"] = "tampered"
        save(path, receipt)
        if failure != "hash":
            rows[0]["reward_receipt_sha256"] = sha(path)
            write_csv(root / "trajectory_scores.csv", rows)
    with pytest.raises(ValueError):
        load_version(root, {})


def test_complete_run_matches_reference_and_tracks_new_support(archives, tmp_path):
    old_dir = tmp_path / "reference"
    reference_run(*archives, old_dir, iterations=20)
    out = tmp_path / "midpoint"
    summary = run(
        *archives, out, reference_summary=old_dir / "summary.json", iterations=20
    )
    wb = summary["analyses"]["all"]["workbuddy"]["all_selected"]["repaired"]
    assert wb["independent"]["original"]["correlations_defined"] == 1
    assert wb["independent"]["midpoint"]["correlations_defined"] == 2
    assert wb["new_correlation_tasks"]["metrics"]["tau_b"]["value"] == -1
    assert wb["common"]["paired_delta"]["tau_b"]["midpoint_minus_original"] == 0
    claw = summary["analyses"]["all"]["claw_eval"]["all_selected"]["repaired"]
    assert claw["new_scalar_tasks"]["tasks"] == 1
    assert (
        claw["new_correlation_tasks"]["tasks"] == 0
    )  # All-unknown midpoint remains tied.
    assert claw["independent"]["midpoint"]["pair_counts"]["official_tie"] == 1
    status = json.loads((out / "status.json").read_text())
    assert status["reference_statistics_verified"] and status["common_deltas_zero"]
    assert status["model_calls"] == 0 and status["receipt_count"] == 24
    assert (out / "RESULTS.md").is_file()
    with pytest.raises(FileExistsError):
        run(*archives, out, iterations=20)


def test_reference_drift_rejected_before_output(archives, tmp_path):
    old_dir = tmp_path / "reference"
    reference_run(*archives, old_dir, iterations=5)
    reference = json.loads((old_dir / "summary.json").read_text())
    reference["analyses"]["all"]["workbuddy"]["all_selected"]["independent"][
        "original"
    ]["tasks"] = 99
    save(old_dir / "summary.json", reference)
    with pytest.raises(ValueError, match="reference"):
        run(
            *archives,
            tmp_path / "new",
            reference_summary=old_dir / "summary.json",
            iterations=5,
        )
    assert not (tmp_path / "new").exists()
