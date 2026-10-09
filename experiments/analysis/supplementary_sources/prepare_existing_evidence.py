"""Offline accounting only: read frozen receipts, never import experiment runners.

--collect-logs copies only accounting fields and source hashes from existing files.
The default rebuild uses this portable ledger; --verify-sources verifies raw files.
No model client, actor, evaluator, subprocess, or network operation is used.
"""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT.parent / "jiuwenswarm-swe-runner"
LEDGER = ROOT / "data/existing_cost_receipts.json"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def csv_rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def table(name, spec, heading, rows):
    (ROOT / "tables" / (name + ".tex")).write_text(
        "\n".join(
            [
                r"\begin{tabular}{" + spec + "}",
                r"\toprule",
                heading + r" \\",
                r"\midrule",
                *[row + r" \\" for row in rows],
                r"\bottomrule",
                r"\end{tabular}",
            ]
        )
        + "\n"
    )


def collect_logs(runner):
    ledger = {"sources": {}, "groups": {}}

    def bind(path, base):
        key = ("runner/" if base == runner else "paper/") + str(path.relative_to(base))
        ledger["sources"][key] = sha(path)
        return key

    for folder in ("rewardpack_joint", "rewardpack_joint_repair"):
        for name in (
            "trajectory_scores.csv",
            "criterion_receipts.csv",
            "protocol.json",
            "summary.json",
        ):
            bind(ROOT / "data" / folder / name, ROOT)
    for name in (
        "data/rewardpack_joint/manifest_index.json",
        "data/rewardpack_joint/execution.json",
        "data/rewardpack_joint_repair/lineage.json",
    ):
        bind(ROOT / name, ROOT)

    index = read(ROOT / "data/rewardpack_joint/manifest_index.json")
    baseline = {
        r["identity"]: r
        for r in read(ROOT / "data/rewardpack_joint/execution.json")["receipts"]
    }
    repaired = {
        r["identity"]: r
        for r in read(ROOT / "data/rewardpack_joint_repair/lineage.json")
    }
    for version, directory in [
        ("original", "fast_diagnostic_v1"),
        ("repair", "technical_repair_v1"),
    ]:
        for row in index:
            group = ledger["groups"].setdefault(version + "/" + row["benchmark"], [])
            directory_path = (
                runner
                / "local/historical_run"
                / directory
                / "evaluations"
                / row["identity"]
            )
            path = directory_path / "receipt.json"
            expected = (
                baseline[row["identity"]]["receipt_sha256"]
                if version == "original"
                else repaired[row["identity"]]["repair_receipt_sha256"]
            )
            assert sha(path) == expected, f"Changed frozen receipt: {path}"
            receipt = read(path)
            records = []
            for log in sorted(directory_path.glob("model/**/result.json")):
                call = read(log)
                records.append(
                    {
                        "source": bind(log, runner),
                        "status": call["status"],
                        "error_type": call.get("error_type"),
                        "elapsed_s": call.get("elapsed_s"),
                        "usage": {
                            k: v
                            for k, v in (call.get("usage") or {}).items()
                            if k
                            in (
                                "prompt_tokens",
                                "completion_tokens",
                                "total_tokens",
                                "cost",
                            )
                        },
                    }
                )
            group.append(
                {
                    "identity": row["identity"],
                    "source": bind(path, runner),
                    "wire_requests": receipt["wire_requests"],
                    "elapsed_s": receipt["elapsed_s"],
                    "terminal_failure": False,
                    "calls": records,
                }
            )
    for study in ("historical_run", "historical_run"):
        bind(ROOT / "experiments" / study / "protocol.json", ROOT)
        bind(ROOT / "experiments" / study / "results/summary.json", ROOT)
        for path in sorted(
            (ROOT / "experiments" / study / "results/raw").glob("*.json")
        ):
            record = read(path)
            calls = []
            for i, attempt in enumerate(record["attempts"]):
                calls.append(
                    {
                        "attempt": i,
                        "status": attempt["status"],
                        "elapsed_s": attempt.get("elapsed_s"),
                        "usage": {
                            k: v
                            for k, v in (attempt.get("usage") or {}).items()
                            if k
                            in ("prompt_tokens", "completion_tokens", "total_tokens")
                        },
                    }
                )
            ledger["groups"].setdefault(study + "/" + record["mode"], []).append(
                {
                    "identity": path.stem,
                    "source": bind(path, ROOT),
                    "wire_requests": None,
                    "elapsed_s": record.get("elapsed_s"),
                    "terminal_failure": record["status"] != "ok",
                    "calls": calls,
                }
            )
    write(LEDGER, ledger)


def summarize_cost(records):
    calls = [c for r in records for c in r["calls"]]
    usage = [c["usage"] for c in calls]
    result = dict(
        receipts=len(records),
        logged_calls=len(calls),
        failed_logged_calls=sum(c["status"] not in ("completed", "ok") for c in calls),
        terminal_failures=sum(r["terminal_failure"] for r in records),
        physical_requests=sum(r["wire_requests"] for r in records)
        if all(r["wire_requests"] is not None for r in records)
        else None,
        usage_recorded_calls=sum(
            u.get("prompt_tokens") is not None
            and u.get("completion_tokens") is not None
            for u in usage
        ),
        missing_usage_calls=sum(
            u.get("prompt_tokens") is None or u.get("completion_tokens") is None
            for u in usage
        ),
        recorded_prompt_tokens=sum(u.get("prompt_tokens") or 0 for u in usage),
        recorded_completion_tokens=sum(u.get("completion_tokens") or 0 for u in usage),
        recorded_cost_fields=sum(u.get("cost") is not None for u in usage),
        monetary_cost=None,
        summed_call_elapsed_s=sum(
            c["elapsed_s"] for c in calls if c["elapsed_s"] is not None
        )
        if any(c["elapsed_s"] is not None for c in calls)
        else None,
        calls_missing_elapsed=sum(c["elapsed_s"] is None for c in calls),
        summed_receipt_elapsed_s=sum(
            r["elapsed_s"] for r in records if r["elapsed_s"] is not None
        )
        if any(r["elapsed_s"] is not None for r in records)
        else None,
        receipts_missing_elapsed=sum(r["elapsed_s"] is None for r in records),
        failures_by_type=dict(
            Counter(
                c.get("error_type") or c["status"]
                for c in calls
                if c["status"] not in ("completed", "ok")
            )
        ),
    )
    result["recorded_tokens"] = (
        result["recorded_prompt_tokens"] + result["recorded_completion_tokens"]
    )
    # A wire counter and a logical result file are different units, not interchangeable attempts.
    result["physical_minus_logged"] = (
        None
        if result["physical_requests"] is None
        else result["physical_requests"] - len(calls)
    )
    return result


def coverage(root):
    scores = csv_rows(root / "trajectory_scores.csv")
    criteria = csv_rows(root / "criterion_receipts.csv")
    by_id = defaultdict(list)
    for row in criteria:
        by_id[row["identity"]].append(row)
    result = {}
    ledger = []
    for benchmark in ("workbuddy", "claw_eval"):
        selected = [r for r in scores if r["benchmark"] == benchmark]
        totals = Counter()
        for row in selected:
            items = by_id[row["identity"]]
            assert items and len(items) == len({r["criterion_id"] for r in items})
            technical = sum(r["protocol_error"] == "True" for r in items)
            evidence = sum(
                r["status"] == "unknown" and r["protocol_error"] != "True"
                for r in items
            )
            assert technical + evidence == int(row["unknown_criteria"])
            state = (
                "complete"
                if not technical and not evidence
                else "mixed_unknown"
                if technical and evidence
                else "technical_only"
                if technical
                else "evidence_only"
            )
            assert bool(row["reward"]) == (state == "complete")
            totals[state] += 1
            totals["technical_criteria"] += technical
            totals["evidence_criteria"] += evidence
            totals["criteria"] += len(items)
            ledger.append(
                dict(
                    identity=row["identity"],
                    benchmark=benchmark,
                    category=state,
                    technical_unknown=technical,
                    evidence_unknown=evidence,
                )
            )
        totals["eligible"] = len(selected)
        totals["receipts"] = len(selected)
        totals["terminal_failures"] = (
            0  # All frozen identities have verified terminal receipts.
        )
        assert sum(
            totals[k]
            for k in ("complete", "mixed_unknown", "technical_only", "evidence_only")
        ) == len(selected)
        result[benchmark] = {
            k: totals[k]
            for k in (
                "eligible",
                "receipts",
                "complete",
                "evidence_only",
                "technical_only",
                "mixed_unknown",
                "terminal_failures",
                "criteria",
                "technical_criteria",
                "evidence_criteria",
            )
        }
    return result, ledger


def accuracy(study, modes, expected_cases):
    summary = read(ROOT / "experiments" / study / "results/summary.json")
    rows = summary["rows"]
    assert len(rows) == expected_cases
    truth_key = "reference" if expected_cases == 30 else "truth"
    result = {
        "cases": len(rows),
        "tasks": len({r["task"] for r in rows}),
        "reference_counts": dict(Counter(r[truth_key] for r in rows)),
        "modes": {},
    }
    for mode in modes:
        determinate = [r for r in rows if r[mode] in ("satisfied", "violated")]
        result["modes"][mode] = dict(
            covered=len(determinate),
            correct=sum(r[mode] == r[truth_key] for r in determinate),
            unknown=len(rows) - len(determinate),
            false_acceptances=sum(
                r[mode] == "satisfied" and r[truth_key] == "violated" for r in rows
            ),
            false_rejections=sum(
                r[mode] == "violated" and r[truth_key] == "satisfied" for r in rows
            ),
            defects_found=sum(r[mode] == r[truth_key] == "violated" for r in rows),
        )
        for row in rows:
            identity = "__".join(
                [row["task"], row["arm"]]
                + ([row["variant"]] if "variant" in row else [])
                + [mode]
            )
            raw = read(
                ROOT / "experiments" / study / "results/raw" / (identity + ".json")
            )
            assert raw["status"] == "ok" and raw["response"]["verdict"] == row[mode]
    common = [r for r in rows if all(r[m] in ("satisfied", "violated") for m in modes)]
    result["common_coverage"] = len(common)
    result["common_correct"] = {
        m: sum(r[m] == r[truth_key] for r in common) for m in modes
    }
    if expected_cases == 24:
        result["variants"] = {
            v: {
                "cases": sum(r["variant"] == v for r in rows),
                "reference_counts": dict(
                    Counter(r["truth"] for r in rows if r["variant"] == v)
                ),
                "correct": {
                    m: sum(r[m] == r["truth"] for r in rows if r["variant"] == v)
                    for m in modes
                },
            }
            for v in ("original", "layout_shift", "numeric_defect")
        }
    return result


def build():
    frozen = read(LEDGER)
    for name, expected in frozen["sources"].items():
        if name.startswith("paper/"):
            assert sha(ROOT / name.removeprefix("paper/")) == expected, (
                f"Changed frozen input: {name}"
            )
    costs = {key: summarize_cost(records) for key, records in frozen["groups"].items()}
    coverages, states = {}, {}
    table_rows = []
    for version, folder in [
        ("original", "rewardpack_joint"),
        ("repair", "rewardpack_joint_repair"),
    ]:
        coverages[version], states[version] = coverage(ROOT / "data" / folder)
        summary = read(ROOT / "data" / folder / "summary.json")
        for benchmark, values in coverages[version].items():
            assert (
                values["complete"] == summary[benchmark]["reward_complete_trajectories"]
            )
            c = costs[version + "/" + benchmark]
            for key, actual in [
                ("prompt_tokens", c["recorded_prompt_tokens"]),
                ("completion_tokens", c["recorded_completion_tokens"]),
                ("logged_model_calls", c["logged_calls"]),
                ("wire_requests", c["physical_requests"]),
            ]:
                assert summary[benchmark]["execution"][key] == actual, (
                    version,
                    benchmark,
                    key,
                )
    for benchmark, label in [("workbuddy", "WorkBuddy"), ("claw_eval", "claw-eval")]:
        for version in ("original", "repair"):
            s = coverages[version][benchmark]
            table_rows.append(
                f"{label} & {version.title()} & {s['complete']}/{s['eligible']} & {s['evidence_only']} & {s['technical_only']} & {s['mixed_unknown']} & {s['terminal_failures']} & {s['technical_criteria']}/{s['evidence_criteria']}"
            )
    table(
        "rewardpack_joint_coverage",
        "llrrrrrr",
        r"Benchmark & Version & Complete/all & Evidence only & Technical only & Mixed & Terminal fail & Tech./evid. $U$",
        table_rows,
    )
    labels = {
        "original/workbuddy": "Joint: WorkBuddy",
        "original/claw_eval": "Joint: claw-eval",
        "repair/workbuddy": "Repair increment: WorkBuddy",
        "repair/claw_eval": "Repair increment: claw-eval",
        "historical_run/structured": "30 units: structured",
        "historical_run/text": "24 cases: text",
        "historical_run/generic": "24 cases: generic",
        "historical_run/bound_no_verdict": "24 cases: bound",
    }
    cost_rows = []
    for key, label in labels.items():
        s = costs[key]
        # Receipt timers in the 30-unit pilot cover all attempts; other rows use call timers.
        elapsed = (
            s["summed_receipt_elapsed_s"]
            if key.startswith("e1_structured")
            else s["summed_call_elapsed_s"]
        )
        wire = "NR" if s["physical_requests"] is None else str(s["physical_requests"])
        cost_rows.append(
            f"{label} & {wire}/{s['logged_calls']} & {s['failed_logged_calls']} & {s['usage_recorded_calls']}/{s['logged_calls']} & {s['recorded_tokens']:,} & {elapsed / 60:.2f}"
        )
    table(
        "existing_evidence_costs",
        "lrrrrr",
        r"Archive & Wire/logs & Failed logs & Usage/logs & Recorded tokens & Summed min.",
        cost_rows,
    )
    studies = {
        "real_30": accuracy("historical_run", ["text", "structured"], 30),
        "controlled_24": accuracy(
            "historical_run", ["text", "generic", "bound_no_verdict"], 24
        ),
    }
    s = studies["real_30"]
    rows = []
    for mode, label in [("text", "Text"), ("structured", "Structured readout")]:
        m = s["modes"][mode]
        rows.append(
            f"{label} & {m['covered']}/30 & {m['correct']}/{m['covered']} & {m['false_acceptances']}/4 & {m['false_rejections']}/26 & {m['unknown']} & {s['common_correct'][mode]}/{s['common_coverage']}"
        )
    table(
        "objective_requirement_accuracy",
        "lrrrrrr",
        r"Evidence & Coverage & Correct/covered & False accept & False reject & Unknown & Common correct",
        rows,
    )
    sources = [
        LEDGER,
        *[
            ROOT / "data" / folder / name
            for folder in ("rewardpack_joint", "rewardpack_joint_repair")
            for name in (
                "trajectory_scores.csv",
                "criterion_receipts.csv",
                "protocol.json",
                "summary.json",
            )
        ],
        *[
            ROOT / "experiments" / name / "results/summary.json"
            for name in ("historical_run", "historical_run")
        ],
    ]
    result = dict(
        scope="Existing artifacts only; 0 new model requests, actor rollouts or experimental evaluations.",
        sources={str(p.relative_to(ROOT)): sha(p) for p in sources},
        coverage=coverages,
        candidate_coverage=states,
        costs=costs,
        accuracy=studies,
        limitations={
            "cost": "Recorded token subtotals only. Missing usage is unknown. Wire requests differ from logical logs; elapsed sums are not batch wall time. Monetary and end-to-end pipeline costs unavailable.",
            "selection": "Runtime A/B/C pool is distinct from the score-stratified joint-ranking pool; claw A_run is selectable, A_mean is not.",
            "accuracy": "30 real candidate-requirement units and 24 controlled cases are separate; neither establishes full Harness accuracy or superiority over a generic judge.",
        },
    )
    write(ROOT / "data/existing_evidence_accounting.json", result)
    print(
        json.dumps(
            {"coverage": coverages, "costs": costs, "accuracy": studies}, indent=2
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collect-logs",
        action="store_true",
        help="Read existing raw files; no experiments execute",
    )
    parser.add_argument("--verify-sources", action="store_true")
    parser.add_argument("--runner-root", type=Path, default=RUNNER)
    args = parser.parse_args()
    if args.collect_logs:
        collect_logs(args.runner_root)
    if args.verify_sources:
        for name, expected in read(LEDGER)["sources"].items():
            prefix, path = name.split("/", 1)
            actual = (args.runner_root if prefix == "runner" else ROOT) / path
            assert sha(actual) == expected, f"Changed raw source: {actual}"
    build()
