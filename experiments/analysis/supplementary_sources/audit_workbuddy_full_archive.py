"""Reconcile every final WorkBuddy Office candidate with its archived evidence."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
REPORT_REL = Path("reports/workbuddy/office50")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def score_fields(row):
    return {key: row[key] for key in ("A", "B", "C", "selected_arm", "selected_score")}


def linked_path(report_dir, href, source, *, file_required=False):
    path = (report_dir / unquote(href)).resolve()
    require(path.is_relative_to(source) and path.exists() and
            (not file_required or path.is_file()), f"Missing report link: {href}")
    return path


def check_csv(report_dir, tasks, round_rows):
    for filename in ("scores.csv", "report_summary.csv"):
        with (report_dir / filename).open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        require(len(rows) == 50, f"{filename}: expected 50 tasks")
        for row in rows:
            task = tasks[row["task"]]
            require(row["selected_arm"] == task["selected_arm"], f"{filename}: selector mismatch")
            for key in ("A", "B", "C", "selected_score"):
                require(float(row[key]) == task[key], f"{filename}: {row['task']} {key}")
    with (report_dir / "rounds.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 150, "rounds.csv: expected 150 task rounds")
    for row in rows:
        key = (row["task"], int(row["round"]))
        source = round_rows[key]
        require(row["selected_arm"] == source["selected_arm"], f"rounds.csv: selector {key}")
        for arm in ("A", "B", "C", "selected_score"):
            require(float(row[arm]) == source[arm], f"rounds.csv: score {key} {arm}")
        invalid = sorted(filter(None, row["invalid_scoring_arms"].split(",")))
        require(invalid == sorted(source["invalid_scoring_arms"]), f"rounds.csv: invalid {key}")


def audit(source):
    source = source.resolve()
    report_dir = source / REPORT_REL
    detail_path = report_dir / "report_detail.json"
    detail = read_json(detail_path)
    report = read_json(report_dir / "results.json")
    report_data = read_json(report_dir / "report_data.json")
    manuscript = read_json(ROOT / "data/office50.json")
    inputs = read_json(ROOT / "data/semantic_input_evidence.json")
    semantic = read_json(ROOT / "data/semantic_archive.json")
    measurements = read_json(ROOT / "data/archive_review_evidence.json")
    receipt_audit = read_json(ROOT / "data/workbuddy_office_evidence.json")

    require(inputs["source_files"][str(REPORT_REL / "report_detail.json")] == digest(detail_path),
            "Report changed since file-binding audit")
    for rel, expected_hash in inputs["source_files"].items():
        require(digest(source / rel) == expected_hash, f"Input audit source changed: {rel}")
    require(len(detail["tasks"]) == len(report["tasks"]) == len(report_data["tasks"]) ==
            len(manuscript["tasks"]) == 50, "Task count")
    tasks = {task["task"]: task for task in detail["tasks"]}
    score_rows = {task["task"]: task for task in report["tasks"]}
    data_rows = {task["task"]: task for task in report_data["tasks"]}
    paper_rows = {task["task"]: task for task in manuscript["tasks"]}
    paper_rounds = {task["task"]: task for task in manuscript["rounds"]}
    artifacts = {(row["task"], row["arm"]): row for row in inputs["artifacts"]}
    final_measurements = {(row["task"], row["arm"]): row for row in measurements["measurements"]
                          if row["benchmark"] == "WorkBuddy Office"}
    receipts = {(row["task"], row["arm"]): row for row in receipt_audit["receipt_audit"]}
    judgments = Counter((row["task"], row["arm"]) for row in semantic["records"])
    require(all(len(index) == n for index, n in ((tasks, 50), (score_rows, 50), (data_rows, 50),
            (paper_rows, 50), (paper_rounds, 50), (artifacts, 150),
            (final_measurements, 150), (receipts, 150), (judgments, 150))), "Index cardinality")
    require(set(tasks) == set(score_rows) == set(data_rows) == set(paper_rows) == set(paper_rounds),
            "Task identity mismatch")
    require(sum(judgments.values()) == 1770, "Semantic judgment count")

    round_rows = {}
    rows = []
    file_count = artifact_links = score_links = invalid_count = 0
    for task_id, task in tasks.items():
        require(score_fields(task) == score_fields(score_rows[task_id]) ==
                score_fields(data_rows[task_id]) == score_fields(paper_rows[task_id]),
                f"Final score mismatch: {task_id}")
        require(len(task["rounds"]) == len(data_rows[task_id]["rounds"]) ==
                len(paper_rounds[task_id]["rounds"]) == 3, f"Round count: {task_id}")
        for index, round_record in enumerate(task["rounds"]):
            number = index + 1
            require(round_record["round"] == number, f"Round order: {task_id}")
            for other in (data_rows[task_id]["rounds"][index],
                          paper_rounds[task_id]["rounds"][index]):
                require(score_fields(round_record) == score_fields(other),
                        f"Round score mismatch: {task_id} {number}")
                require(sorted(round_record["invalid_scoring_arms"]) ==
                        sorted(other["invalid_scoring_arms"] if isinstance(other["invalid_scoring_arms"], list)
                               else filter(None, other["invalid_scoring_arms"].split(","))),
                        f"Invalid score mismatch: {task_id} {number}")
            round_rows[task_id, number] = round_record
            invalid_count += len(round_record["invalid_scoring_arms"])
            require(round_record["selected_score"] == round_record[round_record["selected_arm"]],
                    f"Round selected score: {task_id} {number}")
            for arm in "ABC":
                actor = round_record["actors"][arm]
                receipt_path = source / "local/workbuddy/outcome/office50" / actor["source"]
                require(receipt_path.is_file(), f"Round receipt missing: {task_id} {number} {arm}")
                receipt = read_json(receipt_path)
                require(actor["score"] == round_record[arm] and receipt["status"] == "scored" and
                        abs(receipt["rewards"]["overall"] * 100 - round_record[arm]) < 1e-7,
                        f"Round receipt score: {task_id} {number} {arm}")
        final = task["rounds"][2]
        for arm in "ABC":
            key = (task_id, arm)
            actor = final["actors"][arm]
            binding = artifacts[key]
            measurement = final_measurements[key]
            audit_receipt = receipts[key]
            require(actor["score"] == task[arm] == audit_receipt["score"], f"Arm score: {key}")
            require(measurement["selected"] == (arm == task["selected_arm"]), f"Selection: {key}")
            require(len(binding["listed_files"]) == len(actor["files"]), f"File count: {key}")
            require(judgments[key] == binding["n_judgments"], f"Text judgments: {key}")

            receipt_path = source / "local/workbuddy/outcome/office50" / actor["source"]
            require(receipt_path.is_file(), f"Official receipt missing: {key}")
            official = read_json(receipt_path)
            require(official["status"] == actor["official"]["status"] == "scored", f"Receipt status: {key}")
            require(abs(official["rewards"]["overall"] * 100 - task[arm]) < 1e-7,
                    f"Official receipt score: {key}")
            require((source / "local/workbuddy/outcome/office50" / actor["native_source"]).is_file(),
                    f"Native score file missing: {key}")
            require(len(actor["score_links"]) == 2, f"Score links: {key}")
            native_path = source / "local/workbuddy/outcome/office50" / actor["native_source"]
            linked_scores = [linked_path(report_dir, link["href"], source, file_required=True)
                             for link in actor["score_links"]]
            require(linked_scores == [receipt_path.resolve(), native_path.resolve()],
                    f"Score links point to wrong receipts: {key}")
            score_links += len(linked_scores)
            for link in actor["links"]:
                linked_path(report_dir, link["href"], source)
                artifact_links += 1

            files = []
            by_path = {item["path"]: item for item in binding["listed_files"]}
            require(len(by_path) == len(actor["files"]), f"Duplicate file path: {key}")
            for item in actor["files"]:
                bound = by_path[item["path"]]
                path = source / bound["bound_path"]
                require(path.is_file() and path.stat().st_size == item["bytes"],
                        f"Bound file missing or changed: {key} {item['path']}")
                require(digest(path) == item["sha256"] == bound["sha256"],
                        f"File hash mismatch: {key} {item['path']}")
                files.append({"path": item["path"], "bytes": item["bytes"],
                              "sha256": item["sha256"], "bound_path": bound["bound_path"]})
                file_count += 1

            filtered = any(item["filtered_from_artifact_slot"] for item in binding["listed_files"])
            root_mismatch = any(not item["historical_root_exists"] for item in binding["listed_files"])
            if binding["artifact_chars"]:
                input_status = "nonempty_artifact_text"
            elif filtered:
                input_status = "output_routed_as_input"
            elif root_mismatch:
                input_status = "workspace_root_mismatch"
            else:
                input_status = "no_extracted_deliverable"
            exception = any(judge["evaluator_exception"] for judge in audit_receipt["judges"])
            measure_status = Counter(item["status"] for item in measurement["results"])
            measurement_path = source / measurement["source"]
            require(measurement_path.is_file(), f"Measurement source missing: {key}")
            raw_measurement = read_json(measurement_path)
            require(len(raw_measurement.get("results", [])) == len(measurement["results"]),
                    f"Measurement row count changed: {key}")
            rows.append({"task": task_id, "name": task["name"],
                         "arm": arm, "score": task[arm], "selected_arm": task["selected_arm"],
                         "selected": arm == task["selected_arm"], "selected_score": task["selected_score"],
                         "official_receipt": str(receipt_path.relative_to(source)),
                         "official_receipt_sha256": digest(receipt_path),
                         "native_score_file": str((source / "local/workbuddy/outcome/office50" /
                                                   actor["native_source"]).relative_to(source)),
                         "files": files, "file_count": len(files), "artifact_links": len(actor["links"]),
                         "score_links": len(actor["score_links"]),
                         "text_judgments": judgments[key], "artifact_chars": binding["artifact_chars"],
                         "input_chars": binding["input_chars"], "text_input_status": input_status,
                         "measurement_source": measurement["source"],
                         "measurement_rows": len(measurement["results"]),
                         "measurement_statuses": dict(measure_status),
                         "measurement_errors": sum(bool(item.get("error")) for item in measurement["results"]),
                         "review_status": measurement["review_status"],
                         "receipt_evaluator_exception": exception,
                         "receipt_missing_extracted_value": any(
                             judge["missing_extracted_value"] for judge in audit_receipt["judges"])})

    check_csv(report_dir, tasks, round_rows)
    require(len(round_rows) == 150 and invalid_count == 1 and detail["valid_score_records"] == 449,
            "Round/valid-score totals")
    require(len(rows) == 150 and file_count == 388 and artifact_links == 287 and score_links == 300,
            "Final archive totals")
    for arm in "ABC":
        require(abs(mean(task[arm] for task in tasks.values()) - detail["metrics"]["means"][arm]) < 1e-8,
                f"Mean mismatch: {arm}")
    require(abs(mean(task["selected_score"] for task in tasks.values()) -
                detail["metrics"]["historical_selected_mean"]) < 1e-8, "Selected mean mismatch")
    exceptions = [row for row in rows if row["receipt_evaluator_exception"]]
    require([(row["task"], row["arm"]) for row in exceptions] ==
            [("priority-sync-notification-pipeline", "A")], "Exception set changed")
    kept = [task for task in tasks.values() if task["task"] != exceptions[0]["task"]]
    sensitivity = {arm: round(mean(task[arm] - task["A"] for task in kept), 4)
                   for arm in ("B", "C")}
    sensitivity["selected"] = round(mean(task["selected_score"] - task["A"] for task in kept), 4)
    summary = {"tasks": len(tasks), "task_rounds": len(round_rows), "score_slots": 450,
               "valid_scores": 450 - invalid_count, "final_candidates": len(rows),
               "verified_report_files": file_count, "verified_artifact_links": artifact_links,
               "verified_score_links": score_links, "text_judgments": sum(judgments.values()),
               "text_input_status": dict(Counter(row["text_input_status"] for row in rows)),
               "zero_file_candidates": [f"{row['task']}:{row['arm']}" for row in rows if not row["files"]],
               "receipt_evaluator_exceptions": [f"{row['task']}:{row['arm']}" for row in exceptions],
               "receipt_missing_extracted_value": sum(row["receipt_missing_extracted_value"] for row in rows),
               "final_measurement_rows": sum(row["measurement_rows"] for row in rows),
               "selected_measurement_candidates": sum(row["selected"] and row["measurement_rows"] > 0
                                                      for row in rows),
               "means": {arm: round(mean(task[arm] for task in tasks.values()), 4) for arm in "ABC"},
               "selected_mean": round(mean(task["selected_score"] for task in tasks.values()), 4),
               "exception_task_removed_differences": sensitivity}
    payload = {"scope": "Complete Office50 archive reconciliation; no new grading or full-task correctness labels.",
               "source_hashes": {str(REPORT_REL / name): digest(report_dir / name)
                                 for name in ("report_detail.json", "results.json", "report_data.json",
                                              "scores.csv", "rounds.csv", "report_summary.csv")},
               "summary": summary, "candidates": rows}
    return payload


def write_outputs(payload):
    (ROOT / "data/workbuddy_full_archive_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    columns = ("task", "name", "arm", "score", "selected_arm", "selected",
               "selected_score", "official_receipt", "official_receipt_sha256", "native_score_file",
               "file_count", "artifact_links", "score_links", "text_judgments", "artifact_chars",
               "input_chars", "text_input_status", "measurement_source", "measurement_rows",
               "measurement_statuses", "measurement_errors", "review_status",
               "receipt_evaluator_exception", "receipt_missing_extracted_value", "files")
    with (ROOT / "data/workbuddy_full_archive_audit.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in payload["candidates"]:
            writer.writerow({key: json.dumps(row[key], ensure_ascii=False) if isinstance(row[key], (dict, list))
                             else row[key] for key in columns})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.source)
    write_outputs(result)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
