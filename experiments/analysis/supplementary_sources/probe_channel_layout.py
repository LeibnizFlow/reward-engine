"""Local deterministic extraction probe; never replace accepted benchmark scores.

--source replays the archived rule grader on the three original workbooks and
changes only the row at which its existing sheet reader starts. No model calls.
Default mode recomputes summaries from the retained per-check probe results.
"""

import argparse
import hashlib
import json
import re
import types
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "data/channel_layout_probe.json"


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def collect(source):
    task = "channel-period-compare-L4-017"
    cohort = source / "local/workbuddy/outcome/office50/sources/historical_run"
    dataset = cohort / "main28/dataset/wb-bench-office-v1.0/tasks" / task
    code = dataset / "tests/grading/eval_core.py"
    gold = dataset / "tests/gold/gold_answer.json"
    module = types.ModuleType("archived_channel_grader")
    exec(compile(code.read_text(), str(code), "exec"), module.__dict__)
    module.load_gold = lambda: read(gold)
    original_reader = module._read_sheet
    aliases = {x.lower() for x in module._SETTLE_COL_ALIASES["channel"]}

    class HeaderView:
        def __init__(self, ws):
            self.ws = ws

        def iter_rows(self, values_only=True):
            rows = list(self.ws.iter_rows(values_only=True))
            start = next(
                (
                    i
                    for i, row in enumerate(rows)
                    if row and str(row[0]).strip().lower() in aliases
                ),
                0,
            )
            return iter(rows[start:])

    result = []
    for arm, delivery in [("A", 1), ("B", 3), ("C", 1)]:
        actor = cohort / "channel_recovery/tasks" / task / "round_3/actors" / arm
        book = actor / f"snapshots/delivery-{delivery}/workspace/output/result.xlsx"
        verifier = next((actor / "harbor").glob("*/verifier"))
        # The inspected workbook must be exactly the file retained by the grader.
        assert sha(book) == sha(verifier / "raw_artifacts/comparison_workbook.xlsx")
        before = sha(book)
        module._read_sheet = original_reader
        original = module.evaluate(str(book))
        logged = read(verifier / "reward.json")
        assert (
            original["passed_count"] == logged["tests_passed"]
            and original["total_count"] == logged["tests_total"]
        )
        status = {
            m.group(1): m.group(2) == "PASSED"
            for m in re.finditer(
                r"test_atomic_check\[\d+_([^\]]+)\]\s+(PASSED|FAILED)",
                (verifier / "test_output.txt").read_text(),
            )
        }
        assert status == {c["name"]: c["passed"] for c in original["checks"]}
        module._read_sheet = lambda ws: original_reader(HeaderView(ws))
        shifted = module.evaluate(str(book))
        assert sha(book) == before
        wb = openpyxl.load_workbook(book, read_only=True, data_only=False)
        sheets = []
        for ws in wb:
            rows = list(ws.values)
            start = next(
                (
                    i
                    for i, row in enumerate(rows)
                    if row and str(row[0]).strip().lower() in aliases
                ),
                0,
            )
            sheets.append(
                dict(
                    sheet=ws.title,
                    detected_header_row=start + 1,
                    native_header=list(rows[0]),
                    detected_header=list(rows[start]),
                    formula_cells=sum(c.data_type == "f" for row in ws for c in row),
                )
            )
        wb.close()
        result.append(
            dict(
                arm=arm,
                workbook=str(book.relative_to(source)),
                sha256=before,
                verifier_hashes={
                    f: sha(verifier / f) for f in ["test_output.txt", "reward.json"]
                },
                sheets=sheets,
                original=original,
                header_row_probe=shifted,
            )
        )
    payload = dict(
        task=task,
        public_instruction=(dataset / "instruction.md").read_text(),
        selection="Previously documented largest runtime-selection gap; selected using archived external scores.",
        source_hashes={
            str(p.relative_to(source)): sha(p)
            for p in [code, gold, dataset / "instruction.md"]
        },
        intervention="Only the starting row passed to the native sheet reader changes to the first row whose first cell exactly matches an existing channel alias. No cells, column aliases, numeric rules, model judgments or accepted scores change.",
        arms=result,
    )
    SNAPSHOT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def build():
    data = read(SNAPSHOT)
    summary = []
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Candidate & Header row & Native checks & Header-row probe \\",
        r"\midrule",
    ]
    for row in data["arms"]:
        a = row["original"]
        b = row["header_row_probe"]
        changes = []
        assert [c["name"] for c in a["checks"]] == [c["name"] for c in b["checks"]]
        for x, y in zip(a["checks"], b["checks"], strict=False):
            if x["passed"] != y["passed"]:
                changes.append(
                    dict(name=x["name"], before=x["passed"], after=y["passed"])
                )
        summary.append(
            dict(
                arm=row["arm"],
                before=a["passed_count"],
                after=b["passed_count"],
                changed_checks=changes,
            )
        )
        positions = sorted({s["detected_header_row"] for s in row["sheets"]})
        label = {
            "A": "Task only",
            "B": "Reward-guided revision",
            "C": "Hypothesis guidance",
        }[row["arm"]]
        lines.append(
            f"{label} & {','.join(map(str, positions))} & {a['passed_count']}/{a['total_count']} & {b['passed_count']}/{b['total_count']}"
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "tables/workbuddy_layout_probe.tex").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    if args.source:
        collect(args.source.resolve())
    build()
