"""Rebuild a scoped requirement ledger from frozen Office artifacts and sources."""

import argparse
import csv
import hashlib
import json
import re
import subprocess
from collections import Counter
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "review-stage/REQUIREMENT_AUDIT_PROTOCOL.json"
SNAPSHOT = ROOT / "data/requirement_audit.json"
REFERENCE = ROOT / "data/safety_notice_reference.json"
RUNNER = Path("machine_path_78fc56f4d8a9ea39")
SCORE_ARCHIVE = ROOT / "data/office50.json"
TEXT_RECORDS = "experiments/workbuddy/studies/reward_pack_semantic_gap_20260918/output/semantic_records.json"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def workbook(path):
    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        return {sheet.title: list(sheet.values) for sheet in book}
    finally:
        book.close()


def result(ok, evidence, detail="", *, tested=True):
    return dict(
        reference="satisfied" if ok else "violated" if tested else "uncertain",
        evidence=evidence,
        detail=detail,
    )


def unknown(evidence, detail):
    return dict(reference="uncertain", evidence=evidence, detail=detail)


def header(rows):
    return [str(x) if x is not None else "" for x in rows[0]] if rows else []


def valid_csv_quoting(value):
    # csv.reader accepts bare quotes inside unquoted fields; RFC-style CSV does not.
    state = "start"
    for char in value:
        if state == "start":
            state = "quoted" if char == '"' else "start" if char in ",\r\n" else "plain"
        elif state == "plain":
            if char == '"':
                return False
            if char in ",\r\n":
                state = "start"
        elif state == "quoted":
            if char == '"':
                state = "after_quote"
        else:
            if char == '"':
                state = "quoted"
            elif char in ",\r\n":
                state = "start"
            else:
                return False
    return state != "quoted"


def decimal(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        return Decimal(str(value).replace(",", "").strip())
    except (InvalidOperation, ValueError):
        return None


def normalized_date(value):
    if isinstance(value, (date, datetime)):
        return value.strftime("%Y-%m-%d")
    s = str(value).strip()
    m = re.fullmatch(r"(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})(?:日)?", s)
    if m:
        try:
            return date(*map(int, m.groups())).isoformat()
        except ValueError:
            pass
    return s


def paths_for(evidence, source):
    out = {}
    for item in evidence["listed_files"]:
        path = source / item["bound_path"]
        assert path.is_file() and sha(path) == item["sha256"], path
        out[item["path"]] = path
    return out


def xia(evidence, files, source):
    root = source / evidence["historical_root"]
    expected = {"content_plan.csv", "xiaohongshu_note_pack.md"}
    actual = {p.name for p in root.iterdir() if p.is_file()}
    out = {
        "delivery": result(
            actual == expected,
            [str(root.relative_to(source))],
            f"root files: {sorted(actual)}",
        )
    }
    path = files.get("content_plan.csv")
    if path:
        raw = path.read_text(encoding="utf-8-sig")
        try:
            rows = list(csv.reader(raw.splitlines(keepends=True), strict=True))
            parser_error = None
        except csv.Error as exc:
            parser_error = str(exc)
            rows = list(csv.reader(raw.splitlines(keepends=True)))
        columns = "稿件代号,暂定标题,主要读者,生活场景,主素材线索,辅助素材线索,参考碎片或弃用碎片,采用与改写说明,本稿避开的旧意见或风险,与其他稿件的复用协调,互动边界,成稿位置,责编备注".split(
            ","
        )
        quoting = valid_csv_quoting(raw)
        ok = (
            parser_error is None
            and bool(rows)
            and rows[0] == columns
            and len(rows) == 5
            and all(len(row) == 13 for row in rows[1:])
            and quoting
        )
        out["csv_schema"] = result(
            ok,
            [str(path.relative_to(source))],
            f"row widths: {[len(row) for row in rows]}; valid quoting={quoting}; strict parser error={parser_error}; expected five rows of width 13",
        )
    else:
        out["csv_schema"] = result(False, [str(root.relative_to(source))], "CSV absent")
    out["notes"] = unknown(
        [str(root.relative_to(source))],
        "Editorial and safety content needs separate source-level review.",
    )
    return out


RESEARCH_COLUMNS = {
    "research_points": "point_id,canonical_company_id,canonical_company_name,canonical_sector_name,point_type,effective_source_time,summary_text,source_document,section_or_table,superseded_source",
    "factor_mentions": "factor_row_id,canonical_company_id,factor_id,factor_name,direction,horizon,evidence_text,effective_source_time,source_document,section_or_table,superseded_source",
    "risk_disclosures": "risk_id,canonical_company_id,risk_type,risk_text,source_document,section_or_table",
}


def research(evidence, files, source):
    root = source / evidence["historical_root"]
    xlsx = "output/research_factor_extract.xlsx"
    js = "output/research_factor_extract.json"
    out = {
        "delivery": result(
            xlsx in files and js in files,
            [str(root.relative_to(source))],
            f"present: {sorted(files)}",
        )
    }
    if xlsx in files and js in files:
        try:
            sheets = workbook(files[xlsx])
            content = read(files[js])
            good = (
                list(sheets) == list(RESEARCH_COLUMNS)
                and all(
                    header(sheets[name]) == cols.split(",")
                    for name, cols in RESEARCH_COLUMNS.items()
                )
                and isinstance(content.get("citation_map"), dict)
            )
            out["schema"] = result(
                good,
                [
                    str(files[xlsx].relative_to(source)),
                    str(files[js].relative_to(source)),
                ],
                f"sheets: {list(sheets)}; citation_map object: {isinstance(content.get('citation_map'), dict)}",
            )
        except (ValueError, KeyError, json.JSONDecodeError, OSError) as exc:
            out["schema"] = result(
                False, [str(root.relative_to(source))], f"parse error: {exc}"
            )
    else:
        out["schema"] = result(
            False, [str(root.relative_to(source))], "required artifact missing"
        )
    out["content"] = unknown(
        [str(root.relative_to(source))],
        "No source-level extraction or citation accuracy label assigned.",
    )
    return out


def drug(evidence, files, initial, source):
    rel = "output/result.xlsx"
    out = {
        "delivery": result(
            rel in files,
            [str((source / evidence["historical_root"]).relative_to(source))],
        )
    }
    if rel not in files:
        for key in ["partition", "nonpositive", "warning"]:
            out[key] = result(
                False,
                [str((source / evidence["historical_root"]).relative_to(source))],
                "output absent",
            )
        return out
    src = next(iter(workbook(initial / "药品出入库明细.xlsx").values()))
    sheets = workbook(files[rel])
    base = str(files[rel].relative_to(source))
    if (
        len(sheets) != 3
        or "出库盘亏" not in sheets
        or any(header(rows) != header(src) for rows in sheets.values())
    ):
        for key in ["partition", "nonpositive", "warning"]:
            out[key] = unknown(
                [base], "Three matching source-column sheets could not be identified."
            )
        return out
    srcrows = [tuple(row) for row in src[1:]]
    names = list(sheets)
    first = [tuple(row) for row in sheets["出库盘亏"][1:]]
    second = [tuple(row) for row in sheets[names[1]][1:]]
    third = [tuple(row) for row in sheets[names[2]][1:]]
    out["partition"] = result(
        Counter(srcrows) == Counter(first + second + third),
        [str((initial / "药品出入库明细.xlsx").relative_to(source)), base],
        f"source={len(srcrows)}, delivered={len(first) + len(second) + len(third)}; exact source rows preserved",
    )
    h = header(src)
    qty = h.index("STOCK_CHANGE")
    expiry = h.index("EXPIRY_ALERT_MONTHS")
    category = h.index("DRUG_CLASS")
    vendor = h.index("VENDOR")
    channel = h.index("CHANNEL")
    expected = Counter(
        row
        for row in srcrows
        if decimal(row[qty]) is not None and decimal(row[qty]) <= 0
    )
    out["nonpositive"] = result(
        Counter(first) == expected,
        [base, str((initial / "药品出入库明细.xlsx").relative_to(source))],
        f"expected={sum(expected.values())}, delivered={len(first)}",
    )
    agreed = Counter()
    ambiguous = 0
    for row in srcrows:
        value = decimal(row[qty])
        months = decimal(row[expiry])
        code = str(row[category] or "")
        supplier = str(row[vendor] or "")
        route = str(row[channel] or "")
        if value is None or months is None:
            ambiguous += 1
            continue
        if value <= 0:
            continue
        code_match = code in {"A01", "A02", "B01", "C01", "D01"} or code.startswith("N")
        other = "国药" in supplier or "华润" in supplier or route == "集采"
        reading1 = months < 3 and (code_match or other)
        reading2 = (months < 3 and code_match) or other
        if reading1 == reading2:
            agreed[row] += 1
        else:
            ambiguous += 1
    second_count = Counter(second)
    third_count = Counter(third)
    failures = sum(
        n
        for row, n in agreed.items()
        if ((second_count[row] > 0) == (third_count[row] > 0))
    )
    for row, n in agreed.items():
        months = decimal(row[expiry])
        code = str(row[category] or "")
        supplier = str(row[vendor] or "")
        route = str(row[channel] or "")
        should = months < 3 and (
            code in {"A01", "A02", "B01", "C01", "D01"}
            or code.startswith("N")
            or "国药" in supplier
            or "华润" in supplier
            or route == "集采"
        )
        if (second_count[row] > 0) != should:
            failures += n
    out["warning"] = result(
        failures == 0,
        [base, str((initial / "药品出入库明细.xlsx").relative_to(source))],
        f"agreed rows={sum(agreed.values())}, ambiguous rows={ambiguous}, failures={failures}",
    )
    return out


def fund(evidence, files, initial, source):
    rel = "output/fund_product_normalized.xlsx"
    out = {
        "delivery": result(
            rel in files,
            [str((source / evidence["historical_root"]).relative_to(source))],
        )
    }
    if rel not in files:
        for key in ["schema", "fee_units", "risk_mapping"]:
            out[key] = result(
                False,
                [str((source / evidence["historical_root"]).relative_to(source))],
                "output absent",
            )
        out["content"] = unknown([], "No output to compare.")
        return out
    sheets = workbook(files[rel])
    schema = read(initial / "standard_schema.json")["sheets"]
    base = str(files[rel].relative_to(source))
    required_order = [
        "product_comparison",
        "fee_schedule",
        "risk_level_mapping",
        "exceptions",
    ]
    good = list(sheets) == required_order and all(
        header(sheets[name]) == schema[name]["required_columns"]
        for name in required_order
    )
    out["schema"] = result(
        good,
        [base, str((initial / "standard_schema.json").relative_to(source))],
        f"sheets: {list(sheets)}",
    )
    if good:
        fees = [
            dict(zip(header(sheets["fee_schedule"]), row, strict=False))
            for row in sheets["fee_schedule"][1:]
        ]
        allowed = set(schema["fee_schedule"]["allowed_fee_units"])
        failed = []
        for i, row in enumerate(fees, 2):
            section = row["fee_section"]
            unit = row["fee_unit"]
            defaults = schema["fee_schedule"]["field_defaults_by_fee_section"].get(
                section, {}
            )
            if unit not in allowed or any(row.get(k) != v for k, v in defaults.items()):
                failed.append(i)
            if row["fee_status"] == "waived" and decimal(row["fee_value"]) != 0:
                failed.append(i)
        alias = workbook(initial / "raw_fee_table.xlsx")["product_aliases"]
        alias_ids = {row[0]: row[1] for row in alias[1:]}
        aliases = {row[2]: row[1] for row in alias[1:]}
        source_fees = workbook(initial / "raw_fee_table.xlsx")
        expected = Counter()
        missing_source = []

        def fee_key(product, share, section, fee_type, group, condition, value, unit):
            return (
                product,
                share,
                section,
                fee_type,
                group,
                condition,
                str(value.normalize()) if value is not None else "missing",
                unit,
            )

        for row in source_fees["sales_fee_export_raw"][1:]:
            name, share, fee_type, condition, raw, group, _, note = row
            if note and "promotional waiver" in note:
                continue
            product = alias_ids.get(name) or aliases.get(name)
            if raw == "不收取":
                value, unit = Decimal(0), "percent"
            elif str(raw).startswith("每笔"):
                value, unit = (
                    decimal(re.search(r"\d+(?:\.\d+)?", raw)[0]),
                    "CNY_per_txn",
                )
            else:
                value, unit = decimal(str(raw).rstrip("%")), "percent"
            assert product and value is not None
            expected[
                fee_key(
                    product, share, "sales_fee", fee_type, group, condition, value, unit
                )
            ] += 1
        for row in source_fees["operation_fee_export_raw"][1:]:
            name, share, fee_type, raw, raw_unit, _party, _date, _, _note = row
            if raw is None:
                missing_source.append((name, share, fee_type))
                continue
            product = alias_ids.get(name) or aliases.get(name)
            value = decimal(raw) / 100 if raw_unit == "bp/year" else decimal(raw)
            assert product and value is not None
            expected[
                fee_key(
                    product,
                    share,
                    "operation_fee",
                    fee_type,
                    "全部投资群体",
                    "annual_rate",
                    value,
                    "percent_per_year",
                )
            ] += 1
        for row in source_fees["comprehensive_fee_export_raw"][1:]:
            name, share, raw, raw_unit, *_ = row
            if raw is None:
                missing_source.append((name, share, "comprehensive_fee"))
                continue
            product = alias_ids.get(name) or aliases.get(name)
            assert product and raw_unit == "%/year"
            expected[
                fee_key(
                    product,
                    share,
                    "comprehensive_fee",
                    "any",
                    "全部投资群体",
                    "annualized_estimate",
                    decimal(raw),
                    "percent_per_year",
                )
            ] += 1
        observed = Counter()
        for row in fees:
            fee_type = (
                "any" if row["fee_section"] == "comprehensive_fee" else row["fee_type"]
            )
            observed[
                fee_key(
                    row["canonical_product_id"],
                    row["share_class"],
                    row["fee_section"],
                    fee_type,
                    row["investor_group"],
                    row["condition"],
                    decimal(row["fee_value"]),
                    row["fee_unit"],
                )
            ] += 1
        absent = expected - observed
        extra = observed - expected
        out["fee_units"] = result(
            not failed and not absent and not extra,
            [
                base,
                str((initial / "standard_schema.json").relative_to(source)),
                str((initial / "raw_fee_table.xlsx").relative_to(source)),
            ],
            f"{len(fees)} output fee rows; expected={sum(expected.values())}; invalid format/default rows={sorted(set(failed))}; missing source rows={sum(absent.values())}; extra output rows={sum(extra.values())}; missing raw source values={len(missing_source)}",
        )
        codebook = workbook(initial / "risk_codebook.xlsx")["risk_label_map"]
        mapping = {row[0]: tuple(row[1:4]) for row in codebook[1:]}
        risk = [
            dict(zip(header(sheets["risk_level_mapping"]), row, strict=False))
            for row in sheets["risk_level_mapping"][1:]
        ]
        bad = [
            i
            for i, row in enumerate(risk, 2)
            if row["source_risk_label"] not in mapping
            or (
                row["standard_risk_level"],
                row["risk_order"],
                row["sales_suitability_bucket"],
            )
            != mapping.get(row["source_risk_label"])
        ]
        out["risk_mapping"] = result(
            not bad,
            [base, str((initial / "risk_codebook.xlsx").relative_to(source))],
            f"{len(risk)} risk rows; mismatched rows: {bad}",
        )
    else:
        out["fee_units"] = unknown(
            [base], "Fee rows could not be mapped to the required schema."
        )
        out["risk_mapping"] = unknown(
            [base], "Risk rows could not be mapped to the required schema."
        )
    out["content"] = unknown(
        [base, str((initial / "raw_fee_table.xlsx").relative_to(source))],
        "PDF prose, source-row fee conversions and conflict priority not fully source-labeled.",
    )
    return out


SAFETY_COLUMNS = [
    "通报编号",
    "项目名称",
    "建设单位",
    "施工单位",
    "检查日期",
    "违规类型",
    "处罚金额",
    "整改期限",
]


def safety(evidence, files, initial, source):
    rel = "output/安全检查通报汇总.xlsx"
    root = source / evidence["historical_root"]
    out = {"delivery": result(rel in files, [str(root.relative_to(source))])}
    if rel not in files:
        out.update(
            {
                key: result(False, [str(root.relative_to(source))], "output absent")
                for key in ["records", "inspection_dates", "fines"]
            }
        )
        out["other_fields"] = unknown(
            [str(root.relative_to(source))], "No output to compare."
        )
        return out
    sheets = workbook(files[rel])
    base = str(files[rel].relative_to(source))
    rows = next(iter(sheets.values()))
    if header(rows) != SAFETY_COLUMNS:
        out["records"] = unknown([base], "Required eight-column header missing.")
        out["inspection_dates"] = unknown([base], "Source fields cannot be aligned.")
        out["fines"] = unknown([base], "Source fields cannot be aligned.")
        out["other_fields"] = unknown([base], "Source fields cannot be aligned.")
        return out
    refs = read(REFERENCE)["notices"]
    by_notice = {}
    for ref in refs.values():
        pdf = initial / ref["pdf"]
        extracted = subprocess.run(
            ["pdftotext", "-layout", str(pdf), "-"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        compact = re.sub(r"\s+", "", extracted)
        assert all(
            re.sub(r"\s+", "", ref[key]) in compact
            for key in ["date_anchor", "fine_anchor"]
        ), pdf
    for row in rows[1:]:
        m = re.search(r"(?<!\d)0?([1-5])(?!\d)", str(row[0]))
        if m:
            by_notice.setdefault(f"{int(m[1]):03d}", []).append(row)
    complete = len(rows) == 6 and all(len(by_notice.get(key, [])) == 1 for key in refs)
    out["records"] = result(
        complete,
        [base]
        + [str((initial / ref["pdf"]).relative_to(source)) for ref in refs.values()],
        f"rows={len(rows) - 1}, matched notices={sorted(by_notice)}",
    )
    if not complete:
        out["inspection_dates"] = unknown(
            [base], "Notice rows cannot be aligned one to one."
        )
        out["fines"] = unknown([base], "Notice rows cannot be aligned one to one.")
        out["other_fields"] = unknown(
            [base], "Notice rows cannot be aligned one to one."
        )
        return out
    checks = []
    for key, ref in refs.items():
        row = by_notice[key][0]
        actual_date = normalized_date(row[4])
        expected_date = ref["inspection_date"]
        actual_fine = (
            decimal(row[6]) if ref["fine_yuan"] != "未提及" else str(row[6]).strip()
        )
        date_ok = actual_date == expected_date
        fine_ok = actual_fine == (
            Decimal(ref["fine_yuan"])
            if isinstance(ref["fine_yuan"], int)
            else ref["fine_yuan"]
        )
        checks.append(
            dict(
                pdf=ref["pdf"],
                date_expected=expected_date,
                date_observed=actual_date,
                date_ok=date_ok,
                fine_expected=ref["fine_yuan"],
                fine_observed=str(row[6]),
                fine_ok=fine_ok,
                date_anchor=ref["date_anchor"],
                fine_anchor=ref["fine_anchor"],
            )
        )
    citation = [base, str(REFERENCE.relative_to(ROOT))] + [
        str((initial / ref["pdf"]).relative_to(source)) for ref in refs.values()
    ]
    out["inspection_dates"] = result(
        all(c["date_ok"] for c in checks),
        citation,
        f"date matches={sum(c['date_ok'] for c in checks)}/5",
    )
    out["fines"] = result(
        all(c["fine_ok"] for c in checks),
        citation,
        f"imposed-fine matches={sum(c['fine_ok'] for c in checks)}/5",
    )
    out["inspection_dates"]["checks"] = checks
    out["fines"]["checks"] = checks
    out["other_fields"] = unknown(
        [base]
        + [str((initial / ref["pdf"]).relative_to(source)) for ref in refs.values()],
        "Notice number, project, owner, constructor, violation categories and deadlines not source-transcribed in this audit.",
    )
    return out


def contract(evidence, files, initial, source):
    rel = "output/result.xlsx"
    root = source / evidence["historical_root"]
    base = str(root.relative_to(source))
    out = {"delivery": result(rel in files, [base])}
    if rel not in files:
        out.update(
            {key: result(False, [base], "output absent") for key in ["dedup", "month"]}
        )
        out["fields"] = unknown([base], "Source coverage not assessed.")
        return out
    sheets = workbook(files[rel])
    filename = str(files[rel].relative_to(source))
    main = next(
        (
            rows
            for name, rows in sheets.items()
            if name != "异常文件"
            and "合同编号" in header(rows)
            and "月份" in header(rows)
        ),
        None,
    )
    exceptions = sheets.get("异常文件")
    if not main or not exceptions:
        out["dedup"] = result(False, [filename], "Main or 异常文件 sheet missing")
        out["month"] = unknown([filename], "Main rows unavailable")
        out["fields"] = unknown([filename], "Source coverage not assessed.")
        return out
    h = header(main)
    ids = [
        str(row[h.index("合同编号")]).strip()
        for row in main[1:]
        if row[h.index("合同编号")] is not None
    ]
    out["dedup"] = result(
        len(ids) == len(set(ids))
        and all(key in h for key in ["甲方", "乙方", "签约日期", "金额", "付款方式"]),
        [filename],
        f"main rows={len(ids)}, unique contract IDs={len(set(ids))}, exception rows={len(exceptions) - 1}",
    )
    failures = []
    for i, row in enumerate(main[1:], 2):
        d = normalized_date(row[h.index("签约日期")])
        m = re.fullmatch(r"\d{4}-(\d{2})-\d{2}", d)
        month = decimal(row[h.index("月份")])
        expected = int(m[1]) if m else None
        if expected is None or month != expected:
            failures.append(
                dict(row=i, date=d, month=str(row[h.index("月份")]), expected=expected)
            )
    out["month"] = result(
        not failures, [filename], f"checked={len(main) - 1}; failures={failures[:12]}"
    )
    out["fields"] = unknown(
        [filename, str((initial / "采购合同").relative_to(source))],
        "Record-level source coverage and six extracted fields need multi-format source transcription; no label from header or dedup alone.",
    )
    return out


NEW_CHECKS = {
    "xiaohongshu-psychology-note-pack": xia,
    "research-factor-table-extract-L3-016": research,
    "drug-inventory-split-L2-006": drug,
    "fund-product-table-normalize-L3-017": fund,
    "safety-inspection-extract-L4-019": safety,
    "contract-extract-L3-014": contract,
}

TEXT_MATCH = {
    ("xiaohongshu-psychology-note-pack", "delivery"): (
        ["t-deliverable-files-scope"],
        "same",
    ),
    ("xiaohongshu-psychology-note-pack", "csv_schema"): (
        ["t-csv-schema-and-parse"],
        "same",
    ),
    ("research-factor-table-extract-L3-016", "delivery"): (
        ["T_C1_xlsx_structure", "T_C1_json_toplevel"],
        "partial",
    ),
    ("research-factor-table-extract-L3-016", "schema"): (
        ["T_C1_xlsx_structure", "T_C1_json_toplevel"],
        "same",
    ),
    ("drug-inventory-split-L2-006", "delivery"): (
        ["t1_result_file_exists_and_readable"],
        "partial",
    ),
    ("drug-inventory-split-L2-006", "partition"): (
        ["t5_mutually_exclusive_exhaustive_partition", "t6_cell_fidelity_by_record_id"],
        "partial",
    ),
    ("drug-inventory-split-L2-006", "nonpositive"): (
        ["t3_sheet1_exact_stock_loss_set"],
        "same",
    ),
    ("drug-inventory-split-L2-006", "warning"): (
        ["t4_sheet2_exact_alert_set"],
        "partial",
    ),
    ("fund-product-table-normalize-L3-017", "schema"): (
        ["T1_deliverable_structure"],
        "same",
    ),
    ("fund-product-table-normalize-L3-017", "fee_units"): (
        ["T4_fee_normalization"],
        "partial",
    ),
    ("fund-product-table-normalize-L3-017", "risk_mapping"): (
        ["T5_risk_mapping_codebook"],
        "partial",
    ),
    ("safety-inspection-extract-L4-019", "delivery"): (
        ["check_output_xlsx_exists_parseable_summary"],
        "same",
    ),
    ("safety-inspection-extract-L4-019", "records"): (
        ["check_five_notice_records_one_to_one"],
        "same",
    ),
    ("safety-inspection-extract-L4-019", "inspection_dates"): (
        ["check_inspection_date_normalization"],
        "same",
    ),
    ("safety-inspection-extract-L4-019", "fines"): (
        ["check_penalty_integer_yuan"],
        "same",
    ),
    ("contract-extract-L3-014", "delivery"): (
        ["T_C1_workbook_and_exception_sheet"],
        "partial",
    ),
    ("contract-extract-L3-014", "dedup"): (["T_C4_dedup_contract_number"], "partial"),
    ("contract-extract-L3-014", "month"): (["T_C5_month_column"], "partial"),
}


def text_mapping(task, arm, requirement, records):
    mapping = TEXT_MATCH.get((task, requirement))
    if not mapping:
        return dict(scope="unmapped", judgments=[])
    ids, scope = mapping
    selected = [
        dict(
            test_id=r["test_id"],
            verdict=r["verdict"],
            score=r["score"],
            artifact_chars=r["artifact_chars"],
            include_inputs=r["include_inputs"],
        )
        for r in records
        if r["test_id"] in ids
    ]
    assert len(selected) == len(ids), (task, arm, requirement, ids)
    if not all(r["artifact_chars"] and r["include_inputs"] for r in selected):
        scope = "input_unavailable"
    return dict(
        scope=scope,
        judgments=selected,
        source="experiments/workbuddy/studies/reward_pack_semantic_gap_20260918/output/semantic_records.json",
    )


def native_spots(task, arm, group, source_arm, native):
    out = []
    if task == "channel-period-compare-L4-017":
        for item in native:
            name = item["name"]
            m = re.fullmatch(
                r"settlement_(gmv_26q1|gmv_change|revenue_26q1|order_count)_spot_ch[12]",
                name,
            )
            if not m:
                continue
            channel = item["detail"].split(":", 1)[0]
            metric, period = {
                "gmv_26q1": ("GMV", "26Q1"),
                "gmv_change": ("GMV", "change"),
                "revenue_26q1": ("营收", "26Q1"),
                "order_count": ("订单量", "26Q1"),
            }[m[1]]
            hits = [
                c
                for c in source_arm["checks"]
                if c["dimension"] == "settlement"
                and c["channel"] == channel
                and c["metric"] == metric
                and c["period"] == period
            ]
            assert len(hits) == 1
            check = hits[0]
            out.append(
                dict(
                    task=task,
                    arm=arm,
                    group=group,
                    requirement_id=name,
                    public_requirement=f"{channel}: settlement {metric} {period}",
                    public_source="public_task:channel comparison",
                    reference="satisfied" if check["matches"] else "violated",
                    evidence=[
                        "data/channel_source_checks.json",
                        "data/channel_layout_probe.json",
                    ],
                    detail=f"source={check['expected']}; workbook={check['observed']}; original native={item['detail']}",
                    checks=[check],
                    native_mapping=dict(scope="same", name=name, passed=item["passed"]),
                    text_mapping=dict(scope="unmapped", judgments=[]),
                    internal_measurement="not_compared",
                    fault_category="extraction_failure"
                    if check["matches"] and not item["passed"]
                    else "none",
                    confirmed_effect="false_rejection"
                    if check["matches"] and not item["passed"]
                    else "none_inferred",
                )
            )
    elif task == "ticket-weekly-L3-010":
        weeks = sorted({c["key"].split("/")[0] for c in source_arm["numeric"]})
        for item in native:
            m = re.fullmatch(r"week([1-4])_(plan|actual|rate)_correct", item["name"])
            if not m:
                continue
            key = f"{weeks[int(m[1]) - 1]}/{m[2]}"
            hits = [c for c in source_arm["numeric"] if c["key"] == key]
            assert len(hits) == 1
            check = hits[0]
            out.append(
                dict(
                    task=task,
                    arm=arm,
                    group=group,
                    requirement_id=item["name"],
                    public_requirement=f"Week {m[1]} {m[2]} from original ticket records",
                    public_source="public_task:weekly aggregation",
                    reference="satisfied" if check["matches"] else "violated",
                    evidence=[
                        "data/additional_source_checks.json",
                        "data/additional_grader_audit.json",
                    ],
                    detail=f"source={check['expected']}; workbook={check['observed']}; original native={item['detail']}",
                    checks=[check],
                    native_mapping=dict(
                        scope="same", name=item["name"], passed=item["passed"]
                    ),
                    text_mapping=dict(scope="unmapped", judgments=[]),
                    internal_measurement="not_compared",
                    fault_category=(
                        "date_or_row_extraction" if arm == "B" else "percentage_unit"
                    )
                    if check["matches"] and not item["passed"]
                    else "none",
                    confirmed_effect="false_rejection"
                    if check["matches"] and not item["passed"]
                    else "none_inferred",
                )
            )
    return out


def collect(source):
    protocol = read(PROTOCOL)
    tasks = (
        protocol["development_tasks"]
        + protocol["diagnostic_tasks"]
        + protocol["validation_tasks"]
    )
    manifest = read(ROOT / "data/semantic_input_evidence.json")["artifacts"]
    evidence = {(r["task"], r["arm"]): r for r in manifest}
    scores = {r["task"]: r for r in read(SCORE_ARCHIVE)["tasks"]}
    raw_text = read(source / TEXT_RECORDS)
    text_index = {}
    for r in raw_text:
        if r["task"] in tasks:
            text_index.setdefault((r["task"], r["arm"]), []).append(
                dict(
                    test_id=r["test_id"],
                    criterion_id=r["criterion_id"],
                    question=r["question"],
                    verdict=r["result"]["parsed"]["verdict"],
                    score=r["result"]["parsed"]["score"],
                    reasoning=r["result"]["parsed"]["reasoning"],
                    artifact_chars=r["artifact_chars"],
                    include_inputs=r["include_inputs"],
                )
            )
    channel = read(ROOT / "data/channel_source_checks.json")
    official_path = source / "reports/workbuddy/office50/report_detail.json"
    official = {r["task"]: r for r in read(official_path)["tasks"]}
    measurements = {
        (r["task"], r["arm"]): r
        for r in read(ROOT / "data/archive_review_evidence.json")["measurements"]
        if r["benchmark"] == "WorkBuddy Office"
    }
    additional = {
        r["task"]: r for r in read(ROOT / "data/additional_source_checks.json")["tasks"]
    }
    channel_native = {
        r["arm"]: r["original"]["checks"]
        for r in read(ROOT / "data/channel_layout_probe.json")["arms"]
    }
    additional_native = {
        r["task"]: {a["arm"]: a["native"]["checks"] for a in r["arms"]}
        for r in read(ROOT / "data/additional_grader_audit.json")["tasks"]
    }
    channel_arms = {r["arm"]: r for r in channel["arms"]}
    catalog = []
    rows = []
    hashes = {}
    input_identity = {}
    for task in tasks:
        group = (
            "development"
            if task in protocol["development_tasks"]
            else "diagnostic"
            if task in protocol["diagnostic_tasks"]
            else "validation"
        )
        for arm in "ABC":
            item = evidence[task, arm]
            root = source / item["historical_root"]
            assert root.is_dir(), root
            initial = root.parents[2] / "snapshots/initial/workspace/input"
            assert initial.is_dir(), initial
            for p in initial.rglob("*"):
                if p.is_file():
                    digest = sha(p)
                    hashes[str(p.relative_to(source))] = digest
                    key = (task, str(p.relative_to(initial)))
                    assert input_identity.setdefault(key, digest) == digest, (
                        task,
                        arm,
                        p,
                    )
                    final = root / "input" / p.relative_to(initial)
                    assert final.is_file() and sha(final) == digest, final
            files = paths_for(item, source)
            hashes.update({str(p.relative_to(source)): sha(p) for p in files.values()})
            if task in NEW_CHECKS:
                outcomes = (
                    NEW_CHECKS[task](item, files, source)
                    if task in protocol["diagnostic_tasks"]
                    else NEW_CHECKS[task](item, files, initial, source)
                )
            elif task == "channel-period-compare-L4-017":
                checks = channel_arms[arm]["checks"]
                assert len(checks) == 300
                outcomes = {
                    "source_arithmetic": result(
                        all(c["matches"] for c in checks),
                        [
                            "data/channel_source_checks.json",
                            str(next(iter(files.values())).relative_to(source)),
                        ],
                        f"{sum(c['matches'] for c in checks)}/300 public-source numeric comparisons; commission, headcount and relative rates excluded",
                    )
                }
            else:
                checked = next(a for a in additional[task]["arms"] if a["arm"] == arm)
                comparisons = checked["numeric"] + checked["sets"]
                outcomes = {
                    "source_arithmetic": result(
                        all(c["matches"] for c in comparisons),
                        [
                            "data/additional_source_checks.json",
                            str(next(iter(files.values())).relative_to(source)),
                        ],
                        f"{sum(c['matches'] for c in comparisons)}/{len(comparisons)} scoped numeric/set comparisons",
                    )
                }
            if task in protocol["requirements"]:
                requirements = protocol["requirements"][task]
            else:
                requirements = [
                    {
                        "id": "source_arithmetic",
                        "source": "frozen public-source check snapshot",
                        "rule": "The explicitly enumerated arithmetic and membership checks agree with original inputs.",
                    }
                ]
            native = (
                channel_native[arm]
                if task == "channel-period-compare-L4-017"
                else additional_native.get(task, {}).get(arm, [])
            )
            official_actor = official[task]["rounds"][-1]["actors"][arm]
            assert (
                scores[task][arm] == official[task][arm] == official_actor["score"]
            ), (task, arm)
            assert scores[task]["selected_arm"] == official[task]["selected_arm"]
            assert scores[task]["selected_score"] == official[task]["selected_score"]
            measurement = measurements[task, arm]
            catalog.append(
                dict(
                    task=task,
                    arm=arm,
                    native_checks=native,
                    official_status=official_actor["official"],
                    official_verdicts=official_actor.get("verdicts", []),
                    final_response=item["final_response"],
                    cached_text=text_index.get((task, arm), []),
                    internal_measurement=dict(
                        review_status=measurement["review_status"],
                        source=measurement["source"],
                        results=measurement["results"],
                    ),
                    original_score=scores[task][arm],
                    selected_arm=scores[task]["selected_arm"],
                    selected_score=scores[task]["selected_score"],
                    source_artifacts=[
                        dict(path=f["bound_path"], sha256=f["sha256"])
                        for f in item["listed_files"]
                    ],
                )
            )
            for req in requirements:
                outcome = outcomes[req["id"]]
                direct_text = text_mapping(
                    task, arm, req["id"], text_index.get((task, arm), [])
                )
                rows.append(
                    dict(
                        task=task,
                        arm=arm,
                        group=group,
                        requirement_id=req["id"],
                        public_requirement=req["rule"],
                        public_source=req["source"],
                        reference=outcome["reference"],
                        evidence=outcome["evidence"],
                        detail=outcome["detail"],
                        checks=outcome.get("checks", []),
                        native_mapping=dict(
                            scope="partial" if native else "composite_only",
                            original_check_names=[n["name"] for n in native],
                            original_score=scores[task][arm],
                        ),
                        text_mapping=direct_text,
                        internal_measurement=dict(
                            scope="unmapped",
                            review_status=measurement["review_status"],
                            result_count=len(measurement["results"]),
                            source=measurement["source"],
                        ),
                        fault_category=(
                            "artifact_error"
                            if task
                            in [
                                "xiaohongshu-psychology-note-pack",
                                "fund-product-table-normalize-L3-017",
                            ]
                            else "non_delivery"
                        )
                        if outcome["reference"] == "violated"
                        else "insufficient_evidence"
                        if outcome["reference"] == "uncertain"
                        else "none",
                        confirmed_effect="none_inferred",
                    )
                )
            if task == "channel-period-compare-L4-017":
                rows.extend(native_spots(task, arm, group, channel_arms[arm], native))
            elif task == "ticket-weekly-L3-010":
                checked = next(a for a in additional[task]["arms"] if a["arm"] == arm)
                rows.extend(native_spots(task, arm, group, checked, native))
    assert len(catalog) == 30 and len(
        {(r["task"], r["arm"], r["requirement_id"]) for r in rows}
    ) == len(rows)
    data = dict(
        scope=protocol["scope"],
        protocol_sha256=sha(PROTOCOL),
        safety_reference_sha256=sha(REFERENCE),
        archive_hashes={
            "office50.json": sha(SCORE_ARCHIVE),
            "semantic_records.json": sha(source / TEXT_RECORDS),
            "report_detail.json": sha(official_path),
            "channel_source_checks.json": sha(ROOT / "data/channel_source_checks.json"),
            "additional_source_checks.json": sha(
                ROOT / "data/additional_source_checks.json"
            ),
        },
        source_hashes=hashes,
        rows=rows,
        judgment_catalog=catalog,
        accepted_scores={
            task: {
                key: scores[task][key]
                for key in ["A", "B", "C", "selected_arm", "selected_score"]
            }
            for task in tasks
        },
    )
    SNAPSHOT.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def build():
    data = read(SNAPSHOT)
    assert data["protocol_sha256"] == sha(PROTOCOL) and data[
        "safety_reference_sha256"
    ] == sha(REFERENCE)
    rows = data["rows"]
    catalog = data["judgment_catalog"]
    assert len(catalog) == 30
    with (ROOT / "data/requirement_audit.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "task",
                "arm",
                "group",
                "requirement_id",
                "reference",
                "public_requirement",
                "public_source",
                "detail",
                "evidence",
                "native_mapping",
                "text_mapping",
                "internal_measurement",
                "fault_category",
                "confirmed_effect",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: json.dumps(row[k], ensure_ascii=False)
                    if isinstance(row[k], (list, dict))
                    else row[k]
                    for k in writer.fieldnames
                }
            )
    summary = []
    for task in data["accepted_scores"]:
        group = next(r["group"] for r in rows if r["task"] == task)
        rr = [r for r in rows if r["task"] == task]
        counts = Counter(r["reference"] for r in rr)
        summary.append(
            dict(
                task=task,
                group=group,
                candidates=3,
                requirements=len(rr),
                **{k: counts[k] for k in ["satisfied", "violated", "uncertain"]},
            )
        )
    comparable_native = [
        r
        for r in rows
        if r["native_mapping"].get("scope") == "same" and r["reference"] != "uncertain"
    ]
    false_rejections = [
        r
        for r in comparable_native
        if r["reference"] == "satisfied" and not r["native_mapping"]["passed"]
    ]
    false_acceptances = [
        r
        for r in comparable_native
        if r["reference"] == "violated" and r["native_mapping"]["passed"]
    ]
    comparable_text = [
        r
        for r in rows
        if r["text_mapping"]["scope"] == "same"
        and r["reference"] != "uncertain"
        and all(
            j["verdict"] in ["full", "fail"] for j in r["text_mapping"]["judgments"]
        )
    ]
    text_false_rejections = [
        r
        for r in comparable_text
        if r["reference"] == "satisfied"
        and any(j["verdict"] == "fail" for j in r["text_mapping"]["judgments"])
    ]
    text_false_acceptances = [
        r
        for r in comparable_text
        if r["reference"] == "violated"
        and all(j["verdict"] == "full" for j in r["text_mapping"]["judgments"])
    ]
    stats = dict(
        tasks=len(summary),
        candidates=len(catalog),
        requirements=len(rows),
        satisfied=sum(r["satisfied"] for r in summary),
        violated=sum(r["violated"] for r in summary),
        uncertain=sum(r["uncertain"] for r in summary),
        comparable_native_checks=len(comparable_native),
        native_false_rejections=len(false_rejections),
        native_false_acceptances=len(false_acceptances),
        comparable_native_violations=sum(
            r["reference"] == "violated" for r in comparable_native
        ),
        comparable_cached_text_checks=len(comparable_text),
        text_false_rejections=len(text_false_rejections),
        text_false_acceptances=len(text_false_acceptances),
        comparable_cached_text_violations=sum(
            r["reference"] == "violated" for r in comparable_text
        ),
        known_violation_tasks=sorted(
            {r["task"] for r in rows if r["reference"] == "violated"}
        ),
        known_violation_candidates=sorted(
            {r["task"] + ":" + r["arm"] for r in rows if r["reference"] == "violated"}
        ),
        native_false_rejection_cases=dict(
            Counter(r["task"] + ":" + r["arm"] for r in false_rejections)
        ),
        by_task=summary,
        interpretation="Requirement rows within a task and candidate are dependent. No population rate, revised task score or new ranking is inferred.",
    )
    (ROOT / "data/requirement_audit_summary.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n"
    )
    labels = {
        "channel-period-compare-L4-017": "Channel",
        "health-quote-L2-003": "Quote",
        "procurement-reconcile-L3-015": "Procurement",
        "ticket-weekly-L3-010": "Weekly tickets",
        "xiaohongshu-psychology-note-pack": "Editing package",
        "research-factor-table-extract-L3-016": "Research extract",
        "drug-inventory-split-L2-006": "Inventory split",
        "fund-product-table-normalize-L3-017": "Fund table",
        "safety-inspection-extract-L4-019": "Safety notices",
        "contract-extract-L3-014": "Contracts",
    }
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Task & Role & Candidates & Satisfied & Violated & Uncertain \\",
        r"\midrule",
    ]
    for r in summary:
        role = {"development": "Dev.", "diagnostic": "Diag.", "validation": "Fixed"}[
            r["group"]
        ]
        lines.append(
            f"{labels[r['task']]} & {role} & 3 & {r['satisfied']} & {r['violated']} & {r['uncertain']}"
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (ROOT / "tables/workbuddy_requirement_audit.tex").write_text(
        "\n".join(lines) + "\n"
    )
    print(
        json.dumps(
            {k: v for k, v in stats.items() if k != "by_task"},
            ensure_ascii=False,
            indent=2,
        )
    )
    for r in summary:
        print(
            f"{r['task']}: {r['satisfied']} satisfied, {r['violated']} violated, {r['uncertain']} uncertain"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    if args.source:
        collect(args.source.resolve())
    build()
