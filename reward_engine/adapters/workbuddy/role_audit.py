"""Read-only role diagnostics and observed upstream accounting, including legacy gaps."""

import math
import re
from collections import defaultdict
from pathlib import Path

from reward_engine.adapters.workbuddy.io import file_hash, read
from reward_engine.adapters.workbuddy.role_requests import gateway_receipt_path


def task_failures(task_root):
    result = []
    for pattern in (
        "round_*/models/*/*/failure.json",
        "round_*/models/*/*/contract_rejected.json",
    ):
        for path in sorted(Path(task_root).glob(pattern)):
            row = read(path)
            identity_path = path.parent / "request_identity.json"
            identity = read(identity_path) if identity_path.exists() else {}
            result.append(
                {
                    "role": path.parent.parent.name,
                    "receipt": str(path),
                    "error_type": row.get("error_type"),
                    "reason": row.get("reason"),
                    "details": row.get("details"),
                    **identity,
                }
            )
    return result


def request_audit(root, *, config_identity=None):
    """Never infer a logical budget from uncorrelated historical HTTP calls."""
    root = Path(root)
    operations = defaultdict(
        lambda: {
            "gateway_calls": 0,
            "upstream_attempts": 0,
            "rejected_before_execution": 0,
            "receipts": [],
            "physical_ids": [],
            "unproven_receipts": [],
            "records": [],
        }
    )
    legacy = {
        "gateway_calls": 0,
        "observed_upstream_attempts": 0,
        "unproven_attempt_count": [],
        "logical_budget": "unproven: no operation IDs",
    }
    for path in sorted((root / "transport" / "roles").glob("*/receipt.json")):
        row = read(path)
        key = row.get("logical_operation_id")
        attempts = row.get("upstream_attempts", row.get("attempts"))
        if not key:
            legacy["gateway_calls"] += 1
            if type(attempts) is int:
                legacy["observed_upstream_attempts"] += attempts
            else:
                legacy["unproven_attempt_count"].append(str(path))
            continue
        entry = operations[key]
        entry["records"].append((str(path), row))
        entry["gateway_calls"] += 1
        entry["receipts"].append(str(path))
        entry["physical_ids"].append(row.get("physical_request_id"))
        entry["rejected_before_execution"] += (
            row.get("status") == "rejected_before_execution"
        )
        if type(attempts) is int and attempts in (0, 1):
            entry["upstream_attempts"] += attempts
        else:
            entry["unproven_receipts"].append(str(path))
        if row.get("status") not in {
            "completed",
            "failed",
            "rejected_before_execution",
        }:
            entry["unproven_receipts"].append(str(path))
        if (row.get("status") == "completed" and attempts != 1
                or row.get("status") == "rejected_before_execution" and attempts != 0):
            entry["unproven_receipts"].append(str(path))
    # Include role operations that never reached the gateway; absence of a
    # gateway receipt cannot silently turn an uncertain dispatch into zero calls.
    indexes = {}
    for path in (root / "transport" / "role_budget_index").glob("*.json"):
        indexes[path.stem] = read(path)
        operations[path.stem]
    current = []
    for operation, entry in operations.items():
        entry["attempt_count_verified"] = (
            not entry["unproven_receipts"] and entry["upstream_attempts"] <= 8
        )
        records = entry.pop("records")
        index = indexes.get(operation)
        strict = bool(index) or any(r.get("contract_policy") == "canonical_recovery_v1" for _, r in records)
        issues = []
        entry["audit_issues"] = issues
        if not strict:
            issues.append("Historical attempt count available; complete correlation/time proof unavailable")
            entry["budget_verified"] = False
            continue
        # A mismatch must fail the selected revision's audit, not make its
        # operation disappear from the scope being verified.
        selected = (config_identity is None
                    or index is not None and index.get("config_identity") == config_identity
                    or any(r.get("config_identity") == config_identity for _, r in records))
        if selected:
            current.append(entry)
        try:
            if index is None:
                raise ValueError("Missing terminal role budget index")
            path = Path(index["receipt"])
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("Role budget receipt lies outside campaign")
            budget = read(path)
            entry["role_budget_receipt"] = str(path)
            entry["config_identity"] = budget.get("config_identity")
            if not selected and budget.get("config_identity") == config_identity:
                current.append(entry)
                selected = True
            if budget.get("logical_operation_id") != operation or budget.get("contract_policy") != "canonical_recovery_v1":
                issues.append("Role operation identity/policy mismatch")
            if index.get("config_identity") != budget.get("config_identity"):
                issues.append("Role index configuration mismatch")
            start, deadline = budget.get("started_at"), budget.get("deadline")
            if not all(type(t) in (int, float) and math.isfinite(t) for t in (start, deadline)) or not 0 < deadline - start <= 1800:
                raise ValueError("Missing or invalid original 1800-second interval")
            calls = budget.get("receipts", [])
            dispatched = [r for r in calls if r.get("gateway_calls") == 1]
            ids = [r.get("physical_request_id") for r in calls]
            if any(not isinstance(v, str) or not re.fullmatch(r"[0-9a-f]{32}", v) for v in ids) or len(ids) != len(set(ids)):
                issues.append("Invalid or repeated physical request identity")
            if budget.get("requests") != len(dispatched) or len(dispatched) > 8 or budget.get("transport_recoveries", 0) > 1 or budget.get("syntax_repairs", 0) > 1 or budget.get("contract_repairs", 0) > 3:
                issues.append("Role request/recovery allowance mismatch")
            by_id = {r.get("physical_request_id"): (p, r) for p, r in records}
            if len(by_id) != len(records) or set(by_id) != {r["physical_request_id"] for r in dispatched}:
                issues.append("Gateway and role dispatches are not one-to-one")
            admission_path = root / "transport" / "role_operations" / operation / "admissions.json"
            ledger = read(admission_path) if admission_path.exists() else {"requests": {}}
            admitted = ledger.get("requests", {})
            if admitted and (ledger.get("deadline") != deadline or ledger.get("started_at") != start or len(admitted) > 8):
                issues.append("Gateway admission interval/count mismatch")
            if admitted and ledger.get("config_identity") != budget.get("config_identity"):
                issues.append("Gateway admission configuration mismatch")
            if set(admitted) - set(by_id):
                issues.append("Admitted physical request lacks a gateway receipt")
            for r in dispatched:
                physical = r["physical_request_id"]
                if physical not in by_id:
                    continue
                gateway_path, gateway = by_id[physical]
                pointer = gateway_receipt_path(root, budget, r)
                if str(pointer) != gateway_path or r.get("deadline") != deadline:
                    issues.append("Role/gateway receipt pointer or deadline mismatch")
                elif r.get("gateway_receipt") is None:
                    entry.setdefault("recovered_gateway_pointers", []).append({
                        "physical_request_id": physical, "gateway_receipt": str(pointer),
                        "gateway_sha256": file_hash(pointer),
                        "admission_ledger": str(admission_path),
                        "admission_sha256": file_hash(admission_path),
                    })
                if physical in admitted:
                    if gateway.get("config_identity") != budget.get("config_identity"):
                        issues.append("Gateway receipt configuration mismatch")
                    if admitted[physical] != gateway_path or gateway.get("deadline") != deadline or gateway.get("operation_started_at") != start:
                        issues.append("Gateway admission provenance mismatch")
                    at = ledger.get("admitted_at", {}).get(physical)
                    if type(at) not in (int, float) or not start <= at < deadline or at != gateway.get("admitted_at"):
                        issues.append("Unproven admission time")
                elif gateway.get("status") != "rejected_before_execution" or gateway.get("upstream_attempts") != 0:
                    issues.append("Unadmitted request may have executed")
                if gateway.get("upstream_attempts") == 1:
                    at = gateway.get("upstream_started_at")
                    if type(at) not in (int, float) or not start <= at < deadline:
                        issues.append("Unproven upstream start time")
                    end = gateway.get("finished_at")
                    if type(end) not in (int, float) or end < at or (gateway.get("status") == "completed" and end > deadline):
                        issues.append("Unproven upstream completion time")
            if not entry["attempt_count_verified"]:
                issues.append("Upstream attempt count unproven or exceeds eight")
        except (OSError, ValueError, KeyError, TypeError) as error:
            issues.append(str(error))
        entry["budget_verified"] = not issues
    return {
        "operations": dict(operations),
        "historical": legacy,
        "all_correlated_budgets_verified": all(
            r["budget_verified"] for r in operations.values()
        ),
        "all_current_contract_budgets_verified": all(r["budget_verified"] for r in current),
        "current_contract_operations": len(current),
    }
