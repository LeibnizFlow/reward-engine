"""Persistent gateway admission for single-attempt internal role requests."""

import fcntl
import math
import re
import time
from pathlib import Path

from reward_engine.adapters.workbuddy.io import read, write


def gateway_receipt_path(root, budget, receipt):
    """Resolve a lost client pointer only through complete gateway admission proof."""
    if receipt.get("gateway_receipt") is not None:
        return Path(receipt["gateway_receipt"])
    operation, physical = budget.get("logical_operation_id"), receipt.get("physical_request_id")
    if (not all(isinstance(v, str) and re.fullmatch(r"[0-9a-f]{32}", v)
                for v in (operation, physical))
            or receipt.get("logical_operation_id") != operation
            or receipt.get("gateway_calls") != 1
            or budget.get("contract_policy") != "canonical_recovery_v1"):
        raise ValueError("Missing canonical dispatch identity for gateway correlation")
    root = Path(root).resolve()
    ledger = read(root / "transport/role_operations" / operation / "admissions.json")
    start, deadline = budget.get("started_at"), budget.get("deadline")
    if (not all(type(v) in (int, float) and math.isfinite(v) for v in (start, deadline))
            or not 0 < deadline - start <= 1800
            or receipt.get("deadline") != deadline
            or ledger.get("started_at") != start or ledger.get("deadline") != deadline
            or ledger.get("config_identity") != budget.get("config_identity")):
        raise ValueError("Gateway correlation interval or configuration differs")
    requests = ledger.get("requests", {})
    if not isinstance(requests, dict) or not 1 <= len(requests) <= 8 or physical not in requests:
        raise ValueError("Gateway admission does not uniquely identify the physical request")
    path = Path(requests[physical])
    try:
        relative = path.resolve().relative_to(root / "transport/roles")
    except ValueError:
        raise ValueError("Gateway correlation points outside campaign transport") from None
    if (not path.is_absolute() or path.is_symlink() or not path.is_file()
            or len(relative.parts) != 2 or relative.name != "receipt.json"
            or not re.fullmatch(r"[0-9a-f]{32}", relative.parts[0])):
        raise ValueError("Invalid correlated gateway receipt path")
    gateway = read(path)
    if (gateway.get("logical_operation_id") != operation
            or gateway.get("physical_request_id") != physical
            or gateway.get("config_identity") != budget.get("config_identity")
            or gateway.get("contract_policy") != budget.get("contract_policy")
            or gateway.get("deadline") != deadline or gateway.get("operation_started_at") != start):
        raise ValueError("Gateway correlation identity or provenance differs")
    admitted = ledger.get("admitted_at", {}).get(physical)
    finished = gateway.get("finished_at")
    if (not all(type(v) in (int, float) and math.isfinite(v) for v in (admitted, finished))
            or not start <= admitted < deadline or admitted != gateway.get("admitted_at")
            or finished < admitted):
        raise ValueError("Unproven correlated gateway admission or termination time")
    status = gateway.get("status")
    attempts = gateway.get("upstream_attempts")
    if (type(attempts) is not int or attempts not in (0, 1)
            or type(gateway.get("attempts")) is not int or gateway["attempts"] != attempts
            or status not in {"completed", "failed", "rejected_before_execution"}
            or status == "completed" and attempts != 1
            or status == "rejected_before_execution" and attempts != 0):
        raise ValueError("Unproven correlated gateway execution count or terminal status")
    if attempts:
        executed = gateway.get("upstream_started_at")
        if (type(executed) not in (int, float) or not math.isfinite(executed)
                or not admitted <= executed < deadline or finished < executed
                or status == "completed" and finished > deadline):
            raise ValueError("Correlated upstream execution falls outside its original interval")
    return path


def reserve(config, headers, receipt_path):
    """Reserve at most eight upstream attempts, including across gateway restarts."""
    operation = headers.get("x-role-operation-id", "")
    physical = headers.get("x-role-request-id", "")
    if not all(re.fullmatch(r"[0-9a-f]{32}", v) for v in (operation, physical)):
        raise ValueError("Canonical roles require operation and physical request IDs")
    try:
        deadline = float(headers.get("x-role-operation-deadline", ""))
    except ValueError:
        raise ValueError("Invalid operation deadline") from None
    now = time.time()
    if not math.isfinite(deadline) or not now < deadline <= now + 1800:
        raise ValueError("Operation deadline expired or exceeds 1800 seconds")
    strict = config.get("role_contract_policy") == "canonical_recovery_v1"
    started_at = None
    if strict:
        try:
            started_at = float(headers.get("x-role-operation-started", ""))
        except ValueError:
            raise ValueError("Missing operation start time") from None
        if not math.isfinite(started_at) or not started_at <= now < deadline or not 0 < deadline - started_at <= 1800:
            raise ValueError("Operation must use its original 1800-second deadline")
    root = Path(config["output"]) / "transport" / "role_operations" / operation
    root.mkdir(parents=True, exist_ok=True)
    with (root / "admission.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = root / "admissions.json"
        ledger = read(path) if path.exists() else {"deadline": deadline, "requests": {}}
        if ledger["deadline"] != deadline:
            raise ValueError("Operation deadline changed")
        if strict:
            if ledger.get("started_at", started_at) != started_at:
                raise ValueError("Operation start time changed")
            ledger.update(started_at=started_at, contract_policy=config["role_contract_policy"],
                          config_identity=config["identity"])
            ledger.setdefault("admitted_at", {})[physical] = now
        if physical in ledger["requests"]:
            raise ValueError("Physical request ID already dispatched")
        if len(ledger["requests"]) >= 8:
            raise ValueError("Operation exhausted eight gateway admissions")
        ledger["requests"][physical] = str(receipt_path)
        write(path, ledger)
    return {
        "logical_operation_id": operation,
        "physical_request_id": physical,
        "deadline": deadline,
        "request_ordinal": len(ledger["requests"]),
        **({"operation_started_at": started_at, "admitted_at": now,
            "contract_policy": config["role_contract_policy"], "config_identity": config["identity"]} if strict else {}),
    }
