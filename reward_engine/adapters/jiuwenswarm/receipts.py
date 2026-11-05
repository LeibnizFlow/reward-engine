"""Count only tool outcomes explicitly reported by Process events."""

import json


def tool_failure(value):
    if isinstance(value, str):
        if value.startswith("[ERROR]:"):
            return True
        try:
            value = json.loads(value)
        except ValueError:
            return None
    if not isinstance(value, dict):
        return None
    if (
        value.get("success") is False
        or value.get("is_error") is True
        or value.get("status") in ("error", "failed", "failure")
    ):
        return True
    for key in ("exit_code", "exitCode", "returncode", "return_code"):
        if type(value.get(key)) is int:
            return value[key] != 0
    if value.get("success") is True or value.get("is_error") is False:
        return False
    if "result" in value:
        return tool_failure(value["result"])
    return None


def tool_outcomes(events):
    results = [
        e.get("payload") or {} for e in events if e["event_type"] == "chat.tool_result"
    ]
    outcomes = [tool_failure(r.get("raw_output", r.get("result"))) for r in results]
    return {
        "results": len(results),
        "reported_errors": outcomes.count(True),
        "unknown": outcomes.count(None),
    }
