"""Native verdict errors are unavailable measurements, never ordinary zero scores."""


def scoring_errors(detail):
    errors = []
    seen = set()
    verdicts = list(detail.get("verdicts", []))
    for judge in detail.get("judges", []):
        verdicts.extend(judge.get("verdicts", []))
        if judge.get("errors") or judge.get("ok") is False:
            errors.append({"judge": judge.get("judge_name"), "errors": judge.get("errors", []),
                           "ok": judge.get("ok")})
    for verdict in verdicts:
        kind = verdict.get("metadata", {}).get("error_type")
        key = (verdict.get("item_id"), kind)
        if kind and key not in seen:
            seen.add(key)
            errors.append({"item_id": key[0], "error_type": kind, "reason": verdict.get("reason")})
    return errors
