"""Candidate-bound public observations and deterministic fact checks.

Only a trusted adapter may bind observation descriptors. Values are read from
the supplied public evidence, never from model-authored verdicts or oracle data.
"""

from copy import deepcopy
from datetime import datetime

from reward_engine.core.outcome.identity import digest

CHECKER = "public.fact"


def select(value, path):
    if not isinstance(path, list) or len(path) > 16:
        raise ValueError("Observation path must have at most 16 components")
    for part in path:
        if isinstance(value, dict) and isinstance(part, str):
            value = value[part]
        elif isinstance(value, list) and type(part) is int and 0 <= part < len(value):
            value = value[part]
        else:
            raise ValueError("Observation path does not match public evidence")
    return value


def bind_observations(context, evidence, descriptors):
    """Freeze adapter-declared scope/time/coverage against exact public inputs.

Descriptors do not establish their own truth. In particular an intermediate
tool response must not be marked terminal, or a partial event log complete.
"""
    if len(context.get("products", [])) != 1:
        raise ValueError("Independent reward requires exactly one candidate")
    if "public_reward_observations" in context:
        raise ValueError("Observations are already frozen")
    allowed = {context["products"][0]["environment_ref"]}
    allowed.update(s["ref"] for s in context["products"][0]["sources"])
    rows, ids = [], set()
    for raw in descriptors:
        row = deepcopy(raw)
        if not isinstance(row.get("id"), str) or not row["id"] or row["id"] in ids:
            raise ValueError("Observation IDs must be unique")
        ids.add(row["id"])
        ref = row.get("source_ref")
        if ref not in allowed or ref not in evidence:
            raise ValueError("Observation must reference this candidate's public source/environment")
        if (not isinstance(row.get("object"), str) or not row["object"]
                or not isinstance(row.get("property"), str) or not row["property"]
                or row.get("phase") not in {"intermediate", "terminal", "history"}
                or type(row.get("sequence")) is not int or row["sequence"] < 0
                or type(row.get("complete")) is not bool
                or row.get("status") not in {"observed", "error", "unavailable"}
                or not isinstance(row.get("observed_at"), str) or not row["observed_at"]):
            raise ValueError("Observation needs object, property, time, phase and explicit coverage")
        if datetime.fromisoformat(row["observed_at"]).tzinfo is None:
            raise ValueError("Observation timestamp requires a timezone")
        if row["status"] == "observed":
            select(evidence[ref], row.get("path", []))
        row["source_sha256"] = digest(evidence[ref])
        rows.append(row)
    bound = deepcopy(context)
    refs = sorted(evidence)
    bound["public_reward_observations"] = {
        "input_refs": refs,
        "candidate_digest": digest({"task": context["task"], "products": context["products"],
                                    "evidence": {r: evidence[r] for r in refs}}),
        "rows": rows,
    }
    return bound


def validate_binding(context, evidence):
    bundle = context.get("public_reward_observations")
    if bundle is None:
        return []
    refs = bundle["input_refs"]
    actual = digest({"task": context["task"], "products": context["products"],
                     "evidence": {r: evidence.get(r) for r in refs}})
    if actual != bundle["candidate_digest"] or any(r not in evidence for r in refs):
        raise ValueError("Public observation candidate binding changed")
    return bundle["rows"]


def validate_predicate(predicate):
    if (not isinstance(predicate, dict)
            or not all(isinstance(predicate.get(k), str) and predicate[k] for k in ("object", "property"))
            or predicate.get("phase") not in {"terminal", "history"}
            or predicate.get("op") not in {"equals", "contains", "forbids"}
            or "expected" not in predicate):
        raise ValueError("Fact check needs object/property, terminal/history, op and expected")
    if predicate["phase"] == "history" and predicate["op"] != "forbids":
        raise ValueError("History checks support persistent prohibitions only")


def unknown(reason, *, error=False, refs=()):
    return {"status": "unknown", "reason": reason, "gap": "", "error": error,
            "unknown_kind": "execution" if error else "evidence", "evidence_refs": list(refs)}


def check_fact(predicate, context, evidence):
    validate_predicate(predicate)
    rows = [r for r in validate_binding(context, evidence)
            if (r["object"], r["property"]) == (predicate["object"], predicate["property"])]
    if not rows:
        return unknown("No candidate-bound public observation for the required fact")
    rows.sort(key=lambda r: (r["sequence"], r["id"]))

    def verdict(row):
        ref = row["source_ref"]
        if digest(evidence[ref]) != row["source_sha256"]:
            raise ValueError("Observation source changed")
        if row["status"] == "error":
            return unknown("Public observation failed", error=True, refs=[ref])
        if row["status"] == "unavailable":
            return unknown("Public observation is unavailable", refs=[ref])
        actual = select(evidence[ref], row.get("path", []))
        expected, op = predicate["expected"], predicate["op"]
        # Membership is structural; a substring match is not an exact path,
        # recipient, table, or recorded action.
        if op in {"contains", "forbids"}:
            if not isinstance(actual, list):
                return unknown("Membership check requires a structured list", error=True, refs=[ref])
            present = any(digest(v) == digest(expected) for v in actual)
            passed = present if op == "contains" else not present
            determined = present or row["complete"]
        else:
            passed = digest(actual) == digest(expected)
            determined = row["complete"]
        if not determined:
            return unknown("Partial evidence cannot establish absence or complete equality", refs=[ref])
        return {"status": "pass" if passed else "fail", "error": False,
                "reason": "Public fact satisfies requirement" if passed else "Public fact contradicts requirement",
                "gap": "" if passed else f"{predicate['object']} / {predicate['property']} violates {op}",
                "evidence_refs": [ref], "observation_ids": [row["id"]],
                "observed": deepcopy(actual), "expected": deepcopy(expected)}

    if predicate["phase"] == "history":
        history = [r for r in rows if r["phase"] == "history"]
        for row in history:
            result = verdict(row)
            if result["status"] == "fail":
                return result  # Later deletion/disclosure cannot undo an unauthorized action.
        if history and rows[-1]["sequence"] > history[-1]["sequence"]:
            return unknown("Action-history coverage predates a later observation")
        rows = history
    if not rows:
        return unknown("Required observation phase is unavailable")
    latest = [r for r in rows if r["sequence"] == rows[-1]["sequence"]]
    if len(latest) != 1:
        return unknown("Conflicting or ambiguous observation order")
    row = latest[0]
    if row["phase"] != predicate["phase"]:
        return unknown("Only intermediate or stale terminal evidence is available")
    result = verdict(row)
    if predicate["phase"] == "history" and result["status"] == "pass" and not row["complete"]:
        return unknown("Absence of prohibited actions requires complete history")
    return result


def install(registry):
    """Opt-in registry extension; no model calls, host IO, or recipe execution."""
    if CHECKER in registry.checkers:
        if registry.checkers[CHECKER][0] is not run:
            raise ValueError("Public fact checker cannot be replaced")
        return
    registry.register(CHECKER, run, "Compare candidate-bound public terminal facts or action history.")


async def run(test, context, evidence):
    return check_fact(test["arguments"], context, evidence)
