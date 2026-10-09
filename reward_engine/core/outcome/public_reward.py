"""Versioned, opt-in independent reward; legacy checks and selection stay intact."""

import re
from copy import deepcopy
from datetime import date as calendar_date

from reward_engine.core.outcome.identity import digest
from reward_engine.core.outcome.public_facts import (
    CHECKER,
    check_fact,
    validate_predicate,
)
from reward_engine.core.outcome.task_inputs import refs as task_input_refs

VERSION = "public-task-reward-v1"


def public_requirements_digest(context, evidence):
    refs = {"task"} | task_input_refs(context)
    refs.update(u["ref"] for p in context["products"] for u in p["user_updates"])
    return digest({r: evidence[r] for r in sorted(refs)})


def validate_sources(sources, context, evidence):
    allowed = {"task"} | task_input_refs(context)
    allowed.update(u["ref"] for p in context["products"] for u in p["user_updates"])
    if not isinstance(sources, list) or not sources:
        raise ValueError("Scored requirements need public Task/user sources")
    for source in sources:
        ref, quote = source.get("ref"), source.get("quote")
        if (
            ref not in allowed
            or ref not in evidence
            or not isinstance(quote, str)
            or not quote.strip()
            or not contains_quote(evidence[ref], quote)
        ):
            raise ValueError(
                "Requirement quote must occur in a public Task/current-user source"
            )


def contains_quote(value, quote):
    if isinstance(value, str):
        return quote in value
    if isinstance(value, dict):
        return any(contains_quote(v, quote) for v in value.values())
    if isinstance(value, list):
        return any(contains_quote(v, quote) for v in value)
    return False


def with_reward_protocol(pack, requirements, context, evidence):
    """Compile reviewed requirement bindings without rewriting a historical Pack.

    Exact quotations establish provenance, not semantic entailment. Classifying a
    prohibition or delivery prerequisite remains a reviewable construction choice.
    All original criteria must be accounted for, including unscored advice.
    """
    if pack.get("reward_protocol"):
        raise ValueError(
            "Reward protocol already frozen; construct a new version from the original Pack"
        )
    if (
        len(context.get("products", [])) != 1
        or pack["task_id"] != context["task"]["task_id"]
        or pack["task_prompt"] != context["task"]["prompt"]
    ):
        raise ValueError(
            "Reward construction requires the same task and one current candidate"
        )
    result = deepcopy(pack)
    criteria = {c["id"]: c for c in result["criteria"]}
    if len(criteria) != len(result["criteria"]):
        raise ValueError("Duplicate criterion")
    if len(requirements) != len(criteria) or {
        r["criterion_id"] for r in requirements
    } != set(criteria):
        raise ValueError("Account for every criterion exactly once")
    tests = {t["test_id"]: t for t in result["tests"]}
    if len(tests) != len(result["tests"]):
        raise ValueError("Duplicate test")
    used, scored = set(), 0
    for original in requirements:
        requirement = deepcopy(original)
        cid, role = requirement["criterion_id"], requirement.get("role")
        if role not in {"quality", "gate", "advisory"}:
            raise ValueError("Requirement role must be quality/gate/advisory")
        criteria[cid]["reward_requirement"] = requirement
        if role == "advisory":
            if not requirement.get("reason") or requirement.get("obligations"):
                raise ValueError(
                    "Advice needs a reason and cannot carry scored obligations"
                )
            continue
        scored += 1
        validate_sources(requirement.get("sources"), context, evidence)
        if role == "gate" and (
            requirement.get("gate_kind") not in {"prohibition", "delivery_prerequisite"}
            or not requirement.get("gate_reason")
        ):
            raise ValueError(
                "Gate needs an explicit public prohibition/prerequisite and justification"
            )
        scope = requirement.setdefault("scope", {})
        if not isinstance(scope, dict):
            raise ValueError("Requirement scope must be an object")
        for condition in scope.get("conditions", []):
            validate_sources(condition.get("sources"), context, evidence)
            validate_predicate(condition["fact"])
        date = scope.get("reference_date")
        if date is not None:
            if not isinstance(date, dict) or not isinstance(date.get("sources"), list):
                raise ValueError("Reference date needs explicit public provenance")
            if date["sources"]:
                validate_sources(date["sources"], context, evidence)
        obligations = requirement.get("obligations")
        if not isinstance(obligations, list) or not obligations:
            raise ValueError("Scored criterion needs explicit obligations")
        ids, descriptions = set(), set()
        for obligation in obligations:
            oid, description = obligation.get("id"), obligation.get("description")
            if (
                not isinstance(oid, str)
                or not oid
                or oid in ids
                or not isinstance(description, str)
                or not description.strip()
                or description.strip() in descriptions
            ):
                raise ValueError("Duplicate or missing obligation identity/description")
            ids.add(oid)
            descriptions.add(description.strip())
            if obligation.get("fact") is not None:
                validate_predicate(obligation["fact"])
                key = f"public-reward:{cid}:{oid}"
                if key in tests:
                    raise ValueError(
                        "Generated fact test collides with an existing test"
                    )
                test = {
                    "test_id": key,
                    "criterion_id": cid,
                    "checker": CHECKER,
                    "question": description,
                    "arguments": deepcopy(obligation["fact"]),
                    "rubric": {
                        k: deepcopy(v)
                        for k, v in criteria[cid].items()
                        if k != "reward_requirement"
                    },
                    "binding_status": "callable",
                }
                result["tests"].append(test)
                tests[key] = test
                obligation.setdefault("test_ids", []).append(key)
            bound = obligation.get("test_ids", [])
            if not bound or len(set(bound)) != len(bound):
                raise ValueError("Obligation needs distinct executable tests")
            for key in bound:
                if key not in tests or tests[key]["criterion_id"] != cid or key in used:
                    raise ValueError(
                        "Cross-criterion, missing, or multiply counted test"
                    )
                used.add(key)
    if not scored:
        raise ValueError("Reward needs at least one scored requirement")
    result["reward_protocol"] = {
        "version": VERSION,
        "parent_pack_id": pack["pack_id"],
        "test_ids": sorted(used),
        "public_requirements_digest": public_requirements_digest(context, evidence),
    }
    result["pack_id"] = digest({k: v for k, v in result.items() if k != "pack_id"})
    return result


def validate_pack(pack):
    if pack.get("reward_protocol", {}).get("version") != VERSION:
        raise ValueError("Unsupported independent reward protocol")
    if pack["pack_id"] != digest({k: v for k, v in pack.items() if k != "pack_id"}):
        raise ValueError("Frozen independent reward Pack changed")


def requirement_scope(requirement, context, evidence):
    try:
        validate_sources(requirement["sources"], context, evidence)
        scope = requirement.get("scope", {})
        conditions = []
        for condition in scope.get("conditions", []):
            validate_sources(condition["sources"], context, evidence)
            conditions.append(
                check_fact(condition["fact"], context, evidence)["status"]
            )
        if "fail" in conditions:
            return "not_applicable", "Public applicability condition is false"
        if "unknown" in conditions:
            return "unknown", "Public applicability condition is unresolved"
        if "reference_date" in scope:
            date = scope["reference_date"]
            if not isinstance(date, dict) or not date.get("sources"):
                return "unknown", "Public reference date is missing"
            validate_sources(date["sources"], context, evidence)
            dates = {
                d
                for s in date["sources"]
                for d in re.findall(r"\d{4}-\d{2}-\d{2}", s["quote"])
            }
            if dates != {date.get("value")}:
                return "unknown", "Public reference date is missing or conflicting"
            calendar_date.fromisoformat(date["value"])
    except (ValueError, KeyError, TypeError):
        return (
            "unknown",
            "Requirement is not grounded in this candidate's public requirements",
        )
    return "applicable", "Public requirement binding checked"


def score(pack, results, context, evidence):
    """Aggregate only executed receipts for this protocol's obligation bindings.

    Call through measure() for candidate-bound receipts. This pure function also
    supports offline tests; arbitrary model JSON is not a trusted receipt source.
    """
    validate_pack(pack)
    requirements_changed = pack["reward_protocol"][
        "public_requirements_digest"
    ] != public_requirements_digest(context, evidence)
    tests = {t["test_id"]: t for t in pack["tests"]}
    observed = {}
    for row in results:
        key = row["test_id"]
        if (
            key not in tests
            or key in observed
            or row["criterion_id"] != tests[key]["criterion_id"]
            or row.get("status") not in {"pass", "fail", "unknown"}
        ):
            raise ValueError("Unexpected, duplicate, or invalid check receipt")
        observed[key] = row
    criteria, intervals, gates = [], [], []
    uncertain_scope = False
    for criterion in pack["criteria"]:
        req = criterion["reward_requirement"]
        role = req["role"]
        entry = {
            "criterion_id": criterion["id"],
            "role": role,
            "sources": deepcopy(req.get("sources", [])),
            "scope": deepcopy(req.get("scope", {})),
        }
        criteria.append(entry)
        if role == "advisory":
            entry.update(fulfillment="not_scored", reason=req["reason"])
            continue
        scope, reason = requirement_scope(req, context, evidence)
        if requirements_changed:
            scope, reason = (
                "unknown",
                "Public requirements changed; review and compile a new binding",
            )
        entry.update(applicability=scope, reason=reason)
        if scope == "not_applicable":
            entry["fulfillment"] = "not_applicable"
            continue
        details = []
        for obligation in req["obligations"]:
            receipts = [observed.get(t) for t in obligation["test_ids"]]
            known = [r for r in receipts if r and not r.get("error")]
            status = (
                "fail"
                if any(r["status"] == "fail" for r in known)
                else "pass"
                if len(known) == len(receipts)
                and all(r["status"] == "pass" for r in known)
                else "unknown"
            )
            if scope == "unknown":
                status = "unknown"
            details.append(
                {
                    "id": obligation["id"],
                    "description": obligation["description"],
                    "status": status,
                    "test_ids": obligation["test_ids"],
                    "evidence_refs": sorted(
                        {ref for r in known for ref in r.get("evidence_refs", [])}
                    ),
                    "unknown_kinds": sorted(
                        {
                            "execution" if not r or r.get("error") else "evidence"
                            for r in receipts
                            if not r or r.get("error") or r["status"] == "unknown"
                        }
                    )
                    if status == "unknown"
                    else [],
                }
            )
        passed = [d["id"] for d in details if d["status"] == "pass"]
        failed = [d["id"] for d in details if d["status"] == "fail"]
        unresolved = [d["id"] for d in details if d["status"] == "unknown"]
        level = (
            "unknown"
            if unresolved
            else "partial"
            if passed and failed
            else "full"
            if passed
            else "none"
        )
        low = 0.5 if passed and (failed or unresolved) else 1.0 if passed else 0.0
        high = 0.5 if failed and (passed or unresolved) else 0.0 if failed else 1.0
        entry.update(
            fulfillment=level,
            satisfied=passed,
            missing=failed,
            unresolved=unresolved,
            obligations=details,
            q_min=low,
            q_max=high,
        )
        uncertain_scope |= scope == "unknown"
        if role == "gate":
            gates.append("fail" if failed else "unknown" if unresolved else "pass")
        else:
            intervals.append((low, high))
    q_min = sum(a for a, _ in intervals) / len(intervals) if intervals else 1.0
    q_max = sum(b for _, b in intervals) / len(intervals) if intervals else 1.0
    if not intervals and not gates:
        lo, hi = 0.0, 1.0
    elif "fail" in gates:
        lo = hi = 0.0
    elif uncertain_scope:
        lo, hi = (
            0.0,
            1.0,
        )  # Applicability may change the denominator; conservative bound.
    else:
        lo = 0.0 if "unknown" in gates else 0.5 + 0.5 * q_min
        hi = 0.5 + 0.5 * q_max
    reward = lo if lo == hi else None
    return {
        "protocol": VERSION,
        "pack_id": pack["pack_id"],
        "criteria": criteria,
        "reward": reward,
        "r_min": lo,
        "r_max": hi,
        "training_mask": reward is not None,
        "gate_status": "fail"
        if "fail" in gates
        else "unknown"
        if "unknown" in gates
        else "pass",
        "q_min": q_min,
        "q_max": q_max,
        "interpretation": "Public-task ordinal reward; not an oracle score or success probability",
    }
