"""Outcome expectations stay visible beyond the tests that happen to exist."""

from copy import deepcopy


def expectations(pack):
    return pack.get("independent_expectations", []) + pack.get("proposal_expectations", [])


def report(pack, results):
    mappings = {r["expectation_id"]: r for r in pack.get("expectation_coverage", [])}
    rows = []
    for expected in expectations(pack):
        mapping = mappings.get(expected["expectation_id"], {})
        ids = mapping.get("criterion_ids", [])
        tests = [t for t in pack["tests"] if t["criterion_id"] in ids]
        measured = {r["test_id"]: r for r in results}
        states = [measured.get(t["test_id"], {}).get("status", "unknown") for t in tests]
        parts = mapping.get("subrequirements", [])
        pending = any(p.get("status") == "deferred" for p in parts)
        bound = mapping.get("status") == "mapped" and bool(ids) and set(ids) <= {t["criterion_id"] for t in tests}
        status = ("unknown" if pending or mapping.get("status") == "deferred" else
                  "excluded" if mapping.get("status") == "excluded" else
                  "fail" if "fail" in states else
                  "pass" if bound and states and all(s == "pass" for s in states) else "unknown")
        rows.append({"expectation_id": expected["expectation_id"],
                     "expected": expected["standard"], "why": expected.get("why", ""),
                     "priority": expected.get("priority", "important"), "origin": expected.get("origin", "unresolved"),
                     "verification": expected.get("measurement_question", "Not yet defined"), "criterion_ids": deepcopy(ids),
                     "subrequirements": deepcopy(parts),
                     "status": status, "reason": mapping.get("reason", "No coverage decision"),
                     "evidence_refs": list(dict.fromkeys(ref for t in tests for ref in
                         measured.get(t["test_id"], {}).get("evidence_refs", [])))})
    return rows


def complete(rows):
    return bool(rows) and all(r["status"] in {"pass", "excluded"} for r in rows)


def final_test(pack):
    """One global completeness review, only at delivery; not every action/test."""
    return {"test_id": "__outcome_completeness__", "criterion_id": "__outcome__",
            "checker": "evidence.review", "arguments": {},
            "question": "What major Task outcome is still missing despite the listed tests passing?",
            "rubric": {"id": "__outcome__", "kind": "soft",
                "standard": "Review the complete actual Task and current user's amendments, not just "
                "the Pack checklist. Inspect consequential omissions in the deliverable and environment. "
                "Check domain completeness with the available evidence; do not demand every imaginable "
                "optimization or impose one solution. Explain WHY any new gap matters to this Task "
                "and HOW to verify it. Common historical omissions still matter. If no supported gap "
                "is found, say so with boundaries; never declare official success or a probability. "
                "Missing expertise/evidence is unknown, not confirmed failure or automatic pass.",
                "expectations": deepcopy(expectations(pack)),
                "anchors": {"full": "No material unresolved outcome gap found in the available evidence",
                            "partial": "Unresolved consequential outcome or insufficient evidence",
                            "fail": "Concrete consequential omission or contradiction with Task evidence"}}}
