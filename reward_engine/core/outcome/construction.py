"""Bounded independent construction audits; modules own their semantic Reduce."""

import hashlib
import json
from copy import deepcopy

from reward_engine.core.batch import BatchPolicy, collect_batch
from reward_engine.core.outcome.context import ask, compact_jury_context, require_refs

VERSION = "outcome-construction/2026-09-10-task-first"

REVIEW_SCHEMA = {
    "assessment": "keep, revise or uncertain",
    "reason": "why, judged against the actual Task",
    "issues": [{
        "scope": "current or upstream",
        "target": "exact affected field/clause in the supplied artifact",
        "problem": "specific omission, contradiction or unsupported assumption",
        "suggested_change": "concrete correction, or verification needed if uncertain",
        "evidence_refs": ["supplied exact evidence handles"],
    }],
}


def validate_rounds(value):
    if type(value) is not int or not 1 <= value <= 3:
        raise ValueError("construction_rounds must be an integer from one to three")
    return value


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def validate_review(result, evidence):
    if not isinstance(result, dict) or result.get("assessment") not in {"keep", "revise", "uncertain"}:
        raise ValueError("Construction review needs keep/revise/uncertain assessment")
    if not isinstance(result.get("reason"), str) or not result["reason"].strip():
        raise ValueError("Construction review needs a reason grounded in Task")
    issues = result.get("issues")
    if not isinstance(issues, list):
        raise ValueError("Construction review issues must be a list")
    for issue in issues:
        if not isinstance(issue, dict) or issue.get("scope") not in {"current", "upstream"}:
            raise ValueError("Review issue must distinguish current artifact from upstream concern")
        for key in ("target", "problem", "suggested_change"):
            if not isinstance(issue.get(key), str) or not issue[key].strip():
                raise ValueError(f"Review issue needs {key}")
        require_refs(issue.get("evidence_refs"), evidence)
    if result["assessment"] == "revise" and not issues:
        raise ValueError("A revise assessment must identify a concrete issue")
    if result["assessment"] == "keep" and issues:
        raise ValueError("Unresolved issues cannot be labelled keep")
    return deepcopy(result)


async def audit_round(module, number, context, evidence, artifact, members, *, instruction,
                      policy=None, extra=None, record=lambda *_: None, max_bytes=None):
    """Three isolated reviews of ONE frozen artifact, with one contract redo per call."""
    if len(members) != 3 or number not in (2, 3):
        raise ValueError("Construction refinement needs three reviewers in round two or three")
    prefix = f"construction/{module}/round_{number}"
    # Index large product/source bodies without compressing the original Task.
    view = compact_jury_context(context, evidence)
    source = {"version": VERSION, "module": module, "round": number,
              "context": deepcopy(view), "evidence": deepcopy(evidence),
              "artifact": deepcopy(artifact), "extra": deepcopy(extra or {}),
              "instruction": instruction}
    record(prefix + "/input", {**source, "fingerprint": fingerprint(source)})
    prompt = (
        "Review the entire supplied current artifact independently against the ORIGINAL Task and "
        "scoped actual user requirements. The previous artifact is under review, not authority. "
        "Knowledge is optional fallible support; knowledge coverage is not Task outcome quality. "
        "Do not turn examples or one possible method into mandatory requirements. "
        "Identify concrete consequential defects and corrections with exact supplied source refs. "
        "Missing evidence means uncertainty, not a proven defect or full quality. "
        "No invented score, majority-based truth or issue quota. Return keep with issues=[] when "
        "no supported correction or unresolved concern remains. "
        + ("Also check whether prior corrections introduced errors, omitted requirements or weakened "
           "full quality. " if number == 3 else "") + instruction
    )

    async def one(key, call):
        result = await ask(
            call, f"outcome_{module}_audit", deepcopy(view), deepcopy(evidence), prompt,
            deepcopy(REVIEW_SCHEMA),
            extra={**deepcopy(extra or {}), "construction_round": number,
                   "construction_version": VERSION, "current_artifact": deepcopy(artifact)},
            max_bytes=max_bytes, validate=lambda r: validate_review(r, evidence),
            max_contract_retries=1, max_evidence_rounds=2, close_evidence=True,
        )
        record(prefix + "/review_" + key, result)
        return result

    batch = await collect_batch({key: lambda k=key, c=call: one(k, c)
                                 for key, call in members.items()}, policy or BatchPolicy())
    record(prefix + "/map", batch.summary())
    reviews = list(batch.require_all())
    record(prefix + "/reviews", reviews)
    return reviews
