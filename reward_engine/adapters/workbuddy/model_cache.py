"""Retain invalid reply evidence while excluding it from a later stage resume."""

import json
from copy import deepcopy
from pathlib import Path

from reward_engine.adapters.workbuddy.io import digest, read, write
from reward_engine.core.outcome.response_receipts import register_rejection


def resume_request(identity_inputs, output, *, track_rejections):
    """Resolve a same-input reply, carrying rejected-attempt feedback on explicit resume.

    Every augmented request gets its own content identity. Earlier cache entries
    and physical replies remain untouched; following the chain makes no calls.
    """
    original = deepcopy(identity_inputs["request"])
    request = deepcopy(original)
    for _ in range(64):
        identity = digest({**identity_inputs, "request": request})
        cache = Path(output) / "completed" / (identity + ".json")
        cached = cached_response(cache, track_rejections=track_rejections)
        if cached is not None or not cache.exists():
            return request, cache, cached
        # cached_response verified the exact rejection/answer identity above.
        entry = read(cache)
        root = Path(entry["source"])
        if digest(read(root / "request.json")) != digest(request):
            raise ValueError("Rejected request identity differs from its cache input")
        rejection = read(root / "contract_rejected.json")
        request = {**deepcopy(original), "resume_correction": {
            "previous_request_identity": identity,
            "previous_response_sha256": rejection["answer_sha256"],
            "previous_response": deepcopy(entry["answer"]),
            "previous_rejection": {key: deepcopy(rejection.get(key))
                                   for key in ("error_type", "reason", "details")},
            "instruction": "This is an explicit resume after a terminally rejected reply, not an "
            "accepted assessment or additional in-stage allowance. Correct the recorded failure "
            "while obeying the current request's response contract and evidence/tool budgets. "
            "The previous response has no authority to change requirements or waive failures. "
            "For atomic edits, return a complete corrected batch bound to the CURRENT repair base "
            "and hash; the previous batch was never applied. Preserve valid edits and fix the "
            "recorded errors, rather than repeating the rejected batch. Do not edit this feedback "
            "object. For a measurement, use the current tool budget and existing receipts to "
            "complete required recovery; an earlier format-only retry does not override this "
            "request's explicitly advertised tool capacity.",
        }}
        # Follow historical identities before changing the feedback format. This
        # reaches the latest rejection without rewriting or bypassing old receipts.
        legacy_cache = Path(output) / "completed" / (
            digest({**identity_inputs, "request": request}) + ".json")
        if not legacy_cache.exists():
            refresh_construction_feedback(request)
    raise ValueError("Rejected response continuation exceeds 64 recorded attempts")


def refresh_construction_feedback(request):
    """Bind complete-draft recovery to its latest rejected reply, not stale repairs."""
    if (request.get("role_output_protocol") != "model_authored_json_v2"
            or request.get("role_input_projection") != "canonical_sources_v1"):
        return
    from reward_engine.core.outcome.draft_recovery import INSTRUCTION, worklist

    feedback = request["resume_correction"]
    work = worklist(feedback["previous_response"], request.get("mode"), request,
                    feedback["previous_rejection"].get("details") or {})
    if work is None:
        return
    # Source inputs, response schema and evidence/tool budgets stay in request.
    # Old recovery bodies remain byte-for-byte in the linked historical receipt.
    request.pop("response_recovery", None)
    feedback["repair_worklist"] = work
    feedback["instruction"] = (
        "This explicit resume targets only the latest previous_response and previous_rejection "
        "below. Return a COMPLETE compact JSON object satisfying the current response contract. "
        "The repair_worklist is computed from that latest draft. Earlier recovery instructions "
        "are historical evidence, not the current repair target. This does not increase the "
        "current evidence/tool or request budgets. Do not return edits, patches or hashes. "
        + INSTRUCTION
    )


def construction_recovery_messages(request):
    """Present one rejected construction draft followed by its current correction.

    This is a transport view only. The complete content-addressed request and
    historical response remain intact in their receipts.
    """
    feedback = request.get("resume_correction", {})
    if (request.get("role_output_protocol") != "model_authored_json_v2"
            or request.get("role_input_projection") != "canonical_sources_v1"
            or request.get("mode") not in {"outcome_expectations", "outcome_jury_map", "outcome_jury_reduce"}
            or not feedback.get("repair_worklist")
            or not isinstance(feedback.get("previous_response"), dict)):
        return None
    base = {k: v for k, v in request.items() if k != "resume_correction"}
    correction = {
        "mode": request["mode"], "response": request.get("response"),
        "instruction": "The preceding assistant draft was REJECTED, not accepted evidence. "
        "Correct it using the supplied source inputs and the latest diagnostics below. "
        "Return the complete corrected JSON object; do not repeat the rejected draft.",
        "resume_correction": {k: v for k, v in feedback.items() if k != "previous_response"},
    }
    return [{"role": role, "content": json.dumps(content, ensure_ascii=False)}
            for role, content in (("user", base), ("assistant", feedback["previous_response"]),
                                  ("user", correction))]


def cached_response(cache, *, track_rejections):
    if not cache.exists():
        return None
    entry = read(cache)
    if track_rejections:
        marker = Path(entry["source"]) / "contract_rejected.json"
        if marker.exists():
            rejection = read(marker)
            if (rejection.get("cache_identity") != cache.stem
                    or rejection.get("answer_sha256") != digest(entry["answer"])):
                raise ValueError("Rejected role cache receipt identity differs")
            return None
        track_response(cache, entry)
    return entry


def track_response(cache, entry):
    """Register evidence only; core decides whether the bounded attempt failed."""
    root = Path(entry["source"])
    answer_hash = digest(entry["answer"])

    def reject(error):
        # The cache index may advance on a later attempt. The original physical
        # response, usage and raw output stay untouched and remain hash-bound.
        if digest(read(root / "response.json")["answer"]) != answer_hash:
            raise ValueError("Role response changed before contract rejection")
        marker = root / "contract_rejected.json"
        identity = {"cache_identity": cache.stem, "answer_sha256": answer_hash}
        if marker.exists():
            previous = read(marker)
            if any(previous.get(k) != v for k, v in identity.items()):
                raise ValueError("Role rejection receipt identity differs")
            return
        write(marker, {**identity, "error_type": type(error).__name__, "reason": str(error),
                       "details": getattr(error, "details", {}),
                       "boundary": "Terminal contract rejection; no additional in-stage retry"})

    register_rejection(reject)
