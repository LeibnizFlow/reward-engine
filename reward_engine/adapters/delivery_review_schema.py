"""Opt-in generation shape for one existing review protocol; never a verdict.

Nested action/evidence semantics remain owned by the core validators. This is a
fixed contract, not a compiler for the prose schemas of other Outcome operations.
"""

from copy import deepcopy

from reward_engine.adapters.workbuddy.io import digest

CONFIG_KEY = "delivery_review_json_schema_models"
MODE = "outcome_delivery_review"
VERSION = "delivery-review-generation-v1"
FIELDS = (
    "outcome_goal",
    "full_score_gaps",
    "improvements",
    "claim_checks",
    "reason",
    "status",
    "gaps",
    "verify",
)


def response_format(*, evidence_closed=False):
    """Preserve final reviews, evidence requests and the existing transport wrappers."""
    text = {"type": "string", "minLength": 1}
    objects = {
        "type": "array",
        "items": {"type": "object", "additionalProperties": True},
    }
    properties = {field: deepcopy(objects) for field in FIELDS}
    properties.update(
        outcome_goal=deepcopy(text),
        reason=deepcopy(text),
        status={
            "type": "string",
            "enum": [
                "target_satisfied",
                "needs_improvement",
                "needs_verification",
                "unknown",
            ],
        },
    )
    final = {
        "type": "object",
        "properties": properties,
        "required": list(FIELDS),
        "additionalProperties": False,
    }
    refs = {"type": "array", "items": deepcopy(text)}
    if evidence_closed:
        refs["maxItems"] = 0
    # A full assessment can carry the optional read list, including an empty one.
    assessment = deepcopy(final)
    assessment["properties"]["need_evidence"] = deepcopy(refs)
    payloads = [assessment]
    if not evidence_closed:
        payloads.append(
            {
                "type": "object",
                "properties": {"need_evidence": {**deepcopy(refs), "minItems": 1}},
                "required": ["need_evidence"],
                "additionalProperties": False,
            }
        )
    # Existing normalization accepts one response wrapper with an optional exact
    # type=json_object marker. It does not accept nested or competing wrappers.
    wrapped = {
        "type": "object",
        "properties": {
            "response": {"anyOf": deepcopy(payloads)},
            "type": {"type": "string", "const": "json_object"},
        },
        "required": ["response"],
        "additionalProperties": False,
    }
    # The existing transport also accepts a full assessment inside response and
    # a read request beside it, but never two read lists at different levels.
    outer_read = deepcopy(wrapped)
    outer_read["properties"].update(response=final, need_evidence=deepcopy(refs))
    outer_read["required"] = ["response", "need_evidence"]
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "outcome_delivery_review_v1",
            "strict": True,
            "schema": {"type": "object", "anyOf": [*payloads, wrapped, outer_read]},
        },
    }


def for_request(request):
    if (
        request.get("mode") != MODE
        or not isinstance(request.get("response"), dict)
        or set(request["response"]) != set(FIELDS)
    ):
        raise ValueError(
            "Unsupported delivery review response shape for frozen JSON schema"
        )
    closed = request.get("evidence_access_closed", False)
    if type(closed) is not bool:
        raise ValueError("Delivery review evidence_access_closed must be boolean")
    return response_format(evidence_closed=closed)


def binding():
    return {
        "version": VERSION,
        "mode": MODE,
        "response_format_sha256": {
            "evidence_open": digest(response_format()),
            "evidence_closed": digest(response_format(evidence_closed=True)),
        },
    }
