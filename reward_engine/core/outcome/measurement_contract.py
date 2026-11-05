"""Single field contract for model-authored measurement requests and validation."""

from copy import deepcopy

# (type, description, required): semantic consistency remains in the Harness.
CLAIM = {
    "claim": (str, "assertion or required outcome under this standard", True),
    "claim_refs": (list, ["exact refs locating the assertion or requirement"], True),
    "claim_location": (
        str,
        "optional chapter/quote, not independently verified",
        False,
    ),
    "evidence_refs": (list, ["observed support; [] only for unverified"], True),
    "observation": (
        str,
        "what evidence establishes, contradicts or leaves unresolved",
        True,
    ),
    "judgment": (str, "supported/contradicted/unverified", True),
    "verdict_relevance": (str, "material/advisory; omitted means material", False),
    "relevance_reason": (
        str,
        "required for advisory: why outside this standard",
        False,
    ),
}
VERDICT = {
    "status": (str, "pass/fail/unknown", True),
    "reason": (str, "evidence-grounded explanation", True),
    "gap": (str, "specific defect if fail; otherwise empty", True),
    "evidence_refs": (list, ["exact observed evidence refs"], True),
}


def template(inquiry=False):
    value = {k: deepcopy(v[1]) for k, v in VERDICT.items()}
    value["claims"] = [{k: deepcopy(v[1]) for k, v in CLAIM.items()}]
    value["measurement_recovery"] = [
        {
            "failed_ref": "failed tool receipt",
            "replacement_refs": [
                "later observed same-tool receipts covering failed inputs"
            ],
            "reason": "why recovery resolves failure",
        }
    ]
    if inquiry:
        value["inquiry_result"] = {k: deepcopy(v[1]) for k, v in VERDICT.items()}
    return value


def issues(row, inquiry=False):
    errors = []

    def fields(value, schema, prefix):
        if not isinstance(value, dict):
            errors.append({"field": prefix, "reason": "Expected object"})
            return
        for key, (kind, _, required) in schema.items():
            if (required or key in value) and not isinstance(value.get(key), kind):
                errors.append(
                    {"field": prefix + key, "reason": "Expected " + kind.__name__}
                )

    fields(row, VERDICT, "")
    claims = row.get("claims")
    if not isinstance(claims, list):
        errors.append({"field": "claims", "reason": "Expected list"})
    else:
        for i, claim in enumerate(claims):
            fields(claim, CLAIM, f"claims[{i}].")
    if inquiry:
        fields(row.get("inquiry_result"), VERDICT, "inquiry_result.")
    return errors
