"""A closed-read correction cannot reroll already supported claim assessments."""


def preserve_supported_claims(previous, current, context):
    from reward_engine.core.outcome.context import ResponseContractError, used_previews

    original, unread = previous
    claims = original.get("claims")
    if not isinstance(claims, list):
        return  # There was no structured claim assessment to preserve.
    revised = current.get("claims")
    if not isinstance(revised, list) or len(revised) != len(claims):
        raise ResponseContractError("claims", "Closed evidence repair must preserve claim count and order")
    for index, (before, after) in enumerate(zip(claims, revised, strict=True)):
        if not isinstance(before, dict):
            continue
        affected = set(used_previews(before, context)) & set(unread)
        if not affected and before != after:
            raise ResponseContractError(f"claims[{index}]", "Closed evidence repair must preserve unaffected claims")
        if affected and (not isinstance(after, dict) or before.get("claim") != after.get("claim")
                         or before.get("verdict_relevance") != after.get("verdict_relevance")):
            raise ResponseContractError(f"claims[{index}]", "Repair unsupported evidence, not the claim or its relevance")
