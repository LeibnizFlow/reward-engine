"""Decode complete replies and record narrowly defined presentation repairs."""

import json
import re

from reward_engine.adapters.claw_eval.reply_format import decode_completed_reply


def decode_role_reply(text):
    """Never repair verdict values, references, tool arguments, or incomplete arrays."""
    raw = text.strip()
    if raw.startswith("{") and raw.endswith("\n```"):
        # Some completed replies omit the opening Markdown fence. Remove only
        # the closing fence after ONE already-complete, unambiguous JSON object.
        # Do not combine this with brace/punctuation repair or discard prose.
        def unique_object(pairs):
            value = {}
            for key, item in pairs:
                if key in value:
                    raise json.JSONDecodeError("Duplicate key before trailing fence", raw, 0)
                value[key] = item
            return value

        def invalid_constant(_):
            raise json.JSONDecodeError("Non-JSON constant before trailing fence", raw, 0)

        value = json.loads(raw[:-4], object_pairs_hook=unique_object, parse_constant=invalid_constant)
        return value, [{"operation": "remove_trailing_json_fence", "offset": len(raw) - 3}]
    try:
        return decode_completed_reply(text)
    except json.JSONDecodeError:
        match = re.fullmatch(r"\{\s*(need_tools|need_evidence)\s*:\s*(\[.*\])\s*\}", text.strip(), re.S)
        if match is None:
            raise
        # The array still has to be strict, complete JSON. Core protocol checks
        # retain authority over tools, argument types, references, and budgets.
        values = json.loads(match.group(2))
        return {match.group(1): values}, [{"operation": "quote_protocol_request_key", "key": match.group(1)}]
