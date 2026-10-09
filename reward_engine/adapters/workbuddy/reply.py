"""Decode complete replies and record narrowly defined presentation repairs."""

import json
import re

from reward_engine.adapters.claw_eval.reply_format import decode_completed_reply
from reward_engine.core.shared.provider_reply import json_answer_text


def _strict_json(text):
    """Reject ambiguous keys at every depth, including inside argument arrays."""

    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise json.JSONDecodeError("Duplicate JSON object key", text, 0)
            value[key] = item
        return value

    def invalid_constant(_):
        raise json.JSONDecodeError("Non-JSON constant", text, 0)

    return json.loads(
        text, object_pairs_hook=unique_object, parse_constant=invalid_constant
    )


def _strict_completed_reply(text):
    # The shared decoder defines the allowed presentation repairs, but its
    # decoded dict has already lost duplicate keys. Reapply ONLY its recorded
    # syntax edit to the original text and strictly parse that source instead.
    _, notes = decode_completed_reply(text)
    if notes == [{"operation": "unwrap_complete_json_presentation"}]:
        fence = re.fullmatch(r"(?::\s*)?```(?:json)?[ \t]*\n([\s\S]+)", text.strip())
        body = fence.group(1).strip()
        if body.endswith("\n```"):
            body = body[:-4].rstrip()
        return _strict_json(body), notes
    source = json_answer_text(text.strip())
    if notes:
        if len(notes) != 1:
            raise ValueError("Unsupported completed reply presentation repairs")
        note = notes[0]
        operation, offset = note["operation"], note.get("offset")
        if operation == "remove_duplicate_root_closing_brace":
            source = source[:offset]
        elif operation == "remove_one_trailing_comma":
            source = source[:offset] + source[offset + 1 :]
        elif operation in {
            "replace_value_separator_semicolon",
            "normalize_one_closing_quote",
        }:
            replacement = (
                "," if operation == "replace_value_separator_semicolon" else '"'
            )
            source = source[:offset] + replacement + source[offset + 1 :]
        elif operation == "close_missing_root_object":
            source += "}"
        else:
            raise ValueError("Unsupported completed reply presentation repair")
    return _strict_json(source), notes


def decode_native_tool_reply(request, chunks, *, content, finish_reason):
    """Translate tool-only measurement reads, never execute them or infer a verdict."""
    if finish_reason != "tool_calls":
        raise ValueError("Native tool request must finish with tool_calls")
    if content.strip():
        # A native tool message may include narration, but never choose between
        # a JSON answer and tools or extract JSON by dropping surrounding prose.
        if any(marker in content for marker in "{}[]") or content.strip().lower() in {
            "pass",
            "fail",
            "unknown",
        }:
            raise ValueError("Native tool request contains a mixed answer")
        try:
            json.loads(content)
        except json.JSONDecodeError:
            pass
        else:
            raise ValueError("Native tool request contains a mixed JSON answer")
    catalog = request.get("measurement_tools")
    response = request.get("response", {})
    protocol = request.get("tool_protocol", "")
    allows_tools = (
        isinstance(response, dict)
        and "need_tools" in response
        or isinstance(protocol, str)
        and '"need_tools"' in protocol
    )
    if (
        request.get("mode") != "outcome_measure"
        or not isinstance(catalog, dict)
        or not allows_tools
    ):
        raise ValueError("Native tools require an advertised measurement protocol")
    calls = {}
    for batch in chunks:
        if not isinstance(batch, list) or not batch:
            raise ValueError("Malformed native tool-call chunk")
        for chunk in batch:
            if not isinstance(chunk, dict) or set(chunk) - {
                "index",
                "id",
                "type",
                "function",
            }:
                raise ValueError("Malformed native tool-call fields")
            index = chunk.get("index")
            if type(index) is not int or index < 0:
                raise ValueError("Native tool-call index must be a nonnegative integer")
            call = calls.setdefault(index, {"name": "", "arguments": ""})
            for key in ("id", "type"):
                if key in chunk:
                    if key in call or not isinstance(chunk[key], str) or not chunk[key]:
                        raise ValueError(
                            "Duplicate or malformed native tool-call identity"
                        )
                    call[key] = chunk[key]
            function = chunk.get("function")
            if (
                not isinstance(function, dict)
                or not function
                or set(function) - {"name", "arguments"}
            ):
                raise ValueError("Malformed native tool-call function")
            for key in ("name", "arguments"):
                if key in function:
                    if not isinstance(function[key], str):
                        raise ValueError("Native function fragments must be strings")
                    call[key] += function[key]
    if not calls or sorted(calls) != list(range(len(calls))):
        raise ValueError("Native tool-call indices must be complete and contiguous")

    allowed = {
        "workspace.read": {"paths", "offset"},
        "workspace.inspect": {"paths"},
        "document.read": {"ref", "url", "offset"},
    }
    probes, identities = [], []
    for index in sorted(calls):
        call = calls[index]
        name, identity = call["name"], call.get("id")
        if call.get("type") != "function" or not identity or identity in identities:
            raise ValueError("Native tool calls require unique function identities")
        advertised = catalog.get(name)
        if (
            name not in allowed
            or not isinstance(advertised, dict)
            or advertised.get("available") is not True
        ):
            raise ValueError(
                "Native tool is not an advertised available read-only measurement tool"
            )
        arguments = _strict_json(call["arguments"])
        if not isinstance(arguments, dict) or set(arguments) - allowed[name]:
            raise ValueError("Malformed native read-only tool arguments")
        identities.append(identity)
        probes.append({"tool": name, "arguments": arguments})
    # Argument values, evidence access and execution budgets remain core-owned.
    return {"need_tools": probes}, [
        {
            "operation": "native_measurement_tools_to_need_tools",
            "call_ids": identities,
            "tools": [probe["tool"] for probe in probes],
            "commentary": content,
            "executed_by_adapter": False,
        }
    ]


def decode_role_reply(text):
    """Never repair verdict values, references, tool arguments, or incomplete arrays."""
    raw = text.strip()
    if raw.startswith("{") and raw.endswith("\n```"):
        # Some completed replies omit the opening Markdown fence. Remove only
        # the closing fence after ONE already-complete, unambiguous JSON object.
        # Do not combine this with brace/punctuation repair or discard prose.
        value = _strict_json(raw[:-4])
        return value, [
            {"operation": "remove_trailing_json_fence", "offset": len(raw) - 3}
        ]
    try:
        return _strict_completed_reply(text)
    except json.JSONDecodeError:
        match = re.fullmatch(
            r"\{\s*(need_tools|need_evidence)\s*:\s*(\[.*\])\s*\}", text.strip(), re.S
        )
        if match is None:
            raise
        # The array still has to be strict, complete JSON. Core protocol checks
        # retain authority over tools, argument types, references, and budgets.
        values = _strict_json(match.group(2))
        return {match.group(1): values}, [
            {"operation": "quote_protocol_request_key", "key": match.group(1)}
        ]
