"""A dangling Markdown fence must not discard or invent any JSON content."""

import json

import pytest

from reward_engine.adapters.workbuddy.reply import decode_role_reply


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_complete_object_with_only_trailing_fence_preserves_all_values(newline):
    expected = {
        "criteria": [{"id": "C1", "standard": "Literal ``` and } are data"}],
        "coverage": [],
        "unknown": None,
        "need_evidence": ["product_1:artifact:1"],
    }
    body = json.dumps(expected, ensure_ascii=False, indent=2)
    raw = body + newline + "```"
    value, notes = decode_role_reply(raw)
    assert value == expected
    assert notes == [
        {"operation": "remove_trailing_json_fence", "offset": len(raw) - 3}
    ]


@pytest.mark.parametrize(
    "raw",
    [
        '{"x": 1\n```',
        '{"x": [1,]}\n```',
        '{"x": 1}\n{"x": 2}\n```',
        '{"x": 1}\nexplanation\n```',
        '{"x": 1}\n```\n```',
        '{"x": 1}```',
        '{"x": 1, "x": 2}\n```',
        '{"x": NaN}\n```',
        '[{"x": 1}]\n```',
    ],
)
def test_trailing_fence_does_not_hide_incomplete_ambiguous_or_invalid_json(raw):
    with pytest.raises(json.JSONDecodeError):
        decode_role_reply(raw)


def test_fence_inside_json_string_is_not_a_presentation_marker():
    expected = {"text": "a code fence\n```", "status": "unknown"}
    assert decode_role_reply(json.dumps(expected)) == (expected, [])


def presentation(body, style):
    if style == "plain":
        return body
    if style == "fenced":
        return "```json\n" + body + "\n```"
    if style == "opening_fence":
        return ": ```json\n" + body
    if style == "closing_fence":
        return body + "\n```"
    if style == "extra_brace":
        return body + "}"
    if style == "trailing_comma":
        return body[:-1] + ",}"
    if style == "semicolon":
        return body[:-1] + ';"reason":"observed"}'
    if style == "closing_quote":
        return body[:-1] + ',\n"reason":"observed”,\n"gap":""}'
    if style == "missing_brace":
        return body[:-1]
    raise AssertionError(style)


PRESENTATIONS = [
    "plain",
    "fenced",
    "opening_fence",
    "closing_fence",
    "extra_brace",
    "trailing_comma",
    "semicolon",
    "closing_quote",
    "missing_brace",
]


@pytest.mark.parametrize("style", PRESENTATIONS)
@pytest.mark.parametrize(
    "body",
    [
        '{"status":"fail","status":"pass"}',
        '{"claims":[{"judgment":"unsupported","judgment":"supported"}]}',
        '{"need_tools":[{"tool":"workspace.read","arguments":'
        '{"paths":["first"],"paths":["second"]}}]}',
    ],
)
def test_duplicate_keys_rejected_through_every_presentation_repair(body, style):
    with pytest.raises(json.JSONDecodeError, match="Duplicate"):
        decode_role_reply(presentation(body, style))


@pytest.mark.parametrize("style", PRESENTATIONS)
def test_unambiguous_presentations_still_preserve_values(style):
    expected = {"status": "pass", "claims": [{"judgment": "supported"}]}
    raw = presentation(json.dumps(expected), style)
    if style in {"semicolon", "closing_quote"}:
        expected["reason"] = "observed"
    if style == "closing_quote":
        expected["gap"] = ""
    result, notes = decode_role_reply(raw)
    assert result == expected
    assert bool(notes) == (style != "plain")


@pytest.mark.parametrize("key", ["need_tools", "need_evidence"])
def test_bare_protocol_key_cannot_hide_nested_duplicates(key):
    raw = "{" + key + ':[{"arguments":{"paths":["first"],"paths":["second"]}}]}'
    with pytest.raises(json.JSONDecodeError, match="Duplicate"):
        decode_role_reply(raw)


def test_escaped_duplicate_keys_are_compared_after_json_unescaping():
    with pytest.raises(json.JSONDecodeError, match="Duplicate"):
        decode_role_reply(r'{"status":"fail","sta\u0074us":"pass"}')
