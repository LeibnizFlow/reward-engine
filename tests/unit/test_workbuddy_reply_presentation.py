"""A dangling Markdown fence must not discard or invent any JSON content."""

import json

import pytest

from reward_engine.adapters.workbuddy.reply import decode_role_reply


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_complete_object_with_only_trailing_fence_preserves_all_values(newline):
    expected = {"criteria": [{"id": "C1", "standard": "Literal ``` and } are data"}],
                "coverage": [], "unknown": None, "need_evidence": ["product_1:artifact:1"]}
    body = json.dumps(expected, ensure_ascii=False, indent=2)
    raw = body + newline + "```"
    value, notes = decode_role_reply(raw)
    assert value == expected
    assert notes == [{"operation": "remove_trailing_json_fence", "offset": len(raw) - 3}]


@pytest.mark.parametrize("raw", [
    '{"x": 1\n```',
    '{"x": [1,]}\n```',
    '{"x": 1}\n{"x": 2}\n```',
    '{"x": 1}\nexplanation\n```',
    '{"x": 1}\n```\n```',
    '{"x": 1}```',
    '{"x": 1, "x": 2}\n```',
    '{"x": NaN}\n```',
    '[{"x": 1}]\n```',
])
def test_trailing_fence_does_not_hide_incomplete_ambiguous_or_invalid_json(raw):
    with pytest.raises(json.JSONDecodeError):
        decode_role_reply(raw)


def test_fence_inside_json_string_is_not_a_presentation_marker():
    expected = {"text": "a code fence\n```", "status": "unknown"}
    assert decode_role_reply(json.dumps(expected)) == (expected, [])
