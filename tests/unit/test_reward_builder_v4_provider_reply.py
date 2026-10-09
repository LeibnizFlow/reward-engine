"""Provider markers are presentation, never another candidate answer or new vote."""
import json

import pytest

from reward_engine.core.shared.provider_reply import json_answer_text


@pytest.mark.parametrize("raw", [
    '{"ok": true}',
    '```json\n{"ok": true}\n```',
    '<think>reasoning</think>{"ok": true}',
    'reasoning tail</think>{"ok": true}',
    '<think>{"ok": false}</think>```json\n{"ok": true}\n```',
])
def test_explicit_wrappers_leave_only_actual_answer(raw):
    assert json.loads(json_answer_text(raw)) == {"ok": True}


def test_reasoning_markers_inside_json_are_preserved():
    value = {"quote": "Example: <think>work</think>", "reason": "Keep the original evidence."}
    assert json.loads(json_answer_text(json.dumps(value))) == value


@pytest.mark.parametrize("raw", [
    '<think>{"ok": true}',
    'thinking {"ok": true}',
    '```json\n{"ok": true}',
    '```json\n{"ok": true}\n```\nextra answer',
    'tail</think>{"ok": true',
    'tail</think>{"ok": true}{"ok": false}',
    '{"reason": "broken </think>{"ok": true}',
])
def test_missing_markers_broken_json_and_multiple_answers_are_not_repaired(raw):
    with pytest.raises(ValueError):
        json.loads(json_answer_text(raw))
