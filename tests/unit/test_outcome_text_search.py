"""Bounded text locating avoids materializing a large range in evidence probes."""
# Unicode escapes preserve CJK coverage and original UTF-8/character boundaries.

import pytest

from reward_engine.core.outcome.numeric_program import run_numeric


@pytest.mark.parametrize("function", ["find", "rfind"])
@pytest.mark.parametrize("bounds", [(), (2,), (-8,), (0, 8), (-9, -1), (99,), (5, 2)])
@pytest.mark.parametrize("needle", ["\u6807\u9898", "", "missing"])
def test_search_matches_string_semantics(function, bounds, needle):
    text = "a\u6807\u9898b\u6807\u9898xyz"
    args = ", ".join(['inputs["text"]', 'inputs["needle"]', *map(str, bounds)])
    assert run_numeric(
        f"result = {function}({args})", {"text": text, "needle": needle}
    ) == getattr(text, function)(needle, *bounds)


def test_existing_long_text_is_searchable_without_large_output():
    text = "x" * 32000 + "\u6807\u9898" + "y" * 2000
    assert run_numeric(
        'i = find(inputs["text"], "\u6807\u9898")\nresult = [i, inputs["text"][i:i+2]]',
        {"text": text},
    ) == [32000, "\u6807\u9898"]
    with pytest.raises(ValueError, match="step budget"):
        run_numeric('result = find(inputs["text"], "\u6807\u9898")', {"text": text}, steps=100)


@pytest.mark.parametrize(
    "program", ['find(1, "x")', 'find("x", 1)', 'find("x", "x", True)', 'rfind("x")']
)
def test_invalid_search_arguments(program):
    with pytest.raises(ValueError, match="require"):
        run_numeric("result = " + program)
