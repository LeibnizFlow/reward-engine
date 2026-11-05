"""Synthetic file and bounded interpreter cases needed by Office reward checks."""

from copy import deepcopy

import pytest

from reward_engine.adapters.workbuddy.evidence import collect
from reward_engine.adapters.workbuddy.measurement import WorkBuddyTools
from reward_engine.core.outcome.context import build_context
from reward_engine.core.outcome.harness import defaults
from reward_engine.core.outcome.numeric_program import run_numeric


def test_numeric_text_methods_and_captured_output():
    output = []
    data = {"record": {"text": "a:b:a"}}
    result = run_numeric(
        "s = inputs['record'].get('text', '')\nprint(s)\nresult = [s.find('b'), s.rfind('a'), s.count('a'), s.split(':')]",
        data,
        stdout=output,
    )
    assert result == [2, 4, 2, ["a", "b", "a"]]
    assert output == ["a:b:a\n"] and data == {"record": {"text": "a:b:a"}}
    with pytest.raises(ValueError, match="10000"):
        run_numeric("for i in range(1001):\n    print('abcdefghij')\nresult = 1")
    with pytest.raises(ValueError):
        run_numeric("print('x', file=inputs)\nresult = 1")


@pytest.mark.asyncio
async def test_json_file_read_is_bound_to_exact_inventory_bytes(tmp_path):
    (tmp_path / "output.json").write_text('{"answer": 7}')
    candidate, _ = collect(
        tmp_path, candidate_id="synthetic", final_response="Done", initial_hashes={}
    )
    context, evidence = build_context(
        {"task_id": "synthetic", "prompt": "Deliver JSON"}, [candidate]
    )
    ref = next(
        k
        for k, v in evidence.items()
        if isinstance(v, list) and v and isinstance(v[0], dict) and "sha256" in v[0]
    )
    registry = defaults(None, tools=WorkBuddyTools())
    test = {
        "test_id": "json",
        "criterion_id": "syntax",
        "checker": "json.parse",
        "arguments": {"ref": ref, "path": "output.json"},
    }
    assert (await registry.run(test, context, evidence))["status"] == "pass"
    changed = deepcopy(evidence)
    changed[ref][0]["sha256"] = "0" * 64
    assert (await registry.run(test, context, changed))["status"] == "unknown"
    changed = deepcopy(evidence)
    changed[ref].append(deepcopy(changed[ref][0]))
    assert (await registry.run(test, context, changed))["status"] == "unknown"


def test_pptx_merged_tables_and_nested_group_geometry(tmp_path):
    from pptx import Presentation
    from pptx.util import Inches

    from reward_engine.adapters.workbuddy.evidence import inspect_file

    deck = Presentation()
    slide = deck.slides.add_slide(deck.slide_layouts[6])
    group = slide.shapes.add_group_shape()
    nested = group.shapes.add_group_shape()
    box = nested.shapes.add_textbox(Inches(1), Inches(2), Inches(3), Inches(1))
    box.text = "Nested evidence"
    table = slide.shapes.add_table(2, 2, 0, 0, Inches(4), Inches(2)).table
    table.cell(0, 0).text = "Merged heading"
    table.cell(0, 0).merge(table.cell(0, 1))
    table.cell(1, 1).text = "Observed value"
    path = tmp_path / "synthetic.pptx"
    deck.save(path)
    parsed = inspect_file(path)
    shapes = parsed["slides"][0]["shapes"]
    assert shapes[0]["shapes"][0]["shapes"][0]["text"] == "Nested evidence"
    assert shapes[0]["children_coordinate_system"].startswith("group-local")
    assert shapes[1]["table"][1][1] == "Observed value"
    assert shapes[1]["table_geometry"]["cells"][0][0]["span_width"] == 2
    assert shapes[1]["table_geometry"]["cells"][0][1]["is_spanned"] is True
    assert parsed["slide_width"] == deck.slide_width
