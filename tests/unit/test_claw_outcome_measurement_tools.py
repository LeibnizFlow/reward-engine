"""Execute real numerical/source tools, not simulated tool success from a model."""

import json
from copy import deepcopy

import pytest

from reward_engine.core.outcome.builder import build
from reward_engine.core.outcome.context import build_context
from reward_engine.core.outcome.harness import defaults, measure, measurement_context
from reward_engine.core.outcome.measurement_tools import MeasurementTools
from reward_engine.core.outcome.numeric_program import run_numeric


def test_loop_control_stays_in_innermost_loop_and_preserves_budget():
    program = """
result = []
for row in inputs['rows']:
    if row < 0:
        continue
    for value in range(5):
        if value == 1:
            continue
        if value == 3:
            break
        result.append(row * 10 + value)
    result.append(row * 10 + 9)
"""
    assert run_numeric(program, {"rows": [-1, 1, 2]}) == [10, 12, 19, 20, 22, 29]
    with pytest.raises(ValueError, match="step budget"):
        run_numeric("for i in range(1000):\n    continue\nresult = 1", steps=20)
    assert run_numeric("for i in range(1000):\n    break\nresult = i", steps=20) == 0
    for keyword in ("break", "continue"):
        with pytest.raises(ValueError, match="enclosing for"):
            run_numeric("if True:\n    " + keyword + "\nresult = 1")


def test_numeric_none_checks_and_text_loops_are_bounded():
    program = "text = inputs['text']\nresult = ''\nif text is not None:\n for char in text:\n  result = result + char\nelse:\n result = 'missing'"
    assert run_numeric(program, {"text": "工程 QA"}) == "工程 QA"
    assert run_numeric(program, {"text": None}) == "missing"
    assert run_numeric("result = [None is None, None is not 1, 1 is None]") == [
        True,
        True,
        False,
    ]
    with pytest.raises(ValueError, match="None checks only"):
        run_numeric("result = 1 is 1")
    with pytest.raises(ValueError, match="step budget"):
        run_numeric(program, {"text": "x" * 100}, steps=20)
    with pytest.raises(ValueError, match="Value too large"):
        run_numeric(program, {"text": "x" * 10001})


def test_large_text_probe_reports_intermediate_failure_and_supports_direct_inspection():
    inputs = {"parts": ["<slide>" + "x" * 10000 + "</slide>"]}
    program = "text = ''\nfor part in inputs['parts']:\n text = text + part\nresult = '<slide>' in text"
    with pytest.raises(ValueError) as caught:
        run_numeric(program, inputs)
    message = str(caught.value)
    assert (
        "line 3" in message
        and "str concatenation length 0 + 10015 exceeds 10000" in message
    )
    assert "intermediate operation" in message
    assert run_numeric(
        "result = ['<slide>' in part for part in inputs['parts']]", inputs
    ) == [True]


def test_numeric_scalar_text_conversion_preserves_bounds_and_reports_function():
    assert run_numeric(
        "result = [str(v) for v in inputs['values']]",
        {"values": ["2026/4/6", 20260406, 0.125, True, None]},
    ) == ["2026/4/6", "20260406", "0.125", "True", "None"]
    for code in ("result = str()", "result = str(1, 2)"):
        with pytest.raises(ValueError, match="one JSON value"):
            run_numeric(code)
    with pytest.raises(ValueError, match="step budget"):
        run_numeric("result = str(inputs['value'])", {"value": "x" * 100}, steps=20)
    with pytest.raises(ValueError, match="Value too large"):
        run_numeric("result = str(inputs['value'])", {"value": "x" * 10001})
    with pytest.raises(
        ValueError, match="Unsupported numerical function 'sum' at line 1"
    ):
        run_numeric("result = sum([1, 2])")


def test_numeric_container_text_conversion_is_bounded_and_python_compatible():
    for value in [[], {}, {"cells": [None, True, 0.25, "a'b", "周汇总"]}]:
        assert run_numeric("result = str(inputs['value'])", {"value": value}) == str(
            value
        )
    with pytest.raises(ValueError, match="Value too large"):
        run_numeric("result = str(inputs['value'])", {"value": ["x" * 100] * 100})
    with pytest.raises(ValueError, match="step budget"):
        run_numeric(
            "result = str(inputs['value'])", {"value": list(range(50))}, steps=20
        )


def test_real_arithmetic_and_bounded_program():
    assert run_numeric("result = (2**31-1)//(127*127)") == 133144
    assert (
        run_numeric(
            "total = 0\nfor i in range(4):\n total = total + inputs['x'][i]\nresult = total",
            {"x": [2, 3, 5, 7]},
        )
        == 17
    )
    assert (
        run_numeric(
            "result = inputs['rows'] < inputs['columns']", {"rows": 2, "columns": 5}
        )
        is True
    )


def test_small_table_two_way_membership_fits_default_but_large_quadratic_work_stays_bounded():
    program = (
        "missing = [x for x in inputs['source'] if x not in inputs['actual']]\n"
        "extra = [x for x in inputs['actual'] if x not in inputs['source']]\n"
        "result = {'missing': len(missing), 'extra': len(extra)}"
    )
    inputs = {"source": list(range(300)), "actual": list(range(280))}
    with pytest.raises(ValueError, match="step budget"):
        run_numeric(program, inputs, steps=100000)
    assert run_numeric(program, inputs) == {"missing": 20, "extra": 0}
    with pytest.raises(ValueError, match="step budget"):
        run_numeric(program, {"source": list(range(1000)), "actual": list(range(1000))})


@pytest.mark.asyncio
async def test_author_uses_current_product_and_repairs_from_actual_execution_error():
    from reward_engine.core.outcome.measurement_tools import author_checker

    context, evidence = build_context(
        {"task_id": "x", "prompt": "Inspect the delivered numeric value"},
        [{"artifact": {"value": 7}}, {"artifact": {"value": 900}}],
    )
    original = deepcopy((context, evidence))
    requests, trials = [], []

    async def model(request):
        requests.append(request)
        assert [p["product_id"] for p in request["context"]["products"]] == [
            "product_1"
        ]
        assert not any(
            ref.startswith("product_2:")
            for ref in request["context"]["available_evidence_refs"]
        )
        if len(requests) == 1:
            program = "result = d + 1"
        else:
            recovery = request["execution_recovery"]
            assert recovery["receipts"][0]["error_type"] == "KeyError"
            assert recovery["receipts"][0]["status"] == "unknown"
            program = "result = inputs['product_1:artifact:value'] + 1"
        return {
            "probes": [
                {
                    "tool": "code.run",
                    "arguments": {
                        "program": program,
                        "input_refs": ["product_1:artifact:value"],
                    },
                }
            ],
            "unavailable_reason": "",
        }

    registry = defaults(
        model, tools=MeasurementTools(record=lambda name, row: trials.append(row))
    )
    await author_checker(
        {"checker": "custom.current_value"}, registry, context, evidence, model
    )
    assert (context, evidence) == original and len(requests) == 2
    recipe = registry.recipes["custom.current_value"]
    # A fresh current product changes the actual calculation; the original
    # pool's value or answer cannot have been compiled into the recipe.
    measured = await MeasurementTools().run(
        recipe["probes"][0], {"product_1:artifact:value": 40}
    )
    assert measured["result"] == 41
    assert any(row.get("attempt") == 0 and row.get("error") for row in trials)
    assert any(row.get("attempt") == 1 and row.get("receipts") for row in trials)


@pytest.mark.asyncio
async def test_author_stops_after_two_failed_dry_runs_without_binding():
    from reward_engine.core.outcome.measurement_tools import author_checker

    context, evidence = build_context(
        {"task_id": "x", "prompt": "Inspect numeric data"}
    )
    calls = []

    async def model(request):
        calls.append(request)
        return {
            "probes": [
                {"tool": "code.run", "arguments": {"program": "result = nonexistent"}}
            ]
        }

    registry = defaults(model)
    with pytest.raises(ValueError, match="could not execute") as caught:
        await author_checker(
            {"checker": "custom.broken"}, registry, context, evidence, model
        )
    assert "probe[0] code.run: KeyError" in str(caught.value)
    assert "nonexistent" in str(caught.value)
    assert len(calls) == 2 and "custom.broken" not in registry.checkers


@pytest.mark.parametrize(
    "statement", ["a, b = [1, 2]", "a = b = 1", "a: int = 1", "x += [3]"]
)
def test_assignment_failure_explains_remaining_unsupported_forms(statement):
    with pytest.raises(ValueError, match="Unsupported assignment at line 2") as caught:
        run_numeric("x = [0]\n" + statement + "\nresult = x")
    assert "one plain variable target" in str(caught.value)
    assert run_numeric("x = [0]\nx.append(3)\nresult = x") == [0, 3]


def test_indexed_scalar_accumulation_preserves_input_and_resource_bounds():
    data = {"counts": [0, 0], "values": [2, 3, 4]}
    program = """
counts = inputs['counts']
for value in inputs['values']:
    index = value % 2
    counts[index] = counts[index] + value
counts[-1] = counts[-1] + 1
result = counts
"""
    assert run_numeric(program, data) == [6, 4]
    assert data["counts"] == [0, 0]
    for statement in ("x[0] = x", "x[:] = 1", "x[True] = 1", "inputs['new'] = 1"):
        with pytest.raises(ValueError, match="Indexed assignment"):
            run_numeric("x = [0]\n" + statement + "\nresult = x")
    with pytest.raises(IndexError):
        run_numeric("x = [0]\nx[1] = 1\nresult = x")
    with pytest.raises(ValueError, match="step budget"):
        run_numeric(
            "x = [0]\nfor i in range(1000):\n    x[0] = i\nresult = x", steps=20
        )


@pytest.mark.asyncio
async def test_author_does_not_redraw_explicit_unavailable_capability():
    from reward_engine.core.outcome.measurement_tools import author_checker

    context, evidence = build_context({"task_id": "x", "prompt": "Measure GPU runtime"})
    calls = []

    async def model(request):
        calls.append(request)
        return {"probes": [], "unavailable_reason": "No GPU backend"}

    registry = defaults(model)
    with pytest.raises(ValueError, match="No GPU backend"):
        await author_checker(
            {"checker": "custom.gpu"}, registry, context, evidence, model
        )
    assert len(calls) == 1 and "custom.gpu" not in registry.checkers


@pytest.mark.asyncio
async def test_author_repairs_excess_probe_contract_before_any_tool_execution():
    from reward_engine.core.outcome.measurement_tools import author_checker

    context, evidence = build_context(
        {"task_id": "x", "prompt": "Measure the supplied number"}
    )
    calls, executions = [], []

    async def model(request):
        calls.append(request)
        if len(calls) == 1:
            return {
                "probes": [{"tool": "calculate", "arguments": {"expression": "1+1"}}]
                * 6
            }
        assert "response_recovery" in request and not executions
        return {"probes": [{"tool": "calculate", "arguments": {"expression": "1+1"}}]}

    registry = defaults(
        model, tools=MeasurementTools(record=lambda name, row: executions.append(row))
    )
    await author_checker(
        {"checker": "custom.count"}, registry, context, evidence, model
    )
    assert len(calls) == 2
    assert len(registry.recipes["custom.count"]["probes"]) == 1
    assert sum(row.get("tool") == "calculate" for row in executions) == 1


@pytest.mark.parametrize(
    "code",
    [
        "import os\nresult = 1",
        "result = open('/etc/passwd')",
        "result = inputs.__class__",
        "result = __import__('os')",
        "while True:\n result = 1",
        "result = 2 ** 999999999",
        "result = 'a' * 999999999",
        "for i in range(999999999):\n result = i",
        "a=[1]\nfor i in range(30):\n a=[a,a]\nresult=a",
        "inputs = {}\nresult = 1",
        "result = float('nan')",
        "assert 1 == 2\nresult = 1",
    ],
)
def test_program_cannot_read_host_or_exhaust_unbounded_resources(code):
    with pytest.raises((ValueError, KeyError, AssertionError)):
        run_numeric(code)


def test_numeric_sequence_slices_skip_headers_without_host_python():
    assert run_numeric(
        "rows = inputs['rows']\nresult = [rows[1:], rows[::-1], rows[1:3], 'ABC'[:1]]",
        {"rows": ["header", 1, 2, 3]},
    ) == [[1, 2, 3], [3, 2, 1, "header"], [1, 2], "A"]
    for program in [
        "result = [1,2][::0]",
        "result = [1,2][1.5:]",
        "result = {'a':1}[:]",
    ]:
        with pytest.raises(ValueError):
            run_numeric(program)
    with pytest.raises(ValueError, match="step budget"):
        run_numeric("result = inputs['rows'][:]", {"rows": list(range(100))}, steps=20)


def test_bounded_list_append_supports_table_checks_without_mutating_sources():
    source = {"rows": [["header"], [1], [2]]}
    assert run_numeric(
        "out = []\nfor row in inputs['rows'][1:]:\n out.append(row[0] * 2)\nresult = out",
        source,
    ) == [2, 4]
    assert (
        run_numeric("out = inputs['rows']\nout.append([3])\nresult = len(out)", source)
        == 4
    )
    assert source == {"rows": [["header"], [1], [2]]}
    for code in [
        "out=[]\nout.append(out)\nresult=out",
        "a=[]\nb=[a]\na.append(b)\nresult=a",
        "out={}\nout.append(1)\nresult=out",
        "out=[]\nout.extend([1])\nresult=out",
        "out=[]\nfor i in range(10000):\n out.append(i)\nout.append(0)\nresult=1",
    ]:
        with pytest.raises(ValueError):
            run_numeric(code)


def test_sorted_multiset_comparison_preserves_duplicates_and_charges_work():
    program = "result = sorted(inputs['source']) == sorted(inputs['actual'])"
    assert (
        run_numeric(program, {"source": ["z", "a", "a"], "actual": ["a", "z", "a"]})
        is True
    )
    assert (
        run_numeric(program, {"source": ["z", "a", "a"], "actual": ["z", "z", "a"]})
        is False
    )
    with pytest.raises(ValueError, match="step budget"):
        run_numeric(
            "result = sorted(inputs['rows'])", {"rows": list(range(300))}, steps=100
        )
    with pytest.raises(ValueError, match="JSON scalars"):
        run_numeric("result = sorted(inputs['rows'])", {"rows": [{"x": 1}]})


@pytest.mark.asyncio
async def test_code_failure_returns_real_selector_hints_not_fabricated_input_aliases():
    evidence = {"source:table": {"tables": {"sheets": [{"rows": [["id"], [7]]}]}}}
    result = await MeasurementTools().run(
        {
            "tool": "code.run",
            "arguments": {
                "input_refs": ["source:table"],
                "program": "result = inputs['src']",
            },
        },
        evidence,
    )
    assert result["status"] == "unknown" and result["error_type"] == "KeyError"
    hint = result["recovery_hint"]
    assert set(hint["input_sources"]) == {"source:table"}
    assert {
        "path": ["tables", "sheets", 0, "rows"],
        "kind": "list",
        "length": 2,
    } in hint["input_sources"]["source:table"]
    assert "result" not in result


def test_numeric_predicate_and_sequence_operations_remain_bounded():
    assert run_numeric(
        "x = 99\nrows = [1,2] + [3]\nresult = [[x for x in rows if x > 1], x, "
        "'ab' in 'abcd', 2 not in rows, False and inputs['absent'], True or inputs['absent']]"
    ) == [[2, 3], 99, True, False, False, True]
    for program in [
        "result = [inputs for inputs in [1]]",
        "result = [x for x in [1] for y in [2]]",
        "result = 2 in 3",
        "result = [1] * 1000000",
    ]:
        with pytest.raises(ValueError):
            run_numeric(program)
    with pytest.raises(ValueError, match="step budget"):
        run_numeric(
            "result = 1 in inputs['rows']", {"rows": list(range(100))}, steps=20
        )


@pytest.mark.asyncio
async def test_numeric_input_paths_select_exact_evidence_without_discarding_limits():
    tools = MeasurementTools()
    evidence = {
        "source": {
            "raw_xml": "x" * 300000,
            "tables": [{"values": [["header"], [2], [3]]}],
        }
    }
    args = {
        "input_refs": ["source"],
        "input_paths": {"source": ["tables", 0, "values"]},
        "program": "rows = inputs['source']\ntotal = 0\nfor row in rows[1:]:\n total = total + row[0]\nresult = total",
    }
    result = await tools.run({"tool": "code.run", "arguments": args}, evidence)
    assert result["status"] == "observed" and result["result"] == 5
    assert len(evidence["source"]["raw_xml"]) == 300000
    for path in [["raw_xml"], ["missing"], ["tables", "0"], ["tables", False]]:
        bad = await tools.run(
            {
                "tool": "code.run",
                "arguments": {**args, "input_paths": {"source": path}},
            },
            evidence,
        )
        assert bad["status"] == "unknown"
    bad = await tools.run(
        {"tool": "code.run", "arguments": {**args, "input_paths": {"uncited": []}}},
        evidence,
    )
    assert bad["status"] == "unknown"


@pytest.mark.asyncio
async def test_document_reads_actual_evidence_and_cuda_unavailable_not_pass():
    tools = MeasurementTools()
    evidence = {
        "doc": {
            "text": "A real source explains the instruction.",
            "url": "https://docs.example/a",
        }
    }
    result = await tools.run(
        {"tool": "document.read", "arguments": {"ref": "doc"}}, evidence
    )
    assert result["status"] == "observed" and "real source" in result["result"]["text"]
    assert (
        await tools.run(
            {"tool": "document.read", "arguments": {"ref": "hidden_grader"}}, evidence
        )
    )["status"] == "unknown"
    assert (
        await tools.run({"tool": "cuda.run", "arguments": {"source": "code"}}, evidence)
    )["status"] == "unknown"
    assert tools.catalog()["cuda.run"]["available"] is False


@pytest.mark.asyncio
async def test_numeric_stride_counterexample_is_a_receipt_not_a_quality_verdict():
    # Small generic layout regression: equal dimensions do not make layouts equivalent.
    program = (
        "A=[1,0]\nB=[1,2,3,4]\nK=2\ncol=1\nreference=0\nproposed=0\n"
        "for k in range(K):\n reference=reference+A[k]*B[col*K+k]\n"
        " proposed=proposed+A[k]*B[k*K+col]\n"
        "result={'reference':reference,'proposed':proposed,'equal':reference==proposed}"
    )
    receipt = await MeasurementTools().run(
        {"tool": "code.run", "arguments": {"program": program}}, {}
    )
    assert receipt["status"] == "observed"
    assert receipt["result"] == {"reference": 3, "proposed": 2, "equal": False}
    assert (
        "score" not in receipt
    )  # The caller must still establish relevance to the real artifact.


@pytest.mark.asyncio
async def test_document_external_reader_only_observed_url():
    called = []

    async def reader(url):
        called.append(url)
        return "Fetched real source"

    tools = MeasurementTools(document_reader=reader)
    evidence = {"source": {"url": "https://docs.example/a"}}
    for url in ("https://docs.example/a", "https://invented.example/answer"):
        result = await tools.run(
            {"tool": "document.read", "arguments": {"url": url}}, evidence
        )
        assert (result["status"] == "observed") == (url.endswith("/a"))
    assert called == ["https://docs.example/a"]


def case():
    ctx, ev = build_context(
        {"task_id": "math", "prompt": "Explain the computation"},
        [{"artifact": {"final_response": "The answer is 5"}}],
    )
    criterion = {
        "id": "math",
        "standard": "Correct computation",
        "kind": "hard",
        "origin": "task",
    }
    test = {
        "test_id": "check",
        "criterion_id": "math",
        "checker": "evidence.review",
        "question": "Check the arithmetic",
        "rubric": criterion,
    }
    return ctx, ev, criterion, test


def verdict(tool_ref=None):
    refs = ["product_1:artifact:final_response"] + ([tool_ref] if tool_ref else [])
    return {
        "status": "pass",
        "reason": "Computed result agrees with the artifact",
        "gap": "",
        "evidence_refs": refs,
        "claims": [
            {
                "claim": "The answer is 5",
                "claim_refs": ["product_1:artifact:final_response"],
                "evidence_refs": refs,
                "observation": "2+3 gives 5",
                "judgment": "supported",
            }
        ],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "budget_attribute", ["request_budget_bytes", "max_request_bytes"]
)
async def test_measurement_wrapper_preserves_adapter_budgets_and_valid_json_examples(
    budget_attribute,
):
    ctx, ev, _, test = case()
    ctx["memories"] = ["m" * 170000]
    refs = [f"document:{i}" for i in range(7)]
    ev.update({ref: {"text": ref} for ref in refs})

    class Model:
        evidence_expansion_limit = 12

        def __init__(self):
            self.calls = []

        async def __call__(self, request):
            self.calls.append(request)
            for field, key in (
                ("evidence_access", "need_evidence"),
                ("tool_protocol", "need_tools"),
            ):
                value, _ = json.JSONDecoder().raw_decode(
                    request[field][request[field].index("{") :]
                )
                assert list(value) == [key]
            if len(self.calls) == 1:
                return {"need_evidence": refs}
            assert request["requested_evidence"] == {ref: ev[ref] for ref in refs}
            return verdict()

    model = Model()
    setattr(model, budget_attribute, 240000)
    result = await defaults(model).run(test, ctx, ev)
    assert result["status"] == "pass" and len(model.calls) == 2
    assert "at most 12 supplied refs" in model.calls[0]["evidence_access"]


@pytest.mark.asyncio
async def test_semantic_check_calls_real_tool_and_receipt_reaches_verdict():
    ctx, ev, _, test = case()
    requests = []

    async def model(request):
        requests.append(deepcopy(request))
        if len(requests) == 1:
            return {
                "need_tools": [
                    {"tool": "calculate", "arguments": {"expression": "2+3"}}
                ]
            }
        refs = request["context"]["tool_observations"]
        assert refs["tool:check:0"]["result"] == 5
        result = verdict("tool:check:0")
        result["_tool_evidence"] = {"forged": "made up"}
        return result

    result = await defaults(model).run(test, ctx, ev)
    assert result["status"] == "pass" and len(requests) == 2
    assert result["tool_observations"]["tool:check:0"]["status"] == "observed"
    assert (
        "_tool_evidence" not in result and "forged" not in result["tool_observations"]
    )
    assert "tool:check:0" not in ev


@pytest.mark.asyncio
async def test_unavailable_tool_and_infinite_requests_cannot_manufacture_pass():
    ctx, ev, _, test = case()

    async def model(request):
        if request["context"]["tool_observations"]:
            return verdict()
        return {
            "need_tools": [
                {"tool": "cuda.run", "arguments": {"source": "not executed"}}
            ]
        }

    result = await defaults(model).run(test, ctx, ev)
    assert result["status"] == "unknown"

    async def forever(_):
        return {
            "need_tools": [{"tool": "calculate", "arguments": {"expression": "2+3"}}]
        }

    assert (await defaults(forever).run(test, ctx, ev))["status"] == "unknown"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "recovery_kind", ["valid", "missing", "uncited", "other_tool", "other_input"]
)
async def test_failed_probe_needs_cited_executed_replacement_on_same_inputs(
    recovery_kind,
):
    ctx, ev, _, test = case()
    source = "product_1:artifact:final_response"
    ev["other_source"] = "unrelated"
    calls = []

    async def model(request):
        calls.append(1)
        if len(calls) == 1:
            return {
                "need_tools": [
                    {
                        "tool": "code.run",
                        "arguments": {
                            "program": "result = missing_variable",
                            "input_refs": [source],
                        },
                    }
                ]
            }
        if len(calls) == 2:
            if recovery_kind == "other_tool":
                return {
                    "need_tools": [
                        {"tool": "calculate", "arguments": {"expression": "2+3"}}
                    ]
                }
            return {
                "need_tools": [
                    {
                        "tool": "code.run",
                        "arguments": {
                            "program": "result = 2+3",
                            "input_refs": [
                                "other_source"
                                if recovery_kind == "other_input"
                                else source
                            ],
                        },
                    }
                ]
            }
        result = verdict(None if recovery_kind == "uncited" else "tool:check:1")
        if recovery_kind != "missing":
            result["measurement_recovery"] = [
                {
                    "failed_ref": "tool:check:0",
                    "replacement_refs": ["tool:check:1"],
                    "reason": "Corrected the numerical program using the same evidence.",
                }
            ]
        return result

    result = await defaults(model).run(test, ctx, ev)
    assert result["status"] == ("pass" if recovery_kind == "valid" else "unknown")
    assert result["tool_observations"]["tool:check:0"]["status"] == "unknown"
    assert result["tool_observations"]["tool:check:1"]["status"] == "observed"


@pytest.mark.asyncio
async def test_measurement_cap_preserves_receipts_and_allows_bounded_final_verdict():
    ctx, ev, _, test = case()
    probe = {"tool": "calculate", "arguments": {"expression": "2+3"}}
    test["execution_recipe"] = {"probes": [probe, probe, probe]}
    requests = []

    async def model(request):
        requests.append(deepcopy(request))
        if len(requests) == 1:
            assert request["measurement_budget"]["remaining_tool_calls"] == 1
            return {"need_tools": [probe, probe]}  # Would exceed the total cap of four.
        assert request["measurement_feedback"]["status"] == "request_not_executed"
        assert len(request["context"]["tool_observations"]) == 3
        return verdict("tool:check:0")

    result = await defaults(model).run(test, ctx, ev)
    assert result["status"] == "pass"
    assert result["execution_usage"] == {"model_requests": 2, "tool_calls": 3}
    assert len(result["tool_observations"]) == 3


@pytest.mark.asyncio
async def test_exhausted_measurement_protocol_stops_unknown_without_extra_execution():
    ctx, ev, _, test = case()
    probe = {"tool": "calculate", "arguments": {"expression": "2+3"}}
    test["execution_recipe"] = {"probes": [probe, probe, probe]}
    requests = []

    async def model(request):
        requests.append(deepcopy(request))
        return {"need_tools": [probe]}  # Ignores the final-only instruction.

    result = await defaults(model).run(test, ctx, ev)
    assert result["status"] == "unknown"
    assert result["execution_usage"] == {"model_requests": 3, "tool_calls": 4}
    assert len(result["tool_observations"]) == 4
    assert requests[-1]["measurement_budget"]["final_verdict_required"] is True
    assert requests[-1]["measurement_budget"]["remaining_tool_calls"] == 0
    assert "not executed" in result["reason"]


@pytest.mark.asyncio
async def test_rejected_claim_after_bounded_repair_retains_executed_receipt():
    ctx, ev, _, test = case()
    test["execution_recipe"] = {
        "probes": [{"tool": "calculate", "arguments": {"expression": "2+3"}}]
    }
    calls = []

    async def model(request):
        calls.append(request)
        result = verdict("tool:check:0")
        result["claims"][0]["judgment"] = "unverified"
        return result  # An unverified material claim cannot support pass.

    model.bounded_stage = True
    result = await defaults(model).run(test, ctx, ev)
    assert result["status"] == "unknown" and result["error"]
    assert len(calls) == 2  # Initial answer and one bounded contract repair.
    assert "Unresolved material claims" in result["diagnostic"]
    assert result["tool_observations"]["tool:check:0"]["result"] == 5
    assert result["execution_usage"] == {"model_requests": 2, "tool_calls": 1}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "defect", ["partial_status", "missing_reason", "missing_gap", "task_only_refs"]
)
async def test_final_verdict_contract_is_repaired_without_reexecuting_probe(defect):
    ctx, ev, _, test = case()
    test["execution_recipe"] = {
        "probes": [{"tool": "calculate", "arguments": {"expression": "2+3"}}]
    }
    calls = []

    async def model(request):
        calls.append(request)
        row = verdict("tool:check:0")
        row.update(
            status="fail",
            reason="The required explanation is omitted; the number is correct.",
            gap="Explain how the number was computed.",
        )
        if len(calls) == 1:
            if defect == "partial_status":
                row["status"] = "partial"
            elif defect == "missing_reason":
                row.pop("reason")
            elif defect == "missing_gap":
                row.pop("gap")
            else:
                row["evidence_refs"] = ["task"]
        else:
            assert "response_recovery" in request
            assert (
                request["context"]["tool_observations"]["tool:check:0"]["result"] == 5
            )
        return row

    model.bounded_stage = True
    result = await defaults(model).run(test, ctx, ev)
    assert result["status"] == "fail" and not result.get("error")
    assert result["gap"] == "Explain how the number was computed."
    assert result["execution_usage"] == {"model_requests": 2, "tool_calls": 1}


@pytest.mark.asyncio
async def test_repeated_invalid_verdict_is_unknown_after_one_repair_with_receipt_retained():
    ctx, ev, _, test = case()
    test["execution_recipe"] = {
        "probes": [{"tool": "calculate", "arguments": {"expression": "2+3"}}]
    }
    calls = []

    async def model(request):
        calls.append(request)
        return {**verdict("tool:check:0"), "status": "partial"}

    model.bounded_stage = True
    result = await defaults(model).run(test, ctx, ev)
    assert result["status"] == "unknown" and result["error"]
    assert "typed status" in result["diagnostic"]
    assert len(calls) == 2
    assert result["execution_usage"] == {"model_requests": 2, "tool_calls": 1}
    assert result["tool_observations"]["tool:check:0"]["result"] == 5


@pytest.mark.asyncio
async def test_missing_checker_authored_executed_saved_and_reloaded_in_fresh_registry():
    ctx, ev, criterion, test = case()
    test["checker"] = "custom.numeric.audit"
    calls = []

    async def model(request):
        calls.append(request["mode"])
        if request["mode"] == "outcome_builder":
            assert request["measurement_tools"]["code.run"]["available"]
            return {
                "rubrics": {
                    "math": {"fail": "wrong", "partial": "unclear", "full": "correct"}
                },
                "tests": [test],
            }
        if request["mode"] == "outcome_checker_author":
            return {
                "probes": [
                    {"tool": "code.run", "arguments": {"program": "result = 2+3"}}
                ]
            }
        assert request["context"]["tool_observations"]["tool:check:0"]["result"] == 5
        return verdict("tool:check:0")

    pack = await build(ctx, ev, {"criteria": [criterion]}, defaults(model), model)
    assert "outcome_checker_author" in calls
    assert pack["tests"][0]["execution_recipe"]["probes"]
    result = await measure(pack, defaults(model), ctx, ev)
    assert result["results"][0]["status"] == "pass"
    assert result["results"][0]["measurement_kind"] == "model_assisted"


@pytest.mark.asyncio
@pytest.mark.parametrize("repairable", [True, False])
async def test_checker_recipe_repairs_probe_count_and_actual_execution_with_bounded_attempts(
    repairable,
):
    from reward_engine.core.outcome.measurement_tools import author_checker

    ctx, ev, _, test = case()
    test["checker"] = "custom.repaired"
    ev["n"] = 5
    ctx["available_evidence_refs"].append("n")
    calls, records = [], []
    tools = MeasurementTools(record=lambda name, row: records.append(row))
    registry = defaults(lambda request: None, tools=tools)

    async def model(request):
        calls.append(request)
        if len(calls) == 1 or not repairable:
            return {
                "probes": [{"tool": "calculate", "arguments": {"expression": "2+3"}}]
                * 4
            }
        assert request["mode"] == "outcome_checker_author"
        if len(calls) == 2:
            assert "one to three" in request["response_recovery"]["details"]["reason"]
            assert not records  # Repair the contract before any probe executes.
            return {
                "probes": [
                    {
                        "tool": "code.run",
                        "arguments": {
                            "input_refs": ["n"],
                            "program": "result = inputs[0]",
                        },
                    }
                ]
            }
        receipt = request["execution_recovery"]["receipts"][0]
        assert receipt["status"] == "unknown" and receipt["error_type"] == "KeyError"
        return {
            "probes": [
                {
                    "tool": "code.run",
                    "arguments": {
                        "input_refs": ["n"],
                        "program": 'result = inputs["n"]',
                    },
                }
            ]
        }

    if repairable:
        await author_checker(test, registry, ctx, ev, model)
        assert "custom.repaired" in registry.checkers
        assert records[-1]["receipts"][0]["result"] == 5
    else:
        with pytest.raises(ValueError, match="one to three"):
            await author_checker(test, registry, ctx, ev, model)
        assert "custom.repaired" not in registry.checkers
    assert len(calls) == (3 if repairable else 2)


def test_published_sources_survive_fresh_runtime_context_without_becoming_user_authority():
    ctx, ev, _, _ = case()
    pack = {
        "measurement_sources": {
            "reference:task_review:s0": {"text": "Original evidence"}
        },
        "measurement_priors": [
            {
                "ref": "skill:task_review:k",
                "content": {"authority": "domain_advice_not_user_requirement"},
            }
        ],
    }
    view, refs = measurement_context(pack, ctx, ev)
    assert refs["reference:task_review:s0"]["text"] == "Original evidence"
    assert (
        view["skills"][0]["content"]["authority"]
        == "domain_advice_not_user_requirement"
    )
    assert "reference:task_review:s0" not in ev


@pytest.mark.asyncio
async def test_unavailable_gpu_recipe_not_published_as_callable():
    ctx, ev, criterion, test = case()
    test["probes"] = [{"tool": "cuda.run", "arguments": {"source": "not executed"}}]

    async def model(request):
        return {
            "rubrics": {
                "math": {"fail": "wrong", "partial": "unclear", "full": "correct"}
            },
            "tests": [test],
        }

    pack = await build(ctx, ev, {"criteria": [criterion]}, defaults(model), model)
    assert pack["status"] == "needs_implementation"
    assert pack["tests"][0]["required_tools"] == {"cuda.run": False}


@pytest.mark.asyncio
async def test_known_measurement_tool_used_as_checker_needs_no_author():
    ctx, ev, criterion, test = case()
    test.update(checker="calculate", arguments={"expression": "2+3"})

    async def model(request):
        assert request["mode"] == "outcome_builder"
        return {
            "rubrics": {
                "math": {"fail": "wrong", "partial": "unclear", "full": "correct"}
            },
            "tests": [test],
        }

    pack = await build(ctx, ev, {"criteria": [criterion]}, defaults(model), model)
    compiled = pack["tests"][0]
    assert (
        compiled["checker"] == "evidence.review"
        and compiled["measurement_tool"] == "calculate"
    )
    assert compiled["execution_recipe"]["probes"] == [
        {"tool": "calculate", "arguments": {"expression": "2+3"}}
    ]


@pytest.mark.asyncio
async def test_adapter_wires_reader_and_keeps_each_real_tool_receipt(
    tmp_path, monkeypatch
):
    from reward_engine.adapters.outcome_measurement import measurement_registry
    from reward_engine.adapters.tavily_review import TavilyReview

    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    fetched = []

    async def extract(self, url):
        fetched.append(url)
        return {"results": [{"url": url, "raw_content": "Real returned text"}]}

    monkeypatch.setattr(TavilyReview, "extract", extract)
    registry = measurement_registry(
        None, {"measurement": {"external_documents": True}}, tmp_path
    )
    urls = [
        "https://docs.example/a",
        "https://127.0.0.1/secret",
        "https://localhost/secret",
    ]
    for url in urls:
        receipt = await registry.tools.run(
            {"tool": "document.read", "arguments": {"url": url}},
            {"source": {"url": url}},
        )
        assert (receipt["status"] == "observed") == (url == urls[0])
    assert fetched == urls[:1]
    assert len(list((tmp_path / "measurements").glob("measurement_tool_*.json"))) == 3


@pytest.mark.asyncio
async def test_cuda_adapter_requires_execution_receipt_not_a_model_verdict():
    async def fake_verdict(*_):
        return {"status": "pass", "reason": "Should work"}

    tools = MeasurementTools(cuda_runner=fake_verdict)
    assert (await tools.run({"tool": "cuda.run", "arguments": {"source": "code"}}, {}))[
        "status"
    ] == "unknown"


@pytest.mark.asyncio
async def test_real_tools_feed_delivery_failure_revision_and_retest(tmp_path):
    from reward_engine.adapters.outcome_measurement import measurement_registry
    from reward_engine.core.outcome.runner import DeliveryReview, run_delivery

    ctx, _, criterion, test = case()
    test["execution_recipe"] = {
        "probes": [{"tool": "calculate", "arguments": {"expression": "2+3"}}]
    }
    pack = {
        "pack_id": "local-smoke",
        "task_id": ctx["task"]["task_id"],
        "task_prompt": ctx["task"]["prompt"],
        "criteria": [criterion],
        "tests": [test],
    }

    async def check(request):
        # Test substitute supplies interpretation; the numeric tool is real.
        actual = request["context"]["products"][0]["artifact"]["final_response"]
        expected = request["context"]["tool_observations"]["tool:check:0"]["result"]
        result = verdict("tool:check:0")
        if actual != str(expected):
            result.update(
                status="fail",
                reason="Number disagrees with executed calculation",
                gap="Correct the sum",
            )
            result["claims"][0].update(
                judgment="contradicted",
                claim=actual,
                observation=f"Calculator returned {expected}",
            )
        return result

    class Actor:
        async def start(self, prompt):
            return {"artifact": {"final_response": "4"}}

        async def resume(self, feedback):
            assert "Correct the sum" in feedback
            return {"artifact": {"final_response": "5"}}

    review = DeliveryReview(
        ctx["task"], pack, measurement_registry(check, {}, tmp_path), audit_mode="full"
    )
    final = await run_delivery(Actor(), review)
    assert final["artifact"]["final_response"] == "5"
    assert [row["results"][0]["status"] for row in review.history] == ["fail", "pass"]
    assert len(list((tmp_path / "measurements").glob("measurement_tool_*.json"))) == 2


@pytest.mark.asyncio
async def test_evidence_limit_retains_actual_measurements_and_bounded_usage():
    ctx, ev, _, test = case()
    test["execution_recipe"] = {
        "probes": [{"tool": "calculate", "arguments": {"expression": "2+3"}}]
    }
    calls = []
    ref = next(iter(ev))

    async def model(request):
        calls.append(deepcopy(request))
        return {"need_evidence": [ref]}

    model.bounded_stage = True
    result = await defaults(model).run(test, ctx, ev)
    assert result["status"] == "unknown" and result["error"]
    assert len(calls) == 3  # One read, one closure, then reject further reads.
    assert calls[-1]["evidence_access_closed"] is True
    assert result["unresolved_evidence_refs"] == [ref]
    assert result["tool_observations"]["tool:check:0"]["result"] == 5
    assert result["execution_usage"] == {"model_requests": 3, "tool_calls": 1}


@pytest.mark.asyncio
async def test_closed_final_pass_citing_unread_preview_retains_receipts_as_unknown():
    ctx, ev = build_context(
        {"task_id": "boundary", "prompt": "Verify the output"},
        [
            {
                "artifact": {"final_response": "The answer is 5"},
                "source_documents": [{"text": str(i) * 1000} for i in range(7)],
            }
        ],
    )
    _, _, _, test = case()
    test["execution_recipe"] = {
        "probes": [{"tool": "calculate", "arguments": {"expression": "2+3"}}]
    }
    calls = []

    async def model(request):
        calls.append(deepcopy(request))
        if len(calls) == 1:
            return {"need_evidence": [f"product_1:source:{i}" for i in range(6)]}
        row = verdict("tool:check:0")
        row["evidence_refs"].append("product_1:source:6")
        return row

    model.bounded_stage = True
    result = await defaults(model).run(test, ctx, ev)
    assert len(calls) == 3 and calls[-1]["evidence_access_closed"]
    assert result["status"] == "unknown" and result["error"]
    assert result["unresolved_evidence_refs"] == ["product_1:source:6"]
    assert result["tool_observations"]["tool:check:0"]["result"] == 5
    assert result["execution_usage"] == {"model_requests": 3, "tool_calls": 1}


@pytest.mark.asyncio
async def test_request_rejected_flag_distinguishes_malformed_from_unavailable():
    tools = MeasurementTools()
    evidence = {"source:table": {"rows": [["id"], [7]]}}
    malformed = await tools.run(
        {
            "tool": "code.run",
            "arguments": {"input_refs": ["missing:ref"], "program": "result = 1"},
        },
        evidence,
    )
    assert malformed["status"] == "unknown" and malformed["request_rejected"] is True
    assert malformed["error_type"] == "MeasurementRequestRejected"
    oversized = await tools.run(
        {
            "tool": "code.run",
            "arguments": {
                "input_refs": ["source:table"],
                "program": "result = len(inputs['source:table']['rows'])",
            },
        },
        {"source:table": {"blob": "x" * 300000, "rows": [["id"], [7]]}},
    )
    assert oversized["status"] == "unknown" and oversized["request_rejected"] is True
    assert "too large" in oversized["reason"]
    # A genuinely executed program failure is not a request rejection.
    executed = await tools.run(
        {
            "tool": "code.run",
            "arguments": {
                "input_refs": ["source:table"],
                "program": "result = inputs['src']",
            },
        },
        evidence,
    )
    assert executed["status"] == "unknown" and executed["request_rejected"] is False
    # A missing capability is an unavailable measurement, not a malformed request.
    unavailable = await tools.run(
        {"tool": "cuda.run", "arguments": {"source": "code"}}, {}
    )
    assert (
        unavailable["status"] == "unknown" and unavailable["request_rejected"] is False
    )


@pytest.mark.asyncio
async def test_rejected_probe_does_not_force_unknown_when_corrected_probe_executes():
    # Replay of the ticket weekly_totals_all_rows trajectory: probe :2 cited a
    # ref that was not supplied evidence (rejected, never executed), probe :3
    # then computed over the real evidence. The rejected request must not leave
    # an unresolved measurement that forces the whole check to unknown.
    ctx, ev, _, test = case()
    ev["other_source"] = "unrelated"
    requests = []

    async def model(request):
        requests.append(deepcopy(request))
        if len(requests) == 1:
            return {
                "need_tools": [
                    {
                        "tool": "code.run",
                        "arguments": {
                            "program": "result = inputs['product_1:source:0']",
                            "input_refs": ["product_1:source:0"],
                        },
                    }
                ]
            }
        if len(requests) == 2:
            return {
                "need_tools": [
                    {
                        "tool": "code.run",
                        "arguments": {
                            "program": "result = len(inputs['other_source'])",
                            "input_refs": ["other_source"],
                        },
                    }
                ]
            }
        return verdict("tool:check:1")

    result = await defaults(model).run(test, ctx, ev)
    assert result["status"] == "pass", result
    assert result["tool_observations"]["tool:check:0"]["request_rejected"] is True
    assert result["tool_observations"]["tool:check:0"]["status"] == "unknown"
    assert result["tool_observations"]["tool:check:1"]["status"] == "observed"
    assert result["unresolved_measurement_refs"] == []
    assert result["execution_usage"] == {"model_requests": 3, "tool_calls": 1}


@pytest.mark.asyncio
async def test_oversized_selection_is_correctable_without_consuming_executed_budget():
    # Replay of the inventory t_sheet2_predicate_correct trajectory: the model
    # first cited a whole oversized source, then reselected a bounded path. The
    # rejected first attempt must not consume the four-execution budget.
    ctx, ev, _, test = case()
    ev["big_source"] = {"blob": "x" * 300000, "rows": [["h"], [1], [2]]}
    requests = []

    async def model(request):
        requests.append(deepcopy(request))
        if len(requests) == 1:
            return {
                "need_tools": [
                    {
                        "tool": "code.run",
                        "arguments": {
                            "input_refs": ["big_source"],
                            "program": "result = len(inputs['big_source']['rows'])",
                        },
                    }
                ]
            }
        if len(requests) == 2:
            assert request["measurement_budget"]["remaining_tool_calls"] == 4
            return {
                "need_tools": [
                    {
                        "tool": "code.run",
                        "arguments": {
                            "input_refs": ["big_source"],
                            "input_paths": {"big_source": ["rows"]},
                            "program": "rows = inputs['big_source']\nresult = len(rows)",
                        },
                    }
                ]
            }
        return verdict("tool:check:1")

    result = await defaults(model).run(test, ctx, ev)
    assert result["status"] == "pass", result
    assert result["tool_observations"]["tool:check:0"]["request_rejected"] is True
    assert "too large" in result["tool_observations"]["tool:check:0"]["reason"]
    assert result["tool_observations"]["tool:check:1"]["status"] == "observed"
    assert result["unresolved_measurement_refs"] == []


@pytest.mark.asyncio
async def test_executed_probe_failure_still_requires_observed_recovery():
    # The rejected-request exemption must not weaken the executed-failure rule:
    # a KeyError inside a real program still forces unknown until a cited,
    # executed replacement covers the same inputs.
    ctx, ev, _, test = case()
    requests = []

    async def model(request):
        requests.append(deepcopy(request))
        if len(requests) == 1:
            return {
                "need_tools": [
                    {
                        "tool": "code.run",
                        "arguments": {
                            "input_refs": ["product_1:artifact:final_response"],
                            "program": "result = inputs['invented_alias']",
                        },
                    }
                ]
            }
        return verdict()  # No recovery declared, no replacement receipt cited.

    result = await defaults(model).run(test, ctx, ev)
    assert result["status"] == "unknown", result
    assert result["tool_observations"]["tool:check:0"]["request_rejected"] is False
    assert result["unresolved_measurement_refs"] == ["tool:check:0"]
