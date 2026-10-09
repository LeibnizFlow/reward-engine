import asyncio
from copy import deepcopy

import pytest

from reward_engine.core.outcome.builder import build
from reward_engine.core.outcome.harness import Registry, defaults, measure
from tests.support.claw_outcome_builder_harness import (
    build_context,
    setup,
)


def test_build_and_execute_real_json():
    context, evidence, jury, reply = setup()

    async def call(request):
        return reply

    async def no_model(request):
        raise AssertionError("JSON needs no model")

    registry = defaults(no_model)
    pack = asyncio.run(build(context, evidence, jury, registry, call))
    result = asyncio.run(measure(pack, registry, context, evidence))
    assert result["ready_under_estimator"] and result["coverage"] == 1
    evidence["product_1:artifact:final_response"] = "{broken"
    result = asyncio.run(measure(pack, registry, context, evidence))
    assert "score_on_observed" not in result and not result["ready_under_estimator"]
    assert result["results"][0]["gap"].startswith("Invalid JSON")


def test_builder_recovers_unreadable_future_handle_without_exposing_historical_product():
    context, evidence, jury, reply = setup()
    calls = []

    async def call(request):
        calls.append(deepcopy(request))
        if len(calls) == 1:
            return {"need_evidence": ["product_1:artifact:final_response"]}
        assert request["response_recovery"]["details"]["field"] == "need_evidence"
        assert request["context"]["products"] == []
        assert "requested_evidence" not in request
        return reply

    pack = asyncio.run(build(context, evidence, jury, defaults(call), call))
    assert len(calls) == 2
    assert pack["tests"][0]["arguments"]["ref"] == "product_1:artifact:final_response"
    assert calls[0]["context"] == calls[1]["context"]


def test_builder_unreadable_handle_retry_is_bounded():
    context, evidence, jury, _ = setup()
    calls = []

    async def call(request):
        calls.append(request)
        return {"need_evidence": ["product_1:artifact:final_response"]}

    with pytest.raises(ValueError, match="Evidence references must exist"):
        asyncio.run(build(context, evidence, jury, defaults(call), call))
    assert len(calls) == 2


def test_historical_artifact_argument_binds_new_product_not_old_answer():
    context, evidence, jury, reply = setup()
    reply["tests"][0]["arguments"]["ref"] = "product_6:artifact:final_response"
    original = deepcopy(reply)
    records = {}

    async def call(_):
        return reply

    pack = asyncio.run(
        build(
            context,
            evidence,
            jury,
            defaults(call),
            call,
            record=lambda k, v: records.update({k: v}),
        )
    )
    assert reply == original
    assert (
        records["builder_artifact_bindings/0"]["bindings"][0]["before"]
        == "product_6:artifact:final_response"
    )
    current, current_evidence = build_context(
        context["task"], [{"artifact": {"final_response": "{broken"}}]
    )
    measured = asyncio.run(measure(pack, defaults(call), current, current_evidence))
    assert measured["results"][0]["status"] == "fail"
    assert measured["results"][0]["evidence_refs"] == [
        "product_1:artifact:final_response"
    ]


def test_semantic_argument_and_probe_bindings_keep_test_and_rubric():
    context, evidence, jury, reply = setup()
    test = reply["tests"][0]
    test.update(
        checker="evidence.review",
        arguments={
            "evidence_refs": [
                "task",
                "product_1:artifact:final_response",
                "product_2:artifact:final_response",
                "product_6:artifact:final_response",
            ]
        },
        probes=[
            {
                "tool": "document.read",
                "arguments": {"ref": "product_3:artifact:final_response", "offset": 0},
            }
        ],
    )

    async def call(_):
        return reply

    pack = asyncio.run(build(context, evidence, jury, defaults(call), call))
    bound = pack["tests"][0]
    assert bound["arguments"]["evidence_refs"] == [
        "task",
        "product_1:artifact:final_response",
    ]
    assert bound["probes"][0]["arguments"] == {
        "ref": "product_1:artifact:final_response",
        "offset": 0,
    }
    assert bound["question"] == test["question"]
    assert pack["criteria"][0]["standard"] == jury["criteria"][0]["standard"]


@pytest.mark.parametrize(
    "ref", ["product_2:public:0", "product_2:source:4", "product_2:artifact:undeclared"]
)
def test_historical_user_source_and_unknown_artifacts_cannot_be_rebound(ref):
    context, evidence, jury, reply = setup()
    reply["tests"][0]["arguments"]["ref"] = ref

    async def call(_):
        return reply

    with pytest.raises(ValueError, match="cannot bind a historical product"):
        asyncio.run(build(context, evidence, jury, defaults(call), call))


def test_historical_reference_in_question_is_not_rewritten():
    context, evidence, jury, reply = setup()
    reply["tests"][0]["question"] = "Is this identical to product_2?"

    async def call(_):
        return reply

    with pytest.raises(ValueError, match="cannot bind a historical product"):
        asyncio.run(build(context, evidence, jury, defaults(call), call))


def test_compile_excludes_historical_handles_but_published_provenance_is_lossless():
    _, _, jury, reply = setup()
    task = {"task_id": "x", "prompt": "Deliver JSON"}
    context, evidence = build_context(
        task,
        [
            {"artifact": {"final_response": "old answer"}},
            {
                "artifact": {"final_response": "another old answer"},
                "source_documents": ["old manual"],
            },
        ],
    )
    jury["criteria"][0]["evidence_refs"] = ["task", "product_2:source:0"]
    original = deepcopy(jury)
    records = {}

    async def call(request):
        assert request["context"]["products"] == []
        assert request["jury"]["criteria"][0]["evidence_refs"] == ["task"]
        assert (
            request["jury"]["criteria"][0]["standard"]
            == original["criteria"][0]["standard"]
        )
        assert (
            "product_1:artifact:final_response"
            in request["context"]["current_product_contract"]["artifact_refs"]
        )
        return reply

    pack = asyncio.run(
        build(
            context,
            evidence,
            jury,
            defaults(call),
            call,
            record=lambda k, v: records.update({k: v}),
        )
    )
    assert jury == original
    assert pack["criteria"][0]["evidence_refs"] == ["task", "product_2:source:0"]
    assert records["builder_provenance_scope"]["excluded_from_executable_context"] == {
        "syntax": ["product_2:source:0"]
    }


def test_compile_preserves_actual_scoped_user_requirements():
    _, _, jury, reply = setup()
    context, evidence = build_context(
        {"task_id": "x", "prompt": "Deliver JSON"},
        [
            {"artifact": {"final_response": "old"}},
            {
                "artifact": {
                    "final_response": "old",
                    "public_consultation": {
                        "earlier_exchanges": [
                            {"speaker": "user", "text": "Use UTF-8 for my file"}
                        ]
                    },
                }
            },
        ],
    )
    jury["criteria"][0].update(origin="user", evidence_refs=["product_2:public:0"])

    async def call(request):
        assert [p["product_id"] for p in request["context"]["products"]] == [
            "product_2"
        ]
        assert request["jury"]["criteria"][0]["evidence_refs"] == ["product_2:public:0"]
        assert (
            request["context"]["products"][0]["user_updates"][0]["text"]
            == "Use UTF-8 for my file"
        )
        return reply

    pack = asyncio.run(build(context, evidence, jury, defaults(call), call))
    assert pack["criteria"][0]["evidence_refs"] == ["product_2:public:0"]


@pytest.mark.parametrize("shape", ["keyed", "list", "bad_anchors", "bad_container"])
def test_builder_anchor_table_normalization_is_not_fabrication(shape):
    context, evidence, jury, reply = setup()
    anchors = reply["rubrics"][0]["anchors"]
    if shape == "keyed":
        reply["rubrics"] = {"syntax": anchors}
    elif shape == "bad_anchors":
        reply["rubrics"] = {"syntax": "not anchors"}
    elif shape == "bad_container":
        reply["rubrics"] = "not rubrics"

    async def call(_):
        return reply

    if shape.startswith("bad"):
        with pytest.raises(ValueError):
            asyncio.run(build(context, evidence, jury, defaults(call), call))
    else:
        pack = asyncio.run(build(context, evidence, jury, defaults(call), call))
        assert pack["criteria"][0]["anchors"] == anchors
        assert pack["criteria"][0]["standard"] == jury["criteria"][0]["standard"]


def test_open_key_unbound_stays_unknown_then_real_author():
    context, evidence, jury, reply = setup()
    reply["tests"][0]["checker"] = "new.custom.reader.v99"

    async def call(request):
        return reply

    registry = Registry()
    pack = asyncio.run(build(context, evidence, jury, registry, call))
    result = asyncio.run(measure(pack, registry, context, evidence))
    assert pack["status"] == "needs_implementation"
    assert result["coverage"] == 0 and "score_on_observed" not in result
    calls = []

    async def author(test, registry):
        calls.append(test["checker"])

        async def checker(test, context, evidence):
            return {
                "status": "pass",
                "reason": "Actual check",
                "evidence_refs": ["product_1:artifact:final_response"],
            }

        registry.register(test["checker"], checker, "test checker")

    pack = asyncio.run(build(context, evidence, jury, registry, call, author=author))
    assert calls == ["new.custom.reader.v99"] and pack["status"] == "callable"
    assert asyncio.run(measure(pack, registry, context, evidence))["coverage"] == 1


def test_builder_cannot_drop_requirement_and_check_errors_not_fail():
    context, evidence, jury, reply = setup()

    async def call(request):
        return reply

    registry = Registry()

    async def broken(*args):
        raise RuntimeError("environment unavailable")

    registry.register("json.parse", broken, "broken")
    pack = asyncio.run(build(context, evidence, jury, registry, call))
    result = asyncio.run(measure(pack, registry, context, evidence))
    assert result["results"][0]["status"] == "unknown" and result["results"][0]["error"]
    reply["rubrics"] = []
    with pytest.raises(ValueError):
        asyncio.run(build(context, evidence, jury, registry, call))
