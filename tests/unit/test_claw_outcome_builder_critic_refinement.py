"""Bounded compilation, cheap uncertainty handling, and multi-turn Outcome advice."""

import json
from copy import deepcopy

import pytest

from reward_engine.core.outcome.builder import build
from reward_engine.core.outcome.context import build_context, measurement_test_view
from reward_engine.core.outcome.delivery_triage import review
from reward_engine.core.outcome.harness import defaults, measure
from reward_engine.core.outcome.measurement_tools import author_checker
from reward_engine.core.outcome.policy_critic import PolicyCritic
from reward_engine.core.outcome.runner import DeliveryReview
from tests.support.claw_outcome_builder_harness import setup
from tests.support.claw_outcome_pruning import TASK, candidate, pack
from tests.support.outcome_policy_critic import answer


@pytest.mark.asyncio
async def test_builder_repairs_once_and_validates_all_before_author_side_effects():
    context, evidence, jury, reply = setup()
    reply["tests"][0]["checker"] = "custom.parser"
    broken = deepcopy(reply)
    broken["tests"].append(
        deepcopy(broken["tests"][0])
    )  # Duplicate after a valid first row.
    requests, authored, records = [], [], {}

    async def model(request):
        requests.append(request)
        assert not authored
        return deepcopy(broken if len(requests) == 1 else reply)

    async def author(test, registry):
        authored.append(test["checker"])

    result = await build(
        context,
        evidence,
        jury,
        defaults(model),
        model,
        author=author,
        record=records.__setitem__,
    )
    assert len(requests) == 2 and authored == ["custom.parser"]
    assert requests[1]["repair"]["original_output"] == broken
    assert "contract_error" in requests[1]["repair"]
    assert result["status"] == "needs_implementation"
    assert result["measurement_profile"]["counts"]["unavailable"] == 1
    assert records["builder_reuse"]["reuse_fraction"] == 0
    assert "builder_error_1" in records and "builder_attempt_2" in records


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", [False, True])
async def test_no_unbounded_repair_or_reinterpretation_of_transport_error(transport):
    context, evidence, jury, reply = setup()
    requests, records = [], {}
    reply["rubrics"] = []

    async def model(request):
        requests.append(request)
        if transport:
            raise TimeoutError("service unavailable")
        return reply

    with pytest.raises(TimeoutError if transport else ValueError):
        await build(
            context, evidence, jury, defaults(model), model, record=records.__setitem__
        )
    assert len(requests) == (1 if transport else 2)
    if not transport:
        assert "builder_error_2" in records and "pack" not in records


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ref", ["product_1:public:0", "product_2:artifact:final_response"]
)
async def test_historical_bindings_rejected_at_compile_and_runtime(ref):
    context, evidence, jury, reply = setup()
    reply["tests"][0]["arguments"]["ref"] = ref

    async def model(_):
        return reply

    registry = defaults(model)
    from reward_engine.core.outcome.builder import compile_contract

    with pytest.raises(ValueError):
        compile_contract(reply, jury, registry)
    if ref == "product_2:artifact:final_response":
        bundle = await build(context, evidence, jury, registry, model)
        assert (
            bundle["tests"][0]["arguments"]["ref"]
            == "product_1:artifact:final_response"
        )
        assert reply["tests"][0]["arguments"]["ref"] == ref
    else:
        with pytest.raises(ValueError):
            await build(context, evidence, jury, registry, model)
    with pytest.raises(ValueError):
        registry.bind_recipe(
            "unsafe", {"probes": [{"tool": "document.read", "arguments": {"ref": ref}}]}
        )
    row = await registry.run(reply["tests"][0], context, {**evidence, ref: "{}"})
    assert row["status"] == "unknown" and row["error"]


@pytest.mark.asyncio
async def test_structural_contract_does_not_leak_product_values_and_reuse_is_recorded():
    context, evidence, jury, reply = setup()
    evidence["product_1:environment"] = {"secret_key_name": "PRIVATE_VALUE"}
    context["products"][0]["artifact"]["final_response"] = "PRIVATE_ARTIFACT"
    requests, records = [], {}

    async def model(request):
        requests.append(request)
        return reply

    registry = defaults(model)
    first = await build(context, evidence, jury, registry, model)
    second = await build(
        context,
        evidence,
        jury,
        registry,
        model,
        previous=first,
        record=records.__setitem__,
    )
    contract = requests[0]["context"]["current_product_contract"]
    assert contract["artifact_types"]["final_response"] == ["str"]
    assert contract["environment_keys"] == ["secret_key_name"]
    assert "PRIVATE_" not in json.dumps(requests[0])
    assert len(requests) == 1 and records["builder_reuse"]["reuse_fraction"] == 1
    assert second["measurement_profile"]["counts"]["deterministic"] == 1
    assert second["measurement_profile"]["executed"] is False


@pytest.mark.asyncio
async def test_measurement_projection_keeps_semantics_but_not_drafting_ledger():
    context, evidence, jury, reply = setup()
    rubric = {
        **jury["criteria"][0],
        "anchors": reply["rubrics"][0]["anchors"],
        "subrequirements": ["Respect actual request"],
        "applicability": "Only when user requested JSON",
        "measurement_question": "Is actual current output valid?",
        "why": "User requirement",
        "evidence_refs": ["task"],
        "unknown_future_semantics": "KEEP",
        "part_revisions": ["DRAFT_LEDGER" * 100],
        "draft_validation_issues": ["OLD_DRAFT"],
    }
    test = {**reply["tests"][0], "checker": "evidence.review", "rubric": rubric}
    original = deepcopy(test)
    requests = []

    async def model(request):
        requests.append(request)
        if len(requests) == 1:
            return {"need_evidence": ["syntax"]}
        assert (
            request["requested_evidence"]["syntax"]
            == measurement_test_view(test)["rubric"]
        )
        return {
            "claims": [],
            "status": "unknown",
            "reason": "Insufficient proof",
            "gap": "",
            "evidence_refs": [],
        }

    result = await defaults(model).run(test, context, evidence)
    assert result["execution_usage"] == {"model_requests": 2, "tool_calls": 0}
    assert test == original and "DRAFT_LEDGER" not in json.dumps(requests)
    for field in (
        "standard",
        "anchors",
        "subrequirements",
        "applicability",
        "measurement_question",
        "why",
        "evidence_refs",
        "unknown_future_semantics",
    ):
        assert requests[0]["test"]["rubric"][field] == rubric[field]


def overview(*, verify=False, gap=False):
    return {
        "outcome_goal": "Fully satisfy the declared task with a correct, complete result.",
        "full_score_gaps": [],
        "improvements": [],
        "status": "needs_improvement" if gap else "target_satisfied",
        "reason": "Current review",
        "claim_checks": [],
        "gaps": [
            {
                "gap": "Invalid JSON",
                "evidence_refs": ["product_1:artifact:final_response"],
            }
        ]
        if gap
        else [],
        "verify": [{"test_id": "parse", "reason": "Check syntax"}] if verify else [],
    }


@pytest.mark.asyncio
async def test_unavailable_only_preserves_delivery_without_actor_rewrite():
    async def model(_):
        return overview(verify=True)

    registry = defaults(model)

    async def unavailable(*_):
        raise TimeoutError("checker down")

    registry.checkers["json.parse"] = (unavailable, "test outage")
    runner = DeliveryReview(TASK, pack(), registry)
    assert await runner(candidate()) is None
    assert runner.status == "check_unavailable" and runner.feedback == []
    assert not runner.history[-1]["delivery_clear"]
    assert runner.artifacts[-1] == candidate()
    assert not runner.progress.verified


@pytest.mark.asyncio
async def test_error_not_automatically_retried_or_forgotten_explicit_retry_can_resolve():
    context, evidence = build_context(TASK, [candidate()])
    selection, count = True, 0

    async def model(_):
        return overview(verify=selection)

    registry = defaults(model)
    parser = registry.checkers["json.parse"]

    async def unavailable(*_):
        nonlocal count
        count += 1
        raise TimeoutError("offline")

    registry.checkers["json.parse"] = (unavailable, "outage")
    result = await review(pack(), registry, context, evidence)
    selection = False
    for _ in range(2):
        result = await review(
            pack(), registry, context, evidence, previous=(context, evidence, result)
        )
        assert not result["delivery_clear"] and result["results"] == []
        assert len(result["unresolved_checks"]) == 1
        assert result["checks_unavailable_only"]
    assert count == 1
    registry.checkers["json.parse"] = parser
    selection = True
    result = await review(
        pack(), registry, context, evidence, previous=(context, evidence, result)
    )
    assert result["delivery_clear"] and result["unresolved_checks"] == []


@pytest.mark.asyncio
async def test_actual_defect_retests_even_if_other_check_is_unavailable():
    context, evidence = build_context(TASK, [candidate("{")])

    async def model(_):
        return overview()

    prior = {
        "results": [
            {
                "test_id": "parse",
                "criterion_id": "syntax",
                "status": "fail",
                "gap": "Bad syntax",
            },
            {
                "test_id": "other",
                "criterion_id": "syntax",
                "status": "unknown",
                "error": True,
                "reason": "offline",
            },
        ]
    }
    result = await review(
        pack(), defaults(model), context, evidence, previous=(context, evidence, prior)
    )
    assert [r["test_id"] for r in result["results"]] == ["parse"]
    assert result["results"][0]["status"] == "fail"
    assert not result["checks_unavailable_only"] and not result["delivery_clear"]


@pytest.mark.asyncio
async def test_critic_three_part_advice_uses_current_state_without_extra_calls():
    requests = []

    async def model(request):
        requests.append(request)
        assert (
            "stage goal" in request["instruction"]
            and "changing direction" in request["instruction"]
        )
        assert "Open questions are unverified leads" in request["instruction"]
        text = request["context"]["products"][0]["artifact"]["final_response"]
        return answer(
            global_assessment="Deliver valid JSON",
            local_guidance=(
                "Inspect delimiters; correct syntax; if parsing still fails, isolate the failing fragment."
                if text == "{"
                else "Syntax is repaired; check required content; ask the user if the required fields remain unclear."
            ),
        )

    critic = PolicyCritic(TASK, pack(), defaults(model))
    first = await critic(candidate("{"), turn=3)
    second = await critic(candidate("{}"), turn=6)
    assert len(requests) == 2 and first["feedback"] != second["feedback"]
    assert (
        "if parsing still fails" in first["feedback"]
        and "ask the user" in second["feedback"]
    )
    assert all(r["open_questions"] == [] for r in requests)
    assert first["measurement"] is None and second["measurement"] is None


@pytest.mark.asyncio
async def test_execution_counts_do_not_count_reused_parser_as_new_work():
    context, evidence, jury, reply = setup()

    async def model(_):
        return reply

    registry = defaults(model)
    bundle = await build(context, evidence, jury, registry, model)
    first = await measure(bundle, registry, context, evidence)
    second = await measure(
        bundle, registry, context, evidence, previous=(context, evidence, first)
    )
    assert first["execution_summary"]["submitted_checks"] == 1
    assert second["execution_summary"]["submitted_checks"] == 0
    assert second["execution_summary"]["reused_checks"] == 1
    assert second["execution_summary"]["reported_model_requests"] == 0


@pytest.mark.asyncio
async def test_critic_receives_unverified_common_omission_without_promoting_it_to_defect():
    bundle = pack()
    bundle.update(
        expectation_coverage=[
            {
                "expectation_id": "e",
                "status": "deferred",
                "reason": "Needs applicability check",
            }
        ],
        independent_expectations=[
            {
                "expectation_id": "e",
                "standard": "Consider schema relevance",
                "origin": "domain",
                "evidence_refs": ["task"],
            }
        ],
    )

    async def model(request):
        question = request["open_questions"][0]
        assert question["questions"] == ["Consider schema relevance"]
        assert (
            question["status"] == "unverified_proposal"
            and question["origin"] == "domain"
        )
        assert question["pending_sources"][0]["reason"] == "Needs applicability check"
        return answer(
            status="observe",
            local_guidance="",
            global_assessment="No intervention needed",
        )

    result = await PolicyCritic(TASK, bundle, defaults(model))(candidate(), turn=3)
    assert (
        result["status"] == "observe"
        and result["feedback"] == ""
        and result["gaps"] == []
    )


@pytest.mark.asyncio
async def test_author_cannot_dry_run_historical_user_ordinal():
    context, evidence, _, reply = setup()
    invoked = []

    async def model(_):
        return {
            "probes": [
                {"tool": "document.read", "arguments": {"ref": "product_1:public:0"}}
            ]
        }

    registry = defaults(model)

    async def tool(*_):
        invoked.append(True)
        raise AssertionError("Must reject before tool execution")

    registry.tools.run = tool
    test = {**reply["tests"][0], "checker": "custom.reader"}
    with pytest.raises(ValueError, match="historical user-message"):
        await author_checker(test, registry, context, evidence, model)
    assert not invoked and "custom.reader" not in registry.checkers


@pytest.mark.asyncio
async def test_unsafe_saved_template_rebuilt_not_reused():
    context, evidence, jury, reply = setup()
    requests = []

    async def model(request):
        requests.append(request)
        return reply

    registry = defaults(model)
    original = await build(context, evidence, jury, registry, model)
    original["tests"][0]["arguments"]["ref"] = "product_1:public:0"
    current = await build(context, evidence, jury, registry, model, previous=original)
    assert len(requests) == 2
    assert (
        current["tests"][0]["arguments"]["ref"] == "product_1:artifact:final_response"
    )
