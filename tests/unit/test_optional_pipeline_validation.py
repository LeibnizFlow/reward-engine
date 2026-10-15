"""Offline behavioral coverage of optional Outcome paths; NOT live validation.

Actors and model roles below are deterministic fakes. The core controller, request
contracts, JSON checker, publication and selection execute normally, without any
provider, native runtime or paid call. Critic scenarios restore/adapt the archived
test_outcome_policy_critic and test_claw_outcome_feedback_arms behaviors.
"""
# Unicode escapes preserve CJK coverage and original UTF-8/character boundaries.

import asyncio
import json
from copy import deepcopy

import pytest

from reward_engine.core.outcome import (
    arms,
    call_budget,
    hypotheses,
    improvement,
    loop,
    request_projection,
    review_policy,
    validated_construction,
    version_selection,
)
from reward_engine.core.outcome.admission import critical_tests, require_publication
from reward_engine.core.outcome.construction import fingerprint
from reward_engine.core.outcome.context import (
    RequestBudgetExceeded,
    ResponseContractError,
    ask,
    build_context,
)
from reward_engine.core.outcome.counterexamples import verify_issue
from reward_engine.core.outcome.harness import (
    MEASUREMENT_INPUT_SCHEMA,
    defaults,
    measure,
    measurement_context,
    measurement_input_fingerprint,
    summarize_results,
)
from reward_engine.core.outcome.policy_critic import (
    PolicyCritic,
    check_summaries,
    failed_check,
)
from reward_engine.core.outcome.runner import DeliveryReview, run_delivery
from tests.support.claw_outcome_feedback_arms import (
    METHOD,
    PACK,
    TASK,
    candidate,
    model,
)
from tests.support.outcome_clauses_and_disputes import issue
from tests.support.outcome_policy_critic import answer
from tests.support.synthetic_measurements import scalar_inquiries


def structured_pack():
    pack = deepcopy(PACK)
    pack["criteria"][0]["clauses"] = [
        {
            "id": "syntax",
            "text": TASK["prompt"],
            "use": "task_constraint",
            "evidence_refs": ["task"],
            "source_quotes": [{"ref": "task", "quote": TASK["prompt"]}],
            "conditions": [],
            "alternative_group": "",
        }
    ]
    pack["tests"][0]["rubric"] = deepcopy(pack["criteria"][0])
    return pack


def published_pack():
    """Synthetic case support for selection tests, not a live calibration claim."""
    pack = structured_pack()
    return validated_construction.publish_validation(
        pack,
        {
            "status": "completed",
            "artifact_sha256": fingerprint(pack),
            "records": [
                {
                    "target_id": "syntax",
                    "validation_id": "offline-syntax-fixture",
                    "status": "supported_on_cases",
                    "cases": [],
                }
            ],
        },
    )


def version(identity, text):
    value = candidate(text)
    return {
        "candidate_id": identity,
        "candidate": value,
        "artifact_sha256": fingerprint(value["artifact"]),
        "measurement": None,
    }


def quality_advice():
    ref = "product_1:artifact:final_response"
    return answer(
        full_score_gaps=[
            {"id": "syntax-gap", "gap": "JSON is unfinished", "evidence_refs": [ref]}
        ],
        improvements=[
            {
                "id": "repair-json",
                "gap_ids": ["syntax-gap"],
                "action": "Close the JSON object in final_response",
                "expected_gain": "Make the delivered result parseable",
                "verification": "Parse the revised final_response as JSON",
                "evidence_refs": [ref],
            }
        ],
    )


@pytest.mark.parametrize(
    "hypothesis",
    [METHOD, None, {"status": "keep"}, {"status": "propose", "method": " "}],
)
@pytest.mark.asyncio
async def test_feedback_loop_routes_c_and_d_with_real_checks(monkeypatch, hypothesis):
    """Fake construction/actors; real loop routing, review and runtime recheck."""
    config = {"arm_protocol": arms.FEEDBACK}
    enabled = arms.configure(config, ["A", "B", "C"])
    assert enabled == list("ABCD")
    assert config["policy_critic"]["every_k_turns"] == 1
    seen, receipts, records = {}, [], {}
    registry = defaults(model)

    async def propose(*_, **__):
        return {"criteria": deepcopy(PACK["criteria"]), "gaps": []}

    async def build(*_, **__):
        return deepcopy(PACK)

    async def method(*_, **__):
        return deepcopy(hypothesis) if hypothesis else {"status": "keep", "method": ""}

    async def rank(*_, **__):
        return {"status": "unavailable", "reason": "No fake outcome preference"}

    monkeypatch.setattr(loop.jury, "propose", propose)
    monkeypatch.setattr(loop.builder, "build", build)
    monkeypatch.setattr(loop.hypotheses, "propose", method)
    monkeypatch.setattr(loop.jury, "rank", rank)
    active = set("ABCD" if arms.has_hypothesis(hypothesis) else "ACD")
    all_started = asyncio.Event()

    async def sample(arm, round_number, prompt, pack):
        assert round_number == 1
        seen[arm] = (prompt, deepcopy(pack))
        if set(seen) == active:
            all_started.set()
        await asyncio.wait_for(all_started.wait(), 2)
        if arm == "C":
            review = DeliveryReview(TASK, pack, registry, max_revisions=1)
            starts, resumes = [], []

            class FakeActor:
                async def start(self, guidance):
                    starts.append(guidance)
                    return candidate("broken")

                async def resume(self, feedback):
                    resumes.append(feedback)
                    assert "JSON" in feedback
                    return candidate()

            result = await run_delivery(FakeActor(), review, prompt=prompt)
            assert starts == [""] and len(resumes) == 1
            assert [r["results"][0]["status"] for r in review.history] == [
                "fail",
                "pass",
            ]
        elif arm == "D":
            critic = PolicyCritic(TASK, pack, registry, config=config["policy_critic"])
            for turn, text in ((1, "broken"), (2, "{}")):
                receipt = await critic(candidate(text), turn=turn)
                assert receipt["status"] == "suggest"
                assert "Repair JSON" in receipt["feedback"]
                receipts.append(receipt["measurement"]["results"][0]["status"])
            assert critic.valid_runtime_reviews == 2
            result = candidate()
        else:
            assert pack is None
            result = candidate()
        return {"status": "completed", "candidate": result}

    result = await loop.run_task(
        TASK,
        [candidate() for _ in range(3)],
        rounds=1,
        roles=lambda *_: model,
        registry=registry,
        sample=sample,
        enabled_arms=enabled,
        arm_protocol=arms.FEEDBACK,
        record=lambda name, value: records.update({name: deepcopy(value)}),
    )
    assert set(seen) == active
    assert seen["A"][0] == seen["C"][0] == seen["D"][0] == ""
    assert seen["C"][1] == seen["D"][1] == PACK
    assert receipts == ["fail", "pass"]
    assert not any(name.endswith("/failure") for name in records)
    if "B" in active:
        assert METHOD["method"] in seen["B"][0]
    else:
        assert records["round_1/B/skipped"]["status"] == "skipped"
    assert result["completed_rounds"] == 1, result
    assert result["status"] == "completed" and result["stop_reason"] == "round_limit"


@pytest.mark.parametrize("protocol", [arms.FEEDBACK, arms.LEGACY])
@pytest.mark.parametrize("hypothesis", [METHOD, {"status": "keep", "method": ""}])
@pytest.mark.parametrize("broken_arm", ["A", "B", "C", "D"])
@pytest.mark.parametrize("status", ["failed", "skipped", "not_run"])
@pytest.mark.parametrize("retained_artifact", [False, True])
@pytest.mark.asyncio
async def test_only_controller_decided_feedback_b_skip_removes_an_expected_arm(
    monkeypatch, protocol, hypothesis, broken_arm, status, retained_artifact
):
    sampled = []

    async def propose(*_, **__):
        return {"criteria": deepcopy(PACK["criteria"]), "gaps": []}

    async def build(*_, **__):
        return deepcopy(PACK)

    async def method(*_, **__):
        return deepcopy(hypothesis)

    async def rank(*_, **__):
        return {"status": "unavailable"}

    async def sample(arm, *_):
        sampled.append(arm)
        if arm == broken_arm:
            # Even copying the intentional skip's reason cannot grant exemption.
            return {
                "status": status,
                "reason": "No actionable Hypothesis; B would duplicate A",
                **(
                    {"candidate": candidate(), "review_session": {"versions": []}}
                    if retained_artifact
                    else {}
                ),
            }
        return {"status": "completed", "candidate": candidate()}

    monkeypatch.setattr(loop.jury, "propose", propose)
    monkeypatch.setattr(loop.builder, "build", build)
    monkeypatch.setattr(loop.hypotheses, "propose", method)
    monkeypatch.setattr(loop.jury, "rank", rank)
    result = await loop.run_task(
        TASK,
        [candidate() for _ in range(3)],
        rounds=1,
        roles=lambda *_: model,
        registry=defaults(model),
        sample=sample,
        enabled_arms=list("ABCD"),
        arm_protocol=protocol,
    )
    intentional = protocol == arms.FEEDBACK and not arms.has_hypothesis(hypothesis)
    assert set(sampled) == (set("ACD") if intentional else set("ABCD"))
    only_intentional_skip = intentional and broken_arm == "B"
    assert result["status"] == ("completed" if only_intentional_skip else "partial")
    assert result["completed_rounds"] == int(only_intentional_skip)
    assert result["stop_reason"] == (
        "round_limit" if only_intentional_skip else "incomplete_arms"
    )


@pytest.mark.parametrize(
    "construction_strategy", ["legacy", "counterexample_validated"]
)
@pytest.mark.asyncio
async def test_b_only_feedback_keep_cannot_complete_without_a_delivery(
    monkeypatch, construction_strategy
):
    async def propose(*_, **__):
        return {"criteria": deepcopy(PACK["criteria"]), "gaps": []}

    async def build(*_, **__):
        return deepcopy(PACK)

    async def method(*_, **__):
        return {"status": "keep", "method": ""}

    async def construct(*_, **__):
        return published_pack(), await method()

    async def rank(*_, **__):
        return {"status": "unavailable"}

    async def forbidden(*_):
        pytest.fail("An intentionally skipped B must not sample or grade a delivery")

    monkeypatch.setattr(loop.jury, "propose", propose)
    monkeypatch.setattr(loop.builder, "build", build)
    monkeypatch.setattr(loop.hypotheses, "propose", method)
    monkeypatch.setattr(validated_construction, "construct", construct)
    monkeypatch.setattr(loop.jury, "rank", rank)
    result = await loop.run_task(
        TASK,
        [candidate() for _ in range(3)],
        rounds=1,
        construction_strategy=construction_strategy,
        roles=lambda *_: model,
        registry=defaults(model),
        sample=forbidden,
        evaluate=forbidden,
        enabled_arms=("B",),
        arm_protocol=arms.FEEDBACK,
    )
    assert result["rounds"][0]["arms"]["B"]["status"] == "skipped"
    assert result["rounds"][0]["official"] == {}
    assert result["selected_delivery"]["arm"] is None
    assert result["status"] == "partial"
    assert result["completed_rounds"] == 0
    assert result["stop_reason"] == "incomplete_arms"


@pytest.mark.asyncio
async def test_intentional_b_skip_does_not_prevent_next_configured_round(monkeypatch):
    sampled = []

    async def propose(*_, **__):
        return {"criteria": deepcopy(PACK["criteria"]), "gaps": []}

    async def build(*_, **__):
        return deepcopy(PACK)

    async def method(*_, **__):
        return {"status": "keep", "method": ""}

    async def rank(*_, **__):
        return {"status": "unavailable"}

    async def sample(arm, round_number, *_):
        sampled.append((arm, round_number))
        return {"status": "completed", "candidate": candidate()}

    monkeypatch.setattr(loop.jury, "propose", propose)
    monkeypatch.setattr(loop.builder, "build", build)
    monkeypatch.setattr(loop.hypotheses, "propose", method)
    monkeypatch.setattr(loop.jury, "rank", rank)
    result = await loop.run_task(
        TASK,
        [candidate() for _ in range(3)],
        rounds=2,
        roles=lambda *_: model,
        registry=defaults(model),
        sample=sample,
        enabled_arms=list("ABCD"),
        arm_protocol=arms.FEEDBACK,
        continuation_policy="fixed_rounds",
    )
    assert set(sampled) == {(arm, number) for arm in "ACD" for number in (1, 2)}
    assert result["completed_rounds"] == 2
    assert result["rounds"][0]["loop_decision"]["reason"] == "configured_fixed_rounds"
    assert result["stop_reason"] == "round_limit"


@pytest.mark.parametrize(
    "hypothesis",
    [
        {"status": "keep", "method": ""},
        {"status": "keep", "method": "", "availability": "unavailable"},
    ],
)
@pytest.mark.asyncio
async def test_validated_feedback_round_completes_without_optional_b(
    monkeypatch, hypothesis
):
    sampled = []

    async def construct(*_, **__):
        return published_pack(), deepcopy(hypothesis)

    async def rank(*_, **__):
        return {"status": "unavailable"}

    async def sample(arm, *_):
        sampled.append(arm)
        return {
            "status": "completed",
            "candidate": candidate(),
            "feedback_execution": {"status": "completed"},
        }

    monkeypatch.setattr(validated_construction, "construct", construct)
    monkeypatch.setattr(loop.jury, "rank", rank)
    result = await loop.run_task(
        TASK,
        [candidate() for _ in range(3)],
        rounds=1,
        construction_strategy="counterexample_validated",
        roles=lambda *_: model,
        registry=defaults(model),
        sample=sample,
        enabled_arms=list("ABCD"),
        arm_protocol=arms.FEEDBACK,
    )
    assert set(sampled) == set("ACD")
    assert result["completed_rounds"] == 1 and result["status"] == "completed"
    assert result["stop_reason"] == "round_limit"
    assert result["rounds"][0]["arms"]["B"]["status"] == "skipped"


@pytest.mark.parametrize(
    "change",
    [
        {"arm_protocol": "typo"},
        {"arms": []},
        {"arms": ["D", "D"]},
        {"arms": [True]},
        {"policy_critic": {"enabled": False}},
        {"max_revisions": 0},
        {"max_revisions": True},
        {"actor_harness": {"enabled": True}},
    ],
)
def test_invalid_feedback_treatment_rejected_before_role_execution(change):
    with pytest.raises(ValueError):
        arms.configure({"arm_protocol": arms.FEEDBACK, **change}, list("ABCD"))


def test_feedback_without_d_disables_critic_and_legacy_keeps_its_treatments():
    config = {"arm_protocol": arms.FEEDBACK, "arms": ["A", "B"], "max_revisions": 0}
    arms.configure(config, ["A"])
    assert not config["policy_critic"]["enabled"]
    legacy = {}
    assert arms.configure(legacy, ["A", "B", "C"]) == ["A", "B", "C"]
    assert legacy["policy_critic"]["every_k_turns"] == 3
    assert arms.arm_spec(arms.LEGACY, "D", legacy_critic=True) == {
        "hypothesis": True,
        "static_reward": True,
        "delivery_review": True,
        "policy_critic": True,
    }


@pytest.mark.asyncio
async def test_runtime_terminal_and_recheck_use_actual_new_work_and_bounded_history():
    requests, receipts, records = [], [], {}

    async def fake_role(request):
        requests.append(deepcopy(request))
        if request["mode"] == "outcome_policy_critic":
            return answer(
                verify=[{"test_id": "parse", "reason": "Check current syntax"}]
            )
        status = request["measurement"]["results"][0]["status"]
        if status == "fail":
            return {
                **quality_advice(),
                "gaps": [{"gap": "Invalid JSON", "evidence_refs": ["runtime_check:0"]}],
            }
        return answer(status="observe", local_guidance="")

    critic = PolicyCritic(
        TASK,
        PACK,
        defaults(fake_role),
        on_measurement=lambda *row: receipts.append(deepcopy(row)),
        record=lambda name, value: records.update({name: deepcopy(value)}),
    )
    draft = candidate("{")
    before = deepcopy(draft)
    initial = await critic(draft, turn=1, intent="Serialize the response")
    terminal = await critic(draft, turn=2, phase="terminal")
    fixed = await critic(candidate(), turn=3, phase="terminal_recheck")
    assert draft == before
    assert initial["improvements"][0]["id"] == terminal["improvements"][0]["id"]
    assert "repair-json" in initial["feedback"]
    assert fixed["status"] == "observe" and fixed["feedback"] == ""
    assert [r[1]["results"][0]["status"] for r in receipts] == ["fail", "fail", "pass"]
    assert [r["phase"] for r in requests] == [
        "runtime",
        "runtime",
        "terminal",
        "terminal",
        "terminal_recheck",
        "terminal_recheck",
    ]
    assert [r["turn"] for r in requests[-2]["prior_feedback"]] == [1, 2]
    assert [r["turn"] for r in critic.progress] == [2, 3]
    assert records["turn_1_progress"][0]["checks"][0]["status"] == "fail"
    assert critic.valid_runtime_reviews == 1
    assert [r["feedback_available"] for r in critic.terminal_reviews] == [True, False]
    fresh = PolicyCritic(TASK, PACK, defaults(fake_role))
    await fresh(candidate(), turn=1)
    assert requests[-2]["prior_feedback"] == []


@pytest.mark.parametrize("failure", ["transport", "malformed", "receipt_callback"])
@pytest.mark.asyncio
async def test_completed_checks_survive_failed_guidance_without_stale_advice(failure):
    async def fake_role(request):
        if request["mode"] == "outcome_policy_critic":
            return answer(
                local_guidance="STALE_INITIAL_ADVICE",
                verify=[{"test_id": "parse", "reason": "Parse actual output"}],
            )
        if failure == "transport":
            raise TimeoutError("offline transport failure")
        return {"status": "bad"}

    def receipt(*_):
        if failure == "receipt_callback":
            raise RuntimeError("offline logging failure")

    critic = PolicyCritic(TASK, PACK, defaults(fake_role), on_measurement=receipt)
    result = await critic(candidate("{"), turn=1)
    assert result["status"] == "degraded"
    assert result["measurement"]["results"][0]["status"] == "fail"
    assert "Invalid JSON" in result["feedback"]
    assert "STALE_INITIAL_ADVICE" not in result["feedback"]
    assert result["gaps"] == [] and critic.valid_runtime_reviews == 0


@pytest.mark.parametrize("fault", ["bad_test", "invented_memory", "invented_evidence"])
@pytest.mark.asyncio
async def test_critic_rejects_unbound_feedback_before_any_measurement(fault):
    calls, receipts = [], []

    async def fake_role(request):
        calls.append(request)
        return answer(
            **{
                "bad_test": {"verify": [{"test_id": "missing", "reason": "guess"}]},
                "invented_memory": {"memory_ids": ["never_retrieved"]},
                "invented_evidence": {
                    "gaps": [{"gap": "defect", "evidence_refs": ["invented"]}]
                },
            }[fault]
        )

    result = await PolicyCritic(
        TASK,
        PACK,
        defaults(fake_role),
        on_measurement=lambda *row: receipts.append(row),
    )(candidate(), turn=1)
    assert result["status"] == "unavailable" and result["feedback"] == ""
    assert len(calls) == 2 and not receipts


@pytest.mark.parametrize(
    "status,error,accepted",
    [("pass", False, False), ("unknown", True, False), ("fail", False, True)],
)
@pytest.mark.asyncio
async def test_critic_only_accepts_product_defects_from_valid_failed_checks(
    status, error, accepted
):
    async def fake_role(request):
        if request["mode"] == "outcome_policy_critic":
            return answer(verify=[{"test_id": "parse", "reason": "Check syntax"}])
        return answer(
            gaps=[{"gap": "Syntax defect", "evidence_refs": ["runtime_check:0"]}]
        )

    async def fake_checker(*_):
        return {
            "status": status,
            "error": error,
            "reason": "Controlled observation",
            "gap": "Syntax defect" if status == "fail" else "",
            "evidence_refs": ["product_1:artifact:final_response"],
        }

    registry = defaults(fake_role)
    registry.checkers["json.parse"] = (fake_checker, "offline controlled observation")
    result = await PolicyCritic(TASK, PACK, registry)(candidate(), turn=1)
    assert (result["status"] == "suggest") is accepted
    if not accepted:
        assert result["status"] == "degraded" and result["gaps"] == []


@pytest.mark.parametrize(
    "fault,field",
    [
        ("duplicate_gap", "full_score_gaps[1].id"),
        ("duplicate_action", "improvements[1].id"),
        ("unknown_gap", "improvements[0].gap_ids"),
        ("unknown_ref", "improvements[0].evidence_refs"),
        ("missing_action", "improvements"),
        ("empty_verification", "improvements[0].verification"),
        ("too_many_actions", "improvements"),
        ("false_observe", "status"),
    ],
)
def test_actionable_improvement_contract_rejects_ungrounded_or_inconsistent_advice(
    fault, field
):
    _, evidence = build_context(TASK, [candidate("{")])
    row = quality_advice()
    assert improvement.validate(deepcopy(row), evidence) == row
    if fault == "duplicate_gap":
        row["full_score_gaps"] *= 2
    elif fault == "duplicate_action":
        row["improvements"] *= 2
    elif fault == "unknown_gap":
        row["improvements"][0]["gap_ids"] = ["not_in_ledger"]
    elif fault == "unknown_ref":
        row["improvements"][0]["evidence_refs"] = ["invented"]
    elif fault == "missing_action":
        row["improvements"] = []
    elif fault == "empty_verification":
        row["improvements"][0]["verification"] = " "
    elif fault == "too_many_actions":
        row["improvements"] *= 4
    else:
        row["status"] = "observe"
    with pytest.raises(ResponseContractError) as caught:
        improvement.validate(row, evidence)
    assert caught.value.details["field"] == field


@pytest.mark.asyncio
async def test_checkpoint_construction_compiles_checks_and_publishes_only_basic_integrity():
    context, evidence = build_context(TASK, [candidate()])
    original = structured_pack()
    checkpoint = {
        "context_sha256": fingerprint(context),
        "evidence_sha256": fingerprint(evidence),
        "definition": {"criteria": deepcopy(original["criteria"]), "gaps": []},
        "pack": {**deepcopy(original), "criteria": [], "tests": []},
    }
    frozen = deepcopy(checkpoint)
    calls, roles, records = [], [], {}

    async def designer(request):
        calls.append(request)
        assert request["mode"] == "outcome_builder"
        return {
            "rubrics": [
                {
                    "criterion_id": "syntax",
                    "weight": 1,
                    "anchors": original["criteria"][0]["anchors"],
                }
            ],
            "tests": deepcopy(original["tests"]),
        }

    def fake_roles(name):
        roles.append(name)
        assert not name.startswith("attack_")
        return designer

    registry = defaults(designer)
    pack = await validated_construction.construct_reward(
        context,
        evidence,
        registry,
        fake_roles,
        checkpoint=checkpoint,
        record=lambda name, value: records.update({name: deepcopy(value)}),
    )
    assert checkpoint == frozen and len(calls) == 1
    assert records["resumed_draft"]["new_jury_calls"] == 0
    assert records["basic_review"]["basic_integrity"]["model_requests"] == 0
    assert pack["feedback_policy"] == "evidence_admitted"
    assert pack["validation"]["adversarial_review"]["status"] == "skipped"
    assert all(r["status"] == "unverified" for r in pack["validation"]["records"])
    assert critical_tests(pack) == set()
    require_publication(pack)
    bad_context, bad_evidence = build_context(TASK, [candidate("{")])
    checked = await measure(pack, registry, bad_context, bad_evidence)
    assert checked["results"][0]["status"] == "fail"


@pytest.mark.parametrize("changed", ["context", "evidence"])
@pytest.mark.asyncio
async def test_construction_rejects_checkpoint_for_other_public_inputs(changed):
    context, evidence = build_context(TASK)
    checkpoint = {
        "context_sha256": fingerprint(context),
        "evidence_sha256": fingerprint(evidence),
    }
    checkpoint[changed + "_sha256"] = "stale"

    async def forbidden(_):
        pytest.fail("No role execution for mismatched construction inputs")

    with pytest.raises(ValueError, match="different public inputs"):
        await validated_construction.construct_reward(
            context,
            evidence,
            defaults(forbidden),
            lambda _: forbidden,
            checkpoint=checkpoint,
        )


@pytest.mark.parametrize("enabled", [False, None, 0, 1, "true"])
@pytest.mark.asyncio
async def test_adversarial_entry_points_fail_before_roles_or_checkers(enabled):
    expected = (
        review_policy.AdversarialReviewDisabled if enabled is False else ValueError
    )
    with pytest.raises(expected):
        await validated_construction.repair_definition(
            {}, [], {}, {}, None, adversarial_review_enabled=enabled
        )
    with pytest.raises(expected):
        await validated_construction.recheck(
            {},
            {},
            "reward",
            {},
            {},
            None,
            {},
            {},
            record=lambda *_: None,
            adversarial_review_enabled=enabled,
        )


@pytest.mark.parametrize(
    "fault,reason",
    [
        ("wrong_task", "different Task"),
        ("duplicate", "Duplicate checker"),
        ("unknown_criterion", "unknown criterion"),
        ("missing_checker", "not installed"),
        ("rubric_drift", "rubric differs"),
        ("no_tests", "Every criterion"),
    ],
)
def test_basic_review_detects_binding_defects_without_executing_roles(fault, reason):
    async def forbidden(_):
        pytest.fail("Basic review must be model free")

    pack = structured_pack()
    context, evidence = build_context(TASK)
    assert not review_policy.basic_review(pack, context, evidence, defaults(forbidden))[
        "basic_integrity"
    ]["errors"]
    if fault == "wrong_task":
        pack["task_prompt"] = "Another task"
    elif fault == "duplicate":
        pack["tests"] *= 2
    elif fault == "unknown_criterion":
        pack["tests"][0]["criterion_id"] = "missing"
    elif fault == "missing_checker":
        pack["tests"][0]["checker"] = "missing"
    elif fault == "rubric_drift":
        pack["tests"][0]["rubric"]["standard"] = "Unrequested requirement"
    else:
        pack["tests"] = []
    before = deepcopy(pack)
    audit = review_policy.basic_review(pack, context, evidence, defaults(forbidden))
    assert pack == before
    assert audit["basic_integrity"]["model_requests"] == 0
    assert reason in audit["basic_integrity"]["errors"][0]["reason"]
    assert all(
        r["status"] == "unverified" and not r["contrast_qualified"]
        for r in audit["records"]
    )


def test_review_opt_in_is_explicit_and_configuration_is_copied():
    config = {"nested": {"values": [1]}}
    normalized = review_policy.effective_review_config(config)
    assert normalized["adversarial_review_enabled"] is False
    normalized["nested"]["values"].append(2)
    assert config == {"nested": {"values": [1]}}
    assert review_policy.review_enabled({"adversarial_review_enabled": True})
    review_policy.require_adversarial(True)


@pytest.mark.parametrize(
    "scenario", ["gain", "regression", "tie", "ranking_unavailable"]
)
@pytest.mark.asyncio
async def test_version_selection_runs_blind_ranking_and_real_critical_checks(scenario):
    pack = published_pack()
    assert critical_tests(pack) == {"parse"}
    first_text, last_text = ("{", "{}") if scenario == "gain" else ("{}", "{")
    first, last = version("first", first_text), version("last", last_text)
    rows = {
        "A": {"candidate": deepcopy(first["candidate"])},
        "D": {
            "candidate": deepcopy(last["candidate"]),
            "review_session": {"versions": [first, last]},
            "private_grade": "PRIVATE_GRADE",
            "feedback": "SYNTHETIC_REPAIR_HISTORY",
        },
    }
    before = deepcopy(rows)
    requests, records = [], {}

    async def fake_ranking_role(request):
        requests.append(deepcopy(request))
        assert request["mode"] in {"outcome_ranking_map", "outcome_ranking_reduce"}
        assert "PRIVATE_GRADE" not in str(
            request
        ) and "SYNTHETIC_REPAIR_HISTORY" not in str(request)
        assert "outcome_criteria" not in request and "validation_records" not in request
        if scenario == "ranking_unavailable":
            raise TimeoutError("offline unavailable ranking")
        products = request["context"]["products"]
        ids = [p["product_id"] for p in products]
        preferred = next(
            p["product_id"]
            for p in products
            if p["artifact"]["final_response"] == last_text
        )
        return {
            "groups": [ids]
            if scenario == "tie"
            else [[preferred], [p for p in ids if p != preferred]],
            "unknown": [],
            "observations": [
                {
                    "product_id": p,
                    "reason": "Offline comparative observation",
                    "evidence_refs": [p + ":artifact:final_response"],
                }
                for p in ids
            ],
            "reason": "Synthetic preference challenges the deterministic preservation gate",
        }

    result = await version_selection.select_versions(
        TASK,
        pack,
        rows,
        defaults(fake_ranking_role),
        lambda _: fake_ranking_role,
        record=lambda name, value: records.update({name: deepcopy(value)}),
    )
    assert len([name for name in records if name.startswith("version_checks/")]) == 2
    assert len(requests) == (3 if scenario == "ranking_unavailable" else 4)
    assert rows["D"]["candidate"] == before["D"]["candidate"]
    assert rows["D"]["review_session"] == before["D"]["review_session"]
    selected = rows["D"]["delivery_selection"]
    decision = selected["decisions"][0]
    assert decision["accepted"] is (scenario == "gain")
    assert selected["candidate_id"] == ("last" if scenario == "gain" else "first")
    assert result["selection"]["arm"] == ("D" if scenario == "gain" else "A")
    assert selected["native_trace_preserved"]
    if scenario == "regression":
        assert decision["strictly_preferred"] and decision["lost_critical_checks"] == [
            "parse"
        ]
    assert {
        v["measurement"]["results"][0]["status"] for v in rows["D"]["version_checks"]
    } == {"pass", "fail"}


@pytest.mark.asyncio
async def test_version_selection_rejects_tampered_immutable_artifact_before_calls():
    saved = version("first", "{}")
    saved["candidate"]["artifact"]["final_response"] = "changed"
    rows = {
        "D": {"candidate": saved["candidate"], "review_session": {"versions": [saved]}}
    }

    def forbidden(_):
        pytest.fail("No ranking or measurement for a changed artifact")

    with pytest.raises(ValueError, match="Immutable delivery hash mismatch"):
        await version_selection.select_versions(
            TASK, published_pack(), rows, None, forbidden
        )


@pytest.mark.parametrize("status,error", [("unknown", False), ("pass", True)])
def test_preferred_version_with_missing_or_errored_critical_evidence_is_rejected(
    status, error
):
    pack = published_pack()
    first, last = version("first", "{}"), version("last", '{"ok": true}')
    for row, product, outcome in (
        (first, "product_1", "pass"),
        (last, "product_2", status),
    ):
        row.update(
            product_id=product,
            measurement={
                "pack_id": pack["pack_id"],
                "results": [
                    {
                        "test_id": "parse",
                        "status": outcome,
                        "error": error if row is last else False,
                    }
                ],
            },
        )
    selected, decisions = version_selection.choose(
        [first, last],
        {"status": "compared", "groups": [["product_2"], ["product_1"]]},
        pack,
    )
    assert selected is first
    assert (
        decisions[0]["strictly_preferred"]
        and not decisions[0]["critical_checks_complete"]
    )
    assert decisions[0]["lost_critical_checks"] == ["parse"]


@pytest.mark.asyncio
async def test_bounded_knowledge_navigation_can_read_back_exact_canonical_material():
    ref = "skill:task_review:json"
    content = {
        "kind": "task_review",
        "authority": "optional",
        "conditions_unstated": True,
        "mechanisms": [
            {"id": "parse", "source_id": "manual", "source_statement": "\u539f\u6587" * 300}
        ],
        "details": "Full derivation and exceptions " * 100,
    }
    context, evidence = build_context(TASK)
    context["skills"] = [{"ref": ref, "content": deepcopy(content)}]
    evidence[ref] = deepcopy(content)
    original = deepcopy((context, evidence))
    requests = []

    async def fake_role(request):
        call_budget.consume()
        requests.append(deepcopy(request))
        skill = request["context"]["skills"][0]
        assert skill["preview_only"]
        assert skill["content"]["details_ref"] == ref
        assert skill["content"]["conditions_unstated"] is True
        assert skill["content"]["passage_previews"][0]["preview"] == "\u539f\u6587" * 150
        if len(requests) == 1:
            return {"need_evidence": [ref]}
        assert request["requested_evidence"][ref] == content
        return {"ok": True}

    fake_role.bounded_stage = True
    assert await ask(
        fake_role,
        "offline_navigation",
        context,
        evidence,
        "Read complete material",
        {"ok": "boolean"},
    ) == {"ok": True}
    assert len(requests) == 2 and (context, evidence) == original
    assert call_budget.ACTIVE.get() is None


def test_navigation_leaves_unbound_or_non_review_skill_content_unchanged():
    content = {"details": "Keep exact"}
    context = {
        "skills": [
            {"ref": "skill:other", "content": content},
            {"ref": "skill:task_review:stale", "content": content},
            {"ref": "skill:task_review:text", "content": "plain text"},
        ]
    }
    evidence = {
        "skill:other": content,
        "skill:task_review:stale": {"details": "different"},
    }
    projected = request_projection.knowledge_navigation(context, evidence)
    assert projected == context and projected is not context


@pytest.mark.asyncio
async def test_navigation_compression_preserves_protected_fields_and_input():
    canonical = "\u539f\u59cb\u6761\u4ef6\u53ca\u4f8b\u5916" * 100
    protected = {
        key: {"summary": canonical}
        for key in (
            "task",
            "requested_evidence",
            "cases",
            "criteria",
            "clauses",
            "verified_findings",
            "disputed_findings",
            "response_recovery",
        )
    }
    request = {
        "mode": "offline",
        "context": {
            **protected,
            "sources": [
                {"preview": canonical * 20, "preview_only": True},
                {"rationale": canonical * 20, "preview_only": True},
            ],
        },
    }
    before = deepcopy(request)
    calls, records = [], []

    async def compressor(operation):
        calls.append(operation)
        return {
            "summaries": ["Conditions and exceptions remain in the source"]
            * len(operation["fields"])
        }

    compressor.max_request_bytes = 200_000
    cap = len(json.dumps({"context": protected}, ensure_ascii=False).encode()) + 1000
    projected = await request_projection.compress(
        request, compressor, cap, records.append
    )
    assert request == before
    assert all(projected["context"][key] == value for key, value in protected.items())
    assert [f["path"] for f in calls[0]["fields"]] == [
        ["context", "sources", 0, "preview"],
        ["context", "sources", 1, "rationale"],
    ]
    assert len(json.dumps(projected, ensure_ascii=False).encode()) <= cap
    assert records[-1]["canonical_evidence_unchanged"]
    assert records[-1]["after_bytes"] < records[0]["before_bytes"]


@pytest.mark.parametrize(
    "failure,stage",
    [
        ("immutable", "immutable_context_exceeds_capacity"),
        ("input", "compression_input"),
        ("output", "compression_output"),
        ("malformed", None),
    ],
)
@pytest.mark.asyncio
async def test_compression_failure_is_explicit_and_does_not_truncate_input(
    failure, stage
):
    request = {
        "mode": "offline",
        "context": {
            "task": TASK,
            "sources": [{"preview": "\u6761\u4ef6" * 1000, "preview_only": True}],
        },
    }
    if failure == "immutable":
        request["context"]["task"] = {"prompt": "immutable" * 1000}
    before = deepcopy(request)
    calls = []

    async def compressor(operation):
        calls.append(operation)
        return {"summaries": [] if failure == "malformed" else ["too long" * 1000]}

    compressor.max_request_bytes = 100 if failure == "input" else 100_000
    with pytest.raises(RequestBudgetExceeded if stage else ValueError) as caught:
        await request_projection.compress(request, compressor, 1000, lambda _: None)
    assert request == before
    assert len(calls) == (0 if failure in {"immutable", "input"} else 1)
    if stage:
        assert caught.value.details["stage"] == stage


@pytest.mark.asyncio
async def test_six_request_budget_includes_transport_attempts_and_restores_parent_context():
    context, evidence = build_context(TASK)
    parent = {"requests": 2}
    marker = call_budget.ACTIVE.set(parent)
    attempts, budgets = [], []

    async def fake_retrying_transport(_):
        budgets.append(call_budget.ACTIVE.get())
        for attempt in range(7):
            call_budget.consume()
            attempts.append(attempt)
        pytest.fail("Seventh physical attempt must never execute")

    fake_retrying_transport.bounded_stage = True
    try:
        with pytest.raises(RuntimeError, match="six physical requests"):
            await ask(
                fake_retrying_transport,
                "offline",
                context,
                evidence,
                "Test retries",
                {},
            )
        assert attempts == list(range(6))
        assert budgets[0]["requests"] == 6
        assert call_budget.ACTIVE.get() is parent and parent["requests"] == 2
    finally:
        call_budget.ACTIVE.reset(marker)


@pytest.mark.asyncio
async def test_contract_repair_budget_is_shared_across_evidence_rounds():
    context, evidence = build_context(TASK)
    calls = []

    async def fake_role(request):
        call_budget.consume()
        calls.append(deepcopy(request))
        if len(calls) == 1:
            return {"need_evidence": ["invented"]}
        if len(calls) == 2:
            assert "response_recovery" in request
            return {"need_evidence": ["task"]}
        assert request["requested_evidence"]["task"] == evidence["task"]
        return {"need_evidence": ["invented_again"]}

    fake_role.bounded_stage = True
    with pytest.raises(ResponseContractError):
        await ask(fake_role, "offline", context, evidence, "Read exact evidence", {})
    assert len(calls) == 3
    assert call_budget.ACTIVE.get() is None


@pytest.mark.parametrize("decision", ["propose", "keep", "unavailable"])
@pytest.mark.asyncio
async def test_completed_reward_resume_runs_real_hypothesis_pipeline_with_fake_roles(
    decision,
):
    context, evidence = build_context(TASK, [candidate("{")])
    pack = published_pack()
    checkpoint = {
        "context_sha256": fingerprint(context),
        "evidence_sha256": fingerprint(evidence),
        "completed_reward": pack,
    }
    before = deepcopy(checkpoint)
    requests, records = [], {}
    method = {
        "status": "propose",
        "method": "Parse the JSON before delivering it.",
        "target": "Valid output",
        "expected_improvement": "Fewer syntax errors",
        "verification": "Run a JSON parser",
        "evidence_refs": ["task"],
        "target_gap_ids": [],
        "clarifications": [],
        "validation_targets": [{"id": "hypothesis:1", "quote": "Parse the JSON"}],
    }

    def roles(name):
        assert name.startswith("hypothesis_")

        async def fake_role(request):
            requests.append(deepcopy(request))
            assert request["context"]["memories"] == []
            assert request["context"]["gaps"] == []
            if decision == "unavailable":
                raise TimeoutError("offline unavailable hypothesis")
            if decision == "keep":
                return {"status": "keep", "method": ""}
            return deepcopy(method)

        return fake_role

    returned_pack, result = await validated_construction.construct(
        context,
        evidence,
        defaults(None),
        roles,
        checkpoint=checkpoint,
        record=lambda name, value: records.update({name: deepcopy(value)}),
    )
    assert checkpoint == before and returned_pack == pack
    modes = [r["mode"] for r in requests]
    assert modes.count("outcome_hypothesis_map") == 3
    if decision == "unavailable":
        assert result["availability"] == "unavailable"
        assert records["construction_manifest"]["status"] == "partial"
        assert records["construction_manifest"]["reward_available"]
        assert not records["construction_manifest"]["hypothesis_available"]
    else:
        assert modes.count("outcome_hypothesis_reduce") == 1
        assert result["status"] == decision
    if decision == "propose":
        assert modes.count("outcome_hypothesis_actor_projection") == 1
        assert result["validation"]["status"] == "skipped"
        assert result["actor_projection"]["validated"] is False
        assert result["actor_projection"]["source_sha256"] == fingerprint(
            method["method"]
        )
        hypotheses.require_validated_projection(result, allow_unverified=True)
        with pytest.raises(ValueError, match="independently validated"):
            hypotheses.require_validated_projection(result)
        tampered = deepcopy(result)
        tampered["method"] += " Invented authority."
        with pytest.raises(ValueError, match="changed"):
            hypotheses.require_validated_projection(tampered, allow_unverified=True)


@pytest.mark.parametrize(
    "bad_projection", ["historical", "missing_target", "wrong_quote"]
)
@pytest.mark.asyncio
async def test_actor_projection_rejects_history_and_unbound_prescriptions(
    bad_projection,
):
    context, evidence = build_context(TASK)
    calls = []
    method = {
        "status": "propose",
        "method": "Parse the JSON before delivering it.",
        "target": "Valid output",
        "expected_improvement": "Fewer syntax errors",
        "verification": "Run a JSON parser",
        "evidence_refs": ["task"],
        "validation_targets": [{"id": "hypothesis:1", "quote": "Parse the JSON"}],
    }

    async def fake_role(request):
        calls.append(request)
        row = deepcopy(method)
        if bad_projection == "historical":
            row["method"] = "Repair product_1 using its previous mistakes."
        elif bad_projection == "missing_target":
            row["validation_targets"] = []
        else:
            row["validation_targets"][0]["quote"] = "Not present in the method"
        return row

    with pytest.raises(ResponseContractError):
        await validated_construction.project_method(
            method, context, evidence, {"criteria": []}, fake_role
        )
    assert len(calls) == 2 and "response_recovery" in calls[1]


@pytest.mark.asyncio
async def test_recheck_replays_real_parser_cases_adds_fresh_boundary_and_reuses_checkpoints():
    """Real verification/recheck, fake case generator and independent references."""
    pack = structured_pack()
    context, evidence = build_context(TASK, [candidate("PRIVATE_DRAFT")])
    proposal = issue("syntax")
    reference_calls, boundary_calls, records = [], [], {}

    async def reference(request):
        reference_calls.append(deepcopy(request))
        assert "PRIVATE_DRAFT" not in str(request)
        assert all(
            "expected" not in case and "kind" not in case for case in request["cases"]
        )
        verdicts = []
        for case in request["cases"]:
            try:
                json.loads(case["artifact"])
                status = "pass"
            except ValueError:
                status = "fail"
            verdicts.append(
                {
                    "id": case["id"],
                    "status": status,
                    "reason": "Independent offline syntax reference",
                    "evidence_refs": ["task"],
                }
            )
        return {"cases": verdicts}

    registry = defaults(reference)
    original = await verify_issue(
        proposal,
        pack,
        "reward",
        context,
        evidence,
        registry,
        reference,
        record=lambda *_: None,
        adversarial_review_enabled=True,
    )
    assert original["status"] == "supported_on_cases"
    assert {c["case_kind"]: c["current_verdict"] for c in original["cases"]} == {
        "correct_control": "pass",
        "error_variant": "fail",
        "boundary": "pass",
    }

    async def boundary(request):
        boundary_calls.append(deepcopy(request))
        assert "PRIVATE_DRAFT" not in str(request)
        assert "cases" not in request and "disputed_findings" not in request
        return {
            "case": {
                "id": "fresh-array",
                "artifact": "[]",
                "question": "Does a top-level array parse?",
                "why": "Different top-level JSON type",
                "expected": "pass",
                "scope": "whole_check",
                "probes": [],
            }
        }

    class Checkpoints:
        def __init__(self):
            self.values = {}

        def verification_checkpoint(self, binding, name):
            return self.values.get((binding, name))

        def save_verification_checkpoint(self, binding, name, value):
            self.values[binding, name] = deepcopy(value)

    checkpoints = Checkpoints()
    audit = {"status": "completed", "records": [original]}
    frozen = deepcopy(audit)

    async def run():
        return await validated_construction.recheck(
            audit,
            pack,
            "reward",
            context,
            evidence,
            registry,
            dict.fromkeys(("1", "2", "3"), reference),
            dict.fromkeys(("1", "2", "3"), boundary),
            record=lambda name, value: records.update({name: deepcopy(value)}),
            checkpoint_store=checkpoints,
            adversarial_review_enabled=True,
        )

    checked = await run()
    assert audit == frozen
    finding = next(r for r in checked["records"] if r.get("fresh_boundary"))
    assert finding["origin_validation_id"] == original["validation_id"]
    assert (
        finding["status"] == finding["fresh_boundary"]["status"] == "supported_on_cases"
    )
    assert {c["case_id"] for c in finding["cases"]} == {
        c["case_id"] for c in original["cases"]
    }
    assert "[]" in {
        c["material"]["artifact"] for c in finding["fresh_boundary"]["cases"]
    }
    assert any(
        r["target_id"] == "parse" and r["status"] == "unverified"
        for r in checked["records"]
    )
    assert len(reference_calls) == 3 and len(boundary_calls) == 1
    assert await run() == checked
    assert len(reference_calls) == 3 and len(boundary_calls) == 1
    assert len(checkpoints.values) == 3
    assert all(
        records[f"recheck_0/{name}_recovery"]["reused"]
        for name in ("original", "fresh_case", "boundary")
    )


@pytest.mark.parametrize("field", ["summary", "preview", "rationale"])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.asyncio
async def test_request_compression_must_not_rewrite_canonical_artifact_fields(
    field, nested
):
    body = {field: "Canonical deliverable text " * 100, "preview_only": True}
    if nested:
        body = {"sections": [{"details": [body]}]}
    context, evidence = build_context(TASK, [{"artifact": body}])
    request = {"mode": "offline", "context": context}
    original = deepcopy((request, evidence))
    calls = []

    async def compressor(operation):
        calls.append(operation)
        return {"summaries": ["LOSSY REPLACEMENT"] * len(operation["fields"])}

    compressor.max_request_bytes = 100_000
    # There is no optional navigation here: oversized authored work must fail
    # explicitly, never become a summary that differs from its evidence ref.
    with pytest.raises(RequestBudgetExceeded):
        await request_projection.compress(request, compressor, 1800, lambda _: None)
    assert not calls and (request, evidence) == original


@pytest.mark.asyncio
async def test_compression_preserves_nested_canonical_evidence_while_compacting_navigation():
    body = {
        "sections": [
            {
                "details": [
                    {
                        "preview_only": True,
                        "summary": "Exact source conclusion " * 40,
                        "preview": "Exact authored field " * 40,
                        "rationale": "Exact reasoning and exceptions " * 40,
                    }
                ]
            }
        ],
    }
    context, evidence = build_context(
        TASK,
        [
            {
                "artifact": deepcopy(body),
                "environment": deepcopy(body),
                "source_documents": [{"summary": "s" * 515}, deepcopy(body)],
            }
        ],
    )
    # The first source fits build_context's preview limit and is complete evidence.
    exact_source = deepcopy(context["products"][0]["sources"][0])
    assert exact_source["preview_only"] is False
    assert len(exact_source["preview"].encode()) > 512
    # Exercise protection below arbitrary wrapper/list nesting, not only at root.
    context["wrapper"] = [
        {
            key: deepcopy(body)
            for key in (
                "artifact",
                "environment",
                "source_documents",
                "evidence",
                "requested_evidence",
                "content",
                "result",
                "results",
                "checks",
            )
        }
    ]
    context["unmarked"] = {"summary": "Exact unmarked text " * 40}
    context["reference_documents"] = [
        {
            "ref": "reference:manual",
            "preview_only": True,
            "preview": "Optional navigation " * 1000,
        }
    ]
    evidence["reference:manual"] = deepcopy(body)
    request = {
        "mode": "offline",
        "context": context,
        "requested_evidence": deepcopy(evidence),
    }
    original = deepcopy((request, evidence))
    calls = []

    async def compressor(operation):
        calls.append(deepcopy(operation))
        return {
            "summaries": ["Short optional index; read the source"]
            * len(operation["fields"])
        }

    compressor.max_request_bytes = 200_000
    cap = len(json.dumps(request, ensure_ascii=False).encode()) - 10_000
    projected = await request_projection.compress(
        request, compressor, cap, lambda _: None
    )
    assert (request, evidence) == original
    assert len(calls) == 1
    assert [field["path"] for field in calls[0]["fields"]] == [
        ["context", "products", 0, "sources", 1, "preview"],
        ["context", "reference_documents", 0, "preview"],
    ]
    assert (
        projected["context"]["products"][0]["artifact"]
        == context["products"][0]["artifact"]
    )
    assert projected["context"]["products"][0]["environment"] == body
    assert projected["context"]["products"][0]["sources"][0] == exact_source
    assert projected["context"]["wrapper"] == context["wrapper"]
    assert projected["context"]["unmarked"] == context["unmarked"]
    assert projected["requested_evidence"] == evidence
    assert len(json.dumps(projected, ensure_ascii=False).encode()) <= cap


@pytest.mark.asyncio
async def test_bounded_request_compresses_knowledge_index_and_reads_exact_source():
    ref = "skill:task_review:json"
    source = {
        "conditions_unstated": True,
        "applicability": {"summary": "These are exact applicability limits " * 20},
        "mechanisms": [
            {"id": "syntax", "source_id": "manual", "source_statement": "\u539f\u6587" * 2000}
        ],
    }
    context, evidence = build_context(TASK)
    context["skills"] = [{"ref": ref, "content": deepcopy(source)}]
    evidence[ref] = deepcopy(source)
    original = deepcopy((context, evidence))
    requests, projections, compressions = [], [], []

    async def fake_compressor(operation):
        call_budget.consume()
        compressions.append(deepcopy(operation))
        return {"summaries": ["Optional passage index; read the full source"]}

    fake_compressor.max_request_bytes = 200_000

    async def fake_role(request):
        call_budget.consume()
        requests.append(deepcopy(request))
        index = request["context"]["skills"][0]["content"]
        assert index["applicability"] == source["applicability"]
        assert index["conditions_unstated"] is True
        if len(requests) == 1:
            return {"need_evidence": [ref]}
        assert request["requested_evidence"][ref] == source
        return {"ok": True}

    # Force both ask stages through real compression without relying on prompt
    # lengths: prepare_request supplies an oversized, explicitly optional index.
    def prepare(request):
        request["context"]["skills"][0]["content"]["passage_previews"][0]["preview"] = (
            "Navigation " * 5000
        )
        return request

    async def compress(request, available_evidence, cap):
        assert available_evidence == evidence
        return await request_projection.compress(
            request, fake_compressor, cap, projections.append
        )

    fake_role.bounded_stage = True
    fake_role.prepare_request = prepare
    fake_role.compress_request = compress
    result = await ask(
        fake_role,
        "offline_navigation",
        context,
        evidence,
        "Read exact source",
        {"ok": "boolean"},
        max_bytes=25_000,
    )
    assert result == {"ok": True}
    assert len(requests) == len(compressions) == 2
    assert all(
        operation["fields"][0]["path"]
        == ["context", "skills", 0, "content", "passage_previews", 0, "preview"]
        for operation in compressions
    )
    assert all(
        row["after_bytes"] <= 25_000
        for row in projections
        if row["status"] == "completed"
    )
    assert (context, evidence) == original
    assert call_budget.ACTIVE.get() is None


@pytest.mark.parametrize("changed", ["artifact", "task", "pack", "evidence", "context"])
@pytest.mark.asyncio
async def test_measurement_input_binding_changes_with_actual_measured_inputs(changed):
    async def forbidden(_):
        pytest.fail("Deterministic JSON measurement must not call a model")

    task, pack, delivered = deepcopy(TASK), deepcopy(PACK), candidate()
    context, evidence = build_context(task, [delivered])
    baseline = await measure(pack, defaults(forbidden), context, evidence)
    expected = measurement_input_fingerprint(pack, context, evidence)
    assert baseline["measurement_input_schema"] == MEASUREMENT_INPUT_SCHEMA
    assert baseline["measurement_input_sha256"] == expected
    assert len(expected) == 64
    repeated = await measure(pack, defaults(forbidden), context, evidence)
    assert repeated["measurement_input_sha256"] == expected

    if changed == "artifact":
        delivered["artifact"]["final_response"] = '{"different": true}'
        context, evidence = build_context(task, [delivered])
    elif changed == "task":
        task["prompt"] += " Use an object."
        pack["task_prompt"] = task["prompt"]
        context, evidence = build_context(task, [delivered])
    elif changed == "pack":
        # Reusing a pack_id cannot hide changed checker definitions.
        pack["tests"][0]["question"] = "Changed checker question"
    elif changed == "evidence":
        evidence["reference:actual_source"] = {"content": "New measured evidence"}
    else:
        context["measurement_inquiries"] = {
            "parse": {
                "questions": ["Does the actual response parse?"],
                "claim_checks": [],
            }
        }
    frozen = deepcopy((pack, context, evidence))
    changed_hash = measurement_input_fingerprint(pack, context, evidence)
    result = await measure(pack, defaults(forbidden), context, evidence)
    assert changed_hash != expected
    assert result["measurement_input_sha256"] == changed_hash
    assert result["pack_id"] == baseline["pack_id"]
    assert (pack, context, evidence) == frozen
    assert result["results"][0]["status"] == "pass"
    assert "score" not in result and "native_score" not in result


@pytest.mark.asyncio
async def test_measurement_binding_normalization_is_idempotent_for_published_inputs():
    pack = structured_pack()
    source = {"path": "schema.json", "content": '{"type":"object"}'}
    pack["measurement_sources"] = {
        "task_input:schema.json": source,
        "reference:manual": {"title": "JSON manual", "text": "JSON syntax"},
    }
    pack["measurement_priors"] = [
        {
            "ref": "skill:task_review:syntax",
            "content": {"authority": "optional", "method": "Check syntax"},
        }
    ]
    pack["validation"] = {
        "records": [
            {
                "validation_id": "syntax-case",
                "target_id": "syntax",
                "status": "supported_on_cases",
                "cases": [{"artifact": "{}"}],
            }
        ]
    }
    task = {**TASK, "documents": [source]}
    context, evidence = build_context(task, [candidate()])
    before = deepcopy((pack, context, evidence))
    normalized = measurement_context(pack, context, evidence)
    twice = measurement_context(pack, *normalized)
    assert twice == normalized
    assert (
        len(
            [
                d
                for d in twice[0]["reference_documents"]
                if d["ref"] == "reference:validation:syntax-case"
            ]
        )
        == 1
    )
    assert (pack, context, evidence) == before
    expected = fingerprint(
        {
            "schema": MEASUREMENT_INPUT_SCHEMA,
            "pack": pack,
            "context": normalized[0],
            "evidence": normalized[1],
        }
    )
    assert measurement_input_fingerprint(pack, context, evidence) == expected
    assert measurement_input_fingerprint(pack, *normalized) == expected
    assert measurement_input_fingerprint(pack, *twice) == expected
    raw_result = await measure(pack, defaults(None), context, evidence)
    normalized_result = await measure(pack, defaults(None), *twice)
    assert (
        raw_result["measurement_input_sha256"]
        == normalized_result["measurement_input_sha256"]
        == expected
    )
    assert (pack, context, evidence) == before
    assert measurement_context(pack, *twice) == normalized


@pytest.mark.asyncio
async def test_measurement_binding_is_snapshotted_before_async_checker_mutation():
    pack = structured_pack()
    context, evidence = build_context(TASK, [candidate()])
    frozen = deepcopy((pack, context, evidence))
    expected = measurement_input_fingerprint(pack, context, evidence)
    mutated = []
    registry = defaults(None)

    async def mutating_run(test, execution_context, execution_evidence):
        await asyncio.sleep(0)
        test["rubric"]["standard"] = "Changed by checker"
        execution_context["products"][0]["artifact"]["final_response"] = "null"
        execution_evidence["product_1:artifact:final_response"] = "null"
        mutated.append(
            measurement_input_fingerprint(pack, execution_context, execution_evidence)
        )
        return {
            "test_id": "parse",
            "criterion_id": "syntax",
            "status": "pass",
            "reason": "Offline checker mutation fixture",
            "gap": "",
            "evidence_refs": ["product_1:artifact:final_response"],
        }

    # Bypass Registry.run's defensive copies to exercise measure's own boundary.
    registry.run = mutating_run
    result = await measure(pack, registry, context, evidence)
    assert result["measurement_input_sha256"] == expected
    assert mutated and mutated[0] != expected
    assert (pack, context, evidence) == frozen


@pytest.mark.asyncio
async def test_measurement_binding_retains_selection_and_reuse_semantics():
    pack = deepcopy(PACK)
    pack["tests"].append({**deepcopy(pack["tests"][0]), "test_id": "second-parse"})
    context, evidence = build_context(TASK, [candidate()])
    registry = defaults(None)
    calls = []
    original_run = registry.run

    async def recorded_run(test, current, sources):
        calls.append(test["test_id"])
        return await original_run(test, current, sources)

    registry.run = recorded_run
    expected = measurement_input_fingerprint(pack, context, evidence)
    for selection, wanted in (
        (None, {"parse", "second-parse"}),
        ([], set()),
        (["parse"], {"parse"}),
    ):
        calls.clear()
        result = await measure(pack, registry, context, evidence, test_ids=selection)
        assert set(calls) == wanted
        assert {r["test_id"] for r in result["results"]} == wanted
        assert result["execution_summary"]["selected_checks"] == len(wanted)
        assert result["measurement_input_sha256"] == expected
    calls.clear()
    reused = await measure(
        pack,
        registry,
        context,
        evidence,
        test_ids=["parse"],
        previous=(context, evidence, result),
    )
    assert calls == []
    assert reused["execution_summary"]["reused_checks"] == 1
    assert reused["results"][0]["reused_unchanged_input"]
    assert reused["measurement_input_sha256"] == expected
    changed_context, changed_evidence = build_context(TASK, [candidate("{")])
    changed = await measure(
        pack,
        registry,
        changed_context,
        changed_evidence,
        test_ids=["parse"],
        previous=(context, evidence, result),
    )
    assert calls == ["parse"]
    assert changed["results"][0]["status"] == "fail"
    assert changed["measurement_input_sha256"] != expected
    # A summary of old receipts is not a fresh bound execution.
    historic_summary = summarize_results(pack, result["results"])
    assert "measurement_input_sha256" not in historic_summary
    assert "measurement_input_schema" not in historic_summary


INQUIRY_CASES = scalar_inquiries()


@pytest.mark.parametrize("capture", INQUIRY_CASES, ids=lambda row: row["name"])
@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize("requested", [False, True])
@pytest.mark.asyncio
async def test_synthetic_scalar_inquiry_is_diagnosed_unless_requested(
    capture, bounded, requested
):
    """Authored scalar metadata with synthetic evidence handles."""
    context, evidence = build_context(
        TASK,
        [
            {
                "artifact": {
                    "final_response": "Offline contract fixture",
                    "changed_files": [],
                },
                "source_documents": [
                    {"content": "Offline source handle"} for _ in range(8)
                ],
            }
        ],
    )
    assert set(capture["evidence_refs"]) <= set(evidence)
    pack = deepcopy(PACK)
    pack["tests"][0].update(test_id=capture["test_id"], checker="evidence.review")
    if requested:
        context["measurement_inquiries"] = {
            capture["test_id"]: {
                "questions": ["Address this separately"],
                "claim_checks": [],
            }
        }
    calls = []
    raw = deepcopy(capture["response"])
    frozen = deepcopy((pack, context, evidence, raw))

    async def replay(request):
        calls.append(request)
        assert ("inquiry_result" in request["response"]) is requested
        return raw

    replay.bounded_stage = bounded
    measurement = await measure(pack, defaults(replay), context, evidence)
    row = measurement["results"][0]
    # Only explicitly requested inquiries on opted-in roles get one contract
    # repair. Unsolicited metadata never buys another model call.
    assert len(calls) == (2 if requested and bounded else 1)
    if requested and bounded:
        assert calls[1]["response_recovery"]["rejected_response"] == raw
        assert row["response_validation"]["selected_attempts"] == {"global": 1}
    for key in ("status", "reason", "gap", "claims", "evidence_refs"):
        assert row[key] == capture["response"][key]
    assert (pack, context, evidence, raw) == frozen
    if requested:
        assert "inquiry_result_diagnostic" not in row
        assert row["inquiry_result"]["status"] == "unknown"
        assert row["inquiry_result"]["error"] is True
        assert row["inquiry_result"]["reason"] == "Specific inquiry validation failed"
    else:
        assert "inquiry_result" not in row
        assert row["inquiry_result_diagnostic"] == {
            "reason": "no_measurement_inquiry_requested",
            "authoritative": False,
            "original_value": raw["inquiry_result"],
        }
    # Exercise the actual downstream dictionary consumers that scalars broke.
    assert failed_check(row) is False
    assert check_summaries(measurement)[0]["status"] == raw["status"]


@pytest.mark.parametrize(
    "unsolicited",
    [
        None,
        [],
        "pass",
        "unknown",
        {
            "status": "pass",
            "reason": "Unrequested assertion",
            "gap": "",
            "evidence_refs": ["product_1:artifact:final_response"],
        },
    ],
)
@pytest.mark.asyncio
async def test_unsolicited_inquiry_metadata_cannot_hide_real_failed_check(unsolicited):
    context, evidence = build_context(TASK, [candidate("not JSON")])
    ref = "product_1:artifact:final_response"
    raw = {
        "status": "fail",
        "reason": "Delivered text is not JSON",
        "gap": "Repair invalid JSON",
        "evidence_refs": [ref],
        "claims": [
            {
                "claim": "The response is valid JSON",
                "claim_refs": [ref],
                "evidence_refs": [ref],
                "observation": "The text is not JSON",
                "judgment": "contradicted",
            }
        ],
        "inquiry_result": unsolicited,
    }
    original = deepcopy(raw)

    async def replay(_):
        return raw

    replay.bounded_stage = True
    pack = deepcopy(PACK)
    pack["tests"][0]["checker"] = "evidence.review"
    measured = await measure(pack, defaults(replay), context, evidence)
    row = measured["results"][0]
    assert "inquiry_result" not in row
    assert row["inquiry_result_diagnostic"] == {
        "reason": "no_measurement_inquiry_requested",
        "authoritative": False,
        "original_value": unsolicited,
    }
    assert row["status"] == "fail" and row["gap"] == original["gap"]
    assert row["claims"] == original["claims"] and raw == original
    assert not measured["ready_under_estimator"]
    assert failed_check(row) is True
    assert check_summaries(measured)[0]["status"] == "fail"


@pytest.mark.parametrize("capture", INQUIRY_CASES, ids=lambda row: row["name"])
@pytest.mark.asyncio
async def test_captured_unsolicited_inquiry_finishes_real_delivery_review(capture):
    """Real review bookkeeping, captured reply, synthetic evidence; no live grade."""
    delivered = {
        "artifact": {"final_response": "Offline contract fixture", "changed_files": []},
        "source_documents": [{"content": "Offline source handle"} for _ in range(8)],
    }
    pack = deepcopy(PACK)
    pack["tests"][0].update(test_id=capture["test_id"], checker="evidence.review")
    raw = deepcopy(capture["response"])
    original = deepcopy((delivered, pack, raw))
    calls, records = [], {}

    async def replay(request):
        calls.append(request)
        assert "inquiry_result" not in request["response"]
        return raw

    review = DeliveryReview(
        TASK,
        pack,
        defaults(replay),
        audit_mode="full",
        max_revisions=0,
        record=lambda name, value: records.update({name: deepcopy(value)}),
    )
    assert await review(delivered) is None
    assert len(calls) == len(review.history) == len(review.artifacts) == 1
    assert (delivered, pack, raw) == original
    measured = review.history[0]
    row = measured["results"][0]
    assert "inquiry_result" not in row
    assert row["inquiry_result_diagnostic"] == {
        "reason": "no_measurement_inquiry_requested",
        "authoritative": False,
        "original_value": raw["inquiry_result"],
    }
    assert row["status"] == raw["status"]
    assert records["delivery_1"]["measurement"] == measured
    assert review.export_session()["versions"][0]["measurement"] == measured
    assert records["task_progress"]["status"] == review.status
    assert records["task_progress_1"]["status"] == review.status
    assert review.status == (
        "satisfied_under_estimator"
        if raw["status"] == "pass"
        else "budget_exhausted_unresolved"
    )
    assert measured["ready_under_estimator"] is (raw["status"] == "pass")
    assert review.progress.verified is (raw["status"] == "pass")
