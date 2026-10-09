"""Public reward regressions restored from archived test_public_task_reward.py.

Adapted to the current core measure() API; no archived adapter or model calls.
The interval completion test intentionally exercises the pure receipt aggregator.
"""

import asyncio
from copy import deepcopy
from itertools import product

import pytest

from reward_engine.core.outcome.context import build_context
from reward_engine.core.outcome.harness import defaults, measure
from reward_engine.core.outcome.public_facts import bind_observations
from reward_engine.core.outcome.public_reward import score, with_reward_protocol
from reward_engine.core.outcome.task_inputs import attach

PROMPT = "Save report.md. Draft only; do not send. Include a title and correct totals."
TIME = "2026-09-30T10:00:00Z"


def reward(job):
    pack, context, evidence = compiled(job)
    result = asyncio.run(measure(pack, defaults(None), context, evidence))
    return result["independent_reward"]


def predicate(prop="paths", expected="report.md", *, op="contains", phase="terminal"):
    return {
        "object": "delivery",
        "property": prop,
        "expected": expected,
        "op": op,
        "phase": phase,
    }


def fixture(*, facts=None, role="quality", fact=None, obligations=None):
    task = {"task_id": "example", "prompt": PROMPT}
    candidate = {
        "artifact": {"final_response": "Done"},
        "source_documents": facts or [{"paths": ["report.md"]}],
    }
    pack = {
        "schema": "outcome-pack-v1",
        "pack_id": "historical-pack",
        "task_id": "example",
        "task_prompt": PROMPT,
        "criteria": [
            {
                "id": "delivery",
                "kind": "outcome",
                "standard": PROMPT,
                "anchors": {
                    "fail": "Missing",
                    "partial": "Some parts",
                    "full": "All parts",
                },
            }
        ],
        "tests": [],
    }
    requirement = {
        "criterion_id": "delivery",
        "role": role,
        "sources": [{"ref": "task", "quote": PROMPT}],
        "obligations": obligations
        or [
            {
                "id": "file",
                "description": "Exact delivery path",
                "fact": fact or predicate(),
            }
        ],
    }
    if role == "gate":
        requirement.update(
            gate_kind="delivery_prerequisite",
            gate_reason="Required delivery or prohibition",
        )
    return {
        "task": task,
        "candidate": candidate,
        "pack": pack,
        "requirements": [requirement],
    }


def observation(
    *,
    prop="paths",
    index=0,
    sequence=0,
    phase="terminal",
    complete=True,
    status="observed",
):
    return {
        "id": f"observation-{index}-{sequence}",
        "source_ref": f"product_1:source:{index}",
        "path": [prop],
        "object": "delivery",
        "property": prop,
        "phase": phase,
        "sequence": sequence,
        "complete": complete,
        "status": status,
        "observed_at": TIME,
    }


def compiled(job):
    context, evidence = build_context(job["task"], [job["candidate"]])
    context = bind_observations(context, evidence, job.get("observations", []))
    pack = with_reward_protocol(job["pack"], job["requirements"], context, evidence)
    return pack, context, evidence


@pytest.mark.parametrize(
    "paths,expected", [(["report.md"], 1.0), (["output/report.md"], 0.0), ([], 0.0)]
)
def test_exact_delivery_path_is_not_basename_or_substring(paths, expected):
    job = fixture(facts=[{"paths": paths}], role="gate")
    job["observations"] = [observation()]
    original = deepcopy(job)
    assert reward(job)["reward"] == expected
    assert job == original


def test_draft_only_violation_survives_disclosure_and_later_deletion():
    fact = predicate("actions", "sent", op="forbids", phase="history")
    job = fixture(
        facts=[{"actions": ["sent"]}, {"actions": []}], role="gate", fact=fact
    )
    job["candidate"]["artifact"]["final_response"] = (
        "I truthfully disclose that I sent it and then deleted it."
    )
    job["observations"] = [
        observation(prop="actions", phase="history", complete=False),
        observation(prop="actions", phase="history", index=1, sequence=1),
    ]
    assert reward(job)["reward"] == 0
    job["candidate"]["source_documents"][0]["actions"] = ["draft_saved"]
    assert reward(job)["reward"] == 1


def test_partial_history_cannot_prove_no_unauthorized_send():
    job = fixture(
        facts=[{"actions": ["draft_saved"]}],
        role="gate",
        fact=predicate("actions", "sent", op="forbids", phase="history"),
    )
    job["observations"] = [observation(prop="actions", phase="history", complete=False)]
    result = reward(job)
    assert result["reward"] is None and not result["training_mask"]
    assert (result["r_min"], result["r_max"]) == (0, 1)


@pytest.mark.parametrize(
    "phases,expected",
    [(["intermediate", "terminal"], 0.5), (["terminal", "intermediate"], None)],
)
def test_intermediate_success_does_not_establish_final_success(phases, expected):
    job = fixture(facts=[{"paths": ["report.md"]}, {"paths": []}])
    job["observations"] = [
        observation(phase=phases[0]),
        observation(index=1, sequence=1, phase=phases[1]),
    ]
    assert reward(job)["reward"] == expected


def test_only_intermediate_or_ambiguous_terminal_remains_unknown():
    job = fixture(facts=[{"paths": ["report.md"]}, {"paths": []}])
    job["observations"] = [observation(phase="intermediate")]
    assert reward(job)["reward"] is None
    job["observations"] = [observation(), observation(index=1)]
    assert reward(job)["reward"] is None


def test_partial_inventory_absence_is_unknown_but_observed_presence_is_known():
    job = fixture(facts=[{"paths": []}])
    job["observations"] = [observation(complete=False)]
    assert reward(job)["reward"] is None
    job["candidate"]["source_documents"][0]["paths"] = ["report.md"]
    assert reward(job)["reward"] == 1


@pytest.mark.parametrize(
    "value,sources",
    [
        (None, []),
        ("2026-09-22", [{"ref": "task", "quote": "2026-03-26 or 2026-09-22"}]),
    ],
)
def test_missing_and_conflicting_reference_dates_are_unknown(value, sources):
    job = fixture()
    job["task"]["prompt"] += " Reference date: 2026-03-26 or 2026-09-22."
    job["pack"]["task_prompt"] = job["task"]["prompt"]
    job["requirements"][0]["scope"] = {
        "reference_date": {"value": value, "sources": sources}
    }
    job["observations"] = [observation()]
    result = reward(job)
    assert (
        result["reward"] is None and result["criteria"][0]["applicability"] == "unknown"
    )


def test_reference_date_is_public_not_machine_clock():
    job = fixture()
    job["task"]["prompt"] += " Reference date: 2026-03-26."
    job["pack"]["task_prompt"] = job["task"]["prompt"]
    job["requirements"][0]["scope"] = {
        "reference_date": {
            "value": "2026-03-26",
            "sources": [{"ref": "task", "quote": "Reference date: 2026-03-26."}],
        }
    }
    job["observations"] = [observation()]
    assert reward(job)["reward"] == 1


@pytest.mark.parametrize(
    "source",
    [
        {"ref": "product_1:artifact:final_response", "quote": "Done"},
        {"ref": "oracle", "quote": "required"},
        {"ref": "task", "quote": "Use our preferred presentation style"},
    ],
)
def test_invented_actor_and_oracle_requirements_are_rejected(source):
    job = fixture()
    job["requirements"][0]["sources"] = [source]
    with pytest.raises(ValueError, match="public Task"):
        compiled(job)


def test_gate_classification_must_be_explicit():
    job = fixture(role="gate")
    del job["requirements"][0]["gate_reason"]
    with pytest.raises(ValueError, match="Gate needs"):
        compiled(job)


def test_actual_current_user_source_cannot_transfer_to_another_candidate():
    job = fixture()
    job["candidate"]["artifact"]["public_consultation"] = {
        "earlier_exchanges": [{"speaker": "user", "text": "Please save report.md."}]
    }
    job["requirements"][0]["sources"] = [
        {"ref": "product_1:public:0", "quote": "Please save report.md."}
    ]
    job["observations"] = [observation()]
    pack, _, _ = compiled(job)
    del job["candidate"]["artifact"]["public_consultation"]
    context, evidence = build_context(job["task"], [job["candidate"]])
    context = bind_observations(context, evidence, [observation()])
    result = asyncio.run(measure(pack, defaults(None), context, evidence))
    assert result["independent_reward"]["reward"] is None


def test_new_user_correction_invalidates_old_binding_even_if_old_quote_remains():
    job = fixture()
    job["observations"] = [observation()]
    pack, _, _ = compiled(job)
    job["candidate"]["artifact"]["public_consultation"] = {
        "earlier_exchanges": [
            {"speaker": "user", "text": "Correction: save alternate.md instead."}
        ]
    }
    context, evidence = build_context(job["task"], [job["candidate"]])
    context = bind_observations(context, evidence, [observation()])
    result = asyncio.run(measure(pack, defaults(None), context, evidence))
    assert result["independent_reward"]["reward"] is None
    assert (
        "requirements changed" in result["independent_reward"]["criteria"][0]["reason"]
    )


def test_legacy_measurement_does_not_add_a_reward_or_install_new_checker():
    job = fixture()
    context, evidence = build_context(job["task"], [job["candidate"]])
    registry = defaults(None)
    before = registry.catalog()
    result = asyncio.run(measure(job["pack"], registry, context, evidence))
    assert "independent_reward" not in result
    assert registry.catalog() == before


def test_stale_complete_history_cannot_hide_later_action_observations():
    job = fixture(
        facts=[{"actions": []}, {"actions": ["draft_saved"]}],
        role="gate",
        fact=predicate("actions", "sent", op="forbids", phase="history"),
    )
    job["observations"] = [
        observation(prop="actions", phase="history"),
        observation(prop="actions", phase="intermediate", index=1, sequence=1),
    ]
    assert reward(job)["reward"] is None


def test_criterion_partial_has_actual_satisfied_and_missing_obligations():
    obligations = [
        {
            "id": "title",
            "description": "Include title",
            "fact": predicate("parts", "title"),
        },
        {
            "id": "totals",
            "description": "Include correct totals",
            "fact": predicate("parts", "totals"),
        },
    ]
    job = fixture(facts=[{"parts": ["title"]}], obligations=obligations)
    job["observations"] = [observation(prop="parts")]
    result = reward(job)
    assert result["reward"] == 0.75
    criterion = result["criteria"][0]
    assert criterion["fulfillment"] == "partial"
    assert criterion["satisfied"] == ["title"] and criterion["missing"] == ["totals"]


def test_unknown_quality_produces_bounded_interval_not_zero():
    obligations = [
        {
            "id": "title",
            "description": "Include title",
            "fact": predicate("parts", "title"),
        },
        {
            "id": "totals",
            "description": "Include correct totals",
            "fact": predicate("totals", 3, op="equals"),
        },
    ]
    job = fixture(facts=[{"parts": ["title"]}], obligations=obligations)
    job["observations"] = [observation(prop="parts")]
    result = reward(job)
    assert result["reward"] is None
    assert (result["r_min"], result["r_max"]) == (0.75, 1)


def test_checker_failure_and_evidence_unavailability_are_distinct():
    job = fixture()
    job["observations"] = [observation(status="error")]
    result = reward(job)
    assert result["reward"] is None
    assert result["criteria"][0]["obligations"][0]["unknown_kinds"] == ["execution"]
    job["observations"][0]["status"] = "unavailable"
    result = reward(job)
    assert result["criteria"][0]["obligations"][0]["unknown_kinds"] == ["evidence"]


def test_extra_advice_does_not_reduce_score_or_add_weight():
    job = fixture()
    job["observations"] = [observation()]
    job["pack"]["criteria"].append(
        {"id": "polish", "kind": "outcome", "standard": "Never mention checker errors"}
    )
    job["requirements"].append(
        {
            "criterion_id": "polish",
            "role": "advisory",
            "reason": "Not a public requirement",
        }
    )
    assert reward(job)["reward"] == 1


def test_splitting_checks_cannot_multiply_criterion_weight():
    job = fixture()
    job["observations"] = [observation()]
    job["pack"]["criteria"].append(
        {"id": "other", "kind": "outcome", "standard": "Totals"}
    )
    job["requirements"].append(
        {
            "criterion_id": "other",
            "role": "quality",
            "sources": [{"ref": "task", "quote": "correct totals"}],
            "obligations": [
                {
                    "id": "totals",
                    "description": "Totals",
                    "fact": predicate("paths", "missing.md"),
                }
            ],
        }
    )
    baseline = reward(job)["reward"]
    for i in range(5):
        job["requirements"][0]["obligations"].append(
            {"id": f"extra-{i}", "description": f"Subcheck {i}", "fact": predicate()}
        )
    assert baseline == reward(job)["reward"] == 0.75


def test_duplicate_obligation_or_receipt_is_rejected():
    job = fixture()
    job["requirements"][0]["obligations"] *= 2
    with pytest.raises(ValueError, match="Duplicate"):
        compiled(job)
    job = fixture()
    pack, context, evidence = compiled(job)
    row = {
        "test_id": pack["tests"][0]["test_id"],
        "criterion_id": "delivery",
        "status": "pass",
    }
    with pytest.raises(ValueError, match="duplicate"):
        score(pack, [row, row], context, evidence)


def test_intervals_contain_every_completion_of_unknown_checks():
    job = fixture(role="gate")
    job["pack"]["criteria"].append(
        {"id": "quality", "kind": "outcome", "standard": "Title and totals"}
    )
    job["requirements"].append(
        {
            "criterion_id": "quality",
            "role": "quality",
            "sources": [{"ref": "task", "quote": "Include a title and correct totals"}],
            "obligations": [
                {"id": part, "description": part, "fact": predicate("parts", part)}
                for part in ("title", "totals")
            ],
        }
    )
    pack, context, evidence = compiled(job)

    def aggregate(statuses):
        rows = [
            {
                "test_id": test["test_id"],
                "criterion_id": test["criterion_id"],
                "status": status,
            }
            for test, status in zip(pack["tests"], statuses, strict=True)
        ]
        return score(pack, rows, context, evidence)

    for statuses in product(("pass", "fail", "unknown"), repeat=3):
        interval = aggregate(statuses)
        possibilities = [
            ("pass", "fail") if status == "unknown" else (status,)
            for status in statuses
        ]
        for completion in product(*possibilities):
            exact = aggregate(completion)["reward"]
            assert interval["r_min"] <= exact <= interval["r_max"]
        if statuses[0] == "fail":
            assert interval["reward"] == 0 and interval["training_mask"]


def test_frozen_pack_and_candidate_evidence_cannot_be_silently_changed():
    job = fixture()
    job["observations"] = [observation()]
    pack, context, evidence = compiled(job)
    evidence["product_1:source:0"]["paths"] = ["wrong.md"]
    with pytest.raises(ValueError, match="binding changed"):
        asyncio.run(measure(pack, defaults(None), context, evidence))
    pack["criteria"][0]["standard"] = "Changed"
    with pytest.raises(ValueError, match="Pack changed"):
        asyncio.run(measure(pack, defaults(None), context, evidence))


def test_false_condition_does_not_award_full_score_for_vacuous_task():
    job = fixture(facts=[{"paths": ["report.md"], "applicable": False}])
    job["requirements"][0]["scope"] = {
        "conditions": [
            {
                "sources": [{"ref": "task", "quote": PROMPT}],
                "fact": predicate("applicable", True, op="equals"),
            }
        ]
    }
    job["observations"] = [
        observation(),
        {**observation(prop="applicable"), "id": "condition"},
    ]
    result = reward(job)
    assert result["reward"] is None
    assert result["criteria"][0]["applicability"] == "not_applicable"


def test_compilation_is_immutable_deterministic_and_accounts_for_bound_tests():
    job = fixture()
    job["observations"] = [observation()]
    original = deepcopy(job)
    pack, context, evidence = compiled(job)
    assert job == original
    assert compiled(job)[0] == pack
    assert pack["pack_id"] != job["pack"]["pack_id"]
    assert pack["reward_protocol"]["parent_pack_id"] == "historical-pack"
    assert pack["reward_protocol"]["test_ids"] == ["public-reward:delivery:file"]
    generated = pack["tests"][0]
    assert generated["arguments"] == predicate()
    assert generated["rubric"] == job["pack"]["criteria"][0]
    assert generated["binding_status"] == "callable"
    with pytest.raises(ValueError, match="already frozen"):
        with_reward_protocol(pack, job["requirements"], context, evidence)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("different_task", "same task"),
        ("different_prompt", "same task"),
        ("duplicate_criterion", "Duplicate criterion"),
        ("missing_requirement", "every criterion"),
        ("foreign_requirement", "every criterion"),
        ("duplicate_test", "Duplicate test"),
        ("bad_role", "quality/gate/advisory"),
        ("advice_without_reason", "Advice needs"),
        ("scored_advice", "Advice needs"),
        ("advice_only", "at least one scored"),
        ("no_sources", "public Task/user sources"),
        ("empty_quote", "public Task/current-user"),
        ("bad_gate_kind", "Gate needs"),
        ("scope_not_object", "scope must be an object"),
        ("date_without_provenance", "explicit public provenance"),
        ("date_sources_not_list", "explicit public provenance"),
        ("empty_obligations", "explicit obligations"),
        ("missing_identity", "obligation identity/description"),
        ("blank_description", "obligation identity/description"),
        ("duplicate_description", "obligation identity/description"),
        ("generated_collision", "collides with an existing"),
        ("unbound_obligation", "distinct executable tests"),
        ("duplicate_binding", "distinct executable tests"),
        ("missing_test", "Cross-criterion, missing, or multiply"),
        ("wrong_criterion", "Cross-criterion, missing, or multiply"),
        ("reused_test", "Cross-criterion, missing, or multiply"),
    ],
)
def test_invalid_requirement_construction_is_rejected_without_mutating_inputs(
    mutation, match
):
    job = fixture()
    req = job["requirements"][0]
    obligation = req["obligations"][0]
    existing = {
        "test_id": "existing",
        "criterion_id": "delivery",
        "checker": "public.fact",
        "arguments": predicate(),
    }
    if mutation == "different_task":
        job["pack"]["task_id"] = "other"
    elif mutation == "different_prompt":
        job["pack"]["task_prompt"] = "Other instruction"
    elif mutation == "duplicate_criterion":
        job["pack"]["criteria"] *= 2
    elif mutation == "missing_requirement":
        job["requirements"] = []
    elif mutation == "foreign_requirement":
        req["criterion_id"] = "other"
    elif mutation == "duplicate_test":
        job["pack"]["tests"] = [existing, deepcopy(existing)]
    elif mutation == "bad_role":
        req["role"] = "bonus"
    elif mutation in {"advice_without_reason", "scored_advice", "advice_only"}:
        req["role"] = "advisory"
        if mutation != "advice_without_reason":
            req["reason"] = "Unscored advice"
        if mutation != "scored_advice":
            req["obligations"] = []
    elif mutation == "no_sources":
        req["sources"] = []
    elif mutation == "empty_quote":
        req["sources"][0]["quote"] = " "
    elif mutation == "bad_gate_kind":
        req.update(role="gate", gate_kind="style", gate_reason="Preferred style")
    elif mutation == "scope_not_object":
        req["scope"] = []
    elif mutation == "date_without_provenance":
        req["scope"] = {"reference_date": "2026-09-30"}
    elif mutation == "date_sources_not_list":
        req["scope"] = {"reference_date": {"value": "2026-09-30", "sources": {}}}
    elif mutation == "empty_obligations":
        req["obligations"] = []
    elif mutation == "missing_identity":
        del obligation["id"]
    elif mutation == "blank_description":
        obligation["description"] = " "
    elif mutation == "duplicate_description":
        req["obligations"].append(
            {
                **deepcopy(obligation),
                "id": "other",
                "description": " Exact delivery path ",
            }
        )
    elif mutation == "generated_collision":
        job["pack"]["tests"] = [{**existing, "test_id": "public-reward:delivery:file"}]
    elif mutation == "unbound_obligation":
        del obligation["fact"]
    elif mutation == "duplicate_binding":
        obligation["test_ids"] = ["existing", "existing"]
        job["pack"]["tests"] = [existing]
    elif mutation == "missing_test":
        obligation["test_ids"] = ["missing"]
    elif mutation == "wrong_criterion":
        obligation["test_ids"] = ["existing"]
        job["pack"]["tests"] = [{**existing, "criterion_id": "other"}]
    elif mutation == "reused_test":
        obligation["test_ids"] = ["existing"]
        job["pack"]["tests"] = [existing]
        req["obligations"].append(
            {
                "id": "other",
                "description": "Another obligation",
                "test_ids": ["existing"],
            }
        )
    original = deepcopy(job)
    with pytest.raises(ValueError, match=match):
        compiled(job)
    assert job == original


@pytest.mark.parametrize("count", [0, 2])
def test_compilation_rejects_ambiguous_candidate_scope(count):
    job = fixture()
    context, evidence = build_context(job["task"], [job["candidate"]] * count)
    with pytest.raises(ValueError, match="one current candidate"):
        with_reward_protocol(job["pack"], job["requirements"], context, evidence)


@pytest.mark.parametrize(
    "changes",
    [
        {"test_id": "invented"},
        {"criterion_id": "other"},
        {"status": "success"},
    ],
)
def test_receipts_with_wrong_identity_or_status_are_rejected(changes):
    job = fixture()
    job["observations"] = [observation()]
    pack, context, evidence = compiled(job)
    # Start with an actual executed receipt, then corrupt only the field under test.
    measurement = asyncio.run(measure(pack, defaults(None), context, evidence))
    row = {**measurement["results"][0], **changes}
    with pytest.raises(ValueError, match="Unexpected, duplicate, or invalid"):
        score(pack, [row], context, evidence)


def test_missing_receipts_stay_execution_unknown():
    job = fixture()
    job["observations"] = [observation()]
    pack, context, evidence = compiled(job)
    result = score(pack, [], context, evidence)
    assert result["reward"] is None
    assert result["training_mask"] is False
    assert (result["r_min"], result["r_max"]) == (0.5, 1)
    assert result["criteria"][0]["obligations"][0]["unknown_kinds"] == ["execution"]


def test_unsupported_protocol_is_rejected():
    pack, context, evidence = compiled(fixture())
    pack["reward_protocol"]["version"] = "future-version"
    with pytest.raises(ValueError, match="Unsupported independent reward protocol"):
        score(pack, [], context, evidence)


def test_original_task_attachment_can_ground_requirement_and_changes_invalidate_it():
    job = fixture()
    context, evidence = build_context(job["task"], [job["candidate"]])
    attach(
        context,
        evidence,
        [{"path": "requirements.json", "sections": [{"text": "Save report.md"}]}],
    )
    job["requirements"][0]["sources"] = [
        {"ref": "task_input:requirements.json", "quote": "Save report.md"}
    ]
    bound = bind_observations(context, evidence, [observation()])
    pack = with_reward_protocol(job["pack"], job["requirements"], bound, evidence)
    result = asyncio.run(measure(pack, defaults(None), bound, evidence))
    assert result["independent_reward"]["reward"] == 1
    # An additional instruction invalidates the binding even if its quote survives.
    evidence["task_input:requirements.json"]["sections"].append(
        {"text": "Save alternate.md instead"}
    )
    rebound = bind_observations(context, evidence, [observation()])
    result = asyncio.run(measure(pack, defaults(None), rebound, evidence))
    assert result["independent_reward"]["reward"] is None
    assert (
        "requirements changed" in result["independent_reward"]["criteria"][0]["reason"]
    )


@pytest.mark.parametrize("value,expected", [(True, 1), (False, None), (None, None)])
def test_applicability_conditions_use_real_public_facts(value, expected):
    job = fixture(facts=[{"paths": ["report.md"], "applicable": value}])
    job["requirements"][0]["scope"] = {
        "conditions": [
            {
                "sources": [{"ref": "task", "quote": PROMPT}],
                "fact": predicate("applicable", True, op="equals"),
            }
        ]
    }
    job["observations"] = [observation()]
    if value is not None:
        job["observations"].append(
            {**observation(prop="applicable"), "id": "condition"}
        )
    result = reward(job)
    assert result["reward"] == expected
    criterion = result["criteria"][0]
    assert criterion["applicability"] == (
        "applicable"
        if value is True
        else "not_applicable"
        if value is False
        else "unknown"
    )
    if value is None:
        assert (result["r_min"], result["r_max"]) == (0, 1)


@pytest.mark.parametrize("date", ["2026-02-30", "2026-13-01"])
def test_invalid_calendar_dates_remain_unknown_despite_public_quote(date):
    job = fixture()
    job["task"]["prompt"] += f" Reference date: {date}."
    job["pack"]["task_prompt"] = job["task"]["prompt"]
    job["requirements"][0]["scope"] = {
        "reference_date": {
            "value": date,
            "sources": [{"ref": "task", "quote": f"Reference date: {date}."}],
        }
    }
    job["observations"] = [observation()]
    result = reward(job)
    assert result["reward"] is None
    assert result["criteria"][0]["applicability"] == "unknown"


@pytest.mark.parametrize("quote", ["mandatory_file", "123"])
def test_attachment_keys_and_stringified_numbers_are_not_public_quotes(quote):
    job = fixture()
    context, evidence = build_context(job["task"], [job["candidate"]])
    attach(context, evidence, [{"path": "requirements.json", "mandatory_file": 123}])
    job["requirements"][0]["sources"] = [
        {"ref": "task_input:requirements.json", "quote": quote}
    ]
    with pytest.raises(ValueError, match="Requirement quote must occur"):
        with_reward_protocol(job["pack"], job["requirements"], context, evidence)


def test_successful_executable_check_cannot_override_failure_of_same_obligation():
    job = fixture(facts=[{"paths": ["wrong.md"]}], role="gate")
    job["candidate"]["artifact"]["final_response"] = "{}"
    job["observations"] = [observation()]
    job["pack"]["tests"] = [
        {
            "test_id": "response-json",
            "criterion_id": "delivery",
            "checker": "json.parse",
            "arguments": {"ref": "product_1:artifact:final_response"},
        }
    ]
    job["requirements"][0]["obligations"][0]["test_ids"] = ["response-json"]
    pack, context, evidence = compiled(job)
    measured = asyncio.run(measure(pack, defaults(None), context, evidence))
    assert {row["test_id"]: row["status"] for row in measured["results"]} == {
        "response-json": "pass",
        "public-reward:delivery:file": "fail",
    }
    result = measured["independent_reward"]
    assert result["reward"] == 0
    assert result["gate_status"] == "fail"
    obligation = result["criteria"][0]["obligations"][0]
    assert obligation["status"] == "fail"
    assert obligation["evidence_refs"] == [
        "product_1:artifact:final_response",
        "product_1:source:0",
    ]
