"""Real three-round Jury orchestration with controlled model responses and provenance."""

import json

import pytest

from reward_engine.core.outcome import jury
from reward_engine.core.outcome.context import ResponseContractError, ask, build_context
from reward_engine.core.outcome.jury_refinement import (
    check_retired_clauses,
    pending_expectations,
)
from tests.support.claw_outcome_jury_construction import (
    CORRECT,
    OLD,
    REFINED,
    controlled_models,
    deepcopy,
    reduce_reply,
)

KEEP = "Return valid JSON."


def fixture():
    task = {
        "task_id": "construction-test",
        "prompt": "Return valid JSON with an evidence-grounded explanation.",
    }
    candidates = [
        {
            "artifact": {
                "final_response": "OLD_ARTIFACT",
                "public_consultation": {
                    "earlier_exchanges": [
                        {"speaker": "user", "text": "MY_SCOPED_REQUEST"},
                        {"speaker": "assistant", "text": "NOT_A_USER_REQUIREMENT"},
                    ]
                },
            }
        },
        {"artifact": {"final_response": "OTHER_OLD_ARTIFACT"}},
    ]
    context, evidence = build_context(
        task, candidates, skills=["KNOWLEDGE_ONLY"], memories=["MEMORY_ONLY"]
    )
    criterion = {
        "id": "correctness",
        "standard": "Correct evidence-grounded JSON answer",
        "kind": "hard",
        "origin": "task",
        "measurement_question": "Are claims accurate and scoped?",
        "evidence_refs": ["task"],
        "subrequirements": [OLD, KEEP],
    }
    return context, evidence, criterion


@pytest.mark.asyncio
async def test_three_rounds_task_first_and_corrections_follow_original_coverage():
    context, evidence, criterion = fixture()
    members, reducer, calls = controlled_models(criterion)
    records = {}
    result = await jury.propose(
        context,
        evidence,
        members,
        reducer,
        construction_rounds=3,
        record=records.__setitem__,
    )
    assert len(calls) == 15
    assert result["construction"]["completed_rounds"] == 3
    assert result["criteria"][0]["subrequirements"] == [REFINED, KEEP]
    for _, request in calls:
        if request["mode"] == "outcome_expectations":
            encoded = json.dumps(request)
            assert "KNOWLEDGE_ONLY" not in encoded and "MEMORY_ONLY" not in encoded
            assert (
                "OLD_ARTIFACT" not in encoded
                and "NOT_A_USER_REQUIREMENT" not in encoded
            )
            assert "MY_SCOPED_REQUEST" in encoded
            assert (
                request["context"]["products"][0]["user_updates"][0]["ref"]
                == "product_1:public:0"
            )
            assert request["context"]["products"][1]["user_updates"] == []
        if request["mode"] == "outcome_jury_map":
            assert "KNOWLEDGE_ONLY" in json.dumps(request["context"])
        if request.get("construction_round", 1) > 1:
            active = request.get(
                "previous_synthesis", request.get("independent_expectations")
            )
            assert OLD not in json.dumps(active)
            assert "_retired_clauses" not in request
    assert all(
        e["subrequirements"][0]["text"] == OLD
        for e in result["independent_expectations"]
    )
    original_ids = {e["expectation_id"] for e in result["independent_expectations"]}
    rows = [r for r in result["coverage"] if r["expectation_id"] in original_ids]
    for row in rows:
        part = row["subrequirements"][0]
        assert part["proposed_text"] == OLD
        assert part["current_text"] == part["retained_text"] == REFINED
        assert [h["round"] for h in part["lineage"]] == [1, 2, 3]
    assert all(f"construction/jury/round_{n}/output" in records for n in (1, 2, 3))


@pytest.mark.asyncio
async def test_rewritten_then_deferred_clause_keeps_corrected_current_text():
    context, evidence, criterion = fixture()

    async def reducer_hook(request):
        if request.get("construction_round") == 2:
            return reduce_reply(request, criterion, defer_round=2)

    members, reducer, _ = controlled_models(criterion, reducer_hook=reducer_hook)
    result = await jury.propose(
        context, evidence, members, reducer, construction_rounds=2
    )
    pending = pending_expectations(
        result["independent_expectations"] + result["proposal_expectations"],
        result["coverage"],
    )
    assert pending
    assert OLD not in json.dumps(pending)
    assert all(p["text"] == CORRECT for e in pending for p in e["subrequirements"])


@pytest.mark.asyncio
async def test_audit_redo_preserves_expanded_evidence_and_does_not_repeat_other_seats():
    context, evidence, criterion = fixture()
    attempts = []

    async def hook(key, request):
        if key != "a" or request.get("construction_round") != 2:
            return None
        attempts.append(request)
        if len(attempts) == 1:
            return {"need_evidence": ["product_1:artifact:final_response"]}
        if len(attempts) == 2:
            return {**request["previous_synthesis"], "criterion_audit": []}
        assert (
            request["requested_evidence"]["product_1:artifact:final_response"]
            == "OLD_ARTIFACT"
        )
        assert request["response_recovery"]["attempt"] == 1

    members, reducer, calls = controlled_models(criterion, audit_hook=hook)
    result = await jury.propose(
        context, evidence, members, reducer, construction_rounds=3
    )
    assert result["construction"]["completed_rounds"] == 3
    assert len(calls) == 17 and len(attempts) == 3


@pytest.mark.asyncio
async def test_three_seats_are_required_and_failed_round_does_not_publish_partial_jury():
    context, evidence, criterion = fixture()

    async def hook(key, request):
        if key == "b" and request.get("construction_round") == 2:
            return {**request["previous_synthesis"], "criterion_audit": []}

    members, reducer, calls = controlled_models(criterion, audit_hook=hook)
    records = {}
    with pytest.raises(RuntimeError, match="All three"):
        await jury.propose(
            context,
            evidence,
            members,
            reducer,
            construction_rounds=3,
            record=records.__setitem__,
        )
    assert "jury" not in records
    assert records["construction/jury/status"]["completed_rounds"] == 1
    assert not any(r.get("construction_round") == 3 for _, r in calls)


@pytest.mark.asyncio
async def test_expectations_cache_ignores_supplemental_knowledge_but_not_task_or_user_changes():
    context, evidence, criterion = fixture()
    cache = {}
    members, reducer, calls = controlled_models(criterion)
    await jury.propose(
        context,
        evidence,
        members,
        reducer,
        construction_rounds=3,
        expectation_cache=cache,
    )
    first_count = len(calls)
    changed = deepcopy(context)
    changed["skills"][0]["content"] = "DIFFERENT_SUPPLEMENT"
    await jury.propose(
        changed,
        {**evidence, "skill:0": "DIFFERENT_SUPPLEMENT"},
        members,
        reducer,
        construction_rounds=3,
        expectation_cache=cache,
    )
    assert len(calls) - first_count == 12
    changed["products"][0]["user_updates"][0]["text"] = "CHANGED_USER_REQUEST"
    second_count = len(calls)
    await jury.propose(
        changed,
        {**evidence, "product_1:public:0": "CHANGED_USER_REQUEST"},
        members,
        reducer,
        construction_rounds=3,
        expectation_cache=cache,
    )
    assert len(calls) - second_count == 15


@pytest.mark.asyncio
async def test_stale_literal_cannot_silently_return_from_a_new_audit_proposal():
    context, evidence, criterion = fixture()

    async def hook(key, request):
        if request.get("construction_round") == 2:
            row = deepcopy(request["previous_synthesis"])
            row["criteria"][0]["subrequirements"][0] = OLD
            row["criterion_audit"] = [
                {
                    "criterion_id": criterion["id"],
                    "assessment": "revise",
                    "reason": "Unsubstantiated return of an old clause",
                    "evidence_refs": ["task"],
                }
            ]
            return row

    async def bad_reduce(request):
        if request.get("construction_round") == 2:
            reply = reduce_reply(request, criterion)
            old_selectors = {
                p["selection_id"]
                for e in request["proposal_expectations"]
                for p in e["subrequirements"]
                if p["text"] == OLD
            }
            for decision in reply["part_decisions"]:
                if set(decision["parts"]) & old_selectors:
                    decision.update(action="retain")
            reply["criteria"][0]["subrequirements"] = [REFINED, KEEP, OLD]
            return reply

    members, reducer, _ = controlled_models(
        criterion, audit_hook=hook, reducer_hook=bad_reduce
    )
    with pytest.raises(ResponseContractError, match="superseded"):
        await jury.propose(context, evidence, members, reducer, construction_rounds=3)


def test_missing_decisions_defer_but_withdrawn_raw_clause_is_rejected():
    context, evidence, criterion = fixture()
    from reward_engine.core.outcome.jury_drafts import (
        apply_part_decisions,
        expectations,
    )

    inputs = {
        "independent_expectations": expectations([criterion], "a", "before"),
        "proposal_expectations": [],
        "proposals": [{}, {}, {}],
    }
    omitted = {"criteria": [criterion], "part_decisions": []}
    check_retired_clauses(omitted, inputs)
    projected = apply_part_decisions(omitted, inputs, evidence, mark_omissions=True)
    assert all(
        p["status"] == "deferred" and p["code_deferred"]
        for row in projected["coverage"]
        for p in row["subrequirements"]
    )
    request = {
        **inputs,
        "independent_expectations": [
            {
                **inputs["independent_expectations"][0],
                "subrequirements": [
                    {"part_id": "a:0/0", "selection_id": "p0", "text": OLD},
                    {"part_id": "a:0/1", "selection_id": "p1", "text": KEEP},
                ],
            }
        ],
    }
    reply = reduce_reply(request, criterion)
    reply["criteria"][0]["subrequirements"].append(OLD)
    with pytest.raises(ValueError, match="withdrawn"):
        check_retired_clauses(reply, inputs)


def test_retired_selector_cannot_restore_text_when_raw_criteria_omits_it():
    _, _, criterion = fixture()
    from reward_engine.core.outcome.jury_drafts import expectations

    inputs = {
        "independent_expectations": expectations([criterion], "a", "before"),
        "proposal_expectations": [],
        "proposals": [{}, {}, {}],
        "_retired_clauses": [OLD],
    }
    raw = {
        "criteria": [{**criterion, "subrequirements": []}],
        "part_decisions": [
            {"parts": ["p0", "p1"], "action": "retain", "criterion_id": "correctness"}
        ],
    }
    with pytest.raises(ValueError, match="superseded"):
        check_retired_clauses(raw, inputs)
    # Reconsidering a prior exclusion must be explicit and evidenced, even if
    # the now-supported wording itself is unchanged.
    raw["part_decisions"] = [
        {
            "parts": ["p0"],
            "action": "rewrite",
            "criterion_id": "correctness",
            "text": OLD,
            "reason": "The observed Task evidence now supports this previously withheld clause",
            "evidence_refs": ["task"],
        },
        {"parts": ["p1"], "action": "retain", "criterion_id": "correctness"},
    ]
    check_retired_clauses(raw, inputs)


@pytest.mark.asyncio
async def test_followup_audits_preserve_applicability_and_future_semantic_fields():
    context, evidence, criterion = fixture()
    criterion.update(
        applicability={
            "product_ids": ["product_1"],
            "when": "Only the actual user's requested scope",
        },
        domain_contract={"condition": "Keep this semantic field intact"},
    )
    members, reducer, calls = controlled_models(criterion)
    result = await jury.propose(
        context, evidence, members, reducer, construction_rounds=3
    )
    audits = [
        r
        for _, r in calls
        if r.get("construction_round", 1) > 1 and r["mode"] == "outcome_jury_map"
    ]
    assert len(audits) == 6
    for request in audits:
        current = request["previous_synthesis"]["criteria"][0]
        assert current["applicability"] == criterion["applicability"]
        assert current["domain_contract"] == criterion["domain_contract"]
        assert "part_revisions" not in current
    assert result["criteria"][0]["applicability"] == criterion["applicability"]


@pytest.mark.asyncio
async def test_initial_partial_batch_is_not_a_three_seat_round():
    context, evidence, criterion = fixture()
    members, reducer, calls = controlled_models(criterion)

    async def unavailable(_):
        raise ConnectionError("one seat unavailable")

    members["b"] = unavailable
    records = {}
    with pytest.raises(RuntimeError, match="Fewer than three"):
        await jury.propose(
            context,
            evidence,
            members,
            reducer,
            construction_rounds=3,
            record=records.__setitem__,
        )
    assert records["construction/jury/status"]["completed_rounds"] == 0
    assert all(r["mode"] == "outcome_expectations" for _, r in calls)


@pytest.mark.asyncio
async def test_omitted_selector_becomes_current_pending_and_next_round_clears_marker():
    context, evidence, criterion = fixture()

    async def omit_second_round(request):
        if request.get("construction_round") != 2:
            return None
        reply = reduce_reply(request, criterion)
        omitted = {
            p["selection_id"]
            for e in request["independent_expectations"]
            + request["proposal_expectations"]
            for p in e["subrequirements"]
            if p["text"] == CORRECT
        }
        reply["part_decisions"] = [
            d for d in reply["part_decisions"] if not set(d["parts"]) & omitted
        ]
        reply["criteria"][0]["subrequirements"] = [CORRECT, KEEP]
        return reply

    members, reducer, calls = controlled_models(
        criterion, reducer_hook=omit_second_round
    )
    records = {}
    result = await jury.propose(
        context,
        evidence,
        members,
        reducer,
        construction_rounds=3,
        record=records.__setitem__,
    )
    assert (
        len(calls) == 15
    )  # Omission is unresolved coverage, not a format-redo trigger.
    middle = records["construction/jury/round_2/output"]
    original_id = result["independent_expectations"][0]["expectation_id"]
    part = next(r for r in middle["coverage"] if r["expectation_id"] == original_id)[
        "subrequirements"
    ][0]
    assert (
        part["code_deferred"]
        and part["status"] == "deferred"
        and part["current_text"] == CORRECT
    )
    audits = [
        r
        for _, r in calls
        if r.get("construction_round") == 3 and r["mode"] == "outcome_jury_map"
    ]
    for request in audits:
        pending = request["unresolved_expectations"]
        assert pending and OLD not in json.dumps(pending)
        assert any(
            p["text"] == CORRECT and p["code_deferred"]
            for e in pending
            for p in e["subrequirements"]
        )
    final = next(r for r in result["coverage"] if r["expectation_id"] == original_id)[
        "subrequirements"
    ][0]
    assert final["status"] == "mapped" and final["current_text"] == CORRECT
    assert "code_deferred" not in final
    assert final["lineage"][-2]["code_deferred"]
    assert "code_deferred" not in final["lineage"][-1]


@pytest.mark.asyncio
async def test_last_round_omission_stays_visible_without_reviving_old_hypothesis_text():
    from reward_engine.core.outcome.hypotheses import compact_jury

    context, evidence, criterion = fixture()

    async def omit_last_round(request):
        if request.get("construction_round") != 3:
            return None
        reply = reduce_reply(request, criterion)
        omitted = {
            p["selection_id"]
            for e in request["independent_expectations"]
            + request["proposal_expectations"]
            for p in e["subrequirements"]
            if p["text"] == REFINED
        }
        reply["part_decisions"] = [
            d for d in reply["part_decisions"] if not set(d["parts"]) & omitted
        ]
        return reply

    members, reducer, calls = controlled_models(criterion, reducer_hook=omit_last_round)
    result = await jury.propose(
        context, evidence, members, reducer, construction_rounds=3
    )
    assert len(calls) == 15 and result["construction"]["completed_rounds"] == 3
    original_id = result["independent_expectations"][0]["expectation_id"]
    part = next(r for r in result["coverage"] if r["expectation_id"] == original_id)[
        "subrequirements"
    ][0]
    assert (
        part["status"] == "deferred"
        and part["code_deferred"]
        and part["current_text"] == REFINED
    )
    questions = compact_jury(result)["open_questions"]
    assert questions and REFINED in json.dumps(questions)
    assert OLD not in json.dumps(questions) and CORRECT not in json.dumps(questions)


@pytest.mark.parametrize("has_current_explanations", [False, True])
def test_pending_preserves_open_semantic_scope_without_old_explanation_replay(
    has_current_explanations,
):
    stale = "SUPERSEDED_CLAUSE_AND_EXPLANATION"
    expected = [
        {
            "id": "scoped",
            "expectation_id": "initial:0",
            "kind": "hard",
            "origin": "user",
            "evidence_refs": ["product_1:public:0"],
            "standard": stale,
            "why": stale,
            "measurement_question": stale,
            "verification": stale,
            "reason": stale,
            "rationale": stale,
            "applicability": {
                "product_ids": ["product_1"],
                "when": "Only if that user requests this format",
            },
            "future_semantics": {"unit": "ms", "strict": False, "limit": 0},
            "part_revisions": [{"reason": stale}],
            "source_expectation_ids": [stale],
            "draft_validation_issues": [stale],
            "subrequirements": [{"part_id": "initial:0/0", "text": stale}],
        }
    ]
    snapshot = deepcopy(expected)
    coverage = [
        {
            "expectation_id": "initial:0",
            "status": "deferred",
            "criterion_ids": [],
            "subrequirements": [
                {
                    "part_id": "initial:0/0",
                    "status": "deferred",
                    "criterion_ids": [],
                    "current_text": "Use the actual requested output format",
                    "code_deferred": True,
                }
            ],
        }
    ]
    if has_current_explanations:
        coverage[0].update(
            current_why="Current unresolved question",
            current_verification="Inspect the current user request",
        )
    pending = pending_expectations(expected, coverage)
    assert expected == snapshot
    assert len(pending) == 1
    assert pending[0]["applicability"] == expected[0]["applicability"]
    assert pending[0]["future_semantics"] == expected[0]["future_semantics"]
    assert pending[0]["evidence_refs"] == ["product_1:public:0"]
    assert pending[0]["standard"] == "Use the actual requested output format"
    assert pending[0]["subrequirements"][0]["code_deferred"]
    assert stale not in json.dumps(pending)
    assert not set(pending[0]) & {
        "verification",
        "part_revisions",
        "source_expectation_ids",
        "draft_validation_issues",
    }
    if has_current_explanations:
        assert pending[0]["why"] == coverage[0]["current_why"]
        assert pending[0]["measurement_question"] == coverage[0]["current_verification"]


def archived_map_structure(criterion):
    # Exact malformed fields from T068 Jury seat 3's 2026-09-10 redo
    # b7f92699480f4a928104aa3c14278f1d. Other content is a minimal valid fixture;
    # tests do not depend on the paid experiment or accept its factual claims.
    texts = [
        [
            "At least three genuinely distinct defects, not three rephrasings of one.",
            "Each defect anchored to the supplied snippet or quantization description.",
            "Each defect given a bottom-level mechanism (math or CUDA hardware).",
        ],
        [
            "int8_t accumulator named as wrong type.",
            "Overflow bound tied to K products (e.g., max ~127*127*K).",
            "INT32 accumulation and post-accumulation float rescale stated.",
        ],
        [
            "Activation outliers / high dynamic range in LLaMA explained.",
            "Mechanism by which one per-tensor scale is dominated by the max and collapses ordinary values.",
            "Weight side also argued to need finer granularity than per-tensor.",
        ],
        [
            "Wrong bound variable (N used for row) detected.",
            "Indexing formulas checked against declared layouts.",
            "Consequence (wrong/out-of-bounds output) explained.",
        ],
        [
            "Concrete memory-access or execution-unit mechanism.",
            "Tied to the supplied thread mapping / inner loop.",
            "No generic advice such as 'add comments' or 'use a library'.",
        ],
        [
            "Correct accumulator/dequantization path.",
            "Finer-grained quantization or outlier mitigation compatible with LLaMA.",
            "Hardware-appropriate kernel direction.",
        ],
        ["No generic code-quality/style advice presented as the answer."],
        [
            "Correct scale and clip/round formulas.",
            "Correct dequantization given int32 accumulation.",
            "No false claim that scale_a*scale_b is itself the core bug.",
        ],
    ]
    bases = [
        "task formula plus domain prior",
        "task code thread mapping",
        "domain prior / research agenda (improvement opportunity, not a task hard requirement)",
    ]
    return {
        "criteria": [
            {
                **deepcopy(criterion),
                "id": f"C{i + 1}",
                "subrequirements": [
                    {"part_id": f"C{i + 1}/{chr(97 + j)}", "text": text}
                    for j, text in enumerate(parts)
                ],
            }
            for i, parts in enumerate(texts)
        ],
        "gaps": [
            {
                "gap_id": f"G{i + 1}",
                "product_ids": ["product_1"],
                "expected": "Correct task-grounded explanation",
                "expected_basis": basis,
                "expected_refs": ["task"],
                "observed": "A claim in the artifact needs checking",
                "gap": "Suspected unsupported claim",
                "evidence_refs": ["product_1:artifact:final_response"],
                "verification": "Compare against the actual Task evidence",
                "status": "needs_check",
            }
            for i, basis in enumerate(bases)
        ],
    }


@pytest.mark.asyncio
async def test_archived_map_redo_reports_all_shape_errors_and_one_repair_can_pass():
    context, evidence, criterion = fixture()
    malformed = archived_map_structure(criterion)
    original = deepcopy(malformed)
    calls = []

    async def call(request):
        calls.append(deepcopy(request))
        if len(calls) == 1:
            return malformed
        details = request["response_recovery"]["details"]
        assert details["field"] == "response"
        message = details["reason"]
        for index, row in enumerate(malformed["criteria"]):
            for part_index, part in enumerate(row["subrequirements"]):
                assert (
                    f"criteria[{index}].subrequirements[{part_index}]: actual={part!r}"
                    in message
                )
        for index, row in enumerate(malformed["gaps"]):
            assert (
                f"gaps[{index}].expected_basis: actual={row['expected_basis']!r}"
                in message
            )
        assert (
            "nonempty string" in message
            and "exactly one of 'task', 'user', 'domain'" in message
        )
        assert request["context"] == calls[0]["context"]
        assert request["response"] == jury.SCHEMA
        # Only the model fixture repairs its answer; validation never coerces
        # compound provenance or drops object fields silently.
        corrected = deepcopy(malformed)
        for row in corrected["criteria"]:
            row["subrequirements"] = [part["text"] for part in row["subrequirements"]]
        for row in corrected["gaps"]:
            row["expected_basis"] = "domain"
        return corrected

    result = await ask(
        call,
        "outcome_jury_map",
        context,
        evidence,
        "Inspect the Task evidence",
        jury.SCHEMA,
        validate=lambda row: jury.validate(row, evidence, context),
        max_contract_retries=1,
    )
    assert len(calls) == 2
    assert malformed == original
    assert len(result["criteria"]) == 8 and len(result["gaps"]) == 3
    assert jury.validate(result, evidence, context) == result


def test_live_first_map_provenance_reports_path_actual_and_allowed_values():
    context, evidence, criterion = fixture()
    reply = archived_map_structure(criterion)
    reply["criteria"] = [criterion]
    # Original completed output 44510b6ee8db4c94b4f3feb3635eedb1.
    reply["gaps"][0]["expected_basis"] = (
        "task (requires hardcore math) + domain; reference:task_review:source_8 derives the exact bound "
        "16256·K < 2^31, valid for K ≤ 132,104."
    )
    with pytest.raises(ResponseContractError) as caught:
        jury.validate(reply, evidence, context)
    message = caught.value.details["reason"]
    assert "gaps[0].expected_basis" in message
    assert repr(reply["gaps"][0]["expected_basis"]) in message
    assert "exactly one of 'task', 'user', 'domain'" in message


@pytest.mark.parametrize(
    "subrequirements",
    ["a single string", [{"part_id": "p0", "text": "Keep me"}], [" "]],
)
def test_subrequirement_shape_diagnostic_does_not_normalize_or_drop_fields(
    subrequirements,
):
    context, evidence, criterion = fixture()
    criterion["subrequirements"] = subrequirements
    reply = {"criteria": [criterion]}
    original = deepcopy(reply)
    with pytest.raises(ResponseContractError, match=r"criteria\[0\].subrequirements"):
        jury.validate(reply, evidence, context)
    assert reply == original


@pytest.mark.parametrize("mixed_structure_errors", [False, True])
def test_recovery_map_unknown_refs_and_structure_errors_are_reported_together(
    mixed_structure_errors,
):
    context, evidence, criterion = fixture()
    evidence.update(
        {
            "skill:task_review:s1_g2": {
                "authority": "domain_advice_not_user_requirement"
            },
            "skill:task_review:s8_g1": {
                "authority": "domain_advice_not_user_requirement"
            },
        }
    )
    context["available_evidence_refs"] = list(evidence)
    gap = archived_map_structure(criterion)["gaps"][0]
    reply = {
        "criteria": [deepcopy(criterion)],
        "gaps": [
            {**deepcopy(gap), "gap_id": f"G{i}", "expected_basis": "domain"}
            for i in range(5)
        ],
    }
    # Exact unresolved handles and output positions from recovery01's only
    # new response 62e4e0cb3574499cbb7bd2c6e8cb7ab7; no paid archive dependency.
    reply["gaps"][4].update(
        gap_id="g_validation_shared",
        expected_refs=["reference:task_review:s1_g2", "reference:task_review:s8_g1"],
    )
    expected_errors = {
        "gaps[4].expected_refs[0]": "reference:task_review:s1_g2",
        "gaps[4].expected_refs[1]": "reference:task_review:s8_g1",
    }
    if mixed_structure_errors:
        reply["criteria"][0]["subrequirements"] = [{"part_id": "c0", "text": KEEP}]
        reply["criteria"][0]["evidence_refs"].append("reference:missing_criterion")
        reply["gaps"][0]["expected_basis"] = "task formula plus domain prior"
        reply["gaps"][1]["evidence_refs"].append("product_1:source:missing")
        reply["criteria"][0]["part_revisions"] = [
            {"evidence_refs": ["reference:missing_revision"]}
        ]
        expected_errors.update(
            {
                "criteria[0].evidence_refs[1]": "reference:missing_criterion",
                "gaps[1].evidence_refs[1]": "product_1:source:missing",
                "criteria[0].part_revisions[0].evidence_refs[0]": "reference:missing_revision",
            }
        )
    original = deepcopy(reply)
    with pytest.raises(ResponseContractError) as caught:
        jury.validate(reply, evidence, context)
    message = caught.value.details["reason"]
    for field, actual in expected_errors.items():
        assert f"{field}: actual={actual!r}" in message
    assert "exact supplied evidence handle" in message
    if mixed_structure_errors:
        assert (
            "criteria[0].subrequirements[0]" in message and "nonempty string" in message
        )
        assert (
            "gaps[0].expected_basis" in message
            and "exactly one of 'task', 'user', 'domain'" in message
        )
    assert reply == original  # In particular, reference: was never mapped to skill:.
    repaired = deepcopy(reply)
    repaired["gaps"][4]["expected_refs"] = [
        "skill:task_review:s1_g2",
        "skill:task_review:s8_g1",
    ]
    if mixed_structure_errors:
        repaired["criteria"][0]["subrequirements"] = [KEEP]
        repaired["criteria"][0]["evidence_refs"] = ["task"]
        repaired["criteria"][0]["part_revisions"][0]["evidence_refs"] = ["task"]
        repaired["gaps"][0]["expected_basis"] = "domain"
        repaired["gaps"][1]["evidence_refs"] = ["product_1:artifact:final_response"]
    assert jury.validate(repaired, evidence, context) == repaired
