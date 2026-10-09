import asyncio

import pytest

from reward_engine.core.batch import BatchPolicy
from reward_engine.core.outcome.harness import defaults
from reward_engine.core.outcome.loop import run_task, run_tasks


@pytest.mark.parametrize("with_question", [False, True])
@pytest.mark.parametrize("start_round", [1, 2])
@pytest.mark.parametrize("with_review", [False, True])
@pytest.mark.parametrize("with_d", [False, True])
@pytest.mark.parametrize("continuation_mode", ["new", "stop", "fixed"])
def test_three_rounds_with_terminal_ranking_no_evaluation_score_leak(with_question, start_round, with_review, with_d,
                                                                  continuation_mode, grading_concurrency=3):
    task = {"task_id": "x", "prompt": "Return JSON"}
    seeds = [{"candidate_id": str(i), "artifact": {"final_response": "{}"}} for i in range(3 * start_round)]
    initial_checks = [{"product_id": "product_1", "pack_id": "prior", "checks": [
        {"test_id": "previous", "status": "pass", "reason": "Observed JSON",
         "evidence_refs": ["product_1:artifact:final_response"]}]}] if start_round == 2 else []
    requests, arms, records = [], [], {}

    async def knowledge(context):
        from reward_engine.core.outcome.task_review import requirements

        return {"status": "ready", "input": requirements(context), "directions": [{"id": "json",
            "direction": "Use a real parser", "citations": [{"source_id": "s", "quote": "Parse"}]}],
            "sources": [{"source_id": "s", "title": "Parsing", "url": "https://example.org/parser"}]}

    def roles(role, prefix):
        async def call(request):
            requests.append(request)
            assert "OFFICIAL_SECRET" not in str(request)
            mode = request["mode"]
            if "ranking" in mode:
                products = request["context"]["products"]
                ranking = {
                    "groups": [[p["product_id"] for p in products]],
                    "unknown": [],
                    "reason": "Same actual JSON",
                    "observations": [
                        {
                            "product_id": p["product_id"],
                            "reason": "Identical outcome",
                            "evidence_refs": [p["product_id"] + ":artifact:final_response"],
                        }
                        for p in products
                    ],
                }
                if mode == "outcome_ranking_reduce" and "loop_review" in request:
                    assert "OFFICIAL_SECRET" not in str(request["loop_review"])
                    current = request["loop_review"]["current_product_ids"]
                    target = current[0] if continuation_mode != "wrong_product" else "product_1"
                    gap = "repeated" if continuation_mode == "repeat" else prefix
                    ranking["continuation"] = {"decision": "continue", "reason": "A useful new direction",
                        "gap_id": gap, "gap": "New test-only missing capability " + gap,
                        "why_important": "A material Task requirement remains unfulfilled",
                        "why_new": "New observation compared with earlier criteria and decisions",
                        "next_step": "Verify the observed gap before changing the definition",
                        "product_ids": [target], "evidence_refs": [target + ":artifact:final_response"]}
                    if continuation_mode == "stop":
                        ranking["continuation"] = {"decision": "stop", "reason": "No important new gap"}
                    elif continuation_mode == "missing":
                        ranking.pop("continuation")
                    elif continuation_mode == "invalid":
                        ranking["continuation"]["evidence_refs"] = ["invented"]
                    elif continuation_mode == "unavailable":
                        raise ConnectionError("Terminal reducer unavailable")
                return ranking
            if "jury" in mode or mode == "outcome_expectations":
                assert ("task_review_prior" in str(request["context"]["skills"])) == with_review
                return {
                    "coverage": [{"expectation_id": e["expectation_id"], "criterion_ids": ["valid"],
                                  "status": "mapped", "reason": "JSON is the requested output",
                                  "subrequirements": [{"part_id": p["part_id"], "status": "mapped",
                                      "criterion_ids": ["valid"], "reason": "Syntax retained",
                                      "retained_text": "Valid JSON"} for p in e["subrequirements"]]}
                                 for e in request.get("independent_expectations", []) + request.get("proposal_expectations", [])],
                    "criteria": [
                        {
                            "id": "valid",
                            "standard": "Valid JSON",
                            "kind": "hard",
                            "origin": "task",
                            "measurement_question": "Parse JSON",
                            "evidence_refs": ["task"],
                        }
                    ],
                    "gaps": [],
                }
            if "hypothesis" in mode:
                if with_question:
                    from tests.support.claw_outcome_hypothesis_questions import proposal
                    return proposal()
                return {"status": "keep", "method": ""}
            if mode == "outcome_builder":
                return {
                    "rubrics": [
                        {
                            "criterion_id": "valid",
                            "weight": 1,
                            "anchors": {"fail": "invalid", "partial": "fragment", "full": "valid"},
                        }
                    ],
                    "tests": [
                        {
                            "test_id": "parse",
                            "criterion_id": "valid",
                            "checker": "json.parse",
                            "question": "parse",
                            "arguments": {"ref": "product_1:artifact:final_response"},
                        }
                    ],
                }
            raise AssertionError(mode)

        return call

    async def forbidden(request):
        raise AssertionError("No unnecessary semantic judge")

    async def sample(arm, number, prompt, pack):
        if arm in {"C", "D"} and with_question:
            assert "Who is the intended audience?" in prompt
        if arm not in {"C", "D"}:
            assert "Who is the intended audience?" not in prompt
        arms.append((arm, number, bool(prompt), pack is not None))
        return {"candidate": {"candidate_id": f"{number}{arm}", "artifact": {"final_response": "{}"}}}

    grade_started, grade_barriers = {}, {}
    grade_active, grade_peak = 0, 0

    async def evaluate(task, row):
        nonlocal grade_active, grade_peak
        number, arm = row['candidate']['candidate_id'][:-1], row['candidate']['candidate_id'][-1]
        grade_active += 1
        grade_peak = max(grade_peak, grade_active)
        grade_started[number] = grade_started.get(number, 0) + 1
        event = grade_barriers.setdefault(number, asyncio.Event())
        if grade_started[number] >= grading_concurrency:
            event.set()
        try:
            await asyncio.wait_for(event.wait(), 1)
            if with_d and arm == 'B':
                raise ConnectionError('One independent grader unavailable')
            return {"status": "official", "score": 70, "private_reason": "OFFICIAL_SECRET"}
        finally:
            grade_active -= 1

    result = asyncio.run(
        run_task(
            task,
            seeds,
            rounds=4 - start_round,
            start_round=start_round,
            initial_checks=initial_checks,
            roles=roles,
            task_review=knowledge if with_review else None,
            enabled_arms=tuple('ABCD' if with_d else 'ABC'),
            registry=defaults(forbidden),
            sample=sample,
            evaluate=evaluate,
            policy=BatchPolicy(concurrency=grading_concurrency),
            continuation_policy="fixed_rounds" if continuation_mode == "fixed" else "jury_new_gap",
            record=lambda name, data: records.update({name: data}),
        )
    )
    count = 4 if with_d else 3
    assert grade_peak == grading_concurrency and grade_active == 0
    assert all(n == count for n in grade_started.values())
    for row in result['rounds']:
        assert row['official']['A']['score'] == 70
        assert row['official']['C']['score'] == 70
        if with_d:
            assert row['official']['B'] == {'status': 'failed', 'error_type': 'ConnectionError'}
            assert row['official']['D']['score'] == 70
    expected_rounds = (4 - start_round if continuation_mode in {"new", "fixed"} else
                       min(2, 4 - start_round) if continuation_mode == "repeat" else 1)
    assert result["status"] == "completed"
    assert result["completed_rounds"] == expected_rounds and len(arms) == count * expected_rounds
    assert min(number for _, number, _, _ in arms) == start_round
    assert bool(records[f"round_{start_round}/context"]["checks"]) == bool(initial_checks)
    assert result["rounds"][-1]["pool_size"] == 3 * start_round + count * expected_rounds
    assert all((p and pack) if arm in {"B", "D"} else (p == (arm == "C" and with_question) and not pack)
               for arm, _, p, pack in arms)
    assert sum(r["mode"] == "outcome_ranking_reduce" for r in requests) == expected_rounds
    last = f"round_{start_round + expected_rounds - 1}"
    assert records[last + "/context"]["products"] and "OFFICIAL_SECRET" not in str(records[last + "/context"])
    assert records[last + "/loop_decision"]["decision"] == "stop"
    if continuation_mode in {"missing", "invalid", "wrong_product", "repeat"}:
        assert result["rounds"][0]["ranking"]["status"] == "compared"
    if expected_rounds > 1:
        assert records[f"round_{start_round + 1}/context"]["next_round_focus"]["decision"] == "continue"

    if continuation_mode in {"new", "fixed"}:
        from reward_engine.core.outcome.memory import Memory

        checkpoint = records[f"round_{start_round}/checkpoint"]
        assert len(checkpoint["pool"]) == 3 * start_round + count
        assert "OFFICIAL_SECRET" not in str(checkpoint)
        restored_records, offset = {}, len(requests)
        resumed = asyncio.run(run_task(task, checkpoint["pool"], rounds=3 - start_round,
            start_round=start_round + 1, memory=Memory(checkpoint["memory"]),
            initial_checks=checkpoint["checks"], initial_pack=checkpoint["previous_pack"],
            initial_expectation_cache=checkpoint["expectation_cache"],
            initial_known_gaps=checkpoint["known_gaps"], initial_decisions=checkpoint["decisions"],
            roles=roles, enabled_arms=tuple('ABCD' if with_d else 'ABC'),
            task_review=knowledge if with_review else None, registry=defaults(forbidden),
            sample=sample, evaluate=evaluate, policy=BatchPolicy(concurrency=grading_concurrency),
            continuation_policy="fixed_rounds" if continuation_mode == "fixed" else "jury_new_gap",
            record=lambda name, data: restored_records.update({name: data})))
        assert resumed["status"] == "completed" and resumed["completed_rounds"] == 3 - start_round
        assert not any(r["mode"] == "outcome_expectations" for r in requests[offset:])
        assert restored_records[f"round_{start_round + 1}/context"] == records[f"round_{start_round + 1}/context"]


@pytest.mark.parametrize("mode", ["missing", "invalid", "repeat", "wrong_product", "unavailable"])
def test_no_automatic_quality_redraw_without_a_new_supported_gap(mode):
    test_three_rounds_with_terminal_ranking_no_evaluation_score_leak(
        False, 1, False, False, continuation_mode=mode)


@pytest.mark.parametrize("mutation", [
    {"decision": "maybe"}, {"why_new": ""}, {"next_step": None},
    {"product_ids": ["product_2"]}, {"evidence_refs": ["task"]},
    {"gap_id": "old"}, {"gap": "  PREVIOUS GAP  "},
])
def test_continuation_requires_novel_evidenced_product_gap(mutation):
    from reward_engine.core.outcome.jury import continuation_decision
    context = {"products": [{"product_id": "product_1"}]}
    evidence = {"task": {}, "product_1:artifact:final_response": {}}
    raw = {"decision": "continue", "reason": "New capability missing", "gap_id": "new",
           "gap": "Different gap", "why_important": "Task requirement", "why_new": "New observation",
           "next_step": "Investigate observed defect", "product_ids": ["product_1"],
           "evidence_refs": ["product_1:artifact:final_response"]}
    known = [{"gap_id": "old", "gap": "Previous gap"}]
    assert continuation_decision(raw, context, evidence, known)["decision"] == "continue"
    assert continuation_decision({**raw, **mutation}, context, evidence, known)["decision"] == "stop"


@pytest.mark.parametrize('limit', [1, 2])
def test_private_grading_honors_smaller_configured_concurrency(limit):
    test_three_rounds_with_terminal_ranking_no_evaluation_score_leak(
        with_question=False, start_round=2, with_review=False, with_d=True,
        grading_concurrency=limit, continuation_mode="new")


def test_batch_failure_isolated():
    async def run(task):
        if task["task_id"] == "bad":
            raise ValueError("bad adapter")
        return {"status": "completed"}

    results = asyncio.run(run_tasks([{"task_id": "bad"}, {"task_id": "good"}], run))
    assert results[0]["status"] == "failed" and results[1]["result"]["status"] == "completed"
