"""Synthetic protocol witnesses: fixed baselines, six-way selection and recovery."""

from copy import deepcopy

import pytest

from reward_engine.core.outcome import hypotheses, three_b

TASK = {"task_id": "synthetic", "prompt": "Write a complete answer."}


def candidate(text):
    return {
        "artifact": {"final_response": text},
        "environment": {},
        "source_documents": [],
    }


@pytest.fixture
def fixture(monkeypatch):
    calls = {
        "baselines": [],
        "actors": [],
        "construction": [],
        "selection": [],
        "grades": [],
    }
    records = {}

    async def forbidden(*args, **kwargs):
        pytest.fail("Hypothesis must have zero calls")

    monkeypatch.setattr(hypotheses, "propose", forbidden)

    async def construct(context, evidence, registry, roles, **kwargs):
        assert kwargs["adversarial_review_enabled"] is True
        assert "PRIVATE_SCORE" not in str(context)
        calls["construction"].append(deepcopy(context))
        return {
            "pack_id": "pack" + str(len(calls["construction"])),
            "criteria": [],
            "tests": [],
        }

    async def measure(pack, registry, context, evidence, **kwargs):
        assert kwargs["test_ids"] is None
        return {"pack_id": pack["pack_id"], "results": []}

    async def smoke(*args):
        return {
            "status": "receipt_path_checked",
            "receipt": {"status": "pass"},
            "bindings": {},
        }

    async def rank(context, evidence, members, reducer, **kwargs):
        assert "PRIVATE_SCORE" not in str(context) and len(context["products"]) == 6
        calls["selection"].append(deepcopy(context))
        ids = [p["product_id"] for p in context["products"]]
        return {
            "status": "compared",
            "groups": [ids],
            "unknown": [],
            "reason": "Equivalent synthetic deliveries",
            "observations": [
                {
                    "product_id": p,
                    "reason": "Observed answer",
                    "evidence_refs": [p + ":artifact:final_response"],
                }
                for p in ids
            ],
        }

    monkeypatch.setattr(three_b, "construct_reward", construct)
    monkeypatch.setattr(three_b, "measure", measure)
    monkeypatch.setattr(three_b, "smoke", smoke)
    monkeypatch.setattr(three_b.jury, "rank", rank)

    def row(text):
        return {
            "status": "completed",
            "session": text,
            "candidate": candidate(text),
            "PRIVATE_SCORE": 999,
        }

    async def baseline(index):
        calls["baselines"].append(index)
        return row("baseline " + str(index))

    async def sample(arm, number, prompt, pack):
        calls["actors"].append((number, arm))
        assert (
            pack
            and "Outcome standards" in prompt
            and "Optional exploration" not in prompt
        )
        return row(f"round {number} {arm}")

    async def evaluate(task, state):
        calls["grades"].append(state["session"])
        return {"status": "scored", "PRIVATE_SCORE": 100}

    async def run(**kwargs):
        return await three_b.run_task(
            TASK,
            rounds=3,
            runtime=kwargs.pop("runtime", {"bundle": "frozen", "mode": "code"}),
            baseline=baseline,
            sample=kwargs.pop("sample", sample),
            roles=lambda *_: None,
            registry=None,
            evaluate=kwargs.pop("evaluate", evaluate),
            record=lambda n, v: records.__setitem__(n, deepcopy(v)),
            restore=lambda n: deepcopy(records.get(n)),
            **kwargs,
        )

    return run, calls, records


@pytest.mark.asyncio
async def test_fixed_baselines_current_six_and_zero_hypothesis(fixture):
    run, calls, records = fixture
    result = await run()
    assert result["completed_rounds"] == 3
    assert calls["baselines"] == [0, 1, 2]
    assert calls["actors"] == [(n, a) for n in (1, 2, 3) for a in three_b.ARMS]
    assert (
        len(calls["grades"]) == 12
    )  # Fixed baseline scores reused, native identities distinct.
    assert [len(c["products"]) for c in calls["construction"]] == [3, 6, 9]
    for n, context in enumerate(calls["selection"], 1):
        assert {p["artifact"]["final_response"] for p in context["products"]} == {
            *("baseline " + str(i) for i in range(3)),
            *(f"round {n} {a}" for a in three_b.ARMS),
        }
        assert "PRIVATE_SCORE" not in str(records[f"round_{n}/checkpoint"])
    before = deepcopy(calls)
    assert await run() == result
    assert before == calls


@pytest.mark.asyncio
async def test_retry_audit_metadata_stays_out_of_model_inputs(fixture):
    run, calls, records = fixture

    async def retried(arm, number, prompt, pack):
        assert "PRIVATE_RETRY_AUDIT" not in prompt + str(pack)
        text = f"round {number} {arm}"
        return {"status": "completed", "session": text, "candidate": candidate(text),
                "actor_retry": {"authorization": "PRIVATE_RETRY_AUDIT"}}

    await run(sample=retried)
    assert records["round_1/B1/stage"]["value"]["actor_retry"] == {
        "authorization": "PRIVATE_RETRY_AUDIT"}
    assert "PRIVATE_RETRY_AUDIT" not in str(calls["construction"])
    assert "PRIVATE_RETRY_AUDIT" not in str(calls["selection"])


@pytest.mark.asyncio
async def test_recovery_after_grading_failure_does_not_resample(fixture):
    run, calls, records = fixture

    async def broken(task, state):
        raise ConnectionError("synthetic interruption")

    with pytest.raises(ConnectionError):
        await run(evaluate=broken)
    assert len(calls["actors"]) == 3
    assert any(
        k.startswith("grades.private/") and k.endswith("/failure") for k in records
    )
    result = await run()
    assert result["completed_rounds"] == 3 and len(calls["actors"]) == 9
    assert len(calls["selection"]) == 3


@pytest.mark.asyncio
async def test_uncertain_actor_stage_is_never_redrawn(fixture):
    run, calls, records = fixture

    async def interrupted(*args):
        raise TimeoutError("lost terminal receipt")

    with pytest.raises(TimeoutError):
        await run(sample=interrupted)
    with pytest.raises(RuntimeError, match="Uncertain actor stage"):
        await run()
    assert records["round_1/B1/failure"]["error_type"] == "TimeoutError"


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["bundle", "mode", "policy", "image"])
async def test_changed_runtime_rejected_before_calls(fixture, field):
    run, calls, _ = fixture
    await run()
    before = deepcopy(calls)
    with pytest.raises(ValueError, match="runtime changed"):
        await run(runtime={"bundle": "frozen", "mode": "code", field: "changed"})
    assert calls == before


@pytest.mark.asyncio
async def test_mutated_delivery_rejected(fixture):
    run, calls, records = fixture
    await run()
    records["round_2/B2/stage"]["value"]["candidate"] = candidate("tampered")
    with pytest.raises(ValueError, match="content changed"):
        await run()


@pytest.mark.parametrize(
    "extra",
    [
        {"selection_policy": "independent_pool_v1"},
        {"public_delivery_contract": {"version": 1}},
        {"hypothesis_enabled": True},
    ],
)
def test_incompatible_combinations_fail(extra):
    with pytest.raises(ValueError, match="incompatible"):
        three_b.validate_config({"generation_protocol": three_b.PROTOCOL, **extra})


@pytest.mark.asyncio
async def test_failed_measurement_is_retryable_and_keeps_attempts(fixture, monkeypatch):
    run, calls, records = fixture
    original = three_b.measure

    async def broken(pack, *args, **kwargs):
        return {"pack_id": pack["pack_id"], "error": True, "results": []}

    monkeypatch.setattr(three_b, "measure", broken)
    for _ in range(2):
        with pytest.raises(ValueError, match="measurement did not complete"):
            await run()
    assert not calls["actors"] and len(calls["baselines"]) == 3
    key = "round_1/construction_checks/0"
    assert records[key + "/failures/1"] and records[key + "/failures/2"]
    monkeypatch.setattr(three_b, "measure", original)
    assert (await run())["completed_rounds"] == 3
    assert records[key + "/stage"]["attempt"] == 3


@pytest.mark.asyncio
async def test_nonterminal_native_grade_does_not_complete_stage(fixture):
    run, calls, records = fixture

    async def unavailable(*args):
        return {"status": "unavailable"}

    with pytest.raises(RuntimeError, match="Native grading incomplete"):
        await run(evaluate=unavailable)
    assert len(calls["actors"]) == 3
    assert (await run())["completed_rounds"] == 3
    assert len(calls["actors"]) == 9


@pytest.mark.asyncio
async def test_unavailable_selection_is_not_a_completed_checkpoint(fixture, monkeypatch):
    run, calls, records = fixture
    good_rank = three_b.jury.rank
    unavailable = {"status": "unavailable", "reason": "Synthetic member transport failure"}
    async def failed_rank(*args, **kwargs):
        kwargs["record"]("ranking", deepcopy(unavailable))
        return deepcopy(unavailable)
    monkeypatch.setattr(three_b.jury, "rank", failed_rank)
    with pytest.raises(RuntimeError, match="Six-candidate ranking unavailable"):
        await run()
    assert len(calls["actors"]) == 3
    old_actors = {arm: deepcopy(records[f"round_1/{arm}/stage"]) for arm in three_b.ARMS}
    assert records["round_1/selection/ranking"] == unavailable
    monkeypatch.setattr(three_b.jury, "rank", good_rank)
    result = await run()
    assert result["completed_rounds"] == 3
    assert len(calls["actors"]) == 9 and calls["baselines"] == [0, 1, 2]
    assert all(records[f"round_1/{arm}/stage"] == row for arm, row in old_actors.items())
    assert any(value == unavailable for key, value in records.items()
               if key.startswith("round_1/selection/unavailable/"))


@pytest.mark.asyncio
async def test_malformed_completed_ranking_cannot_trigger_another_judgment(fixture, monkeypatch):
    run, _, records = fixture
    good_rank = three_b.jury.rank
    async def failed_rank(*args, **kwargs):
        kwargs["record"]("ranking", {"status": "compared", "groups": [["invented"]]})
        raise RuntimeError("synthetic interruption")
    monkeypatch.setattr(three_b.jury, "rank", failed_rank)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        await run()
    monkeypatch.setattr(three_b.jury, "rank", good_rank)
    with pytest.raises(ValueError, match="Ranking"):
        await run()
    assert records["round_1/selection/ranking"]["groups"] == [["invented"]]
