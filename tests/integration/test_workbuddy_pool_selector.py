"""Final round-3 selection wiring, historical score binding and three-round recovery."""

from copy import deepcopy

import pytest

from experiments.workbuddy import experiment
from experiments.workbuddy.pool_selection_report import selected_score
from experiments.workbuddy.report import report
from reward_engine.adapters.workbuddy.io import read, write
from reward_engine.adapters.workbuddy.pool_selector import (
    bind_source,
    collect_candidates,
)
from reward_engine.core.outcome import jury, loop
from reward_engine.core.outcome.harness import defaults
from reward_engine.core.outcome.pool_selector import VERSION, decide, prepare


def setup_pool(root):
    for i in range(1, 4):
        write(
            root / "seeds" / str(i) / "RESULT.json",
            {
                "status": "completed",
                "candidate": {
                    "candidate_id": f"seed-{i}",
                    "artifact": {"final_response": "Seed content"},
                },
            },
        )
    for arm in "ABC":
        path = root / "round_3" / "actors" / arm
        write(
            path / "RESULT.json",
            {
                "status": "completed",
                "candidate": {
                    "candidate_id": f"round-3-{arm}",
                    "artifact": {"final_response": "Actual content"},
                },
            },
        )
        write(
            path / "OFFICIAL.private.json",
            {"status": "scored", "overall": 30 + ord(arm) - ord("A")},
        )
    return collect_candidates(root)


def selected(root, key):
    candidates, sources = collect_candidates(root)
    prepared = prepare(
        {"task_id": "t", "prompt": "Task"},
        candidates,
        runtime={},
        shuffle=lambda _: None,
    )
    winner = next(p for p, k in prepared["mapping"].items() if k == key)
    ids = [winner] + [p for p in prepared["mapping"] if p != winner]
    raw = {
        "groups": [[p] for p in ids],
        "unknown": [],
        "reason": "Best public result",
        "observations": [
            {
                "product_id": p,
                "reason": "Inspected content",
                "evidence_refs": [p + ":artifact:final_response"],
            }
            for p in ids
        ],
    }
    return bind_source(decide(raw, prepared), sources)


@pytest.mark.parametrize(
    "key,expected", [("round/3/A", 30), ("round/3/B", 31), ("round/3/C", 32)]
)
def test_report_uses_exact_historical_source_and_never_borrows_current_arm_score(
    tmp_path, key, expected
):
    root = tmp_path / "tasks/t"
    setup_pool(root)
    selection = selected(root, key)
    write(tmp_path / "config.json", {"task_ids": ["t"], "selection_policy": VERSION})
    write(root / "final_selection/selected_delivery.json", selection)
    result = report(tmp_path)
    assert result["final_deliveries"]["t"]["overall"] == expected
    assert result["valid_final_selected"] == 1
    assert result["valid_scores"] == 3
    assert "round_3_selected" not in result["means"]
    selection["candidate_id"] = "wrong-trajectory"
    with pytest.raises(ValueError, match="differs from source"):
        selected_score(root, selection)


@pytest.mark.asyncio
async def test_only_final_round_skips_old_jury_and_all_round_checkpoints_remain(
    monkeypatch,
):
    records, rankings, sampled = {}, [], []
    criterion = {
        "id": "json",
        "kind": "hard",
        "origin": "task",
        "standard": "Valid JSON",
        "measurement_question": "Parse",
        "evidence_refs": ["task"],
        "anchors": {"full": "Valid"},
    }

    async def propose(*args, **kwargs):
        return {"criteria": [criterion], "gaps": []}

    async def build(*args, **kwargs):
        return {"pack_id": "p", "criteria": [criterion], "tests": []}

    async def method(*args, **kwargs):
        return {"status": "keep", "method": ""}

    async def smoke(*args, **kwargs):
        return {"status": "callable"}

    async def rank(context, *args, **kwargs):
        rankings.append(len(context["products"]))
        return {
            "status": "compared",
            "groups": [[p["product_id"] for p in context["products"]]],
        }

    async def sample(arm, number, *args):
        sampled.append((number, arm))
        return {
            "status": "completed",
            "candidate": {
                "candidate_id": f"{number}-{arm}",
                "artifact": {"final_response": "{}"},
            },
            "final_measurement": {"pack_id": "p", "results": []},
        }

    for owner, name, value in [
        (jury, "propose", propose),
        (jury, "rank", rank),
        (loop.builder, "build", build),
        (loop.hypotheses, "propose", method),
        (loop, "smoke", smoke),
    ]:
        monkeypatch.setattr(owner, name, value)
    result = await loop.run_task(
        {"task_id": "t", "prompt": "Task"},
        [
            {"candidate_id": str(i), "artifact": {"final_response": "{}"}}
            for i in range(3)
        ],
        roles=lambda *_: None,
        registry=defaults(None),
        sample=sample,
        rounds=3,
        continuation_policy="fixed_rounds",
        defer_final_selection=True,
        record=records.__setitem__,
    )
    assert rankings == [6, 9] and len(sampled) == 9
    assert (
        result["completed_rounds"] == 3
        and result["selected_delivery"]["status"] == "pending"
    )
    assert len(records["round_3/checkpoint"]["pool"]) == 12


@pytest.mark.asyncio
async def test_completed_rounds_resume_only_final_selection(tmp_path, monkeypatch):
    from experiments.workbuddy import resume
    from reward_engine.adapters.workbuddy import pool_selector

    root = tmp_path / "run/tasks/t"
    setup_pool(root)
    write(tmp_path / "dataset/t/instruction.md", "Task")
    config = {
        "output": str(tmp_path / "run"),
        "tasks_root": str(tmp_path / "dataset"),
        "selection_policy": VERSION,
    }
    history = [
        {"selection": {"status": "pending", "arm": None}, "official": {}}
        for _ in range(3)
    ]
    monkeypatch.setattr(
        resume, "completed_rounds", lambda *_: (deepcopy(history), {"memory": []})
    )

    async def launch(config, task, output, **kwargs):
        assert "seeds" in output.parts and (output / "RESULT.json").exists()
        return read(output / "RESULT.json")

    async def forbidden(*args, **kwargs):
        pytest.fail("Completed rounds must not execute the Actor or construction loop")

    attempts = []

    async def finalize(*args):
        attempts.append(1)
        if len(attempts) == 1:
            raise ConnectionError("Resume final selection later")
        return selected(root, "round/3/C")

    monkeypatch.setattr(experiment, "launch", launch)
    monkeypatch.setattr(experiment, "run_task", forbidden)
    monkeypatch.setattr(pool_selector, "finalize", finalize)
    with pytest.raises(ConnectionError):
        await experiment.one_task(config, "t")
    assert (
        read(root / "result.private.json")["selected_delivery"]["status"] == "pending"
    )
    assert (await experiment.one_task(config, "t"))["status"] == "completed"
    assert read(root / "result.private.json")["selected_delivery"]["source_round"] == 3


def test_new_preparations_default_to_legacy_and_pool_roles_stay_optional():
    import inspect

    assert (
        inspect.signature(experiment.prepare).parameters["selection_policy"].default
        == "legacy"
    )
    assert (
        experiment.role_mapping(include_final=True)["final_selector"]
        == "deepseek-v4-flash"
    )
    assert "final_selector" not in experiment.role_mapping()


@pytest.mark.asyncio
async def test_real_adapter_freezes_model_and_restores_completed_terminal_decision(
    tmp_path, monkeypatch
):
    from reward_engine.adapters.workbuddy import pool_selector

    setup_pool(tmp_path)
    calls = []

    class Role:
        request_budget_bytes = 1_000_000

        def __init__(self, config, role, output):
            assert (
                role == "final_selector"
                and config["roles"][role] == "deepseek-v4-flash"
            )

        async def __call__(self, request):
            calls.append(request)
            ids = [p["product_id"] for p in request["context"]["products"]]
            return {
                "groups": [ids],
                "unknown": [],
                "reason": "Equivalent actual content",
                "observations": [
                    {
                        "product_id": p,
                        "reason": "Same content",
                        "evidence_refs": [p + ":artifact:final_response"],
                    }
                    for p in ids
                ],
            }

    monkeypatch.setattr(pool_selector, "Role", Role)
    config = {
        "output": str(tmp_path),
        "roles": {"final_selector": "deepseek-v4-flash"},
        "gateway_local": "unused-offline",
    }
    task = {"task_id": "t", "prompt": "Task"}
    first = await pool_selector.finalize(config, tmp_path, task)
    again = await pool_selector.finalize(config, tmp_path, task)
    assert first == again and len(calls) == 1
    assert len(read(tmp_path / "final_selection/sources.json")) == 3


def test_pending_task_does_not_publish_a_stale_final_selection(tmp_path):
    root = tmp_path / "tasks/t"
    setup_pool(root)
    write(tmp_path / "config.json", {"task_ids": ["t"], "selection_policy": VERSION})
    write(root / "final_selection/selected_delivery.json", selected(root, "round/3/B"))
    write(root / "result.private.json", {"selected_delivery": {"status": "pending"}})
    assert report(tmp_path)["valid_final_selected"] == 0
