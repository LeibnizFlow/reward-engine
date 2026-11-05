"""A failed batch resumes only failed items under the same frozen public inputs."""

from copy import deepcopy

import pytest

from reward_engine.core.outcome.harness import defaults, measure
from tests.unit.test_claw_outcome_measurement_tools import case, verdict


@pytest.mark.asyncio
async def test_partial_batch_keeps_success_and_reexecutes_only_failure():
    context, evidence, criterion, test = case()
    other = {**test, "test_id": "other"}
    pack = {"task_id": context["task"]["task_id"], "task_prompt": context["task"]["prompt"],
            "pack_id": "synthetic", "criteria": [criterion], "tests": [test, other]}
    calls = []
    broken = True

    async def model(request):
        key = request["test"]["test_id"]
        calls.append(key)
        if broken and key == "other":
            raise RuntimeError("synthetic unavailable measurement")
        return verdict()

    registry = defaults(model)
    first = await measure(pack, registry, context, evidence)
    assert any(row.get("error") for row in first["results"])
    broken = False
    second = await measure(pack, registry, context, evidence, completed=first)
    assert calls.count("check") == 1 and calls.count("other") == 2
    assert all(row["status"] == "pass" for row in second["results"])
    changed = deepcopy(evidence)
    changed["task"] = "Different input"
    with pytest.raises(ValueError, match="inputs changed"):
        await measure(pack, registry, context, changed, completed=first)
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_duplicate_historical_result_is_not_a_completed_check():
    context, evidence, criterion, test = case()
    pack = {"task_id": context["task"]["task_id"], "task_prompt": context["task"]["prompt"],
            "pack_id": "synthetic", "criteria": [criterion], "tests": [test]}
    async def model(request):
        return verdict()
    registry = defaults(model)
    first = await measure(pack, registry, context, evidence)
    first["results"].append(deepcopy(first["results"][0]))
    with pytest.raises(ValueError, match="unknown or repeated"):
        await measure(pack, registry, context, evidence, completed=first)
