"""Evidence-aware preflight cannot turn actual execution errors into rejections."""

import pytest

from reward_engine.adapters.workbuddy.measurement import WorkBuddyTools
from reward_engine.core.outcome.authored_json import RECOVERY, VERSION
from reward_engine.core.outcome.context import build_context
from reward_engine.core.outcome.harness import defaults, unresolved_measurements


@pytest.mark.asyncio
async def test_absent_inventory_is_rejected_before_execution_and_not_charged():
    class Tools(WorkBuddyTools):
        async def run(self, request, evidence):
            pytest.fail("Unavailable workspace inspection must not execute")

    async def model(request):
        assert request["measurement_tools"]["workspace.inspect"]["available"] is False
        assert request["measurement_budget"]["remaining_tool_calls"] == 4
        return {"status": "pass", "reason": "The requested reply is present", "gap": "",
                "evidence_refs": ["product_1:artifact:final_response"],
                "claims": [{"claim": "The reply says done", "claim_refs": ["product_1:artifact:final_response"],
                            "evidence_refs": ["product_1:artifact:final_response"],
                            "observation": "The supplied reply is done", "judgment": "supported"}]}

    model.role_output_protocol, model.role_recovery_policy = VERSION, RECOVERY
    context, evidence = build_context({"task_id": "synthetic", "prompt": "Reply done"},
                                      [{"artifact": {"final_response": "done"}}])
    test = {"test_id": "reply", "criterion_id": "c1", "checker": "evidence.review",
            "execution_recipe": {"probes": [{"tool": "workspace.inspect", "arguments": {"paths": ["*"]}}]}}
    result = await defaults(model, tools=Tools()).run(test, context, evidence)
    assert result["status"] == "pass"
    assert result["execution_usage"]["tool_calls"] == 0
    assert all(r["request_rejected"] for r in result["tool_observations"].values())
    assert not result["unresolved_measurement_refs"]


@pytest.mark.asyncio
async def test_supplied_corrupt_inventory_is_still_a_real_failure():
    tools = WorkBuddyTools()
    evidence = {"product_1:source:0": {"source": "synthetic", "offset": 0, "total_chars": 1, "text": "!"}}
    assert tools.catalog_for(evidence)["workspace.inspect"]["available"] is True
    receipt = await tools.run_checked({"tool": "workspace.inspect"}, evidence)
    assert receipt["status"] == "unknown" and not receipt.get("request_rejected")
    assert unresolved_measurements({}, {"failed": receipt}) == {"failed"}


@pytest.mark.asyncio
async def test_supplied_empty_inventory_is_observed_and_legacy_absence_is_unchanged():
    tools = WorkBuddyTools()
    evidence = {"product_1:source:0": {"source": "synthetic", "offset": 0, "total_chars": 2, "text": "[]"}}
    receipt = await tools.run_checked({"tool": "workspace.inspect"}, evidence)
    assert receipt["status"] == "observed"
    legacy = await tools.run({"tool": "workspace.inspect"}, {})
    assert legacy["status"] == "unknown" and not legacy.get("request_rejected")
