"""Native trial configuration must honor the frozen provider model route."""

import pytest

from experiments.workbuddy.worker import trial_config


@pytest.mark.parametrize("model", ["qwen3.8-flash", "deepseek-v4-flash-0731", None])
def test_trial_actor_and_native_grader_use_frozen_model(tmp_path, model):
    auth = tmp_path / "campaign.auth"
    auth.write_text("local-fixture")
    config = {
        "gateway_container": "http://host.docker.internal:1234",
        "harness_mount": str(tmp_path / "harness"),
        "auth_file": str(auth),
    }
    if model is not None:
        config["roles"] = {"actor": model}
    spec = {
        "candidate_id": "wb-fixture",
        "task_path": str(tmp_path / "task"),
        "output": str(tmp_path / "output"),
        "actor_timeout_s": 60,
        "spec_path": str(tmp_path / "spec.json"),
        "grade": True,
    }
    trial = trial_config(spec, config).model_dump(mode="json")
    expected = model or "deepseek-v4-flash"
    assert trial["agent"]["model_name"] == expected
    assert trial["agent"]["kwargs"]["connection"]["model_route"] == expected
    verifier = trial["verifier"]
    assert verifier["env"]["WORKBUDDY_VERIFIER_LLM_MODEL"] == expected
    assert "/grade-wb-fixture/v1" in verifier["env"]["WORKBUDDY_VERIFIER_LLM_BASE_URL"]
    assert verifier["disable"] is False
