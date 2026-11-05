"""Native trial configuration must honor the frozen provider model route."""

import pytest

from experiments.workbuddy.worker import trial_config


@pytest.mark.parametrize("model", ["test-actor", "test-reasoner-revision", "test-model-b", None])
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


@pytest.mark.asyncio
async def test_grade_only_preserves_credential_through_trial_preparation(tmp_path, monkeypatch):
    from harbor.trial.trial import Trial

    from experiments.workbuddy import worker
    from reward_engine.adapters.jiuwenswarm import checkpoints

    auth = tmp_path / "campaign.auth"
    auth.write_text("synthetic-grader-credential")
    config = {"gateway_container": "http://offline", "harness_mount": str(tmp_path),
              "auth_file": str(auth), "identity": "synthetic-config"}
    spec = {"candidate_id": "wb-synthetic", "task_path": str(tmp_path / "task"),
            "output": str(tmp_path), "actor_timeout_s": 60, "grade": False,
            "campaign_config": "synthetic-config", "spec_identity": "synthetic-spec",
            "task": {"task_id": "synthetic"}}
    actor = {"candidate": {"id": "delivery"}, "artifact_bindings": []}
    def read(path):
        if str(path) == "synthetic-config":
            return config
        return actor if str(path).endswith("RESULT.json") else spec.copy()
    monkeypatch.setattr(worker, "read", read)
    monkeypatch.setattr(checkpoints, "verify_actor", lambda row: None)
    class Prepared(Exception):
        pass
    async def create(value):
        assert value.verifier.env["WORKBUDDY_VERIFIER_LLM_API_KEY"] == auth.read_text()
        assert value.agent.import_path.endswith("FrozenDeliveryAgent")
        assert value.agent.kwargs == {"actor_result": str(tmp_path / "RESULT.json")}
        assert value.environment.mounts == []
        raise Prepared()
    monkeypatch.setattr(Trial, "create", create)
    with pytest.raises(Prepared):
        await worker.run(tmp_path / "spec.json", grade_only=True)
