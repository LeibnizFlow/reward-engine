"""Opt-in Claw routing stays frozen and independent of native scoring."""
import httpx
import pytest

from experiments.claw_eval.opencode.model_settings import configure_roles
from reward_engine.adapters.workbuddy.io import digest
from reward_engine.core.outcome.three_b import PROTOCOL


def defaults():
    return dict(actor="old", jury_reducer="old", harness_designer="old",
                reward_calibration_judge="old", hypothesis_chooser="old",
                **{f"jury_member_{i}": "old" for i in range(1, 4)})


def test_derived_seats_and_frozen_identity():
    original = defaults()
    roles = configure_roles(original, PROTOCOL, ["actor=flash", "jury_member_2=judge", "jury_reducer=max"])
    assert roles["actor"] == "flash"
    assert roles["attack_member_2"] == roles["attack_reference_2"] == "judge"
    assert roles["attack_reducer"] == "max"
    assert original == defaults()
    assert digest({"roles": roles}) != digest({"roles": configure_roles(original, PROTOCOL)})
    assert digest({"roles": roles, "native_grader_model": "judge"}) != digest({"roles": roles})


@pytest.mark.parametrize("entries", [["unknown=x"], ["actor=x", "actor=y"], ["actor="], ["actor"], ["attack_member_1=x"], ["hypothesis_chooser=x"]])
def test_invalid_overrides(entries):
    with pytest.raises(ValueError):
        configure_roles(defaults(), PROTOCOL, entries)


@pytest.mark.parametrize("field", ["roles", "native_grader_model"])
def test_freeze_rejects_model_changes(tmp_path, monkeypatch, field):
    from experiments.claw_eval.opencode.campaign import verify_freeze
    from reward_engine.adapters.jiuwenswarm import config as runtime_config
    from reward_engine.adapters.workbuddy.io import write
    monkeypatch.setattr(runtime_config, "verify", lambda config: None)
    config = dict(output=str(tmp_path), roles={"actor": "flash"}, native_grader_model="judge")
    config["identity"] = digest(config)
    write(tmp_path / "manifest.json", {"config_identity": config["identity"], "files": {}})
    verify_freeze(config)
    config[field] = {"actor": "changed"} if field == "roles" else "changed"
    with pytest.raises(ValueError, match="identity changed"):
        verify_freeze(config)


@pytest.mark.asyncio
async def test_grader_binding_at_gateway(tmp_path):
    from reward_engine.adapters.workbuddy.gateway import application
    auth = tmp_path / "auth"
    auth.write_text("synthetic")
    config = dict(output=str(tmp_path), auth_file=str(auth), identity="synthetic",
                  backend_url="http://synthetic/v1", model_concurrency=1,
                  roles={"actor": "flash"}, native_grader_model="judge")
    sent = []
    async def respond(request):
        sent.append(request)
        return httpx.Response(200, json={"model": "judge", "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]})
    app = application(config, backend_key="synthetic", client_factory=lambda **kw: httpx.AsyncClient(transport=httpx.MockTransport(respond), **kw))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test", headers={"Authorization": "Bearer synthetic"}) as client:
        for channel, model, expected in [("grade", "flash", 400), ("grade-other", "judge", 400), ("grade", "judge", 200)]:
            result = await client.post(f"/{channel}/v1/chat/completions", json={"model": model, "messages": []})
            assert result.status_code == expected, result.text
    assert len(sent) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("override", [None, "judge"])
async def test_native_grade_uses_independent_model(tmp_path, monkeypatch, override):
    from experiments.claw_eval.opencode import campaign
    from reward_engine.adapters.claw_eval import native_grader
    captured = []
    def evaluator(config, output):
        captured.append(config)
        async def evaluate(task, row):
            return {"status": "graded", "task_score": 1}
        return evaluate
    monkeypatch.setattr(native_grader, "evaluator", evaluator)
    monkeypatch.setattr(campaign, "file_hash", lambda path: "synthetic-hash")
    monkeypatch.setattr(campaign, "auth", lambda config: "synthetic")
    config = dict(identity="synthetic", benchmark_root=str(tmp_path), python="python",
                  gateway_local="http://synthetic", roles={"actor": "flash"}, model_aliases={})
    if override:
        config["native_grader_model"] = override
    before = digest(config)
    await campaign.grade(config, {"task_id": "synthetic"}, {"native_grader_trial": str(tmp_path)}, tmp_path)
    assert captured[0]["native_grader"]["judge"]["model"] == (override or "flash")
    assert digest(config) == before
