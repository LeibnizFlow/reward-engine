"""Explicit opt-in retry of a terminated first Actor attempt; no implicit resampling."""

import json
from types import SimpleNamespace

import pytest

from reward_engine.adapters.claw_eval.actor_retry import (
    ActorRetries,
    propose_actor_retry,
    verify_actor_retry,
)
from reward_engine.adapters.workbuddy.io import file_hash, read, write
from reward_engine.core.outcome.construction import fingerprint
from reward_engine.core.outcome.three_b import Stages, public_state
from tests.support.actor_retry import (
    approved,
    failed_actor,
    file_inventory,
    native_timed_out_actor,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("preflight_failure", [False, True])
@pytest.mark.parametrize("native_timeout", [False, True])
async def test_explicit_retry_preserves_failure_and_cached_success(
        tmp_path, preflight_failure, native_timeout):
    factory = native_timed_out_actor if native_timeout else failed_actor
    config, root, old_slot, old_stage = factory(tmp_path)
    old_bytes = file_inventory(old_slot)
    proposal = propose_actor_retry(config, "synthetic_task", "round_1/B1")
    retries = ActorRetries(config, [approved(proposal)])

    def restore(key):
        p = root / (key + ".json")
        return read(p) if p.exists() else None

    stages = Stages({}, lambda k, v: write(root / (k + ".json"), v), restore,
                    recover_actor=lambda name, saved: retries.recover("synthetic_task", name, saved))
    if preflight_failure:
        async def reject_before_start():
            retries.validate_slot(retries.slot("synthetic_task", "round_1/B1", old_slot), "changed")
        with pytest.raises(ValueError, match="input identity"):
            await stages.run("round_1/B1", {}, reject_before_start, actor=True)
        failure_hash = file_hash(root / "round_1/B1/failures/2.json")
        retries = ActorRetries(config, [approved(proposal)])
    calls = []

    async def execute():
        slot = retries.slot("synthetic_task", "round_1/B1", old_slot)
        assert slot != old_slot and not slot.exists()
        retries.validate_slot(slot, "actor")
        calls.append(slot)
        write(slot / "RESULT.json", {"status": "completed"})
        return {"status": "completed", "runtime_directory": str(slot),
                "identity": "actor", "actor_execution_s": 8,
                **retries.result_fields(slot, 8)}

    result = await stages.run("round_1/B1", {}, execute, actor=True)
    assert read(root / "round_1/B1/stage.json")["attempt"] == (3 if preflight_failure else 2)
    if preflight_failure:
        assert file_hash(root / "round_1/B1/failures/2.json") == failure_hash
    assert file_inventory(old_slot) == old_bytes
    assert read(root / "round_1/B1/failures/1.json")["error_type"] == "TimeoutError"
    assert result["actor_retry"]["total_execution_s"] == (44.4 if native_timeout else 33.1)
    assert result["actor_retry"]["additional_attempts"] == 1
    verify_actor_retry(result)
    assert await stages.run("round_1/B1", {}, execute, actor=True) == result
    assert len(calls) == 1
    altered = {**result, "actor_retry": {**result["actor_retry"], "total_execution_s": 8}}
    with pytest.raises(ValueError, match="cost"):
        verify_actor_retry(altered)
    (old_slot / "jiuwenswarm/process/turn-1.stderr").write_text("changed old failure")
    with pytest.raises(ValueError, match="evidence"):
        verify_actor_retry(result)


@pytest.mark.parametrize('change', [
    'completed', 'failed', 'cancelled', 'returncode', 'error_code', 'sequence', 'request',
    'host_budget', 'requested_budget', 'early_timeout', 'no_elapsed', 'no_session',
    'cleanup', 'terminal_result_file', 'delivered', 'session_archive',
])
def test_native_timeout_retry_rejects_unproven_or_completed_execution(tmp_path, change):
    config, _, slot, _ = native_timed_out_actor(tmp_path)
    process = slot / 'jiuwenswarm/process'
    rows = [json.loads(line) for line in (process / 'turn-1.jsonl').read_text().splitlines()]
    terminal = rows[-1]
    if change in ('completed', 'failed', 'cancelled'):
        terminal['status'] = change
        if change == 'completed':
            terminal.update(exit_code=0, error=None)
    elif change in ('returncode', 'sequence', 'request', 'error_code', 'no_session'):
        if change == 'returncode':
            terminal['exit_code'] = 137
        elif change == 'error_code':
            terminal['error']['code'] = 'RUNTIME_ERROR'
        elif change == 'no_session':
            terminal.pop('session_id')
        else:
            terminal['sequence' if change == 'sequence' else 'request_id'] = 999
    elif change in ('host_budget', 'requested_budget', 'early_timeout', 'no_elapsed', 'cleanup'):
        path = process / 'turn-1.execution.json'
        execution = read(path)
        if change == 'host_budget':
            execution['host_timeout_s'] += 1
        elif change == 'requested_budget':
            execution['requested_timeout_s'] += 1
        elif change == 'early_timeout':
            execution['timeout_elapsed_s'] = 20
        elif change == 'no_elapsed':
            execution.pop('timeout_elapsed_s')
        else:
            execution['cleanup_error'] = 'Synthetic unproven container shutdown'
        write(path, execution)
    elif change == 'terminal_result_file':
        write(process / 'turn-1.result.json', terminal)
    elif change == 'delivered':
        write(slot / 'RESULT.json', {'status': 'completed'})
    else:
        write(slot / 'sessions/1/state.json', {})
    (process / 'turn-1.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(ValueError):
        propose_actor_retry(config, 'synthetic_task', 'round_1/B1')


def test_proposal_is_read_only_and_requires_explicit_approval(tmp_path):
    config, root, slot, stage = failed_actor(tmp_path)
    before = file_inventory(root)
    proposal = propose_actor_retry(config, "synthetic_task", "round_1/B1")
    assert file_inventory(root) == before
    assert not ActorRetries(config, []).recover("synthetic_task", "round_1/B1", stage)
    with pytest.raises(ValueError, match="approval"):
        ActorRetries(config, [{"proposal": proposal, "approval": None}])
    envelope = approved(proposal)
    envelope["proposal"] = {**proposal, "actor_timeout_s": 11}
    with pytest.raises(ValueError):
        ActorRetries(config, [envelope])


@pytest.mark.parametrize("change", ["bytes", "identity", "cleanup", "terminal", "delivery", "session"])
def test_changed_or_ambiguous_failure_is_rejected(tmp_path, change):
    config, root, slot, stage = failed_actor(tmp_path)
    proposal = propose_actor_retry(config, "synthetic_task", "round_1/B1")
    if change == "bytes":
        (slot / "jiuwenswarm/process/turn-1.stderr").write_text("changed")
    elif change == "identity":
        write(slot / "STARTED.json", {"identity": "changed"})
    elif change == "cleanup":
        p = slot / "jiuwenswarm/process/turn-1.execution.json"
        write(p, {**read(p), "cleanup_error": "container still alive"})
    elif change == "terminal":
        p = slot / "jiuwenswarm/process/turn-1.jsonl"
        with p.open("a") as stream:
            stream.write(json.dumps({"type": "result", "status": "completed"}) + "\n")
    elif change == "delivery":
        write(slot / "snapshots/1/candidate.json", {"final_response": "already delivered"})
    else:
        p = slot / "FAILURE.json"
        write(p, {**read(p), "session": "prior-completed-turn"})
    with pytest.raises(ValueError):
        ActorRetries(config, [approved(proposal)]).recover("synthetic_task", "round_1/B1", stage)


def test_authorization_cannot_be_consumed_twice_or_change_actor_identity(tmp_path):
    config, root, slot, stage = failed_actor(tmp_path)
    envelope = approved(propose_actor_retry(config, "synthetic_task", "round_1/B1"))
    first = ActorRetries(config, [envelope])
    assert first.recover("synthetic_task", "round_1/B1", stage)
    with pytest.raises(ValueError, match="consumed"):
        ActorRetries(config, [envelope]).recover("synthetic_task", "round_1/B1", stage)
    with pytest.raises(ValueError, match="identity"):
        first.validate_slot(first.slot("synthetic_task", "round_1/B1", slot), "different")


@pytest.mark.asyncio
async def test_failed_additional_attempt_cannot_be_retried_again(tmp_path):
    config, root, old_slot, stage = failed_actor(tmp_path)
    envelope = approved(propose_actor_retry(config, "synthetic_task", "round_1/B1"))
    retries = ActorRetries(config, [envelope])
    def restore(key):
        path = root / (key + ".json")
        return read(path) if path.exists() else None
    stages = Stages({}, lambda k, v: write(root / (k + ".json"), v), restore,
                    recover_actor=lambda name, saved: retries.recover("synthetic_task", name, saved))
    calls = []
    async def fail():
        calls.append(retries.slot("synthetic_task", "round_1/B1", old_slot))
        raise TimeoutError("second and final attempt timed out")
    with pytest.raises(TimeoutError):
        await stages.run("round_1/B1", {}, fail, actor=True)
    with pytest.raises(ValueError, match="consumed"):
        await stages.run("round_1/B1", {}, fail, actor=True)
    assert len(calls) == 1
    assert read(root / "round_1/B1/failures/1.json")["reason"] == ""
    assert read(root / "round_1/B1/failures/2.json")["reason"] == "second and final attempt timed out"


@pytest.mark.asyncio
async def test_campaign_routes_retry_and_review_to_new_slot(tmp_path, monkeypatch):
    from experiments.claw_eval.opencode import campaign
    from reward_engine.core.outcome import three_b

    config, root, original, saved = failed_actor(tmp_path)
    config.update(rounds=1, batch={}, image="synthetic-image")
    manager = ActorRetries(config, [approved(propose_actor_retry(config, "synthetic_task", "round_1/B1"))])
    before = file_inventory(original)
    observed = []
    async def sample(task_id, slot, *, guidance, review):
        assert task_id == "synthetic_task" and slot != original
        assert review["source_prefix"] == str(slot / "checks")
        manager.validate_slot(slot, "actor")
        observed.append(slot)
        return {"session": "synthetic-new-session"}
    runtime = SimpleNamespace(actor_retries=manager, sample=sample)
    monkeypatch.setattr(campaign, "public_task", lambda *args: ({}, {"task_id": "synthetic_task"}))
    monkeypatch.setattr(campaign, "registry", lambda *args: None)
    monkeypatch.setattr(campaign, "DeliveryReview", lambda *args, **kwargs: kwargs)
    async def drive(task, **kwargs):
        stages = Stages({}, kwargs["record"], kwargs["restore"],
                        recover_actor=kwargs["recover_actor"])
        await stages.run("round_1/B1", {}, lambda: kwargs["sample"]("B1", 1, "original guidance", {}), actor=True)
        return {"status": "completed", "completed_rounds": 1}
    monkeypatch.setattr(three_b, "run_task", drive)
    await campaign.three_b_task(config, runtime, "synthetic_task")
    assert len(observed) == 1 and file_inventory(original) == before


@pytest.mark.parametrize("change", [None, "drop_proof", "cost", "identity", "receipt_link"])
def test_completed_retry_lineage_survives_checkpoint_projection(tmp_path, change):
    from reward_engine.adapters.jiuwenswarm.checkpoints import verify_actor

    config, root, old_slot, stage = failed_actor(tmp_path)
    manager = ActorRetries(config, [approved(propose_actor_retry(config, "synthetic_task", "round_1/B1"))])
    manager.recover("synthetic_task", "round_1/B1", stage)
    slot = manager.slot("synthetic_task", "round_1/B1", old_slot)
    candidate = {"artifact": {"final_response": "synthetic retry delivery"}}
    write(slot / "snapshots/1/candidate.json", candidate)
    write(slot / "snapshots/1/inventory.json", [])
    raw = {"status": "completed", "candidate": candidate, "session": "new-session",
           "runtime_directory": str(slot), "identity": "actor", "actor_execution_s": 8,
           "resources": {"turns": 1}, **manager.result_fields(slot, 8)}
    write(slot / "RESULT.json", raw)
    state = public_state(raw)
    assert state.get("actor_retry") == raw["actor_retry"]
    verify_actor(state)
    if change is None:
        from reward_engine.adapters.jiuwenswarm.report import report

        write(root / "round_1/B1/stage.json", {"status": "completed", "value": state,
                                             "value_sha256": fingerprint(state)})
        summary = report({**config, "rounds": 1}, tmp_path / "report")
        assert summary["tasks"][0]["actor_retries"]["round_1/B1"]["total_execution_s"] == 33.1
        return
    if change == "drop_proof":
        state.pop("actor_retry")
    elif change in {"cost", "identity"}:
        write(slot / "RESULT.json", {**raw, **({"actor_execution_s": 0} if change == "cost" else {"identity": "different"})})
    else:
        receipt = slot.parent / "authorization.json"
        copied = slot.parent / "copied.json"
        copied.write_bytes(receipt.read_bytes())
        receipt.unlink()
        receipt.symlink_to(copied)
    with pytest.raises(ValueError):
        verify_actor(state)


@pytest.mark.parametrize("arguments", [
    ["run", "--actor-retry-authorization", "/unused/approval.json"],
    ["run", "--only-task", "synthetic_task", "--actor-retry-authorization", "relative.json"],
    ["report", "--only-task", "synthetic_task", "--actor-retry-authorization", "/unused/approval.json"],
    ["run", "--probe-only", "--only-task", "synthetic_task", "--actor-retry-authorization", "/unused/approval.json"],
])
def test_retry_cli_rejects_ambiguous_scope_before_loading_experiment(tmp_path, monkeypatch, arguments):
    from experiments.claw_eval.opencode import campaign

    monkeypatch.setattr("sys.argv", ["campaign", *arguments, "--output", str(tmp_path / "absent")])
    with pytest.raises(SystemExit) as error:
        campaign.main()
    assert error.value.code == 2
    assert not (tmp_path / "absent").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit_protocol", [True, False])
@pytest.mark.parametrize("preflight_resume", [True, False])
async def test_runtime_checks_retry_input_before_starting_actor(
    tmp_path, monkeypatch, explicit_protocol, preflight_resume,
):
    from experiments.claw_eval.opencode import runtime as module
    from reward_engine.adapters.workbuddy.io import digest

    config, root, original, stage = failed_actor(tmp_path)
    if explicit_protocol:
        config["actor_protocol_identity"] = "frozen-protocol"
    identity = digest({"task": "synthetic_task", "guidance": "same input", "review": False,
                       "protocol": "frozen-protocol" if explicit_protocol else stage["implementation"]})
    for name in ("STARTED.json", "FAILURE.json"):
        path = original / name
        write(path, {**read(path), "identity": identity})
    envelope = approved(propose_actor_retry(config, "synthetic_task", "round_1/B1"))
    manager = ActorRetries(config, [envelope])
    manager.recover("synthetic_task", "round_1/B1", stage)
    preserved = file_inventory(root)
    if preflight_resume:
        stage = {**stage, "attempt": 2, "implementation": config["identity"]}
        failure = {**stage, "error_type": "ValueError", "reason": "Actor retry input identity changed"}
        write(root / "round_1/B1/stage.json", stage)
        write(root / "round_1/B1/failure.json", failure)
        write(root / "round_1/B1/failures/2.json", failure)
        preserved = file_inventory(root)
        manager = ActorRetries(config, [envelope])
        assert manager.recover("synthetic_task", "round_1/B1", stage)
        assert file_inventory(root) == preserved
    slot = manager.slot("synthetic_task", "round_1/B1", original)
    runtime = object.__new__(module.Runtime)
    runtime.config, runtime.actor_retries = config, manager
    calls = []
    runtime.check_resources = lambda: calls.append("resources")
    with pytest.raises(ValueError, match="identity"):
        await runtime.sample("synthetic_task", slot, guidance="changed input")
    assert calls == [] and not slot.exists()

    class SyntheticActor:
        session, execution_s, turn, events, usage = "new-session", 8, 1, [], {}

        def __init__(self, *args):
            calls.append("actor")

        async def setup(self):
            pass

        async def finalize(self):
            pass

        async def close(self):
            pass

    async def deliver(*args, **kwargs):
        return {"artifact": {"final_response": "synthetic"}}

    monkeypatch.setattr(module, "Actor", SyntheticActor)
    monkeypatch.setattr(module, "run_delivery", deliver)
    row = await runtime.sample("synthetic_task", slot, guidance="same input")
    assert row["actor_retry"]["total_execution_s"] == 33.1
    assert read(slot / "RESULT.json") == row
    assert await runtime.sample("synthetic_task", slot, guidance="same input") == row
    assert calls == ["resources", "actor"]
    assert all(file_hash(root / name) == value for name, value in preserved.items())


@pytest.mark.parametrize("change", [
    "started", "failure", "history", "original_history", "archive", "linked_claim", "later_attempt",
])
def test_preflight_resume_rejects_changed_or_started_retry(tmp_path, change):
    config, root, original, stage = failed_actor(tmp_path)
    envelope = approved(propose_actor_retry(config, "synthetic_task", "round_1/B1"))
    manager = ActorRetries(config, [envelope])
    manager.recover("synthetic_task", "round_1/B1", stage)
    slot = manager.slot("synthetic_task", "round_1/B1", original)
    stage = {**stage, "attempt": 2, "implementation": config["identity"]}
    failure = {**stage, "error_type": "ValueError", "reason": "Actor retry input identity changed"}
    write(root / "round_1/B1/stage.json", stage)
    write(root / "round_1/B1/failure.json", failure)
    write(root / "round_1/B1/failures/2.json", failure)
    if change == "started":
        slot.mkdir()
    elif change == "failure":
        write(root / "round_1/B1/failure.json", {**failure, "reason": "something else"})
    elif change == "history":
        write(root / "round_1/B1/failures/2.json", {**failure, "reason": "changed"})
    elif change == "original_history":
        write(root / "round_1/B1/failures/1.json", failure)
    elif change == "archive":
        write(slot.parent / "previous_stage.json", stage)
    elif change == "later_attempt":
        stage = {**stage, "attempt": 3}
        write(root / "round_1/B1/stage.json", stage)
    else:
        claim = slot.parent
        moved = claim.with_name("moved")
        claim.rename(moved)
        claim.symlink_to(moved, target_is_directory=True)
    with pytest.raises(ValueError):
        ActorRetries(config, [envelope]).recover("synthetic_task", "round_1/B1", stage)
