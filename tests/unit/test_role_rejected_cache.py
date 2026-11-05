"""A terminally rejected correction is evidence, not a reusable valid response."""

import asyncio
import json
from copy import deepcopy

import httpx
import pytest

from reward_engine.adapters.workbuddy.io import read
from reward_engine.core.outcome.context import ResponseContractError, ask
from tests.support.role_transport import transport as transport


def test_canonical_resume_replaces_stale_repair_target_without_touching_history(tmp_path):
    from reward_engine.adapters.workbuddy.io import digest, write
    from reward_engine.adapters.workbuddy.model_cache import resume_request

    request = {"mode": "outcome_expectations", "role_output_protocol": "model_authored_json_v2",
               "role_input_projection": "canonical_sources_v1", "response": {"criteria": []},
               "evidence_budget": {"remaining_reads": 0},
               "source_catalog": {}, "source_requirements": {},
               "independent_expectations": [], "proposal_expectations": [],
               "response_recovery": {"rejected_response": {"criteria": [None] * 23},
                                     "repair_worklist": {"criteria_count": 23}}}
    identity_inputs = {"request": request, "model": "synthetic"}
    identity = digest(identity_inputs)
    latest = {"criteria": [None] * 21}
    source = tmp_path / "original"
    write(source / "request.json", request)
    write(source / "response.json", {"answer": latest})
    rejection = {"cache_identity": identity, "answer_sha256": digest(latest),
                 "error_type": "ResponseContractError", "reason": "Invalid draft",
                 "details": {"issues": [{"code": "CRITERIA_COUNT", "field": "criteria",
                                         "actual": 21, "reason": "Require 1–20 requirements"}]}}
    write(source / "contract_rejected.json", rejection)
    write(tmp_path / "completed" / (identity + ".json"), {"answer": latest, "source": str(source)})
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    resumed, cache, answer = resume_request(identity_inputs, tmp_path, track_rejections=True)
    assert answer is None and not cache.exists()
    assert "response_recovery" not in resumed
    feedback = resumed["resume_correction"]
    assert feedback["previous_response"] == latest
    assert feedback["repair_worklist"]["criteria_count"] == 21
    assert feedback["repair_worklist"]["criteria_budget"]["minimum_net_reduction"] == 1
    assert "COMPLETE compact JSON" in feedback["instruction"]
    assert "atomic edits" not in feedback["instruction"]
    assert resumed["evidence_budget"] == request["evidence_budget"]
    assert cache.stem == digest({**identity_inputs, "request": resumed})
    assert all(p.read_bytes() == raw for p, raw in before.items())


def test_canonical_resume_follows_legacy_then_modern_rejections(tmp_path, monkeypatch):
    from reward_engine.adapters.workbuddy import model_cache
    from reward_engine.adapters.workbuddy.io import digest, write

    original = {"mode": "outcome_expectations", "role_output_protocol": "model_authored_json_v2",
                "role_input_projection": "canonical_sources_v1", "source_requirements": {},
                "source_catalog": {}, "response_recovery": {"rejected_response": {"criteria": []}}}
    identity_inputs = {"request": original, "model": "synthetic"}

    def cache_reply(request, count, *, rejected=True):
        identity = digest({**identity_inputs, "request": request})
        source = tmp_path / identity
        answer = {"criteria": [None] * count}
        write(source / "request.json", request)
        write(source / "response.json", {"answer": answer})
        write(tmp_path / "completed" / (identity + ".json"), {"answer": answer, "source": str(source)})
        if rejected:
            write(source / "contract_rejected.json", {
                "cache_identity": identity, "answer_sha256": digest(answer),
                "reason": "Synthetic rejection", "details": {}, "error_type": "ResponseContractError"})
        return identity

    cache_reply(original, 23)
    # Construct the previously recorded format without the new augmentation.
    with monkeypatch.context() as patch:
        patch.setattr(model_cache, "refresh_construction_feedback", lambda _: None)
        legacy, _, _ = model_cache.resume_request(identity_inputs, tmp_path, track_rejections=True)
    legacy_identity = cache_reply(legacy, 21)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    modern, _, _ = model_cache.resume_request(identity_inputs, tmp_path, track_rejections=True)
    assert modern["resume_correction"]["previous_request_identity"] == legacy_identity
    assert modern["resume_correction"]["repair_worklist"]["criteria_count"] == 21
    modern_identity = cache_reply(modern, 20)
    latest, _, _ = model_cache.resume_request(identity_inputs, tmp_path, track_rejections=True)
    assert latest["resume_correction"]["previous_request_identity"] == modern_identity
    assert latest["resume_correction"]["repair_worklist"]["criteria_count"] == 20
    cache_reply(latest, 19, rejected=False)
    restored, _, accepted = model_cache.resume_request(identity_inputs, tmp_path, track_rejections=True)
    assert restored == latest and accepted["answer"] == {"criteria": [None] * 19}
    assert all(p.read_bytes() == raw for p, raw in before.items())


@pytest.mark.asyncio
async def test_resume_feedback_keeps_identities_and_does_not_grow_nested_history(transport):
    role, responses, calls = transport
    role.config["role_stream"] = False
    role.bounded_stage = True
    responses.extend([reply({"value": "bad"}) for _ in range(4)] + [reply({"value": "good"})])
    def validate(row):
        if row["value"] != "good":
            raise ResponseContractError("value", "Use the required value")
        return row
    args = (role, "outcome_builder", {}, {}, "Synthetic", {"value": "required"})
    with pytest.raises(ResponseContractError):
        await ask(*args, validate=validate)
    original_cache = {p: p.read_bytes() for p in (role.output / "completed").glob('*.json')}
    for _ in range(2):
        with pytest.raises(ResponseContractError):
            await ask(*args, validate=validate)
    assert await ask(*args, validate=validate) == {"value": "good"}
    assert await ask(*args, validate=validate) == {"value": "good"}
    assert len(calls) == 5
    feedback = [json.loads(body["messages"][1]["content"])["resume_correction"] for body in calls[2:]]
    assert len({row["previous_request_identity"] for row in feedback}) == 3
    assert all(row["previous_response"] == {"value": "bad"} for row in feedback)
    assert all("resume_correction" not in row for row in feedback)
    assert all(p.read_bytes() == raw for p, raw in original_cache.items())


def reply(answer):
    return httpx.Response(200, json={"model": "test-model-a", "choices": [
        {"index": 0, "finish_reason": "stop", "message": {"content": json.dumps(answer)}}]})


@pytest.mark.asyncio
async def test_explicit_resume_refreshes_only_terminal_rejected_correction(transport):
    role, responses, calls = transport
    role.config["role_stream"] = False
    role.bounded_stage = True
    responses.extend([reply({"value": "invalid draft"}), reply({"value": "invalid correction"}),
                      reply({"value": "valid"})])
    def validate(row):
        if row["value"] != "valid":
            raise ResponseContractError("value", "Synthetic exact contract")
        return row
    args = (role, "outcome_builder", {}, {}, "Synthetic", {"value": "required"})
    with pytest.raises(ResponseContractError):
        await ask(*args, validate=validate)
    assert len(calls) == 2  # No new retry allowance within the failed stage.
    receipts = {p: p.read_bytes() for p in role.output.glob("*/response.json")}
    assert len(receipts) == 2
    assert await ask(*args, validate=validate) == {"value": "valid"}
    assert len(calls) == 3  # Initial draft reused; only invalid correction refreshed.
    prior = json.loads(calls[1]["messages"][1]["content"])
    resumed = json.loads(calls[2]["messages"][1]["content"])
    feedback = resumed.pop("resume_correction")
    assert resumed == prior
    assert feedback["previous_response"] == {"value": "invalid correction"}
    assert feedback["previous_rejection"]["reason"] == "Synthetic exact contract"
    assert feedback["previous_request_identity"]
    assert await ask(*args, validate=validate) == {"value": "valid"}
    assert len(calls) == 3
    assert all(p.read_bytes() == data for p, data in receipts.items())
    rejected = list(role.output.glob("*/contract_rejected.json"))
    assert len(rejected) == 1
    assert read(rejected[0])["reason"] == "Synthetic exact contract"


@pytest.mark.asyncio
async def test_repair_wrapper_rejection_marks_the_actual_edit_reply(transport):
    role, responses, calls = transport
    role.config["role_stream"] = False
    role.bounded_stage = True
    role.protocol_edits = True
    base = {"criteria": [{"id": "synthetic", "kind": "invalid"}]}
    from reward_engine.core.outcome.construction import fingerprint

    invalid = {"base_sha256": fingerprint(base), "edits": [
        {"op": "replace", "path": ["criteria", 0, "kind"], "value": "invalid again", "reason": "Synthetic"}]}
    valid = deepcopy(invalid)
    valid["edits"][0]["value"] = "hard"
    responses.extend([reply(base), reply(invalid), reply(valid)])
    def validate(row):
        if row["criteria"][0]["kind"] != "hard":
            raise ResponseContractError("criteria", "Invalid strength")
        return row
    args = (role, "outcome_jury_map", {}, {}, "Synthetic", {"criteria": []})
    with pytest.raises(ResponseContractError):
        await ask(*args, validate=validate)
    assert await ask(*args, validate=validate) == {"criteria": [{"id": "synthetic", "kind": "hard"}]}
    assert len(calls) == 3
    rejected = next(role.output.glob("*/contract_rejected.json"))
    assert read(rejected.parent / "response.json")["answer"] == invalid


@pytest.mark.asyncio
async def test_concurrent_calls_do_not_reject_another_calls_success(transport):
    from reward_engine.core.outcome.response_receipts import ACTIVE_RESPONSE

    role, responses, calls = transport
    role.config["role_stream"] = False
    role.bounded_stage = True
    responses.extend([reply({"value": "valid"}), reply({"value": "invalid"})])
    async def wrapped(request):
        answer = await role(request)
        await asyncio.sleep(0.01)
        return answer
    def validate(row):
        if row["value"] != "valid":
            raise ResponseContractError("value", "Invalid")
        return row
    args = ("outcome_builder", {}, {}, "Synthetic", {"value": "required"})
    results = await asyncio.gather(
        ask(wrapped, *args, extra={"case": "good"}, validate=validate),
        ask(wrapped, *args, extra={"case": "bad"}, validate=validate),
        return_exceptions=True,
    )
    assert results[0] == {"value": "valid"} and isinstance(results[1], ResponseContractError)
    assert ACTIVE_RESPONSE.get() is None
    assert await ask(wrapped, *args, extra={"case": "good"}, validate=validate) == {"value": "valid"}
    assert len(calls) == 2
    marker = next(role.output.glob("*/contract_rejected.json"))
    assert read(marker.parent / "response.json")["answer"] == {"value": "invalid"}


@pytest.mark.asyncio
async def test_rejected_marker_must_match_exact_cached_answer(transport):
    role, responses, calls = transport
    role.config["role_stream"] = False
    role.bounded_stage = True
    responses.extend([reply({"value": "invalid"}), reply({"value": "invalid correction"})])
    def reject(row):
        raise ResponseContractError("value", "Invalid")
    with pytest.raises(ResponseContractError):
        await ask(role, "outcome_builder", {}, {}, "Synthetic", {"value": "required"}, validate=reject)
    marker = next(role.output.glob("*/contract_rejected.json"))
    data = read(marker)
    data["answer_sha256"] = "another answer"
    marker.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="identity differs"):
        await role(read(marker.parent / "request.json"))
    assert len(calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("accepted_draft", [True, False])
async def test_jury_marks_rejection_only_after_deciding_draft_usability(transport, monkeypatch, accepted_draft):
    from reward_engine.adapters.workbuddy.models import Role
    from reward_engine.core.outcome import jury
    from reward_engine.core.outcome.context import build_context
    from tests.unit.test_outcome_jury_revision import criterion

    original, responses, calls = transport
    config = {**original.config, "role_stream": False, "role_bounded_stage": True}
    members = {str(i): Role(config, "judge", original.output / str(i)) for i in range(3)}
    context, evidence = build_context({"task_id": "synthetic", "prompt": "Produce complete JSON"},
                                     [{"artifact": {"final_response": "{}"}}])
    draft = {"criteria": [criterion()]}
    invalid = {**deepcopy(draft), "gaps": [{"gap_id": "g", "product_ids": ["product_1"],
        "expected": "Complete JSON", "expected_basis": "task", "expected_refs": ["task"],
        "observed": "Synthetic", "gap": "Synthetic", "evidence_refs": ["task"],
        "verification": "Inspect the product", "status": "needs_check"}]}
    if not accepted_draft:
        invalid["revision_decisions"] = [{"source_parts": ["unknown"], "action": "retain", "targets": []}]
    responses.extend([reply(draft) for _ in range(3)] + [reply(invalid) for _ in range(6)])
    reduced = []
    async def reduce(*args, **kwargs):
        reduced.append(True)
        return deepcopy(draft)
    monkeypatch.setattr(jury, "reduce_proposals", reduce)
    async def run():
        return await jury._propose_once(context, evidence, members, None, strict=True, revision_protocol=True)
    if accepted_draft:
        await run()
        assert not list(original.output.glob("*/*/contract_rejected.json"))
        await run()  # New proposal cache; successful draft transport replies remain reusable.
        assert len(reduced) == 2 and len(calls) == 9
    else:
        with pytest.raises(RuntimeError, match="Fewer than three"):
            await run()
        assert len(list(original.output.glob("*/*/contract_rejected.json"))) == 3
        assert not reduced


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit_read", [True, False])
async def test_exhausted_evidence_reply_is_not_replayed_forever_on_resume(transport, explicit_read):
    from reward_engine.core.outcome.context import EvidenceReadLimit, build_context

    role, responses, calls = transport
    role.config["role_stream"] = False
    role.bounded_stage = True
    context, evidence = build_context({"task_id": "synthetic", "prompt": "Inspect supplied records"}, [
        {"source_documents": [{"source": str(i), "text": "x" * 1000} for i in range(3)]}])
    responses.extend([reply({"need_evidence": [f"product_1:source:{i}"]} if explicit_read else
                            {"status": "unknown", "evidence_refs": [f"product_1:source:{i}"]})
                      for i in ((0, 1, 2, 2) if explicit_read else (0, 1, 2, 2, 2))])
    responses.append(reply({"status": "unknown"}))
    args = (role, "outcome_measure", context, evidence, "Inspect records within the read limit", {"status": "unknown"})
    error_type = EvidenceReadLimit if explicit_read else ResponseContractError
    with pytest.raises(EvidenceReadLimit):
        await ask(*args, verify_used_previews=True)
    count = 4 if explicit_read else 5
    assert len(calls) == count
    assert await ask(*args, verify_used_previews=True) == {"status": "unknown"}
    assert len(calls) == count + 1
    prior = json.loads(calls[count - 1]["messages"][1]["content"])
    resumed = json.loads(calls[count]["messages"][1]["content"])
    assert resumed.pop("resume_correction")["previous_request_identity"]
    assert resumed == prior  # Same evidence and read allowance, plus retained failure feedback.
    marker = next(role.output.glob("*/contract_rejected.json"))
    assert read(marker)["error_type"] == error_type.__name__
