"""Bounded core recovery through canonical Roles and offline HTTP only."""

import json
from collections import Counter
from copy import deepcopy

import httpx
import pytest

from reward_engine.adapters.workbuddy import models
from reward_engine.core.outcome import hypotheses, jury
from reward_engine.core.outcome.context import ResponseContractError, build_context


@pytest.fixture
def roles(tmp_path, monkeypatch):
    handlers, calls = {}, []
    client = httpx.AsyncClient
    (tmp_path / "auth").write_text("offline-fixture")

    def upstream(request):
        body = json.loads(request.content)
        req = json.loads(body["messages"][1]["content"])
        calls.append((body["model"], deepcopy(req)))
        value = handlers[body["model"]](req)
        finish = "stop"
        if isinstance(value, tuple):
            finish, value = value
        content = value if isinstance(value, str) else json.dumps(value)
        event = {
            "model": body["model"],
            "choices": [
                {"index": 0, "delta": {"content": content}, "finish_reason": finish}
            ],
        }
        return httpx.Response(
            200, text="data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n"
        )

    monkeypatch.setattr(
        models.httpx,
        "AsyncClient",
        lambda **kw: client(transport=httpx.MockTransport(upstream), **kw),
    )

    def make(name, handler, *, bounded=True):
        handlers[name] = handler
        role = models.Role(
            {
                "output": str(tmp_path),
                "roles": {name: name},
                "auth_file": str(tmp_path / "auth"),
                "gateway_local": "http://offline.invalid",
                "role_bounded_stage": bounded,
                "role_response_format": "json_object",
            },
            name,
            tmp_path / name,
        )
        # Exercise the real configuration, not a fake bounded_stage attribute.
        assert role.bounded_stage is bounded
        return role

    return make, calls


def context():
    return build_context(
        {"task_id": "explain", "prompt": "Explain the interface using public sources."},
        [
            {"candidate_id": f"c{i}", "artifact": {"final_response": f"Answer {i}"}}
            for i in range(2)
        ],
    )


def ranking():
    return {
        "groups": [["product_1", "product_2"]],
        "unknown": [],
        "observations": [
            {
                "product_id": f"product_{i}",
                "reason": "The supplied answer addresses the same interface.",
                "evidence_refs": [f"product_{i}:artifact:final_response"],
            }
            for i in (1, 2)
        ],
        "reason": "No supported difference between these answers.",
    }


def proposal():
    return {
        "status": "propose",
        "method": "Inspect the actual source before asserting any missing option.",
        "target": "Unverified interface detail",
        "expected_improvement": "A more supportable explanation, not yet measured.",
        "verification": "Compare the source with the explanation; report uncertainty.",
        "evidence_refs": ["task"],
        "target_gap_ids": [],
        "clarifications": [],
    }


def draft():
    return {
        "criteria": [
            {
                "id": "explanation",
                "standard": "Explain the interface using public sources.",
                "kind": "hard",
                "origin": "task",
                "evidence_refs": ["task"],
                "measurement_question": "Does the explanation match the source?",
            }
        ],
        "gaps": [],
        "uncertainties": [],
    }


@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.asyncio
async def test_ranking_repairs_all_shape_and_reference_errors_on_same_inputs(
    roles, bounded
):
    make, calls = roles
    ctx, evidence = context()
    invalid = ranking()
    invalid.pop("groups")
    for n, obs in enumerate(invalid["observations"]):
        obs["evidence_refs"] = [f"check:{n}:10"]
    before = deepcopy(invalid)

    def reply(req):
        recovery = req.get("response_recovery")
        if not recovery:
            return deepcopy(invalid)
        assert recovery["rejected_response"] == before
        details = recovery["details"]
        assert details["product_ids"] == ["product_1", "product_2"]
        assert len(details["issues"]) == 3
        assert "check:0:10" in details["reason"] and "check:1:10" in details["reason"]
        return ranking()

    members = {str(i): make(f"member{i}", reply, bounded=bounded) for i in range(3)}
    chooser = make("chooser", reply, bounded=bounded)
    records = {}
    result = await jury.rank(
        ctx, evidence, members, chooser, record=records.__setitem__
    )
    assert result["status"] == "compared" and result["groups"] == ranking()["groups"]
    assert records["ranking_map"]["succeeded"] == 3
    for name in ["member0", "member1", "member2", "chooser"]:
        attempts = [req for model, req in calls if model == name]
        assert len(attempts) == 2
        assert attempts[0] == {
            k: v for k, v in attempts[1].items() if k != "response_recovery"
        }
    assert invalid == before


@pytest.mark.asyncio
async def test_persistent_bad_ranking_never_becomes_ties_or_unknown(roles):
    make, calls = roles
    ctx, evidence = context()
    records = {}
    members = {str(i): make(f"member{i}", lambda _: {"groups": []}) for i in range(3)}

    def never(_):
        pytest.fail("No reducer call is permitted without valid comparisons")

    result = await jury.rank(
        ctx, evidence, members, make("chooser", never), record=records.__setitem__
    )
    assert result["status"] == "unavailable"
    assert "groups" not in result
    assert records["ranking_map"]["succeeded"] == 0
    assert Counter(model for model, _ in calls) == {f"member{i}": 2 for i in range(3)}


@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize("phase", ["outcome_expectations", "outcome_jury_map"])
@pytest.mark.asyncio
async def test_jury_corrects_completed_syntax_and_wrong_draft_shape(
    roles, bounded, phase
):
    make, calls = roles
    ctx, evidence = context()
    records, counts = {}, Counter()

    def member(name):
        def reply(req):
            if req["mode"] == phase:
                counts[name] += 1
                if counts[name] == 1:
                    return '{"criteria":['
                if counts[name] == 2:
                    assert (
                        req["response_recovery"]["rejected_raw_response"]
                        == '{"criteria":['
                    )
                    return {
                        "assessment": "partial",
                        "uncertainties": ["Not a definition"],
                    }
                assert req["response_recovery"]["details"]["field"] == "criteria"
                assert (
                    req["response_recovery"]["rejected_response"]["assessment"]
                    == "partial"
                )
            return draft()

        return reply

    members = {
        str(i): make(f"member{i}", member(str(i)), bounded=bounded) for i in range(3)
    }
    result = await jury.propose(
        ctx,
        evidence,
        members,
        make("chooser", lambda _: draft(), bounded=bounded),
        record=records.__setitem__,
    )
    assert result["criteria"][0]["standard"] == draft()["criteria"][0]["standard"]
    assert (
        records["jury_expectations_map"]["succeeded"]
        == records["jury_map"]["succeeded"]
        == 3
    )
    assert all(n == 3 for n in counts.values())
    assert sum(req["mode"] == phase for _, req in calls) == 9
    for role in members.values():
        assert len(list(role.output.glob("*/failure.json"))) == 1


@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.asyncio
async def test_hypothesis_closes_repeated_reads_and_repairs_real_handle_contract(
    roles, bounded
):
    make, calls = roles
    ctx, evidence = context()
    original = deepcopy((ctx, evidence))
    records = {}

    def reply(req):
        if not req.get("evidence_access_closed"):
            return {"need_evidence": ["product_1:artifact:final_response"]}
        assert (
            req["requested_evidence"]["product_1:artifact:final_response"] == "Answer 0"
        )
        if "response_recovery" not in req:
            return {**proposal(), "evidence_refs": ["input/source.py"]}
        assert req["response_recovery"]["rejected_response"]["evidence_refs"] == [
            "input/source.py"
        ]
        return proposal()

    members = {str(i): make(f"member{i}", reply, bounded=bounded) for i in range(3)}
    result = await hypotheses.propose(
        ctx,
        evidence,
        {},
        members,
        make("chooser", reply, bounded=bounded),
        record=records.__setitem__,
    )
    assert result == proposal()
    assert records["hypothesis_map"]["succeeded"] == 3
    assert Counter(model for model, _ in calls) == {
        name: 4 for name in ["member0", "member1", "member2", "chooser"]
    }
    assert (ctx, evidence) == original


@pytest.mark.parametrize("invalid", ["repeat_read", "bad_reference", "truncated"])
@pytest.mark.asyncio
async def test_unrecoverable_hypotheses_remain_unavailable_not_semantic_keep(
    roles, invalid
):
    make, calls = roles
    ctx, evidence = context()
    records = {}
    value = {
        "repeat_read": {"need_evidence": ["product_1:artifact:final_response"]},
        "bad_reference": {**proposal(), "evidence_refs": ["invented"]},
        "truncated": ("length", proposal()),
    }[invalid]
    members = {str(i): make(f"member{i}", lambda _: value) for i in range(3)}

    def never(_):
        pytest.fail("No synthesis of failed proposals")

    result = await hypotheses.propose(
        ctx, evidence, {}, members, make("chooser", never), record=records.__setitem__
    )
    assert result["availability"] == "unavailable"
    assert result["method"] == ""
    assert records["hypothesis_map"]["succeeded"] == 0
    per_member = {"repeat_read": 3, "bad_reference": 2, "truncated": 1}[invalid]
    assert Counter(model for model, _ in calls) == {
        f"member{i}": per_member for i in range(3)
    }


def test_ranking_shape_errors_are_actionable_without_mutating_or_filling_output():
    ctx, evidence = context()
    invalid = {
        "groups": [[{"product_id": "product_1"}]],
        "unknown": [],
        "observations": [None],
    }
    before = deepcopy(invalid)
    with pytest.raises(ResponseContractError) as exc:
        jury.validate_ranking(invalid, ctx, evidence)
    assert len(exc.value.details["issues"]) == 3
    assert invalid == before
