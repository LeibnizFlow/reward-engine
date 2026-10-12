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
@pytest.mark.parametrize("phase", ["map", "reduce"])
@pytest.mark.parametrize(
    "groups,unknown,changed_groups,changed_unknown",
    [
        ([["product_1"], ["product_2"]], [], [["product_2"], ["product_1"]], []),
        ([["product_1", "product_2"]], [], [["product_1"], ["product_2"]], []),
        ([["product_1"], ["product_2"]], [], [["product_1", "product_2"]], []),
        ([["product_1"]], ["product_2"], [["product_2"]], ["product_1"]),
        ([], ["product_1", "product_2"], [["product_1", "product_2"]], []),
    ],
    ids=["reverse", "split-tie", "merge-tie", "swap-unknown", "remove-unknown"],
)
@pytest.mark.asyncio
async def test_ranking_repair_rejects_unrelated_preference_changes(
    roles, bounded, phase, groups, unknown, changed_groups, changed_unknown
):
    make, calls = roles
    ctx, evidence = context()
    original = {**ranking(), "groups": groups, "unknown": unknown}
    invalid = deepcopy(original)
    invalid["observations"][0].pop("reason")
    changed = {
        **deepcopy(original),
        "groups": changed_groups,
        "unknown": changed_unknown,
    }
    before = deepcopy((original, invalid, changed))
    records = {}

    def reply(req):
        if req["mode"] != f"outcome_ranking_{phase}":
            return deepcopy(original)
        if "response_recovery" not in req:
            return deepcopy(invalid)
        recovery = req["response_recovery"]
        assert recovery["rejected_response"] == invalid
        assert recovery["details"]["preserved_ranking"] == {
            "groups": groups,
            "unknown": unknown,
        }
        return deepcopy(changed)

    members = {str(i): make(f"member{i}", reply, bounded=bounded) for i in range(3)}
    call = jury.rank(
        ctx,
        evidence,
        members,
        make("chooser", reply, bounded=bounded),
        record=records.__setitem__,
    )
    if phase == "reduce":
        with pytest.raises(
            ResponseContractError, match="already-valid ordering"
        ) as exc:
            await call
        assert exc.value.rejected_response == changed
        assert "ranking" not in records  # No repaired or original verdict published.
        assert records["ranking_map"]["succeeded"] == 3
    else:
        result = await call
        assert result["status"] == "unavailable" and "groups" not in result
        assert records["ranking_map"]["succeeded"] == 0
        assert not any(name == "chooser" for name, _ in calls)
    for name in {name for name, _ in calls}:
        attempts = [req for model, req in calls if model == name]
        if attempts[0]["mode"] == f"outcome_ranking_{phase}":
            assert len(attempts) == 2
            assert attempts[0] == {
                k: v for k, v in attempts[1].items() if k != "response_recovery"
            }
    assert (original, invalid, changed) == before


@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize("defect", ["observation", "references", "reason"])
@pytest.mark.asyncio
async def test_ranking_repairs_other_fields_without_changing_preference(
    roles, bounded, defect
):
    make, calls = roles
    ctx, evidence = context()
    original = {**ranking(), "groups": [["product_1"]], "unknown": ["product_2"]}
    invalid = deepcopy(original)
    if defect == "observation":
        invalid["observations"].pop()
    elif defect == "references":
        invalid["observations"][0]["evidence_refs"] = ["check:0:10"]
    else:
        invalid.pop("reason")

    def reply(req):
        return deepcopy(original if "response_recovery" in req else invalid)

    result = await jury.rank(
        ctx,
        evidence,
        {str(i): make(f"member{i}", reply, bounded=bounded) for i in range(3)},
        make("chooser", reply, bounded=bounded),
    )
    assert result == {**original, "status": "compared"}
    assert Counter(name for name, _ in calls) == {
        name: 2 for name in ("member0", "member1", "member2", "chooser")
    }


@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize("defect", ["missing", "empty", "duplicate", "unknown"])
@pytest.mark.asyncio
async def test_invalid_ranking_partition_can_be_corrected(roles, bounded, defect):
    make, calls = roles
    ctx, evidence = context()
    original = {**ranking(), "groups": [["product_2"], ["product_1"]]}
    invalid = deepcopy(original)
    if defect == "missing":
        invalid.pop("groups")
    elif defect == "empty":
        invalid["groups"] = []
    elif defect == "duplicate":
        invalid["groups"] = [["product_1"], ["product_1"]]
    else:
        invalid["unknown"] = ["product_1"]

    def reply(req):
        if "response_recovery" in req:
            assert "preserved_ranking" not in req["response_recovery"]["details"]
            return deepcopy(original)
        return deepcopy(invalid)

    result = await jury.rank(
        ctx,
        evidence,
        {str(i): make(f"member{i}", reply, bounded=bounded) for i in range(3)},
        make("chooser", reply, bounded=bounded),
    )
    assert result == {**original, "status": "compared"}
    assert len(calls) == 8


@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize("syntax_retry", [False, True])
@pytest.mark.asyncio
async def test_ranking_preservation_survives_syntax_retry_and_evidence_closure(
    roles, bounded, syntax_retry
):
    make, calls = roles
    ctx, evidence = context()
    original = {**ranking(), "groups": [["product_1"], ["product_2"]]}
    invalid = deepcopy(original)
    invalid["observations"][0].pop("reason")
    changed = {**deepcopy(original), "groups": list(reversed(original["groups"]))}
    count = 0

    def reply(req):
        nonlocal count
        count += 1
        if syntax_retry:
            return [invalid, '{"groups":', changed][count - 1]
        # A repeated read closes evidence access without supplying new facts.
        if count in (1, 3):
            return {"need_evidence": ["task"]}
        if count == 2:
            return deepcopy(invalid)
        assert req["evidence_access_closed"]
        return deepcopy(changed)

    with pytest.raises(
        ResponseContractError,
        match="already-valid ordering"
        if syntax_retry
        else "Evidence and tools are closed",
    ):
        await jury.rank(
            ctx,
            evidence,
            {
                str(i): make(f"member{i}", lambda _: ranking(), bounded=bounded)
                for i in range(3)
            },
            make("chooser", reply, bounded=bounded),
        )
    # Repair-origin reads stop before a new evidence stage or preference reroll.
    assert count == 3


@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.asyncio
async def test_newly_attached_evidence_allows_ranking_reassessment(roles, bounded):
    make, calls = roles
    ctx, evidence = build_context(
        {"task_id": "compare", "prompt": "Compare against the full source."},
        [
            {
                "artifact": {"final_response": f"Answer {i}"},
                "source_documents": [{"text": "Source detail. " * 100}],
            }
            for i in range(2)
        ],
    )
    original = {**ranking(), "groups": [["product_1"], ["product_2"]]}
    original["observations"][0]["evidence_refs"] = ["product_1:source:0"]
    invalid = deepcopy(original)
    invalid["observations"][0].pop("reason")
    changed = {**deepcopy(original), "groups": list(reversed(original["groups"]))}
    fixed = deepcopy(original)
    count = 0

    def reply(req):
        nonlocal count
        count += 1
        if count == 1:
            return deepcopy(invalid)
        if count == 2:
            return deepcopy(
                fixed
            )  # Valid provisional ranking triggers host preview verification.
        assert (
            req["requested_evidence"]["product_1:source:0"]
            == evidence["product_1:source:0"]
        )
        assert "response_recovery" not in req
        assert req["provisional_check"] == fixed
        return deepcopy(changed)

    result = await jury.rank(
        ctx,
        evidence,
        {
            str(i): make(f"member{i}", lambda _: ranking(), bounded=bounded)
            for i in range(3)
        },
        make("chooser", reply, bounded=bounded),
    )
    assert result == {**changed, "status": "compared"}
    assert count == 3


@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize(
    "ref",
    [
        "task",
        "product_1:artifact:final_response",
        "product_1:environment",
        "product_1:source:0",
    ],
)
@pytest.mark.asyncio
async def test_redundant_inline_read_cannot_unlock_ranking(roles, bounded, ref):
    from reward_engine.core.outcome.context import inline_recent_sources

    make, calls = roles
    ctx, evidence = build_context(
        {"task_id": "compare", "prompt": "Compare the supplied answers."},
        [
            {
                "artifact": {"final_response": f"Answer {i}"},
                "environment": {"observed": "already supplied"},
                "source_documents": [{"text": "Source detail. " * 100}],
            }
            for i in range(2)
        ],
    )
    # This long source would normally require a read, but the real runtime's
    # inlining projection has already supplied it completely to the model.
    ctx = inline_recent_sources(ctx, evidence)
    assert ctx["products"][0]["sources"][0]["preview_only"] is False
    original = {**ranking(), "groups": [["product_1"], ["product_2"]]}
    invalid = deepcopy(original)
    invalid["observations"][0].pop("reason")
    changed = {**deepcopy(original), "groups": list(reversed(original["groups"]))}
    count = 0

    def reply(req):
        nonlocal count
        count += 1
        if count == 1:
            return deepcopy(invalid)
        if count == 2:
            return {"need_evidence": [ref]}
        assert req["requested_evidence"][ref] == evidence[ref]
        return deepcopy(changed)

    with pytest.raises(ResponseContractError, match="Evidence and tools are closed"):
        await jury.rank(
            ctx,
            evidence,
            {
                str(i): make(f"member{i}", lambda _: ranking(), bounded=bounded)
                for i in range(3)
            },
            make("chooser", reply, bounded=bounded),
        )
    assert count == 2  # No source expansion or subsequent model reroll.


@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize("unknown", [False, True])
@pytest.mark.asyncio
async def test_ranking_repair_allows_only_set_reordering_and_valid_replies_need_no_retry(
    roles, bounded, unknown
):
    make, calls = roles
    ctx, evidence = context()
    original = ranking()
    if unknown:
        original.update(groups=[], unknown=["product_1", "product_2"])
    invalid = deepcopy(original)
    invalid.pop("reason")
    changed = deepcopy(original)
    (changed["unknown"] if unknown else changed["groups"][0]).reverse()

    def reply(req):
        return deepcopy(changed if "response_recovery" in req else invalid)

    result = await jury.rank(
        ctx,
        evidence,
        {
            str(i): make(f"member{i}", lambda _: deepcopy(original), bounded=bounded)
            for i in range(3)
        },
        make("chooser", reply, bounded=bounded),
    )
    assert result == {**changed, "status": "compared"}
    assert Counter(name for name, _ in calls) == {
        "member0": 1,
        "member1": 1,
        "member2": 1,
        "chooser": 2,
    }


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
