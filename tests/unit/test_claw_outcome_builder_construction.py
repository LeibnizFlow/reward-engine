"""Multi-round Builder audits must improve drafts before any installed effects."""

import json
from collections import Counter
from copy import deepcopy

import pytest

from reward_engine.core.outcome.builder import build
from reward_engine.core.outcome.construction import fingerprint
from reward_engine.core.outcome.context import ResponseContractError
from reward_engine.core.outcome.harness import defaults
from reward_engine.core.outcome.measurement_tools import MeasurementTools
from tests.support.claw_outcome_builder_harness import setup


def keep():
    return {
        "assessment": "keep",
        "reason": "The full draft implements the Task and frozen Jury.",
        "issues": [],
    }


def revision(draft, summary="No supported defect: preserve the complete draft."):
    return {**deepcopy(draft), "revision_summary": summary, "unresolved_issues": []}


def issue(scope="current"):
    return {
        "scope": scope,
        "target": "syntax.anchors.full",
        "problem": "The check should parse the complete artifact, not a JSON fragment.",
        "suggested_change": "Require the entire requested JSON to parse.",
        "evidence_refs": ["task"],
    }


@pytest.mark.asyncio
async def test_nine_calls_before_author_with_independent_reviews_and_frozen_jury():
    context, evidence, jury, reply = setup()
    reply["tests"][0]["checker"] = "generated.json"
    calls, authored, records = [], [], {}
    baseline_jury = deepcopy(jury)

    async def model(request):
        calls.append(deepcopy(request))
        assert not authored
        if request["mode"] == "outcome_builder":
            return deepcopy(reply)
        if request["mode"] == "outcome_builder_audit":
            assert "reviews" not in request
            assert request["jury"] == {"criteria": jury["criteria"]}
            assert request["context"]["task"] == context["task"]
            request["current_artifact"]["rubrics"][0]["anchors"]["full"] = (
                "PRIVATE_MUTATION"
            )
            return {
                "assessment": "revise",
                "reason": "Full anchors need full-artifact parsing.",
                "issues": [issue()],
            }
        assert request["mode"] == "outcome_builder_reduce"
        assert len(request["reviews"]) == 3
        assert "PRIVATE_MUTATION" not in json.dumps(request)
        current = revision(
            request["draft"], "Full anchors now require the complete JSON document."
        )
        current["rubrics"][0]["anchors"]["full"] = "The whole artifact parses as JSON."
        current["tests"][0]["question"] = "Does the entire current artifact parse?"
        return current

    async def author(test, registry):
        assert len(calls) == 9
        assert test["question"] == "Does the entire current artifact parse?"
        authored.append(test["checker"])
        registry.bind_recipe(
            test["checker"],
            {"probes": [{"tool": "calculate", "arguments": {"expression": "1"}}]},
        )

    registry = defaults(model, tools=MeasurementTools())
    pack = await build(
        context,
        evidence,
        jury,
        registry,
        model,
        author=author,
        construction_rounds=3,
        record=records.__setitem__,
    )
    assert Counter(r["mode"] for r in calls) == {
        "outcome_builder": 1,
        "outcome_builder_audit": 6,
        "outcome_builder_reduce": 2,
    }
    assert len(authored) == 1 and "::builder:" in authored[0]
    assert jury == baseline_jury
    assert pack["criteria"][0]["standard"] == baseline_jury["criteria"][0]["standard"]
    assert [r["round"] for r in pack["construction"]["history"]] == [1, 2, 3]
    assert len([k for k in records if k.endswith("/reviews")]) == 2
    for number in (2, 3):
        independent = [
            r
            for r in calls
            if r["mode"] == "outcome_builder_audit"
            and r["construction_round"] == number
        ]
        assert len({fingerprint(r["current_artifact"]) for r in independent}) == 1


@pytest.mark.asyncio
async def test_keep_still_runs_all_rounds_and_complete_cache_only():
    context, evidence, jury, reply = setup()
    calls = []

    async def model(request):
        calls.append(deepcopy(request))
        if request["mode"].endswith("_audit"):
            return keep()
        return revision(request.get("draft", reply))

    registry = defaults(model)
    old = await build(context, evidence, jury, registry, model)
    assert len(calls) == 1
    current = await build(
        context, evidence, jury, registry, model, previous=old, construction_rounds=3
    )
    assert len(calls) == 10
    reused = await build(
        context,
        evidence,
        jury,
        registry,
        model,
        previous=current,
        construction_rounds=3,
    )
    assert len(calls) == 10 and reused == current
    incomplete = deepcopy(current)
    incomplete["construction"]["history"].pop()
    await build(
        context,
        evidence,
        jury,
        registry,
        model,
        previous=incomplete,
        construction_rounds=3,
    )
    assert len(calls) == 19


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change", ["task", "evidence", "jury", "rounds", "version", "test"]
)
async def test_cache_requires_current_inputs_version_rounds_and_published_contract(
    change,
):
    context, evidence, jury, reply = setup()
    calls = []

    async def model(request):
        calls.append(request["mode"])
        return (
            keep()
            if request["mode"].endswith("_audit")
            else revision(request.get("draft", reply))
        )

    registry = defaults(model)
    old = await build(context, evidence, jury, registry, model, construction_rounds=3)
    rounds = 3
    if change == "task":
        context["task"]["prompt"] += " without Markdown fences"
    elif change == "evidence":
        evidence["reference:task_review:source"] = {
            "text": "Supplemental updated source."
        }
    elif change == "jury":
        jury["criteria"][0]["standard"] += " with no trailing text"
    elif change == "rounds":
        rounds = 2
    elif change == "version":
        old["construction"]["version"] = "old prompt version"
    else:
        old["tests"][0]["question"] = "Unreviewed mutation"
    await build(
        context,
        evidence,
        jury,
        registry,
        model,
        previous=old,
        construction_rounds=rounds,
    )
    assert len(calls) == 9 + (5 if rounds == 2 else 9)


@pytest.mark.asyncio
async def test_same_checker_name_different_contract_cannot_reuse_old_recipe():
    context, evidence, jury, reply = setup()
    reply["tests"][0]["checker"] = "generated.json"
    question = "Parse the whole JSON artifact."
    authored = []

    async def model(request):
        if request["mode"].endswith("_audit"):
            return keep()
        result = revision(request.get("draft", reply))
        result["tests"][0]["question"] = question
        return result

    async def author(test, registry):
        authored.append(deepcopy(test))
        registry.bind_recipe(
            test["checker"],
            {
                "probes": [
                    {
                        "tool": "calculate",
                        "arguments": {"expression": str(len(authored))},
                    }
                ]
            },
        )

    registry = defaults(model, tools=MeasurementTools())
    first = await build(
        context, evidence, jury, registry, model, author=author, construction_rounds=3
    )
    question = "Parse the whole JSON artifact and inspect the requested root type."
    second = await build(
        context, evidence, jury, registry, model, author=author, construction_rounds=3
    )
    assert len(authored) == 2
    assert first["tests"][0]["checker"] != second["tests"][0]["checker"]
    assert (
        first["tests"][0]["execution_recipe"] != second["tests"][0]["execution_recipe"]
    )


@pytest.mark.asyncio
async def test_persistent_invalid_reviewer_stops_before_reduce_and_install():
    context, evidence, jury, reply = setup()
    reply["tests"][0]["checker"] = "new.checker"
    counts, authored, records = Counter(), [], {}

    async def model(request):
        counts[request["mode"]] += 1
        assert request["mode"] == "outcome_builder"
        return deepcopy(reply)

    async def good(request):
        counts["good"] += 1
        return keep()

    async def broken(request):
        counts["broken"] += 1
        if counts["broken"] == 2:
            assert "response_recovery" in request
        return {
            "assessment": "keep",
            "reason": "cannot certify malformed issues",
            "issues": "not a list",
        }

    async def author(test, registry):
        authored.append(test)

    with pytest.raises(ResponseContractError):
        await build(
            context,
            evidence,
            jury,
            defaults(model),
            model,
            author=author,
            construction_rounds=3,
            reviewers={"a": good, "b": good, "c": broken},
            record=records.__setitem__,
        )
    assert counts == {"outcome_builder": 1, "good": 2, "broken": 2}
    assert not authored and "pack" not in records
    assert records["construction/builder/round_2/map"]["succeeded"] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_mode", ["outcome_builder", "outcome_builder_reduce"])
async def test_json_redo_preserves_operation_and_recovered_draft(failed_mode):
    context, evidence, jury, reply = setup()
    requests, failed = [], False

    async def model(request):
        nonlocal failed
        requests.append(deepcopy(request))
        if request["mode"] == failed_mode and not failed:
            failed = True
            raise json.JSONDecodeError("unescaped quote", "{bad", 1)
        if request["mode"].endswith("_audit"):
            return keep()
        return revision(request.get("draft", reply))

    pack = await build(
        context, evidence, jury, defaults(model), model, construction_rounds=3
    )
    assert len(requests) == 10 and pack["status"] == "callable"
    redo = next(r for r in requests if "response_recovery" in r)
    assert redo["mode"] == failed_mode and redo["context"]["task"] == context["task"]


@pytest.mark.asyncio
async def test_upstream_concerns_remain_visible_without_changing_jury():
    context, evidence, jury, reply = setup()
    concern = issue("upstream")
    requests = []

    async def model(request):
        requests.append(deepcopy(request))
        if request["mode"].endswith("_audit"):
            return {
                "assessment": "uncertain",
                "reason": "The frozen standard has an ambiguity.",
                "issues": [deepcopy(concern)],
            }
        return revision(request.get("draft", reply))

    pack = await build(
        context, evidence, jury, defaults(model), model, construction_rounds=3
    )
    assert pack["construction"]["upstream_concerns"] == [concern]
    assert pack["criteria"][0]["standard"] == jury["criteria"][0]["standard"]
    final_audits = [
        r
        for r in requests
        if r["mode"].endswith("_audit") and r["construction_round"] == 3
    ]
    assert all(r["upstream_concerns"] == [concern] for r in final_audits)


@pytest.mark.asyncio
async def test_reducer_attempt_to_replace_frozen_jury_gets_one_contract_correction():
    context, evidence, jury, reply = setup()
    reductions = []

    async def model(request):
        if request["mode"].endswith("_audit"):
            return keep()
        current = revision(request.get("draft", reply))
        if request["mode"].endswith("_reduce"):
            reductions.append(deepcopy(request))
            if len(reductions) == 1:
                current["criteria"] = [
                    {"id": "invented", "standard": "Use a prescribed reference library"}
                ]
        return current

    pack = await build(
        context, evidence, jury, defaults(model), model, construction_rounds=3
    )
    assert len(reductions) == 3 and "response_recovery" in reductions[1]
    assert [c["id"] for c in pack["criteria"]] == ["syntax"]
    assert pack["criteria"][0]["standard"] == jury["criteria"][0]["standard"]


@pytest.mark.asyncio
async def test_source_reread_keeps_limits_and_builder_never_receives_seed_bodies():
    context, evidence, jury, reply = setup()
    source_ref = "reference:task_review:source"
    source = {
        "text": "Incomplete reference example.",
        "reader_limits": ["formula omitted"],
        "excerpt_only": True,
        "total_chars": 1000,
    }
    evidence[source_ref] = deepcopy(source)
    context["products"][0]["artifact"]["final_response"] = "PRIVATE_SEED_BODY"
    requests = []

    async def model(request):
        requests.append(deepcopy(request))
        assert "PRIVATE_SEED_BODY" not in json.dumps(request)
        assert request["context"]["task"] == context["task"]
        if request["mode"].endswith("_audit"):
            assert source_ref in request["context"]["available_evidence_refs"]
            if "requested_evidence" not in request:
                return {"need_evidence": [source_ref]}
            assert request["requested_evidence"][source_ref] == source
            return keep()
        return revision(request.get("draft", reply))

    pack = await build(
        context, evidence, jury, defaults(model), model, construction_rounds=3
    )
    assert (
        len(requests) == 15
    )  # Six independent source reads, not construction restarts.
    assert pack["criteria"][0]["standard"] == jury["criteria"][0]["standard"]


@pytest.mark.asyncio
async def test_complete_cached_pack_restores_its_final_recipe_without_reauditing():
    context, evidence, jury, reply = setup()
    reply["tests"][0]["checker"] = "generated.json"
    calls = []

    async def model(request):
        calls.append(request["mode"])
        return (
            keep()
            if request["mode"].endswith("_audit")
            else revision(request.get("draft", reply))
        )

    async def author(test, registry):
        registry.bind_recipe(
            test["checker"],
            {"probes": [{"tool": "calculate", "arguments": {"expression": "1"}}]},
        )

    first_registry = defaults(model, tools=MeasurementTools())
    pack = await build(
        context,
        evidence,
        jury,
        first_registry,
        model,
        author=author,
        construction_rounds=3,
    )
    next_registry = defaults(model, tools=MeasurementTools())
    cached = await build(
        context,
        evidence,
        jury,
        next_registry,
        model,
        previous=pack,
        construction_rounds=3,
    )
    assert len(calls) == 9 and cached == pack
    assert (
        next_registry.recipes[cached["tests"][0]["checker"]]
        == cached["tests"][0]["execution_recipe"]
    )
