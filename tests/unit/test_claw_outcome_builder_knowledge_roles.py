"""Typed Task Review priors reach the Builder as roles, not filters: criteria stay the standard."""

from copy import deepcopy

import pytest

from reward_engine.core.outcome.builder import bind_knowledge, build, knowledge_roles
from reward_engine.core.outcome.context import build_context, measurement_prior_view
from reward_engine.core.outcome.harness import defaults

RULE, IMPL, CAVEAT, OTHER = ("skill:task_review:" + n for n in ("rule", "impl", "caveat", "other"))


def prior(ref, use, source_id, *, conditions_unstated=False, source_class="official_docs"):
    return {"ref": ref, "content": {"kind": "task_review_prior", "authority": "domain_advice_not_user_requirement",
        "direction": "Selected source passages", "relevance": "Validate records", "applicability": "See cards",
        "verification": "Check against Task materials", "use": use, "conditions_unstated": conditions_unstated,
        "source_class": source_class, "source_contract": "extractive-cards-v3",
        "mechanisms": [{"id": ref + "_c0", "source_id": source_id, "block_ids": ["block_0_9"],
                        "source_statement": "exact text", "need_ids": ["need_0"], "disposition": "retained",
                        "direction_id": ref.split(":")[-1], "citations": []}]}}


def setup(*, unstated=False):
    context, evidence = build_context({"task_id": "x", "prompt": "Validate the uploaded records"},
                                      [{"artifact": {"final_response": "{}"}}])
    for skill in (prior(RULE, "verification_rule", "source_0", conditions_unstated=unstated),
                  prior(IMPL, "reference_implementation", "source_1", source_class="third_party"),
                  prior(CAVEAT, "limitation_note", "source_0"),
                  prior(OTHER, "limitation_note", "source_9")):
        context["skills"].append(skill)
        evidence[skill["ref"]] = deepcopy(skill["content"])
    context["available_evidence_refs"] = list(evidence)
    jury = {"criteria": [
        {"id": "validity", "standard": "Records satisfy the validation rule", "kind": "hard", "origin": "task",
         "measurement_question": "Apply the rule", "evidence_refs": ["task"]},
        {"id": "format", "standard": "Output is JSON", "kind": "hard", "origin": "task",
         "measurement_question": "Parse", "evidence_refs": ["task"]}]}
    reply = {"rubrics": [{"criterion_id": c, "anchors": {"fail": "no", "partial": "some", "full": "yes"}}
                         for c in ("validity", "format")],
             "tests": [
                 {"test_id": "rule-check", "criterion_id": "validity", "checker": "evidence.review",
                  "question": "Does each record satisfy the documented rule, given its stated conditions?",
                  "arguments": {}, "knowledge_refs": [RULE, IMPL]},
                 {"test_id": "syntax", "criterion_id": "format", "checker": "json.parse",
                  "question": "Does it parse?", "arguments": {"ref": "product_1:artifact:final_response"}}]}
    return context, evidence, jury, reply


def test_roles_group_by_use_and_link_same_source_conditions_without_dropping_anything():
    context, *_ = setup(unstated=True)
    roles = knowledge_roles(context)
    assert roles["method_refs"] == [RULE] and roles["reference_refs"] == [IMPL]
    assert roles["condition_refs"] == [CAVEAT, OTHER]  # An unrelated caveat is kept, just not linked.
    assert roles["conditions_by_method"] == {RULE: [CAVEAT]}
    assert roles["conditions_unstated_refs"] == [RULE]
    assert roles["source_class"][IMPL] == "third_party"
    assert knowledge_roles({"skills": []}) is None


def test_legacy_prior_without_use_counts_as_method():
    context, *_ = setup()
    del context["skills"][0]["content"]["use"]
    assert RULE in knowledge_roles(context)["method_refs"]


@pytest.mark.asyncio
async def test_builder_binds_roles_per_test_and_keeps_criterion_as_the_standard():
    context, evidence, jury, reply = setup(unstated=True)
    requests = []
    async def model(request):
        requests.append(request)
        return deepcopy(reply)
    pack = await build(context, evidence, jury, defaults(model), model)
    assert requests[0]["knowledge_roles"]["conditions_by_method"] == {RULE: [CAVEAT]}
    assert "knowledge_refs" in requests[0]["response"]["tests"][0]
    assert "never the standard" in requests[0]["instruction"]
    tests = {t["test_id"]: t for t in pack["tests"]}
    bound = tests["rule-check"]["knowledge"]
    assert bound["method_refs"] == [RULE] and bound["reference_refs"] == [IMPL]
    assert bound["applicability_refs"] == [CAVEAT]  # Same-source caveat travels with the method.
    assert bound["conditions_unstated"] is True and bound["basis"] == "criterion+method"
    plain = tests["syntax"]["knowledge"]
    assert plain == {**plain, "method_refs": [], "reference_refs": [], "applicability_refs": [], "basis": "criterion"}
    assert pack["knowledge_roles"]["reference_refs"] == [IMPL]
    assert {s["ref"] for s in pack["measurement_priors"]} == {RULE, IMPL, CAVEAT, OTHER}  # Nothing filtered.
    assert tests["rule-check"]["rubric"]["standard"] == "Records satisfy the validation rule"


@pytest.mark.asyncio
async def test_unknown_or_model_authored_knowledge_is_a_contract_error_with_one_repair():
    context, evidence, jury, reply = setup()
    broken = deepcopy(reply)
    broken["tests"][0]["knowledge_refs"] = ["skill:task_review:invented"]
    requests = []
    async def model(request):
        requests.append(request)
        return deepcopy(broken if len(requests) == 1 else reply)
    pack = await build(context, evidence, jury, defaults(model), model)
    assert len(requests) == 2 and "knowledge_refs" in requests[1]["repair"]["contract_error"]
    assert pack["tests"][0]["knowledge"]["method_refs"] == [RULE]
    authored = deepcopy(reply)
    authored["tests"][0]["knowledge"] = {"method_refs": [RULE], "applicability_refs": []}
    requests.clear()
    async def insist(request):
        requests.append(request)
        return deepcopy(authored)
    with pytest.raises(ValueError, match="knowledge bindings"):
        await build(context, evidence, jury, defaults(insist), insist)


def test_reference_only_citation_is_visible_and_conditions_come_from_rubric_refs_too():
    context, *_ = setup()
    roles = knowledge_roles(context)
    test = bind_knowledge({"test_id": "t", "criterion_id": "validity", "checker": "evidence.review",
                           "question": "q", "knowledge_refs": [IMPL],
                           "rubric": {"id": "validity", "evidence_refs": [RULE]}}, roles)
    assert test["knowledge"]["method_refs"] == [RULE]  # Jury's own citation of the rule counts.
    assert test["knowledge"]["reference_refs"] == [IMPL] and test["knowledge"]["applicability_refs"] == [CAVEAT]
    only_reference = bind_knowledge({"test_id": "t", "criterion_id": "c", "checker": "evidence.review",
                                     "question": "q", "knowledge_refs": [IMPL], "rubric": {"id": "c"}}, roles)
    assert only_reference["knowledge"]["basis"] == "criterion+reference_only"
    assert only_reference["knowledge"]["applicability_refs"] == []


def test_attached_conditions_stay_readable_in_the_single_check_view():
    context, evidence, jury, reply = setup()
    roles = knowledge_roles(context)
    test = bind_knowledge({**deepcopy(reply["tests"][0]), "rubric": jury["criteria"][0]}, roles)
    view = measurement_prior_view(context, evidence, test)
    inline = {s["ref"]: s for s in view["skills"]}
    assert not inline[RULE].get("preview_only") and not inline[CAVEAT].get("preview_only")
    assert not inline[IMPL].get("preview_only")
    assert inline[OTHER]["preview_only"] and "mechanisms" not in inline[OTHER]["content"]
    assert inline[OTHER]["content"]["use"] == "limitation_note"


@pytest.mark.parametrize('use', ['typo', None, ['procedure']])
def test_invalid_explicit_use_is_kept_but_not_promoted_to_a_method(use):
    context, *_ = setup()
    context['skills'][0]['content']['use'] = use
    roles = knowledge_roles(context)
    assert RULE not in roles['method_refs'] and RULE in roles['unclassified_refs']
    test = bind_knowledge({'knowledge_refs': [RULE]}, roles)
    assert test['knowledge']['unclassified_refs'] == [RULE]
    assert test['knowledge']['basis'] == 'criterion'
    assert test['knowledge']['method_refs'] == []


@pytest.mark.asyncio
async def test_reused_checks_refresh_added_and_removed_rubric_knowledge_without_model_calls():
    context, evidence, jury, reply = setup()
    calls = []
    async def model(request):
        calls.append(request)
        return deepcopy(reply)
    registry = defaults(model)
    first = await build(context, evidence, jury, registry, model)
    changed = deepcopy(jury)
    changed['criteria'][1]['evidence_refs'] = ['task', RULE]
    second = await build(context, evidence, changed, registry, model, previous=first)
    syntax = next(t for t in second['tests'] if t['test_id'] == 'syntax')
    assert syntax['knowledge']['method_refs'] == [RULE]
    assert syntax['knowledge']['applicability_refs'] == [CAVEAT]
    third = await build(context, evidence, jury, registry, model, previous=second)
    assert next(t for t in third['tests'] if t['test_id'] == 'syntax')['knowledge']['method_refs'] == []
    assert len(calls) == 1
    assert next(t for t in first['tests'] if t['test_id'] == 'syntax')['knowledge']['method_refs'] == []


def test_binding_rebuilt_from_recipe_and_same_source_is_not_a_mandatory_condition():
    context, *_ = setup()
    test = {'knowledge': {'method_refs': ['obsolete']},
            'execution_recipe': {'probes': [{'tool': 'document.read', 'arguments': {'ref': RULE}}]}}
    result = bind_knowledge(test, knowledge_roles(context))['knowledge']
    assert result['method_refs'] == [RULE] and result['applicability_refs'] == [CAVEAT]
    assert 'NOT a conjunction' in result['boundary']
    assert OTHER not in result['applicability_refs']
    assert 'knowledge' not in bind_knowledge(test, None)
