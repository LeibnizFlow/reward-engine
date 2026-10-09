"""General regressions: provenance, alternatives, disputed references and bounded trials."""

from tests.support.outcome_clauses_and_disputes import (
    issue,
)

from copy import deepcopy

import pytest

from reward_engine.core.outcome.admission import (
    admit,
    evidence_index,
    validation_payload,
)
from reward_engine.core.outcome.clauses import (
    claim_binding,
    normalize_criteria,
    targets,
)
from reward_engine.core.outcome.construction import fingerprint
from reward_engine.core.outcome.context import ResponseContractError, build_context
from reward_engine.core.outcome.counterexamples import (
    choose_targets,
    targets_for,
    verify_issue,
)
from reward_engine.core.outcome.harness import Registry, defaults
from reward_engine.core.outcome.jury_drafts import apply_part_decisions
from reward_engine.core.outcome.measurement_tools import MeasurementTools
from reward_engine.core.outcome.validated_construction import (
    publish_validation,
    select_repair,
)


def test_reasoned_structured_merge_has_one_canonical_wording_but_no_equivalence_certificate():
    from reward_engine.core.outcome.jury_drafts import normalize_part_bindings
    inputs = {'independent_expectations': [{'expectation_id': 'e', 'subrequirements': [
        {'part_id': 'e:a', 'text': 'Produce a workbook.'}, {'part_id': 'e:b', 'text': 'Produce a JSON file.'}]}],
        'proposal_expectations': []}
    final = 'Produce both a workbook and a JSON file.'
    raw = {'criteria': [{'id': 'files', 'clauses': [{'id': 'both', 'text': final, 'evidence_refs': ['task']}]}],
           'part_decisions': [{'parts': ['p0', 'p1'], 'action': 'retain', 'criterion_id': 'files',
                               'clause_id': 'both', 'reason': 'Consolidate both delivery requirements.'}]}
    frozen = deepcopy(raw)
    normalized = normalize_part_bindings(raw, inputs, {'task': 'TASK'})
    assert raw == frozen
    decision = normalized['part_decisions'][0]
    assert decision['action'] == 'rewrite' and decision['text'] == final
    assert decision['binding_resolution']['original_texts']['p0'] == 'Produce a workbook.'
    assert decision['binding_resolution']['semantic_equivalence_verified'] is False
    result = apply_part_decisions(normalized, inputs, {'task': 'TASK'})
    assert result['criteria'][0]['clauses'][0]['source_part_ids'] == ['e:a', 'e:b']
    invalid = deepcopy(raw)
    invalid['part_decisions'][0].pop('reason')
    with pytest.raises(ValueError, match='exact final clause'):
        apply_part_decisions(normalize_part_bindings(invalid, inputs, {'task': 'TASK'}), inputs, {'task': 'TASK'})
    invalid = deepcopy(raw)
    invalid['part_decisions'][0]['clause_id'] = 'missing'
    with pytest.raises(ValueError, match='exact final clause'):
        apply_part_decisions(normalize_part_bindings(invalid, inputs, {'task': 'TASK'}), inputs, {'task': 'TASK'})

TASK = {'task_id': 'public-task', 'prompt': 'Return valid JSON with a useful explanation.'}


def summary_rewrite_fixture():
    old = 'Write an investigation conclusion and handling recommendation for each case.'
    inputs = {'independent_expectations': [{'expectation_id': 'e', 'subrequirements': [
        {'part_id': 'e:0', 'text': old}]}], 'proposal_expectations': []}
    raw = {'criteria': [{'id': 'investigation', 'standard': old, 'clauses': [
        {'id': 'conclusion', 'text': 'Ground each investigation conclusion in the available evidence.'},
        {'id': 'action', 'text': 'Recommend a concrete action for each case.'}]}],
        'part_decisions': [{'parts': ['p0'], 'action': 'rewrite', 'criterion_id': 'investigation',
            'clause_id': 'conclusion', 'text': 'Ground each investigation conclusion in the available evidence.',
            'reason': 'Separate the conclusion from the action clause.', 'evidence_refs': ['task']}]}
    return old, inputs, raw


def test_explicit_summary_rewrite_projects_all_final_clauses_and_keeps_original():
    from reward_engine.core.outcome.jury_drafts import normalize_part_bindings
    from reward_engine.core.outcome.jury_refinement import check_retired_clauses
    old, inputs, raw = summary_rewrite_fixture()
    frozen = deepcopy(raw)
    with pytest.raises(ValueError, match='withdrawn'):
        check_retired_clauses(raw, inputs)
    normalized = normalize_part_bindings(raw, inputs, {'task': 'TASK'})
    assert raw == frozen
    criterion = normalized['criteria'][0]
    assert criterion['standard'] == '\n'.join(c['text'] for c in criterion['clauses'])
    assert criterion['summary_projections'][0]['original_standard'] == old
    assert criterion['summary_projections'][0]['semantic_equivalence_verified'] is False
    check_retired_clauses(normalized, inputs)
    apply_part_decisions(normalized, inputs, {'task': 'TASK'})


@pytest.mark.parametrize('action', ['exclude', 'defer'])
def test_summary_projection_does_not_erase_an_excluded_or_deferred_obligation(action):
    from reward_engine.core.outcome.jury_drafts import normalize_part_bindings
    from reward_engine.core.outcome.jury_refinement import check_retired_clauses
    old, inputs, raw = summary_rewrite_fixture()
    raw['part_decisions'][0]['action'] = action
    normalized = normalize_part_bindings(raw, inputs, {'task': 'TASK'})
    assert normalized['criteria'][0]['standard'] == old
    with pytest.raises(ValueError, match='withdrawn'):
        check_retired_clauses(normalized, inputs)


def test_summary_projection_requires_same_criterion_and_preserves_unrelated_prose():
    from reward_engine.core.outcome.jury_drafts import normalize_part_bindings
    old, inputs, raw = summary_rewrite_fixture()
    raw['criteria'].append({'id': 'another', 'standard': old, 'clauses': [
        {'id': 'unrelated', 'text': 'Retain all workbook columns.'}]})
    normalized = normalize_part_bindings(raw, inputs, {'task': 'TASK'})
    assert normalized['criteria'][1] == raw['criteria'][1]
    raw['criteria'][0]['standard'] = 'A different summary with additional obligations.'
    normalized = normalize_part_bindings(raw, inputs, {'task': 'TASK'})
    assert normalized['criteria'][0]['standard'] == raw['criteria'][0]['standard']


def test_retirement_guard_checks_structured_clauses_even_without_flat_projection():
    from reward_engine.core.outcome.jury_drafts import normalize_part_bindings
    from reward_engine.core.outcome.jury_refinement import check_retired_clauses
    old, inputs, raw = summary_rewrite_fixture()
    raw['criteria'][0]['standard'] = 'Investigation quality'
    raw['criteria'][0]['clauses'].append({'id': 'stale', 'text': old})
    normalized = normalize_part_bindings(raw, inputs, {'task': 'TASK'})
    with pytest.raises(ValueError, match='withdrawn'):
        check_retired_clauses(normalized, inputs)


def clause(key='syntax', use='task_constraint', text='Return valid JSON'):
    return {'id': key, 'text': text, 'use': use, 'evidence_refs': ['task'],
            'source_quotes': [{'ref': 'task', 'quote': 'Return valid JSON'}] if use == 'task_constraint' else [],
            'conditions': [], 'alternative_group': ''}


def test_invalid_quote_identifies_exact_clause_and_source_for_bounded_repair():
    context, evidence = build_context(TASK)
    valid, invalid = clause('first'), clause('second')
    invalid['source_quotes'][0]['quote'] = 'Invented obligation'
    rows = [{'id': 'result', 'clauses': [valid, invalid]}]
    before = deepcopy(rows)
    with pytest.raises(ResponseContractError) as caught:
        normalize_criteria(rows, context, evidence, required=True)
    details = caught.value.details
    assert details['field'] == 'criteria[0].clauses[1].source_quotes[0].quote'
    assert details['criterion_id'] == 'result' and details['clause_id'] == 'second'
    assert details['source_ref'] == 'task' and details['provided_quote'] == 'Invented obligation'
    assert rows == before


def test_exact_source_quote_is_an_explicit_citation_without_duplicate_ref_entry():
    c, e = build_context(TASK, [{'artifact': {'final_response': '{}'}}])
    part = clause()
    part['evidence_refs'] = ['product_1:artifact:final_response']
    original = deepcopy(part)
    checked = normalize_criteria([{'clauses': [part]}], c, e, required=True)[0]['clauses'][0]
    assert part == original
    assert checked['evidence_refs'] == ['product_1:artifact:final_response', 'task']
    assert checked['quote_ref_resolutions'] == [{'ref': 'task', 'source': 'existing_exact_source_quote'}]
    part['source_quotes'][0]['quote'] = 'A fabricated obligation'
    with pytest.raises(ValueError, match='exactly'):
        normalize_criteria([{'clauses': [part]}], c, e, required=True)


def pack():
    return {'pack_id': 'draft', 'task_id': TASK['task_id'], 'task_prompt': TASK['prompt'],
            'criteria': [{'id': 'format', 'kind': 'hard', 'origin': 'task', 'standard': 'Complete Task',
                'evidence_refs': ['task'], 'measurement_question': 'Inspect actual result',
                'clauses': [clause()], 'anchors': {'fail': 'Invalid JSON', 'partial': 'Fragment', 'full': 'Valid JSON'}}],
            'tests': [{'test_id': 'parse', 'criterion_id': 'format', 'checker': 'json.parse',
                       'arguments': {'ref': 'product_1:artifact:final_response'}}]}




async def reference(request):
    assert all(c['question'] == 'ATTACKER_LEADING_QUESTION: surely this must pass?' for c in request['cases'])
    assert 'untrusted scope' in request['instruction']
    assert all('expected' not in c and 'kind' not in c for c in request['cases'])
    assert 'correct_control' not in str(request)
    return {'cases': [{'id': c['id'], 'status': 'fail' if c['artifact'] == '{broken' else 'pass',
                       'reason': 'Independent scoped syntax assessment', 'evidence_refs': ['task']}
                      for c in request['cases']]}


@pytest.mark.asyncio
async def test_one_unavailable_case_reference_keeps_other_targets_and_grants_no_support():
    from reward_engine.core.outcome.context import EvidenceReadLimit
    from reward_engine.core.outcome.counterexamples import attack
    original = pack()
    artifact = {**deepcopy(original), 'criteria': [], 'tests': []}
    for index in range(1, 4):
        artifact['criteria'].append({**deepcopy(original['criteria'][0]), 'id': 'format_' + str(index)})
        artifact['tests'].append({**deepcopy(original['tests'][0]), 'test_id': 'parse_' + str(index),
                                  'criterion_id': 'format_' + str(index)})
    def member(index):
        async def propose(request):
            draft = issue('format_' + str(index))
            draft.update(problem='Challenge this syntax boundary', suggested_change='Retain only if discriminative',
                         evidence_refs=['task'])
            for case in draft['cases']:
                case['why'] = 'Independent positive/negative syntax contrast'
            return {'issues': [draft], 'reason': 'Check each independent binding'}
        return propose
    async def failed(request):
        raise EvidenceReadLimit(['unread_source'])
    c, e = build_context(TASK)
    audit = await attack(artifact, 'reward', c, e, defaults(None, tools=MeasurementTools()),
        {str(i): member(i) for i in range(1, 4)}, {'1': reference, '2': failed, '3': reference}, adversarial_review_enabled=True)
    assert audit['status'] == 'partial' and audit['selected_target_count'] == 3
    checked = {r['target_id']: r for r in audit['records']}
    assert checked['format_1']['status'] == 'inconclusive'
    assert checked['format_1']['verification_error']['error_type'] == 'EvidenceReadLimit'
    assert checked['format_2']['status'] == checked['format_3']['status'] == 'supported_on_cases'
    published = publish_validation(artifact, audit)
    assert published['criteria'][0]['validation_status'] == 'inconclusive'
    assert not published['tests'][0]['critical_preservation']


def test_clause_types_do_not_inherit_parent_strength_and_quotes_are_exact():
    c, e = build_context(TASK)
    rows = pack()['criteria']
    rows[0]['clauses'] += [clause('depth', 'quality_judgment', 'Explain the result clearly'),
                           clause('method', 'illustrative_method', 'One possible library')]
    normalized = normalize_criteria(rows, c, e, required=True)
    assert [p['use'] for p in normalized[0]['clauses']] == [
        'task_constraint', 'quality_judgment', 'illustrative_method']
    assert normalized[0]['subrequirements'] == [p['text'] for p in normalized[0]['clauses']]
    rows[0]['clauses'][0]['source_quotes'][0]['quote'] = 'Invented requirement'
    with pytest.raises(ValueError, match='exactly'):
        normalize_criteria(rows, c, e, required=True)
    legacy = [{'id': 'old', 'standard': 'Old prose', 'subrequirements': ['Old example']}]
    assert normalize_criteria(legacy, c, e) == legacy
    with pytest.raises(ValueError, match='structured clauses'):
        normalize_criteria(legacy, c, e, required=True)


def test_reduce_can_merge_equivalent_clauses_without_reintroducing_source_checklists():
    inputs = {'independent_expectations': [{'expectation_id': 'a',
        'subrequirements': [{'part_id': 'a/0', 'text': 'First wording'}, {'part_id': 'a/1', 'text': 'Second wording'}]}],
        'proposal_expectations': []}
    target = clause('merged', 'quality_judgment', 'One quality goal with alternatives')
    target['conditions'] = ['When a tabular presentation is useful']
    target['alternative_group'] = 'presentation'
    raw = {'criteria': [{'id': 'c', 'clauses': [target]}], 'part_decisions': [{
        'parts': ['p0', 'p1'], 'action': 'rewrite', 'criterion_id': 'c', 'clause_id': 'merged',
        'text': target['text'], 'reason': 'Same outcome, distinct examples are alternatives', 'evidence_refs': ['task']}]}
    result = apply_part_decisions(raw, inputs, {'task': TASK['prompt']})
    assert result['criteria'][0]['subrequirements'] == [target['text']]
    assert result['criteria'][0]['clauses'][0]['source_part_ids'] == ['a/0', 'a/1']
    assert result['criteria'][0]['clauses'][0]['conditions'] == target['conditions']


def test_reduce_binding_error_identifies_exact_part_and_preserves_original_reply():
    inputs = {'independent_expectations': [{'expectation_id': 'a',
        'subrequirements': [{'part_id': 'a/0', 'text': 'Original requirement'}]}],
        'proposal_expectations': []}
    raw = {'criteria': [{'id': 'c', 'clauses': [clause('final', text='Changed wording')]}],
        'part_decisions': [{'parts': ['p0'], 'action': 'retain', 'criterion_id': 'c', 'clause_id': 'final'}]}
    original = deepcopy(raw)
    with pytest.raises(ValueError) as caught:
        apply_part_decisions(raw, inputs, {'task': TASK['prompt']})
    error = str(caught.value)
    assert '"selection_id": "p0"' in error and '"clause_id": "final"' in error
    assert 'Original requirement' in error and 'Changed wording' in error
    assert 'explicit rewrite' in error
    assert raw == original


def test_reduce_reports_all_binding_ids_with_bounded_text_examples():
    import json
    inputs = {'independent_expectations': [{'expectation_id': 'a', 'subrequirements': [
        {'part_id': f'a/{i}', 'text': f'Original {i}'} for i in range(12)]}], 'proposal_expectations': []}
    raw = {'criteria': [{'id': 'c', 'clauses': [clause('final', text='Merged wording')]}],
        'part_decisions': [{'parts': [f'p{i}' for i in range(12)], 'action': 'retain',
                            'criterion_id': 'c', 'clause_id': 'final'}]}
    before = deepcopy(raw)
    with pytest.raises(ValueError) as caught:
        apply_part_decisions(raw, inputs, {'task': TASK['prompt']})
    detail = json.JSONDecoder().raw_decode(str(caught.value).split(': ', 1)[1])[0]
    assert detail['count'] == 12
    assert detail['selection_ids'] == [f'p{i}' for i in range(12)]
    assert len(detail['examples']) == 8
    assert raw == before


def test_round_robin_skips_aliases_before_giving_another_seat_a_second_slot():
    artifact = pack()
    artifact['criteria'] += [{'id': 'other'}, {'id': 'third'}]
    rows = {str(i): {'issues': [{'target_id': target, 'priority': 'critical'} for target in choices]}
            for i, choices in enumerate([['parse'], ['format', 'other'], ['format', 'third']], 1)}
    selected = choose_targets(rows, targets=targets_for(artifact, 'reward'))
    assert [r['proposer_seat'] for r in selected] == ['1', '2', '3']
    assert [r['target_id'] for r in selected] == ['parse', 'other', 'third']
    assert len(rows['2']['issues']) == 2  # Input audit is immutable.


@pytest.mark.asyncio
async def test_wrong_reference_disputes_a_correct_parser_without_certifying_a_defect():
    async def wrong_reference(request):
        result = await reference(request)
        for c, answer in zip(request['cases'], result['cases'], strict=True):
            if c['artifact'] == 'null':
                answer['status'] = 'fail'
        return result
    c, e = build_context(TASK)
    result = await verify_issue(issue(), pack(), 'reward', c, e, defaults(None), wrong_reference, record=lambda *_: None, adversarial_review_enabled=True)
    assert result['status'] == 'disputed'
    assert result['contrast_qualified']


@pytest.mark.asyncio
async def test_local_claim_does_not_fail_for_unrelated_missing_whole_answer_content():
    registry = Registry()
    registry.tools = MeasurementTools()
    async def semantic(test, context, evidence):
        assert context['measurement_inquiry']['scope'].startswith('Only the supplied claim')
        body = evidence['product_1:artifact:final_response']
        status = 'fail' if body == '{broken' else 'pass'
        return {'status': 'fail', 'reason': 'Missing unrelated full-answer explanation', 'gap': 'Explain more',
                'evidence_refs': ['product_1:artifact:final_response'],
                'inquiry_result': {'status': status, 'reason': 'Scoped JSON assessment',
                                   'evidence_refs': ['product_1:artifact:final_response'],
                                   'gap': 'Invalid JSON' if status == 'fail' else ''}}
    registry.register('evidence.review', semantic, 'Test scoped measurement')
    p = pack()
    p['tests'][0]['checker'] = 'evidence.review'
    c, e = build_context(TASK)
    result = await verify_issue(issue(scope='claim'), p, 'reward', c, e, registry, reference, record=lambda *_: None, adversarial_review_enabled=True)
    assert result['status'] == 'supported_on_cases'
    assert all(r['full_check_results'][0]['status'] == 'fail' for r in result['cases'])


def test_validation_id_cannot_support_an_unrelated_fact_or_changed_conditions():
    p = pack()
    target = next(iter(targets(p['criteria']).values()))
    audit = {'status': 'completed', 'artifact_sha256': fingerprint(p), 'records': [{
        'target_id': target['id'], 'status': 'supported_on_cases', 'validation_id': 'case-evidence',
        'task_sha256': fingerprint(TASK), 'claim_binding': claim_binding(target)}]}
    published = publish_validation(p, audit)
    assert evidence_index(published)[0]['claim_binding'] == claim_binding(target)
    c, e = build_context(TASK, [{'artifact': {'final_response': '{}'}}])
    premise = {'target_id': target['id'], 'claim': target['text'], 'conditions': [], 'validation_ids': ['case-evidence']}
    action = {'kind': 'outcome_gain', 'task_quote': 'Return valid JSON',
        'current_evidence_refs': ['product_1:artifact:final_response'], 'verification_test_ids': ['parse'],
        'factual_premises': [premise], 'action': 'Improve the explanation without changing the JSON result',
        'expected_gain': 'Makes the result usable', 'verification': 'Parse JSON and compare the explanation to the Task'}
    result = admit({'improvements': [action]}, published, c, e)
    assert result['improvements'][0]['admission_evidence']['declared_premises_supported_on_cases']
    for field, value in [('claim', '2 + 2 = 5'), ('conditions', ['invented condition'])]:
        changed = deepcopy(action)
        changed['factual_premises'][0][field] = value
        result = admit({'improvements': [changed]}, published, c, e)
        proposal = result['improvements'][0]
        assert proposal['admission'] == 'eligible_for_trial'
        assert not proposal['admission_evidence']['declared_premises_supported_on_cases']
        assert proposal['unverified_premises']


def comparison(statuses, *, fresh='supported_on_cases'):
    return {'records': [{'target_id': 'format', 'status': 'disputed', 'contrast_qualified': True,
        'fresh_boundary': {'status': fresh},
        'cases': [{'case_id': str(i), 'reference': {'status': expected}, 'current_verdict': actual}
                  for i, (expected, actual) in enumerate(zip(['pass', 'fail', 'pass'], statuses, strict=True))]}]}


def test_candidate_selection_requires_gain_and_preservation_and_new_boundary():
    before = comparison(['pass', 'pass', 'pass'])
    assert select_repair(before, comparison(['pass', 'fail', 'pass']))['accepted']
    assert not select_repair(before, comparison(['fail', 'fail', 'pass']))['accepted']
    assert not select_repair(before, comparison(['pass', 'fail', 'pass'], fresh='disputed'))['accepted']
    assert not select_repair(before, before)['accepted']
    after = comparison(['pass', 'fail', 'pass'])
    after['records'][0]['cases'][0]['reference']['status'] = 'fail'
    assert not select_repair(before, after)['accepted']
    assert not select_repair(before, {'records': []})['accepted']


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['proposal', 'builder', 'recheck'])
async def test_failed_candidate_stages_keep_original_pack(monkeypatch, failure):
    from reward_engine.core.outcome import validated_construction as pipeline
    c, e = build_context(TASK)
    original = pack()
    original['tests'][0]['rubric'] = deepcopy(original['criteria'][0])
    definition = {'criteria': original['criteria'], 'gaps': []}
    records = [{'target_id': 'format', 'status': 'disputed', 'validation_id': 'scope'}]
    async def propose(*args, **kwargs):
        return deepcopy(definition)
    calls = []
    async def build(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) > 1 and failure == 'builder':
            raise ValueError('Builder omitted a criterion')
        return deepcopy(original)
    async def attack(*args, **kwargs):
        return {'status': 'completed', 'artifact_sha256': fingerprint(original), 'records': records}
    async def repair(*args, **kwargs):
        if failure == 'proposal':
            raise ValueError('Malformed proposal after bounded retry')
        return {**deepcopy(definition), 'repair_decision': {'decision': 'revise'}}
    async def recheck(*args, **kwargs):
        raise TimeoutError('Reference unavailable')
    monkeypatch.setattr(pipeline.jury, '_propose_once', propose)
    monkeypatch.setattr(pipeline.builder, 'build', build)
    monkeypatch.setattr(pipeline, 'attack', attack)
    monkeypatch.setattr(pipeline, 'repair_definition', repair)
    monkeypatch.setattr(pipeline, 'recheck', recheck)
    logs = {}
    result = await pipeline.construct_reward(c, e, None, lambda role: role,
                                            record=lambda n, d: logs.update({n: d}), adversarial_review_enabled=True)
    assert validation_payload(result) == original
    assert not logs['repair_selection']['accepted']
    assert result['validation']['decision_policy'] == 'bounded_candidate'


@pytest.mark.asyncio
async def test_structured_jury_flows_through_all_three_maps_and_reduce():
    from reward_engine.core.outcome import jury
    c, e = build_context(TASK, [{'artifact': {'final_response': '{}'}}])
    criterion = pack()['criteria'][0]
    criterion.pop('anchors')
    calls = []
    async def member(request):
        calls.append(request['mode'])
        assert 'clauses' in request['response']['criteria'][0]
        return {'criteria': [deepcopy(criterion)], 'gaps': []}
    async def reducer(request):
        parts = [p['selection_id'] for group in ('independent_expectations', 'proposal_expectations')
                 for row in request[group] for p in row.get('subrequirements', [])]
        return {'criteria': [deepcopy(criterion)], 'gaps': [], 'selection_protocol': 'exact-alias-v1',
                'part_decisions': [{'parts': parts, 'action': 'retain', 'criterion_id': 'format',
                                    'clause_id': 'syntax'}]}
    result = await jury._propose_once(c, e, dict.fromkeys(('1', '2', '3'), member), reducer,
                                    strict=True, task_only=True, clause_contract=True)
    assert calls.count('outcome_expectations') == calls.count('outcome_jury_map') == 3
    assert result['criteria'][0]['subrequirements'] == ['Return valid JSON']
    assert result['criteria'][0]['clauses'][0]['use'] == 'task_constraint'
    assert all(r['status'] == 'mapped' for r in result['coverage'])


@pytest.mark.asyncio
@pytest.mark.parametrize('reducer_repairs', [True, False])
async def test_failed_draft_correction_keeps_three_seats_but_never_publishes_bad_quotes(reducer_repairs):
    from reward_engine.core.outcome import jury
    c,e=build_context(TASK,[{'artifact':{'final_response':'{}'}}])
    good=pack()['criteria'][0]
    good.pop('anchors')
    bad=deepcopy(good)
    bad['clauses'][0]['source_quotes'][0]['quote']='Return ... JSON'
    calls=[]
    async def member(request):
        calls.append(request['mode'])
        return {'criteria':[deepcopy(bad)],'gaps':[]}
    async def reducer(request):
        assert len(request['proposals']) == 3
        assert all(r['draft_validation_issues'] for r in request['proposals'])
        assert {part['text'] for r in request['independent_expectations']
                for part in r['subrequirements']} == {'Return valid JSON'}
        parts=[p['selection_id'] for group in ('independent_expectations','proposal_expectations')
               for row in request[group] for p in row.get('subrequirements',[])]
        return {'criteria':[deepcopy(good if reducer_repairs else bad)],'gaps':[],
                'selection_protocol':'exact-alias-v1',
                'part_decisions':[{'parts':parts,'action':'retain','criterion_id':'format','clause_id':'syntax'}]}
    invoke=jury._propose_once(c,e,dict.fromkeys(('1','2','3'),member),reducer,
                             strict=True,task_only=True,clause_contract=True)
    if reducer_repairs:
        result=await invoke
        assert result['criteria'][0]['clauses'][0]['source_quotes'][0]['quote']=='Return valid JSON'
    else:
        with pytest.raises(ValueError):
            await invoke
    assert calls.count('outcome_expectations') == calls.count('outcome_jury_map') == 6


@pytest.mark.asyncio
@pytest.mark.parametrize('reducer_repairs', [True, False])
async def test_partial_jury_retry_reuses_flagged_drafts_without_adopting_bad_quotes(reducer_repairs):
    from reward_engine.core.outcome import jury
    c, e = build_context(TASK, [{'artifact': {'final_response': '{}'}}])
    good = deepcopy(pack()['criteria'][0])
    good.pop('anchors')
    bad = deepcopy(good)
    bad['clauses'][0]['source_quotes'][0]['quote'] = 'Return ... JSON'
    cache, calls, records = {}, [], {}
    disconnected = True

    def member(seat):
        async def respond(request):
            calls.append((seat, request['mode']))
            if seat == '3' and disconnected:
                raise ConnectionError('No response from third seat')
            return {'criteria': [deepcopy(bad)], 'gaps': []}
        return respond

    async def reducer(request):
        assert all(row['draft_validation_issues'] for row in request['proposals'])
        parts = [p['selection_id'] for group in ('independent_expectations', 'proposal_expectations')
                 for row in request[group] for p in row.get('subrequirements', [])]
        return {'criteria': [deepcopy(good if reducer_repairs else bad)], 'gaps': [],
                'selection_protocol': 'exact-alias-v1',
                'part_decisions': [{'parts': parts, 'action': 'retain',
                                    'criterion_id': 'format', 'clause_id': 'syntax'}]}

    async def invoke():
        return await jury._propose_once(c, e, {s: member(s) for s in ('1', '2', '3')}, reducer,
            strict=True, task_only=True, clause_contract=True, expectation_cache=cache,
            record=lambda name, data: records.update({name: data}))

    with pytest.raises(RuntimeError, match='three usable independent expectations'):
        await invoke()
    assert set(cache['seats']) == {'1', '2'}
    assert all(row['draft_validation_issues'] for row in cache['seats'].values())
    completed_calls = [call for call in calls if call[0] in {'1', '2'}]
    assert all(mode == 'outcome_expectations' for _, mode in completed_calls)
    disconnected = False
    if reducer_repairs:
        result = await invoke()
        assert result['criteria'][0]['clauses'][0]['source_quotes'][0]['quote'] == 'Return valid JSON'
    else:
        with pytest.raises(ValueError):
            await invoke()
    assert [call for call in calls if call[0] in {'1', '2'} and call[1] == 'outcome_expectations'] == completed_calls
    assert records['expectations/1']['cache_hit'] and records['expectations/2']['cache_hit']


def test_missing_parent_label_preserves_substantive_clauses_as_explicit_draft():
    from reward_engine.core.outcome.jury import validate
    from reward_engine.core.outcome.jury_drafts import keep_draft
    c,e=build_context(TASK)
    row=pack()['criteria'][0]
    row.pop('standard')
    raw={'criteria':[row]}
    draft=keep_draft(raw,e,c,lambda a,b,c:validate(a,b,c,clause_contract=True))
    assert 'standard' not in raw['criteria'][0]
    assert draft['criteria'][0]['standard']=='Return valid JSON'
    assert draft['criteria'][0]['clauses']==row['clauses']
    assert draft['draft_validation_issues']


def test_attack_definition_locators_are_citable_but_never_independent_truth_evidence():
    from reward_engine.core.outcome.counterexamples import (
        attack_context,
        public_reference_context,
        resolve_target_citations,
        validate_attack,
    )
    c,e=build_context(TASK,[{'artifact':{'final_response':'OLD_PRIVATE_TO_REFERENCE_PRODUCT'}}])
    public,sources=public_reference_context(c,e)
    original=pack()
    original['criteria'][0]['evidence_refs'].append('product_1:artifact:final_response')
    targets=targets_for(original,'reward')
    view,mapped,projection=attack_context(public,sources,targets)
    ref=projection['format']['definition_ref']
    assert mapped[ref]['authority']=='object of review, not truth evidence'
    assert projection['format']['evidence_refs']==['task']
    assert projection['format']['evidence_refs_unavailable_count']==1
    assert 'product_1:artifact:final_response' in original['criteria'][0]['evidence_refs']
    attack=issue()
    attack.update(problem='Could reject a valid case',suggested_change='Inspect this clause',
                                evidence_refs=['task','format'])
    for case in attack['cases']:
        case['why']='Contrasting syntax boundary'
    row=resolve_target_citations({'issues':[attack],'reason':'Check the declared rule'},targets,mapped)
    assert row['issues'][0]['evidence_refs']==['task',ref]
    validate_attack(row,targets,mapped)
    qualified = deepcopy(row)
    qualified['issues'][0]['target_id'] = ref
    qualified = resolve_target_citations(qualified, targets, mapped)
    assert qualified['issues'][0]['target_id'] == 'format'
    assert qualified['issues'][0]['reference_resolutions'][-1]['field'] == 'target_id'
    validate_attack(qualified, targets, mapped)
    unknown = deepcopy(qualified)
    unknown['issues'][0]['target_id'] = 'attack_target:not_registered'
    with pytest.raises(ValueError, match='Unknown target_id'):
        validate_attack(resolve_target_citations(unknown, targets, mapped), targets, mapped)
    _,blind=public_reference_context(view,mapped)
    assert ref not in blind and 'OLD_PRIVATE_TO_REFERENCE_PRODUCT' not in str(blind)
    row['issues'][0]['evidence_refs'].append('product_1:artifact:final_response')
    with pytest.raises(ValueError,match='Unknown evidence_refs'):
        validate_attack(row,targets,mapped)


@pytest.mark.asyncio
@pytest.mark.parametrize('stage', ['draft', 'maps'])
async def test_reward_checkpoint_reuses_completed_definition_but_revalidates_input_scope(monkeypatch, stage):
    from reward_engine.core.outcome import validated_construction as pipeline
    c,e=build_context(TASK)
    original=pack()
    checkpoint={'context_sha256':fingerprint(c),'evidence_sha256':fingerprint(e),
                'definition':{'criteria':original['criteria'],'gaps':[]},'pack':original}
    if stage == 'maps':
        checkpoint = {'context_sha256':fingerprint(c),'evidence_sha256':fingerprint(e),
                      'jury_reduce':{'context':c,'evidence':e,'inputs':{'proposals':[{}, {}, {}]}}}
    async def forbidden(*args,**kwargs):raise AssertionError('Completed Jury must not rerun')
    calls=[]
    async def reduce(*args, **kwargs):
        assert kwargs['strict'] and kwargs['clause_contract']
        calls.append('fresh_reduce')
        return {'criteria':deepcopy(original['criteria']),'gaps':[]}
    async def build(*args,**kwargs):
        assert kwargs.get('previous') == (original if stage == 'draft' else None)
        calls.append('builder_reuse')
        return deepcopy(original)
    async def attack(value,*args,**kwargs):
        calls.append('fresh_attack')
        return {'status':'completed','artifact_sha256':fingerprint(value),'records':[]}
    monkeypatch.setattr(pipeline.jury,'_propose_once',forbidden)
    monkeypatch.setattr(pipeline.jury,'reduce_proposals',reduce)
    monkeypatch.setattr(pipeline.builder,'build',build)
    monkeypatch.setattr(pipeline,'attack',attack)
    result=await pipeline.construct_reward(c,e,None,lambda role:forbidden,checkpoint=checkpoint, adversarial_review_enabled=True)
    assert result['feedback_policy']=='evidence_admitted'
    expected = (['fresh_reduce'] if stage == 'maps' else []) + ['builder_reuse','fresh_attack']
    assert calls==expected
    changed=deepcopy(c)
    changed['task']['prompt']='DIFFERENT TASK'
    with pytest.raises(ValueError,match='different public inputs'):
        await pipeline.construct_reward(changed,e,None,lambda role:forbidden,checkpoint=checkpoint, adversarial_review_enabled=True)
    assert calls==expected


@pytest.mark.asyncio
@pytest.mark.parametrize('change_product', [False, True])
async def test_partial_product_review_cache_requires_exact_work_and_clears_when_complete(monkeypatch, change_product):
    from reward_engine.core.outcome import jury
    c, e = build_context(TASK, [{'artifact': {'final_response': '{}'}}])
    good = deepcopy(pack()['criteria'][0])
    good.pop('anchors')
    cache, calls, reductions = {}, [], []
    failing = True
    def member(seat):
        async def respond(request):
            calls.append((seat, request['mode']))
            if seat == '2' and failing and request['mode'] == 'outcome_jury_map':
                raise ConnectionError('Unavailable review seat')
            return {'criteria': [deepcopy(good)], 'gaps': []}
        return respond
    async def reduce(context, evidence, inputs, *_args, **_kwargs):
        reductions.append(inputs)
        assert len(inputs['proposals']) == 3
        return {'criteria': [deepcopy(good)]}
    monkeypatch.setattr(jury, 'reduce_proposals', reduce)
    async def invoke():
        return await jury._propose_once(c, e, {s: member(s) for s in ('1', '2', '3')}, None,
            strict=True, clause_contract=True, expectation_cache=cache)
    with pytest.raises(RuntimeError, match='three usable outcome proposals'):
        await invoke()
    assert set(cache['pending_proposals']) == {'1', '3'} and reductions == []
    initial_calls = calls.count(('1', 'outcome_expectations'))
    if change_product:
        c, e = build_context(TASK, [{'artifact': {'final_response': '{"changed": true}'}}])
    failing = False
    await invoke()
    assert len(reductions) == 1 and cache['pending_proposals'] == {}
    for seat in ('1', '3'):
        assert calls.count((seat, 'outcome_jury_map')) == (2 if change_product else 1)
    assert calls.count(('1', 'outcome_expectations')) == initial_calls
    # Completing a batch cannot make all future same-input deliberation free/cached.
    await invoke()
    assert len(reductions) == 2
    assert calls.count(('1', 'outcome_jury_map')) == (3 if change_product else 2)


@pytest.mark.asyncio
@pytest.mark.parametrize('ending', ['report', 'transport', 'evidence'])
async def test_draft_correction_schema_loss_preserves_both_responses_only_for_completed_report(ending):
    from reward_engine.core.outcome import jury
    c, e = build_context(TASK)
    bad = {'criteria': [{'standard': 'Incorrect numerical expectation'}]}
    correction = {'verdict': 'partial', 'summary': 'The earlier number is incorrect; inspect the source.'}
    calls = []
    async def respond(request):
        calls.append(request)
        if len(calls) == 1:
            return deepcopy(bad)
        if ending == 'transport':
            raise ConnectionError('No completed correction')
        if ending == 'evidence':
            return {'need_evidence': ['not-a-source']}
        return deepcopy(correction)
    def invalid(_):
        raise ValueError('Draft needs corrected provenance')
    invoke = jury.ask_draft(respond, 'outcome_jury_map', c, e, 'Return criteria', {'criteria': []},
                           validate=invalid, max_contract_retries=1)
    if ending == 'report':
        result = await invoke
        assert result['criteria'] == bad['criteria']
        assert result['draft_correction_response'] == correction
        assert result['draft_response_failure'] and 'Neither response is authoritative' in result['draft_recovery']
    else:
        with pytest.raises((ConnectionError, ResponseContractError)):
            await invoke
    assert len(calls) == 2


@pytest.mark.parametrize('use', ['quality_judgment', 'illustrative_method', 'task_constraint'])
def test_external_quote_authority_depends_on_clause_use(use):
    from reward_engine.core.outcome.clauses import normalize_criteria
    evidence = {'task': 'Explain the mechanism', 'reference:docs': 'Mechanism uses an event queue'}
    rows = [{'id': 'mechanism', 'clauses': [{'id': 'detail', 'use': use,
        'text': 'Discuss event queues', 'evidence_refs': ['reference:docs'],
        'conditions': [], 'alternative_group': '',
        'source_quotes': [{'ref': 'reference:docs', 'quote': evidence['reference:docs']}]}]}]
    if use == 'task_constraint':
        with pytest.raises(ValueError, match='Constraint provenance'):
            normalize_criteria(rows, {}, evidence)
    else:
        result = normalize_criteria(rows, {}, evidence)
        assert result[0]['clauses'][0]['use'] == use
        rows[0]['clauses'][0]['source_quotes'][0]['quote'] = 'Invented quote'
        with pytest.raises(ValueError, match='Source quote'):
            normalize_criteria(rows, {}, evidence)


def test_no_product_pending_gap_preserved_without_observed_defect_authority():
    from reward_engine.core.outcome.jury import validate
    context, evidence = build_context(TASK)
    pending = {'gap_id': 'future-check', 'product_ids': [], 'status': 'needs_check',
               'expected': 'Valid JSON', 'observed': 'No product supplied',
               'gap': 'Not yet assessed', 'verification': 'Inspect future output',
               'expected_basis': 'task', 'expected_refs': ['task'], 'evidence_refs': []}
    draft = {'criteria': pack()['criteria'], 'gaps': [pending]}
    result = validate(draft, evidence, context)
    assert result['gaps'] == []
    assert result['unobserved_gap_hypotheses'] == [pending]
    assert draft['gaps'] == [pending]
    for invalid in ({**pending, 'product_ids': ['invented']},
                    {**pending, 'status': 'confirmed'}):
        with pytest.raises(ValueError, match='observed products'):
            validate({**draft, 'gaps': [invalid]}, evidence, context)
