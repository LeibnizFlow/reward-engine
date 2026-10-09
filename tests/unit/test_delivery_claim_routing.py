"""Domain-independent uncertainty routing, not a test of model truthfulness."""

from copy import deepcopy

import pytest

from reward_engine.core.outcome.delivery_triage import route_claim_checks
from reward_engine.core.outcome.harness import defaults
from reward_engine.core.outcome.runner import DeliveryReview

REF = 'product_1:artifact:final_response'


def response(claim='The owner accepted', test_id=None):
    return {'outcome_goal': 'Fully satisfy the declared task with a correct, complete result.', 'full_score_gaps': [], 'improvements': [], 'status': 'target_satisfied', 'reason': 'Review complete', 'gaps': [], 'verify': [],
            'claim_checks': [{'claim': claim, 'claim_ref': REF, 'support_refs': [],
                'assessment': 'needs_verification', 'reason': 'No acceptance receipt observed',
                'test_id': test_id}]}


def test_no_gap_cannot_erase_unmeasurable_doubt():
    row = response()
    route_claim_checks(row, {REF: 'The owner accepted. Please verify later.'}, set())
    assert row['status'] == 'unknown' and row['verify'] == []
    assert 'The owner accepted' in row['reason']


def test_claim_selects_real_check_once_even_when_overview_says_no_gap():
    row = response(test_id='acceptance')
    route_claim_checks(row, {REF: 'The owner accepted'}, {'acceptance'})
    route_claim_checks(row, {REF: 'The owner accepted'}, {'acceptance'})
    assert len(row['verify']) == 1 and row['verify'][0]['test_id'] == 'acceptance'


@pytest.mark.parametrize('change', [
    {'claim': ''}, {'claim_ref': 'task'},
    {'test_id': 'invented'}, {'support_refs': ['missing']},
])
def test_invalid_claim_evidence_or_executor_is_not_accepted(change):
    row = response()
    row['claim_checks'][0].update(change)
    with pytest.raises(ValueError):
        route_claim_checks(row, {REF: 'The owner accepted'}, set())


@pytest.mark.parametrize('assessment', ['supported', 'not_required'])
def test_derivation_or_design_does_not_force_tools(assessment):
    row = response('2+2=4')
    row['claim_checks'][0].update(assessment=assessment, reason='Direct arithmetic suffices')
    route_claim_checks(row, {REF: '2+2=4'}, set())
    assert row['status'] == 'target_satisfied' and row['verify'] == []


def test_translation_is_not_an_infra_error():
    row = response('The owner accepted')
    row['claim_checks'][0].update(assessment='supported', reason='Receipt is present')
    route_claim_checks(row, {REF: '负责人已接手'}, set())
    assert row['status'] == 'target_satisfied'
    # This tests source plumbing only, not whether the semantic judgment is true.


@pytest.mark.asyncio
async def test_checker_failure_does_not_manufacture_actor_revision():
    async def broken(_):
        raise ConnectionError('review unavailable')
    task = {'task_id': 'x', 'prompt': 'Write a draft'}
    pack = {'task_id': 'x', 'task_prompt': task['prompt'], 'pack_id': 'v1', 'criteria': [], 'tests': []}
    review = DeliveryReview(task, pack, defaults(broken))
    assert await review({'artifact': {'final_response': 'Draft ready'}}) is None
    assert review.status == 'check_unavailable' and review.feedback == []
    assert not review.progress.verified


@pytest.mark.asyncio
async def test_admitted_doubt_runs_harness_and_failure_reaches_actor():
    task = {'task_id': 'json', 'prompt': 'Deliver valid JSON'}
    pack = {'task_id': 'json', 'task_prompt': task['prompt'], 'pack_id': 'v1',
            'criteria': [{'id': 'syntax', 'kind': 'hard', 'standard': 'Valid JSON'}],
            'tests': [{'test_id': 'parse', 'criterion_id': 'syntax', 'checker': 'json.parse',
                       'arguments': {'ref': REF}}]}
    row = response('broken', 'parse')
    async def model(_):
        return deepcopy(row)
    review = DeliveryReview(task, pack, defaults(model))
    feedback = await review({'artifact': {'final_response': 'broken'}})
    assert feedback and 'Invalid JSON' in feedback
    assert review.history[0]['results'][0]['status'] == 'fail'
    assert review.progress.for_actor()['remaining'][-1]['claim_checks'][0]['claim'] == 'broken'


def four_claims():
    row = response('2+2=4')
    row['claim_checks'][0].update(assessment='supported', reason='Direct arithmetic')
    row['claim_checks'] = [deepcopy(row['claim_checks'][0]) for _ in range(4)]
    return row


@pytest.mark.asyncio
async def test_extra_valid_claim_is_preserved_without_forced_revision():
    answer = four_claims()
    calls = []
    async def model(request):
        calls.append(request)
        return deepcopy(answer)
    task = {'task_id': 'math', 'prompt': 'Give a simple arithmetic example'}
    pack = {'task_id': 'math', 'task_prompt': task['prompt'], 'pack_id': 'v1', 'criteria': [], 'tests': []}
    review = DeliveryReview(task, pack, defaults(model))
    assert await review({'artifact': {'final_response': '2+2=4'}}) is None
    assert len(calls) == 1 and review.status == 'full_score_target_satisfied'
    assert review.feedback == [] and review.progress.verified
    measured = review.history[0]
    assert measured['delivery_review']['claim_checks'] == answer['claim_checks']
    assert measured['review_batch']['complete'] and measured['results'] == []
    assert measured['review_diagnostics'] == [{'reason': 'claim_count_above_guidance',
        'suggested': 3, 'received': 4, 'all_claims_preserved': True}]


@pytest.mark.asyncio
@pytest.mark.parametrize('change,field', [
    ({'claim_ref': 'missing'}, 'claim_ref'), ({'support_refs': ['missing']}, 'support_refs'),
    ({'test_id': 'invented'}, 'test_id'), ({'assessment': ['supported']}, 'assessment'),
])
async def test_invalid_fourth_claim_retains_precise_safe_diagnostic(change, field):
    answer = four_claims()
    answer['claim_checks'][-1].update(change)
    async def model(_):
        return deepcopy(answer)
    task = {'task_id': 'math', 'prompt': 'Give an arithmetic example'}
    pack = {'task_id': 'math', 'task_prompt': task['prompt'], 'pack_id': 'v1', 'criteria': [], 'tests': []}
    review = DeliveryReview(task, pack, defaults(model))
    assert await review({'artifact': {'final_response': '2+2=4'}}) is None
    assert review.status == 'check_unavailable' and not review.feedback
    measured = review.history[0]
    assert not measured['delivery_clear']
    failure = measured['delivery_review']
    assert failure['error_kind'] == 'invalid_review_output'
    assert failure['diagnostic']['field'] == 'claim_checks[3].' + field
    assert measured['review_batch']['items'][0]['error_details'] == failure['diagnostic']


@pytest.mark.asyncio
async def test_claim_count_does_not_increase_actual_measurement_budget():
    answer = four_claims()
    for index, claim in enumerate(answer['claim_checks']):
        claim.update(assessment='needs_verification', test_id=f'parse{index}', reason='Check serialization')
    async def model(_):
        return deepcopy(answer)
    task = {'task_id': 'json', 'prompt': 'Return JSON'}
    pack = {'task_id': 'json', 'task_prompt': task['prompt'], 'pack_id': 'v1',
        'criteria': [{'id': 'syntax', 'kind': 'hard', 'standard': 'Valid JSON'}],
        'tests': [{'test_id': f'parse{i}', 'criterion_id': 'syntax', 'checker': 'json.parse',
                   'arguments': {'ref': REF}} for i in range(4)]}
    review = DeliveryReview(task, pack, defaults(model), max_tests=3)
    await review({'artifact': {'final_response': '{}'}})
    measured = review.history[0]
    assert [row['test_id'] for row in measured['results']] == ['parse0', 'parse1', 'parse2']
    assert measured['deferred_tests'] == ['parse3'] and not measured['delivery_clear']
    assert len(measured['delivery_review']['claim_checks']) == 4
