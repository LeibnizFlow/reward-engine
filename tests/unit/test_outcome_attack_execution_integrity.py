"""Failed measurement drafts cannot certify checks; revealed user requests survive QC."""

from tests.support.outcome_attack_execution_integrity import (
    issue,
    json,
    parser_reference,
)

from copy import deepcopy
from types import SimpleNamespace

import pytest
from tests.support.claw_outcome_feedback_arms import PACK, TASK, candidate

from reward_engine.core.outcome.context import build_context
from reward_engine.core.outcome.counterexamples import (
    admit_attack_draft,
    bind_case_dialogue,
    dialogue_contract,
    public_reference_context,
    validate_attack,
    verify_issue,
)
from reward_engine.core.outcome.measurement_tools import MeasurementTools




def test_missing_redundant_attack_reason_preserves_supplied_problem_and_validation():
    proposal = {'issues': [issue()]}
    original = deepcopy(proposal)
    result = admit_attack_draft(proposal, {'syntax': {}}, {'task': 'Return valid JSON'})
    assert proposal == original
    assert result['issues'] == proposal['issues']
    assert result['reason'] == proposal['issues'][0]['problem']
    assert result['envelope_recovery']['source'] == 'issues[].problem'
    proposal['issues'][0]['cases'][1]['expected'] = 'pass'
    with pytest.raises(ValueError, match='error_variant'):
        admit_attack_draft(proposal, {'syntax': {}}, {'task': 'Return valid JSON'})


@pytest.mark.parametrize('row', [{'issues': []}, {'issues': [], 'reason': ' '},
                               {'issues': [issue()], 'reason': 5}])
def test_attack_reason_cannot_be_invented_without_supplied_rationale(row):
    with pytest.raises(ValueError, match='reason'):
        admit_attack_draft(row, {'syntax': {}}, {'task': 'Return valid JSON'})




@pytest.mark.asyncio
@pytest.mark.parametrize('scope', ['claim', 'whole_check'])
@pytest.mark.parametrize('rejected_status', ['pass', 'fail'])
async def test_errored_checker_draft_never_becomes_support_or_dispute(scope, rejected_status):
    context, evidence = build_context(TASK, [candidate('PRIVATE_ACTOR_ANSWER')])
    pack = deepcopy(PACK)
    pack['tests'][0]['checker'] = 'evidence.review'

    async def run(test, c, e):
        text = c['products'][0]['artifact']['final_response']
        verdict = 'fail' if text == '{' else 'pass'
        if text == '{}':
            return {'test_id': test['test_id'], 'criterion_id': 'syntax', 'status': 'unknown',
                    'error': True, 'diagnostic': 'Unresolved material claims cannot silently become pass',
                    'inquiry_result': {'status': rejected_status, 'reason': 'Rejected model interpretation'}}
        return {'test_id': test['test_id'], 'criterion_id': 'syntax', 'status': verdict,
                'inquiry_result': {'status': verdict, 'reason': 'Executed parser'}}

    result = await verify_issue(issue(scope), pack, 'reward', context, evidence,
                                SimpleNamespace(run=run, tools=MeasurementTools()), parser_reference,
                                record=lambda *_: None, adversarial_review_enabled=True)
    failed = next(c for c in result['cases'] if c['case_kind'] == 'correct_control')
    assert failed['current_verdict'] == 'unknown'
    assert not failed['disagreement']
    assert failed['full_check_results'][0]['inquiry_result']['status'] == rejected_status
    assert failed['current_pack_results'][0]['error'] is True
    assert result['status'] == 'inconclusive'


def dialogue_context():
    products = []
    for value in ('FIRST_BRANCH_ONLY', 'SECOND_BRANCH_ONLY'):
        c = candidate('PRIVATE_ACTOR_ANSWER')
        c['artifact']['public_consultation'] = {'earlier_exchanges': [
            {'speaker': 'assistant', 'text': 'PRIVATE_EARLIER_ANSWER'},
            {'speaker': 'user', 'text': value},
            {'speaker': 'assistant', 'text': 'PRIVATE_LATER_ANSWER'}]}
        products.append(c)
    return build_context(TASK, products)


@pytest.mark.parametrize('placeholder', ['', None, 'None', 'n/a', 'product_1', 'product_999'])
def test_task_only_case_binding_is_engine_owned_without_changing_material(placeholder):
    context, evidence = build_context(TASK, [candidate('PRIVATE_ACTOR_ANSWER')])
    view, sources = public_reference_context(context, evidence)
    schema, instruction = dialogue_contract(view)
    assert 'user_context_product_id' not in schema
    assert 'no actual user amendments' in instruction
    proposal = issue()
    for case in proposal['cases']:
        case['user_context_product_id'] = placeholder
    original = deepcopy(proposal)
    admitted = validate_attack({'issues': [proposal], 'reason': 'Check syntax'}, {'syntax': {}}, sources)
    assert proposal == original  # Raw response remains available for diagnostics.
    for actual, expected in zip(admitted['issues'][0]['cases'], original['cases'], strict=True):
        expected = {k: v for k, v in expected.items() if k != 'user_context_product_id'}
        assert actual == expected
    # This does not turn unavailable historical sources into usable evidence.
    proposal['evidence_refs'] = ['product_1:artifact']
    with pytest.raises(ValueError, match='Unknown evidence_refs'):
        validate_attack({'issues': [proposal], 'reason': 'Check syntax'}, {'syntax': {}}, sources)


@pytest.mark.parametrize('wrong', [None, '', 'n/a', 'product_3', ['product_1', 'product_2']])
def test_actual_dialogue_contract_never_silently_selects_or_combines_branches(wrong):
    context, evidence = dialogue_context()
    view, _ = public_reference_context(context, evidence)
    schema, _ = dialogue_contract(view)
    assert schema['user_context_product_id']['choose_exactly_one'] == ['product_1', 'product_2']
    with pytest.raises(ValueError, match='exact actual user dialogue'):
        bind_case_dialogue({'user_context_product_id': wrong}, {'product_1', 'product_2'})


@pytest.mark.asyncio
async def test_task_only_recheck_normalizes_stale_binding_without_inventing_user_messages():
    from reward_engine.core.outcome.validated_construction import recheck

    context, evidence = build_context(TASK, [candidate('PRIVATE_ACTOR_ANSWER')])
    proposal = issue('whole_check')
    for case in proposal['cases']:
        case['user_context_product_id'] = 'product_1'

    async def reference(request):
        assert not request['context']['products']
        assert all('user_context_product_id' not in c for c in request['cases'])
        assert 'PRIVATE_ACTOR_ANSWER' not in str(request)
        return await parser_reference(request)

    async def run(test, c, e):
        assert not c['products'][0]['user_updates']
        text = c['products'][0]['artifact']['final_response']
        try:
            json.loads(text)
            status = 'pass'
        except ValueError:
            status = 'fail'
        return {'test_id': test['test_id'], 'criterion_id': 'syntax', 'status': status}

    registry = SimpleNamespace(run=run, tools=MeasurementTools())
    first = await verify_issue(proposal, PACK, 'reward', context, evidence, registry, reference,
                               record=lambda *_: None, adversarial_review_enabled=True)
    calls = []

    async def fresh(request):
        calls.append(request)
        assert 'user_context_product_id' not in request['response']['case']
        return {'case': {'id': 'fresh', 'artifact': '42', 'question': 'Valid JSON?',
                         'why': 'Numeric JSON literal', 'expected': 'pass', 'scope': 'whole_check',
                         'probes': [], 'user_context_product_id': 'n/a'}}

    result = await recheck({'records': [first]}, PACK, 'reward', context, evidence, registry,
                          {str(i): reference for i in range(1, 4)},
                          {str(i): fresh for i in range(1, 4)}, record=lambda *_: None, adversarial_review_enabled=True)
    assert len(calls) == 1
    assert result['rechecked'] is True
    checked = next(r for r in result['records'] if r.get('fresh_boundary'))
    assert checked['status'] == 'supported_on_cases'
    assert checked['fresh_boundary']['status'] == 'supported_on_cases'


def test_public_attack_context_keeps_actual_dialogue_evidence_only_and_is_idempotent():
    context, evidence = dialogue_context()
    view, sources = public_reference_context(context, evidence)
    assert [[u['text'] for u in p['user_updates']] for p in view['products']] == [
        ['FIRST_BRANCH_ONLY'], ['SECOND_BRANCH_ONLY']]
    assert set(sources) == {'task', 'product_1:public:1', 'product_2:public:1'}
    assert 'PRIVATE_' not in str(view) + str(sources)
    assert public_reference_context(view, sources) == (view, sources)
    altered = deepcopy(evidence)
    altered['product_1:public:1'] = 'DIFFERENT_REQUIREMENT'
    with pytest.raises(ValueError, match='must match'):
        public_reference_context(context, altered)


@pytest.mark.asyncio
async def test_reference_and_checker_receive_same_selected_dialogue_without_mixing_branches():
    context, evidence = dialogue_context()
    pack = deepcopy(PACK)
    pack['tests'][0]['checker'] = 'evidence.review'
    proposal = issue()
    for case in proposal['cases']:
        case['user_context_product_id'] = 'product_2'
    _, sources = public_reference_context(context, evidence)
    validate_attack({'issues': [proposal], 'reason': 'Check syntax'}, {'syntax': {}}, sources)

    async def reference(request):
        assert all(c['user_context_product_id'] == 'product_2' for c in request['cases'])
        assert 'PRIVATE_' not in str(request)
        assert request['context']['products'][1]['user_updates'][0]['text'] == 'SECOND_BRANCH_ONLY'
        return await parser_reference(request)

    seen = []
    async def run(test, c, e):
        seen.append(c)
        assert [u['text'] for u in c['products'][0]['user_updates']] == ['SECOND_BRANCH_ONLY']
        assert 'FIRST_BRANCH_ONLY' not in str(c) + str(e)
        assert 'PRIVATE_' not in str(c) + str(e)
        text = c['products'][0]['artifact']['final_response']
        verdict = 'fail' if text == '{' else 'pass'
        return {'test_id': test['test_id'], 'criterion_id': 'syntax', 'status': verdict,
                'inquiry_result': {'status': verdict, 'reason': 'Actual parser'}}

    result = await verify_issue(proposal, pack, 'reward', context, evidence,
                                SimpleNamespace(run=run, tools=MeasurementTools()), reference,
                                record=lambda *_: None, adversarial_review_enabled=True)
    assert len(seen) == 3 and result['status'] == 'supported_on_cases'
    for case in proposal['cases']:
        del case['user_context_product_id']
    with pytest.raises(ValueError, match='user_context_product_id'):
        validate_attack({'issues': [proposal], 'reason': 'Check syntax'}, {'syntax': {}}, sources)
    with pytest.raises(ValueError, match='exact actual user dialogue'):
        await verify_issue(proposal, pack, 'reward', context, evidence,
                           SimpleNamespace(run=run, tools=MeasurementTools()), reference, record=lambda *_: None, adversarial_review_enabled=True)
