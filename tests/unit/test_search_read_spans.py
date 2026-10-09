"""Model-selected long reads share the original budget and preserve actual text."""
from copy import deepcopy

import pytest

from reward_engine.core.batch import BatchPolicy
from reward_engine.core.outcome.agentic_search import ResearchBudget, research


async def run_material(model, text, *, reads=6, on_read=None):
    async def no_network(_):
        raise AssertionError('Caller-supplied body needs no fetch/search')
    scope = {'task_prompt': 'Review the supplied protocol and its limitations.',
             'task_materials': [{'ref': 'task:protocol', 'title': 'Protocol', 'text': text}]}
    before = deepcopy(scope)
    result = await research(scope, {'queries': []}, model, no_network, no_network,
        policy=BatchPolicy(deadline_s=2), record=lambda *_: None, lane='unified', on_read=on_read,
        budget=ResearchBudget(reads=reads, passage_chars=80, steps=3))
    assert scope == before
    return result


@pytest.mark.asyncio
async def test_one_span_delivers_original_tail_to_reader_and_charges_all_units():
    text = '# Procedure\n\n' + 'Measure the real input.\n' * 12 + '\n# Limitations\n\nOnly valid with a sealed container.'
    observed, calls = [], []
    async def reader(sources, _):
        observed.extend(deepcopy(sources))
        return {'status': 'read', 'boundary': 'Not a verdict'}
    async def model(request):
        calls.append(request)
        candidate = request['candidates'][0]
        if len(calls) == 1:
            assert candidate['read_ranges'] == [] and not candidate['read']
            assert candidate['total_chars'] == len(text)
            assert request['budget_scope']['read_unit_chars'] == 80
            return {'reads': [{'candidate_id': candidate['candidate_id'], 'offset': 0,
                'end_offset': candidate['total_chars'], 'gap': 'Read method and limits together'}]}
        assert candidate['read_ranges'] == [[0, len(text)]]
        assert request['latest_read_passages'][0]['text'] == text
        return {'stop': True}
    result = await run_material(model, text, on_read=reader)
    assert len(observed) == len(result['sources']) == 1 and observed[0]['text'] == text
    assert not result['sources'][0]['excerpt_only']
    assert result['research']['reads_used'] == (len(text) + 79) // 80
    assert result['research']['physical_read_requests'] == 0
    assert result['research']['history'][0]['actions'][0]['read_units'] == (len(text) + 79) // 80


@pytest.mark.asyncio
async def test_span_and_other_reads_share_budget_and_cannot_overspend():
    calls = []
    async def model(request):
        calls.append(request)
        if len(calls) == 1:
            return {'reads': [
                {'candidate_id': 'doc_0', 'offset': 0, 'end_offset': 160, 'gap': 'Main mechanism'},
                {'candidate_id': 'doc_0', 'offset': 160, 'end_offset': 400, 'gap': 'Too much for remaining unit'},
                {'candidate_id': 'doc_0', 'offset': 320, 'gap': 'One targeted late passage'},
                {'candidate_id': 'doc_0', 'offset': 240, 'gap': 'No remaining budget'}]}
        return {'stop': True}
    result = await run_material(model, 'x' * 401, reads=3)
    assert result['research']['reads_used'] == 3
    assert [len(s['text']) for s in result['sources']] == [160, 80]
    actions = result['research']['history'][0]['actions']
    assert any(a['status'] == 'read_span_budget' and a['required_units'] == 3 for a in actions)
    assert any(a['status'] == 'read_budget' for a in actions)


@pytest.mark.asyncio
@pytest.mark.parametrize('fields', [
    {'end_offset': True}, {'end_offset': '80'}, {'end_offset': 0}, {'end_offset': 401},
    {'offset': 160, 'end_offset': 80}, {'end_offset': 160, 'query': 'move the window'},
])
async def test_bad_span_is_receipted_and_does_not_exhaust_budget_or_poison_source(fields):
    async def model(request):
        if not request['history']:
            return {'reads': [{'candidate_id': 'doc_0', 'gap': 'Invalid interval', **fields},
                             {'candidate_id': 'doc_0', 'gap': 'Valid default passage', 'read_chars': 999999}]}
        return {'stop': True}
    result = await run_material(model, 'x' * 400)
    assert result['research']['reads_used'] == 1
    assert len(result['sources']) == 1 and len(result['sources'][0]['text']) == 80
    assert result['research']['failed_urls'] == []
    assert result['research']['history'][0]['actions'][0]['status'] == 'invalid_read_span'


@pytest.mark.asyncio
async def test_unread_web_source_cannot_claim_an_unobserved_length_but_can_read_normally():
    fetches = []
    async def search(_):
        return {'results': [{'url': 'https://example.test/reference'}]}
    async def extract(url):
        fetches.append(url)
        return {'results': [{'url': url, 'raw_content': 'a' * 320, 'physical_requests': 1}]}
    async def model(request):
        if not request['history']:
            return {'reads': [{'candidate_id': 'doc_0', 'end_offset': 320, 'gap': 'Unknown length'},
                             {'candidate_id': 'doc_0', 'gap': 'First real read'}]}
        if len(request['history']) == 1:
            return {'reads': [{'candidate_id': 'doc_0', 'offset': 80, 'end_offset': 320,
                              'gap': 'Read known remainder'}]}
        return {'stop': True}
    result = await research({}, {'queries': ['protocol']}, model, search, extract,
        policy=BatchPolicy(deadline_s=2), record=lambda *_: None,
        budget=ResearchBudget(reads=4, passage_chars=80))
    assert len(fetches) == 1 and result['research']['reads_used'] == 4
    assert [(s['offset'], len(s['text'])) for s in result['sources']] == [(0, 80), (80, 240)]
