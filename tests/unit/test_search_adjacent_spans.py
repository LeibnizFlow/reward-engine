"""A bounded union of offered passages is recoverable; fabricated offsets are not."""

import pytest

from reward_engine.core.outcome.knowledge_evidence import citation, spans


@pytest.mark.parametrize('count', [2, 3])
def test_adjacent_offered_spans_are_copied_with_explicit_provenance(count):
    text = 'An observed method and its scope.\n' * 70
    offered = spans(text)
    parts = offered[1:1 + count]
    ref = {'source_id': 's', 'span_id': f"span_{parts[0]['start']}_{parts[-1]['end']}",
           'quote': 'Model-invented replacement must be ignored'}
    result = citation(ref, {'s': {'text': text}}, span_only=True)
    assert result['quote'] == ''.join(p['text'] for p in parts)
    assert result['resolution'] == 'adjacent_spans'
    assert result['component_span_ids'] == [p['span_id'] for p in parts]
    assert result['span_id'] == ref['span_id']


@pytest.mark.parametrize('invalid', ['internal_start', 'internal_end', 'four_spans', 'outside_excerpt', 'other_source'])
def test_join_does_not_widen_the_read_evidence_boundary(invalid):
    text = 'Known source passage.\n' * 100
    offered = spans(text)
    start, end, source_id = offered[0]['start'], offered[1]['end'], 's'
    if invalid == 'internal_start':
        start += 1
    elif invalid == 'internal_end':
        end -= 1
    elif invalid == 'four_spans':
        end = offered[3]['end']
    elif invalid == 'outside_excerpt':
        end = len(text) + 400
    else:
        source_id = 'unread_source'
    assert citation({'source_id': source_id, 'span_id': f'span_{start}_{end}', 'quote': text[:20]},
                    {'s': {'text': text}}, span_only=True) is None


def test_navigation_still_cannot_supply_adjacent_evidence():
    text = 'Navigation heading.\n' * 60
    offered = spans(text)
    assert citation({'source_id': 's', 'span_id': f"span_0_{offered[1]['end']}"},
                    {'s': {'text': text, 'read_kind': 'navigation'}}, span_only=True) is None


def test_model_cannot_forge_span_resolution_receipts():
    text = 'A real short passage.'
    result = citation({'source_id': 's', 'span_id': spans(text)[0]['span_id'],
                       'resolution': 'adjacent_spans', 'component_span_ids': ['invented'], 'quote': 'fake'},
                      {'s': {'text': text}}, span_only=True)
    assert result['quote'] == text
    assert 'resolution' not in result and 'component_span_ids' not in result
