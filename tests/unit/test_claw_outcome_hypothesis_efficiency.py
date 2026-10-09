"""One shared preprocessing pass, parallel proposals, semantic synthesis, no lost questions."""

import asyncio
from copy import deepcopy

import pytest

from reward_engine.core.outcome import hypotheses
from reward_engine.core.outcome.context import build_context


def proposal():
    return {'status': 'propose', 'method': 'Check the missing evidence first',
            'target': 'Uncertainty', 'expected_improvement': 'A supportable answer',
            'verification': 'Inspect actual sources', 'target_gap_ids': [], 'evidence_refs': ['task']}


@pytest.mark.asyncio
async def test_three_parallel_proposals_reuse_projection_and_reducer_can_refine(monkeypatch):
    context, evidence = build_context({'task_id': 'x', 'prompt': 'Explain the evidence'})
    original = deepcopy((context, evidence))
    prepare, calls, started = [], [], []
    gate = asyncio.Event()
    real = hypotheses.question_context
    def prepare_once(*args, **kwargs):
        prepare.append(1)
        return real(*args, **kwargs)
    monkeypatch.setattr(hypotheses, 'question_context', prepare_once)
    async def member(request):
        started.append(1)
        if len(started) == 3:
            gate.set()
        await asyncio.wait_for(gate.wait(), 1)
        calls.append(request)
        return proposal()
    async def reducer(request):
        calls.append(request)
        assert len(request['proposals']) == 3
        assert 'refine the strongest' in request['instruction']
        assert 'compatible ideas' in request['instruction']
        return {**proposal(), 'method': 'A synthesized investigation then repair'}
    result = await hypotheses.propose(context, evidence, {}, {str(i): member for i in range(3)}, reducer)
    assert len(prepare) == 1 and len(calls) == 4
    assert result['method'] == 'A synthesized investigation then repair'
    assert (context, evidence) == original
    # A direct replay of only Reduce remains a supported public entry point.
    await hypotheses.reduce(context, evidence, {}, reducer, [proposal()] * 3)
    assert len(prepare) == 2


def test_pending_question_keeps_omission_reason_and_all_shared_missing_requirements():
    jury = {'independent_expectations': [], 'coverage': []}
    for i in range(15):
        jury['independent_expectations'].append({'expectation_id': str(i), 'origin': 'task',
            'kind': 'hard', 'priority': 'critical', 'standard': f'Requirement {i}',
            'subrequirements': [{'part_id': f'p{i}', 'text': f'Important unanswered point {i}'}],
            'evidence_refs': ['task']})
        jury['coverage'].append({'expectation_id': str(i), 'status': 'deferred',
            'reason': 'Reduce omitted this item', 'subrequirements': [{'part_id': f'p{i}',
                'status': 'deferred', 'reason': 'No Reduce decision; not a product failure'}]})
    before = deepcopy(jury)
    questions = hypotheses.compact_jury(jury)['open_questions']
    assert len(questions) == 15  # No arbitrary top-10 loss of common blind spots.
    for q in questions:
        assert q['status'] == 'unverified_proposal'
        source = q['pending_sources'][0]
        assert source['kind'] == 'hard' and source['priority'] == 'critical'
        assert source['reason'] == 'Reduce omitted this item'
        assert 'not a product failure' in source['parts'][0]['reason']
    assert jury == before
