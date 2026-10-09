from copy import deepcopy

import pytest

from reward_engine.adapters.workbuddy.evidence import collect
from reward_engine.adapters.workbuddy.measurement import WorkBuddyTools
from reward_engine.core.outcome.context import build_context
from reward_engine.core.outcome.harness import defaults, has_product_evidence


@pytest.mark.asyncio
@pytest.mark.parametrize('tool,paths,expected', [
    ('workspace.read', ['output.md'], 'pass'),
    ('workspace.read', ['input.md'], 'unknown'),
    ('workspace.read', ['missing.md'], 'unknown'),
    ('workspace.read', ['*.md'], 'unknown'),
    ('workspace.inspect', ['output.md'], 'unknown'),
])
async def test_only_real_delivery_read_receipts_establish_product_observation(tmp_path, tool, paths, expected):
    (tmp_path / 'input.md').write_text('Source value 7')
    _, initial = collect(tmp_path, candidate_id='initial', final_response='')
    (tmp_path / 'output.md').write_text('Delivered value 7')
    product, _ = collect(tmp_path, candidate_id='actual', final_response='', initial_hashes=initial)
    context, evidence = build_context({'task_id': 't', 'prompt': 'Write the source value to output.md'}, [product])
    ref = 'tool:check:0'

    class Model:
        bounded_stage = True

        async def __call__(self, request):
            if not request['context'].get('tool_observations'):
                return {'need_tools': [{'tool': tool, 'arguments': {'paths': paths}}]}
            # Even a real tool handle cannot be embellished by the model into
            # a stronger observation than the executor actually returned.
            forged = deepcopy(request['context']['tool_observations'])
            forged[ref].update(tool='workspace.read', status='observed', result={
                'text': 'invented observation', 'observed_product_refs': ['product_1:source:1']})
            return {'status': 'pass', 'reason': 'Observed delivered value', 'evidence_refs': [ref],
                'claims': [{'claim': 'Delivered 7', 'claim_refs': [ref], 'evidence_refs': [ref],
                            'observation': 'The read contains 7', 'judgment': 'supported'}],
                '_tool_evidence': forged, 'tool_observations': forged}

    result = await defaults(Model(), tools=WorkBuddyTools()).run(
        {'test_id': 'check', 'criterion_id': 'value', 'checker': 'evidence.review'}, context, evidence)
    assert result['status'] == expected
    assert result['tool_observations'][ref]['result']['text'] != 'invented observation'
    assert has_product_evidence([ref], evidence, result['tool_observations']) == (expected == 'pass')


@pytest.mark.asyncio
async def test_read_receipt_preserves_pagination_and_does_not_imply_unread_pages(tmp_path):
    (tmp_path / 'output.md').write_text('x' * 70000)
    product, _ = collect(tmp_path, candidate_id='actual', final_response='', initial_hashes={})
    _, evidence = build_context({'task_id': 't', 'prompt': 'Read output'}, [product])
    tools = WorkBuddyTools()
    receipt = await tools.run({'tool': 'workspace.read', 'arguments': {'paths': ['output.md']}}, evidence)
    result = receipt['result']
    assert result['excerpt_only'] and result['next_offset'] < result['total_chars']
    assert has_product_evidence(['tool:r'], evidence, {'tool:r': receipt})
    receipt['status'] = 'unknown'
    assert not has_product_evidence(['tool:r'], evidence, {'tool:r': receipt})


@pytest.mark.asyncio
async def test_upstream_http_failure_keeps_status_and_is_not_a_product_failure(tmp_path):
    import httpx

    from reward_engine.adapters.workbuddy.gateway import application
    from reward_engine.adapters.workbuddy.io import read
    from reward_engine.core.outcome.harness import Registry

    (tmp_path / 'auth').write_text('campaign-token')
    def upstream(request):
        return httpx.Response(401, json={'error': 'private provider detail'})
    def factory(**kwargs):
        return httpx.AsyncClient(transport=httpx.MockTransport(upstream), **kwargs)
    config = {'model_concurrency': 1, 'roles': {'actor_selector': 'actor'}, 'output': str(tmp_path),
              'auth_file': str(tmp_path / 'auth'), 'backend_url': 'https://fake/v1'}
    app = application(config, backend_key='private-secret', client_factory=factory)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://gateway') as client:
        response = await client.post('/roles/v1/chat/completions', json={'model': 'actor', 'messages': []},
                                     headers={'Authorization': 'Bearer campaign-token'})
    assert response.status_code == 502
    receipt = read(next((tmp_path / 'transport/roles').glob('*/receipt.json')))
    assert receipt['upstream_http_status'] == 401
    assert 'private' not in str(receipt)
    async def checker(test, context, evidence):
        response.raise_for_status()
    registry = Registry()
    registry.register('remote', checker, 'Remote observation', kind='model_assisted')
    context, evidence = build_context({'task_id': 't', 'prompt': 'Check delivery'}, [])
    result = await registry.run({'test_id': 'remote', 'criterion_id': 'c', 'checker': 'remote'}, context, evidence)
    assert result['status'] == 'unknown' and result['error']
    assert result['failure_layer'] == 'infrastructure'
