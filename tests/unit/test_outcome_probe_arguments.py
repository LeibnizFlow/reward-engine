from copy import deepcopy

import pytest

from reward_engine.core.outcome.builder import compile_contract
from reward_engine.core.outcome.harness import defaults


def bundle(checker='evidence.review', arguments=None):
    test = {'test_id':'t', 'criterion_id':'c', 'question':'Check actual values', 'checker':checker, 'arguments':{}}
    if checker == 'code.run':
        test['arguments'] = arguments or {}
    else:
        test['probes'] = [{'tool':'code.run','arguments':arguments or {}}]
    return {'rubrics':[{'criterion_id':'c','anchors':{'fail':'Wrong','partial':'Incomplete','full':'Correct'}}], 'tests':[test]}


@pytest.mark.parametrize('checker', ['evidence.review','code.run'])
@pytest.mark.parametrize('program', [None, '', '  ', 42])
def test_empty_code_probe_rejected_before_recipe_installation(checker, program):
    registry = defaults(None)
    result = bundle(checker, {} if program is None else {'program':program})
    original = deepcopy(result)
    with pytest.raises(ValueError, match='non-empty program'):
        compile_contract(result, {'criteria':[{'id':'c'}]}, registry)
    assert result == original
    assert not registry.recipes


def test_concrete_program_and_dynamic_semantic_measurement_remain_available():
    registry = defaults(None)
    result = bundle(arguments={'program':'result = 1'})
    _, tests = compile_contract(result, {'criteria':[{'id':'c'}]}, registry)
    assert tests[0]['execution_recipe']['probes'][0]['arguments']['program'] == 'result = 1'
    result['tests'][0].pop('probes')
    _, tests = compile_contract(result, {'criteria':[{'id':'c'}]}, registry)
    assert 'execution_recipe' not in tests[0]
