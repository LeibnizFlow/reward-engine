"""The conventional discard variable has no interpreter capability privileges."""
import pytest

from reward_engine.core.outcome.numeric_program import run_numeric


def test_discard_loop_assignment_and_comprehension():
    assert run_numeric('total = 0\nfor _ in range(4):\n    total = total + 2\nresult = total') == 8
    assert run_numeric('_ = 3\nresult = [1 for _ in range(_)]') == [1, 1, 1]
    with pytest.raises(ValueError, match='step budget'):
        run_numeric('for _ in range(100):\n    x = 1\nresult = x', steps=20)


@pytest.mark.parametrize('name', ['inputs', '_private', '__builtins__'])
@pytest.mark.parametrize('template', ['{name} = 0\nresult = 1', 'for {name} in range(2):\n    x = 1\nresult = x', 'result = [1 for {name} in range(2)]'])
def test_reserved_names_stay_unavailable(name, template):
    with pytest.raises(ValueError):
        run_numeric(template.format(name=name))
