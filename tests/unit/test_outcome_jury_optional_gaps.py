from copy import deepcopy

import pytest
from tests.support.claw_outcome_gap_signals import fixture

from reward_engine.core.outcome.jury import validate, validate_reduced_standards


@pytest.mark.parametrize('defect', ['missing_task_ref', 'invented_source', 'false_check_support'])
def test_optional_gap_is_recorded_without_invalidating_valid_standards(defect):
    context, evidence, original = fixture()
    bad = deepcopy(original['gaps'][0])
    bad['gap_id'] = 'bad-observation'
    if defect == 'missing_task_ref':
        bad['expected_refs'] = ['product_1:environment']
    elif defect == 'invented_source':
        bad['evidence_refs'].append('nonexistent')
    else:
        bad['status'] = 'check_supported'
    original['gaps'].append(bad)
    before = deepcopy(original)
    records = {}
    with pytest.raises(ValueError):
        validate(original, evidence, context)
    result = validate_reduced_standards(original, evidence, context,
        record=lambda name, value: records.update({name: value}))
    assert result['criteria'] == original['criteria']
    assert result['gaps'] == original['gaps'][:1]
    assert result['gap_validation_issues'][0]['gap_id'] == 'bad-observation'
    assert records['jury_gap_validation']['rejected'][0]['raw'] == bad
    assert original == before
    # Quarantined diagnostic references cannot poison later canonical validation.
    assert validate(result, evidence, context)


def test_invalid_standard_still_requires_repair():
    context, evidence, original = fixture()
    original['criteria'][0]['origin'] = 'domain'
    original['gaps'][0]['expected_refs'] = []
    with pytest.raises(ValueError, match='Domain preferences cannot become hard'):
        validate_reduced_standards(original, evidence, context)


def test_all_invalid_user_standards_get_a_task_baseline():
    context, evidence, original = fixture()
    invalid = deepcopy(original['criteria'][0])
    invalid.update(id='unsupported_user_requirement', origin='user', evidence_refs=['task'])
    original['criteria'] = [invalid]
    records = {}

    result = validate_reduced_standards(
        original, evidence, context,
        record=lambda name, value: records.update({name: value}),
    )

    assert result['criteria'][0]['id'] == 'task_outcome_requirement'
    assert result['criteria'][0]['origin'] == 'task'
    assert result['criterion_validation_issues'][0]['id'] == 'unsupported_user_requirement'
    assert records['jury_criterion_validation']['rejected'][0]['raw'] == invalid
    assert validate(result, evidence, context)


def test_malformed_optional_gap_list_is_preserved_only_in_diagnostic_record():
    context, evidence, original = fixture()
    original['gaps'] = {'not': 'a list'}
    records = {}
    result = validate_reduced_standards(original, evidence, context,
        record=lambda name, value: records.update({name: value}))
    assert result['gaps'] == [] and result['criteria'] == original['criteria']
    assert records['jury_gap_validation']['rejected'][0]['raw'] == original['gaps']
