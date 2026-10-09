"""Operator-owned review permission and model-free Reward integrity checks."""

from copy import deepcopy

from reward_engine.core.outcome import harness, jury
from reward_engine.core.outcome.construction import fingerprint


class AdversarialReviewDisabled(ValueError):
    """A disabled review must fail before any model or checker execution."""


def validate_review_flag(value):
    if type(value) is not bool:
        raise ValueError('adversarial_review_enabled must be a boolean')
    return value


def review_enabled(config):
    return validate_review_flag(config.get('adversarial_review_enabled', False))


def effective_review_config(config):
    value = deepcopy(config)
    value['adversarial_review_enabled'] = review_enabled(value)
    return value


def require_adversarial(enabled):
    if not validate_review_flag(enabled):
        raise AdversarialReviewDisabled(
            'Adversarial review is disabled; only an explicit user request may set '
            'adversarial_review_enabled=true in the run configuration.')


def basic_review(pack, context, evidence, registry):
    """Low-level QC without buying counterexamples or claiming calibration."""
    errors = []
    try:
        if pack['task_id'] != context['task']['task_id'] or pack['task_prompt'] != context['task']['prompt']:
            raise ValueError('Reward belongs to a different Task')
        context, evidence = harness.measurement_context(pack, context, evidence)
        jury.validate({'criteria': pack['criteria']}, evidence, context)
        criteria = {row['id']: row for row in pack['criteria']}
        seen = set()
        for test in pack['tests']:
            identity = test['test_id']
            if identity in seen:
                raise ValueError('Duplicate checker ID: ' + identity)
            seen.add(identity)
            if test['criterion_id'] not in criteria:
                raise ValueError('Checker references an unknown criterion: ' + identity)
            if test['checker'] not in registry.checkers:
                raise ValueError('Checker is not installed: ' + identity)
            if test.get('rubric') is not None and test['rubric'] != criteria[test['criterion_id']]:
                raise ValueError('Checker rubric differs from its criterion: ' + identity)
        if not seen or {t['criterion_id'] for t in pack['tests']} != set(criteria):
            raise ValueError('Every criterion needs a bound checker')
    except (ValueError, KeyError, TypeError) as error:
        errors.append({'error_type': type(error).__name__, 'reason': str(error)})
    pack_hash = fingerprint(pack)
    records = [{'validation_id': fingerprint({'pack': pack_hash, 'kind': kind, 'target': row[key]}),
                'target_id': row[key], 'status': 'unverified', 'contrast_qualified': False, 'cases': [],
                'reason': 'Basic integrity review only; no counterexample calibration claimed.'}
               for kind, key, rows in [('criterion', 'id', pack.get('criteria', [])),
                                       ('test', 'test_id', pack.get('tests', []))] for row in rows if key in row]
    return {'status': 'completed', 'artifact_sha256': pack_hash, 'records': records,
            'basic_integrity': {'errors': errors, 'model_requests': 0,
                'boundary': 'Structure, source and binding checks only. QC still inspects obvious semantic overreach; '
                            'actual execution is recorded by measurement. Grading alignment belongs to the outer loop.'}}
