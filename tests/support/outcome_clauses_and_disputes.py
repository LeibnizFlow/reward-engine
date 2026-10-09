"""Shared fixtures and reusable scenarios extracted without changing assertions."""

def issue(target='format', scope='whole_check'):
    return {'target_id': target, 'proposer_seat': '1', 'priority': 'critical',
            'cases': [{'id': str(i), 'kind': kind, 'scope': scope, 'artifact': body,
                       'question': 'ATTACKER_LEADING_QUESTION: surely this must pass?', 'expected': expected}
                      for i, (kind, body, expected) in enumerate([
                          ('correct_control', '{}', 'pass'), ('error_variant', '{broken', 'fail'),
                          ('boundary', 'null', 'pass')])]}
