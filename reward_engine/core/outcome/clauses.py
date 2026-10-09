"""Task-independent clause semantics; references locate evidence, not truth."""

import json
from copy import deepcopy
from urllib.parse import quote as quote_id

from reward_engine.core.outcome.construction import fingerprint
from reward_engine.core.outcome.context import ResponseContractError, require_refs

VERSION = 'outcome-clauses/2026-09-11'
USES = {'task_constraint', 'quality_judgment', 'illustrative_method'}
SCHEMA = {
    'id': 'stable clause ID; retain the ID when revising this clause',
    'text': 'one outcome requirement, quality judgment or illustrative method',
    'use': 'task_constraint/quality_judgment/illustrative_method',
    'evidence_refs': ['actual supplied refs, not proof of truth'],
    'source_quotes': [{'ref': 'task or actual scoped user ref', 'quote': 'exact source text'}],
    'conditions': ['applicability conditions; empty means none declared, not proven universal'],
    'alternative_group': 'same nonempty group means alternatives, not cumulative obligations; otherwise empty',
}
INSTRUCTION = (
    'Use structured clauses as the sole detailed definition. Classify EACH clause independently: '
    'task_constraint is an actual user obligation with an exact Task/scoped-user source quote; '
    'quality_judgment estimates correctness, depth, completeness or usefulness; illustrative_method '
    'is a possible technique or example. A hard parent does NOT make its children mandatory. '
    'Retain substantive professional quality goals; do not downgrade all expertise into silence. '
    'State applicability conditions and group equivalent alternatives with alternative_group. '
    'An illustrative method is never the only passing solution. Source quotes locate provenance, '
    'not semantic proof. Do not write a second competing subrequirements list: code derives it. '
    'Conditions not supplied remain unstated, not proven universally applicable. '
)


def normalize_criteria(criteria, context, evidence, *, required=False):
    """Validate metadata and derive the legacy text projection without inventing provenance."""
    rows = deepcopy(criteria)
    user_refs = {u['ref'] for p in (context or {}).get('products', []) for u in p.get('user_updates', [])}
    from reward_engine.core.outcome.task_inputs import refs as task_input_refs
    original_inputs = task_input_refs(context or {})
    for criterion_index, criterion in enumerate(rows):
        if 'clauses' not in criterion:
            if required:
                raise ValueError('New construction requires structured clauses for every criterion')
            continue
        clauses = criterion['clauses']
        if not isinstance(clauses, list) or not clauses:
            raise ValueError('clauses must be a nonempty list')
        seen = set()
        for clause_index, clause in enumerate(clauses):
            if (not isinstance(clause, dict) or clause.get('use') not in USES
                    or not isinstance(clause.get('text'), str) or not clause['text'].strip()):
                raise ValueError('Each clause needs text and an explicit use')
            key = clause.get('id')
            if not isinstance(key, str) or not key.strip() or key in seen:
                raise ValueError('Clause IDs must be nonempty and unique within a criterion')
            seen.add(key)
            require_refs(clause.get('evidence_refs'), evidence)
            conditions = clause.get('conditions')
            if (not isinstance(conditions, list)
                    or any(not isinstance(c, str) or not c.strip() for c in conditions)
                    or not isinstance(clause.get('alternative_group'), str)):
                raise ValueError('Declare clause conditions and alternative_group explicitly')
            quotes = clause.get('source_quotes')
            if not isinstance(quotes, list):
                raise ValueError('source_quotes must be a list')
            for quote_index, quote in enumerate(quotes):
                field = f'criteria[{criterion_index}].clauses[{clause_index}].source_quotes[{quote_index}]'
                allowed_refs = ({'task', *user_refs, *original_inputs}
                                if clause['use'] == 'task_constraint' else set(evidence))
                if not isinstance(quote, dict) or quote.get('ref') not in allowed_refs:
                    error = ResponseContractError(field + '.ref',
                        ('Constraint provenance must cite actual Task/scoped user text or supplied original Task input'
                         if clause['use'] == 'task_constraint' else
                         'Quality/method provenance must cite supplied evidence'))
                    error.details.update(criterion_id=criterion.get('id'), clause_id=key,
                                         provided_ref=quote.get('ref') if isinstance(quote, dict) else None)
                    raise error
                ref, text = quote['ref'], quote.get('quote')
                source = evidence.get(ref)
                def texts(value):
                    if isinstance(value, str):
                        return [value]
                    if isinstance(value, dict):
                        return [s for v in value.values() for s in texts(v)]
                    if isinstance(value, list):
                        return [s for v in value for s in texts(v)]
                    return []
                source_texts = texts(source) + [json.dumps(source, ensure_ascii=False)]
                if not isinstance(text, str) or not text.strip() or not any(text in s for s in source_texts):
                    error = ResponseContractError(field + '.quote',
                        'Source quote must occur exactly in its cited evidence; '
                        'inspect this source and correct the quotation without rewriting unaffected clauses')
                    error.details.update(criterion_id=criterion.get('id'), clause_id=key,
                                         source_ref=ref, provided_quote=str(text)[:500])
                    raise error
                if ref not in clause['evidence_refs']:
                    # The exact quote already explicitly cites this registered
                    # source. Keep the two citation projections consistent rather
                    # than asking a model to repeat an already supplied handle.
                    clause['evidence_refs'].append(ref)
                    clause.setdefault('quote_ref_resolutions', []).append(
                        {'ref': ref, 'source': 'existing_exact_source_quote'})
            if clause['use'] == 'task_constraint' and not quotes:
                raise ValueError('Task constraint requires an exact Task/user quote')
        projection = [c['text'] for c in clauses]
        if 'subrequirements' in criterion and criterion['subrequirements'] != projection:
            raise ValueError('subrequirements must equal the structured clause projection')
        criterion['subrequirements'] = projection
        criterion['clause_protocol'] = VERSION
    return rows


def clause_target_id(criterion_id, clause_id):
    return f'clause:{quote_id(criterion_id, safe="")}:{quote_id(clause_id, safe="")}'


def targets(criteria):
    return {clause_target_id(c['id'], p['id']): {**deepcopy(p), 'criterion_id': c['id'],
            'clause_id': p['id'], 'target_kind': 'clause', 'id': clause_target_id(c['id'], p['id'])}
            for c in criteria for p in c.get('clauses', [])}


def claim_binding(target):
    """Bind exactly the assessed statement, including scope and revision identity."""
    if target.get('target_kind') != 'clause':
        return None
    return {'target_id': target['id'], 'target_sha256': fingerprint(target),
            'claim': target['text'], 'conditions': deepcopy(target['conditions']),
            'alternative_group': target['alternative_group']}


def bound_premise(premise, record, current_targets):
    binding = record.get('claim_binding') or {}
    target = current_targets.get(binding.get('target_id'))
    return bool(target and binding == claim_binding(target)
                and premise.get('target_id') == binding['target_id']
                and premise.get('claim') == binding['claim']
                and premise.get('conditions') == binding['conditions'])
