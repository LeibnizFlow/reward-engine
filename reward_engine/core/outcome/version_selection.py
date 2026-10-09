"""Select immutable delivered versions without rewriting the native Actor trace."""

import asyncio
from copy import deepcopy

from reward_engine.core.batch import BatchPolicy
from reward_engine.core.outcome import jury
from reward_engine.core.outcome.admission import critical_tests, preservation
from reward_engine.core.outcome.construction import fingerprint
from reward_engine.core.outcome.context import build_context
from reward_engine.core.outcome.harness import measure


def choose(versions, ranking, pack):
    """Require strict preference and current critical evidence; otherwise keep prior."""
    selected = versions[0]
    decisions = []
    positions = {p: i for i, group in enumerate(ranking.get('groups', [])) for p in group}
    for version in versions[1:]:
        before, after = selected.get('measurement') or {}, version.get('measurement') or {}
        preserved, lost = preservation(before, after, pack)
        known = {r['test_id'] for r in after.get('results', [])
                 if r.get('status') in {'pass', 'fail'} and not r.get('error')}
        complete = critical_tests(pack) <= known
        preferred = (ranking.get('status') == 'compared' and selected['product_id'] in positions
                     and version['product_id'] in positions
                     and positions[version['product_id']] < positions[selected['product_id']])
        accept = preferred and preserved and complete
        decisions.append({'candidate_id': version['candidate_id'], 'accepted': accept,
                          'strictly_preferred': preferred, 'critical_checks_complete': complete,
                          'lost_critical_checks': lost})
        if accept:
            selected = version
    return selected, decisions


async def select_versions(task, pack, rows, registry, roles, *, record=lambda *_: None, reduce_max_bytes=393216):
    from reward_engine.core.outcome.task_sequence import (
        TaskSequence,
        require_version_scope,
    )

    if any(row.get('effective_task') for row in rows.values()):
        scoped = {}
        for arm, row in rows.items():
            if not row.get('candidate'):
                continue
            current_task = row.get('effective_task') or task
            current_pack = row.get('effective_pack') or (
                TaskSequence(current_task).rebase(pack) if current_task != task else pack)
            local = {k: deepcopy(v) for k, v in row.items() if k not in {'effective_task', 'effective_pack'}}
            versions = local.get('review_session', {}).get('versions', [])
            require_version_scope(versions, local['candidate'])
            for version in versions:
                if version['artifact_sha256'] != fingerprint(version['candidate']['artifact']):
                    raise ValueError('Immutable delivery hash mismatch')
            if len(versions) <= 1:
                if versions and versions[0]['artifact_sha256'] != fingerprint(local['candidate']['artifact']):
                    raise ValueError('Native candidate differs from its recorded current-scope version')
                local['selected_candidate'] = deepcopy(local['candidate'])
                local['delivery_selection'] = {'candidate_id': 'delivery_1',
                    'artifact_sha256': fingerprint(local['candidate']['artifact']),
                    'decisions': [], 'native_trace_preserved': True,
                    'boundary': 'Only one delivered version in the current public Task scope'}
                local['version_checks'] = versions or [{'candidate_id': 'delivery_1',
                    'candidate': deepcopy(local['candidate']),
                    'artifact_sha256': fingerprint(local['candidate']['artifact']), 'measurement': None}]
                scoped[arm] = {'status': 'single_version'}
            else:
                scoped[arm] = await select_versions(current_task, current_pack, {arm: local}, registry, roles,
                    record=lambda n, d, a=arm: record('task_scopes/' + a + '/' + n, d),
                    reduce_max_bytes=reduce_max_bytes)
            for key in ('selected_candidate', 'delivery_selection', 'version_checks'):
                if key in local:
                    row[key] = local[key]
        selection = {'arm': None, 'reason': 'Native dialogues keep their own revealed user requirements and deliveries'}
        record('selected_delivery', selection)
        return {'ranking': {'status': 'per_task_scope', 'arms': scoped}, 'selection': selection}

    groups, unique = {}, {}
    for arm, row in rows.items():
        if not row.get('candidate'):
            continue
        versions = deepcopy(row.get('review_session', {}).get('versions', []))
        if not versions:
            versions = [{'candidate_id': 'delivery_1', 'candidate': deepcopy(row['candidate']),
                         'artifact_sha256': fingerprint(row['candidate']['artifact']), 'measurement': None}]
        require_version_scope(versions, row['candidate'])
        for version in versions:
            digest = fingerprint(version['candidate']['artifact'])
            if version['artifact_sha256'] != digest:
                raise ValueError('Immutable delivery hash mismatch')
            unique.setdefault(digest, version['candidate'])
        groups[arm] = versions
    if not unique:
        return {'ranking': {'status': 'unavailable'}, 'selection': {'arm': None, 'reason': 'no_artifacts'}}
    # Rank complete artifact bodies only: no arm labels, revision history, feedback or prior grades.
    ordered = sorted(unique)
    products = {digest: f'product_{i}' for i, digest in enumerate(ordered, 1)}
    anonymous = [{'artifact': deepcopy(unique[digest]['artifact']),
                  'source_documents': deepcopy(unique[digest].get('source_documents', []))}
                 for digest in ordered]
    for candidate in anonymous:
        candidate['artifact'].pop('public_consultation', None)
        candidate['artifact'].pop('earlier_public_deliveries', None)
    critical = sorted(critical_tests(pack))

    async def check(digest):
        context, evidence = build_context(task, [unique[digest]])
        try:
            result = await measure(pack, registry, context, evidence,
                                   policy=BatchPolicy(deadline_s=480), test_ids=critical)
        except Exception as error:
            result = {'pack_id': pack['pack_id'], 'results': [], 'error': type(error).__name__}
        result['artifact_sha256'] = digest
        record('version_checks/' + digest, result)
        return digest, result
    measured = dict(await asyncio.gather(*(check(digest) for digest in ordered)))
    context, evidence = build_context(task, anonymous)
    # Rank complete outcomes against the original Task, independently of our
    # generated standards, attack rationale and suggested edits.
    try:
        ranking = await jury.rank(context, evidence,
            {str(i): roles(f'jury_member_{i}') for i in range(1, 4)}, roles('jury_reducer'),
            policy=BatchPolicy(deadline_s=900), reduce_max_bytes=reduce_max_bytes,
            record=lambda n, d: record('version_ranking/' + n, d))
    except Exception as error:
        ranking = {'status': 'unavailable', 'error_type': type(error).__name__, 'reason': str(error)}
    record('version_ranking/result', ranking)
    finalists = {}
    for arm, versions in groups.items():
        for version in versions:
            digest = version['artifact_sha256']
            version.update(product_id=products[digest], measurement=deepcopy(measured[digest]))
        selected, decisions = choose(versions, ranking, pack)
        selection = {k: deepcopy(selected[k]) for k in ('candidate_id', 'artifact_sha256', 'product_id')}
        selection.update(check_source='version_checks/' + selected['artifact_sha256'] + '.json',
                         decisions=decisions, native_trace_preserved=True,
                         boundary='Fallible public ranking and scoped checks; no score guarantee')
        rows[arm]['selected_candidate'] = deepcopy(selected['candidate'])
        rows[arm]['delivery_selection'] = selection
        rows[arm]['version_checks'] = versions
        finalists[arm] = selected
        record(arm + '/delivery_selection', selection)
    baseline = 'A' if 'A' in finalists else next(iter(finalists))
    across = [dict(finalists[baseline], candidate_id=baseline)] + [dict(v, candidate_id=a)
                for a, v in finalists.items() if a != baseline]
    selected, decisions = choose(across, ranking, pack)
    selection = {'arm': selected['candidate_id'], 'artifact_sha256': selected['artifact_sha256'],
                 'product_id': selected['product_id'], 'decisions': decisions,
                 'source': selected['candidate_id'] + '/delivery_selection.json'}
    record('selected_delivery', selection)
    return {'ranking': ranking, 'selection': selection}
