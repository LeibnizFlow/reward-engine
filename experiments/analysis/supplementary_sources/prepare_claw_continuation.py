"""Freeze and summarize all three claw-eval rounds, without new model calls.

Use --source RUNNER once to verify raw receipts and freeze a portable ledger.
Offline runs reproduce summaries from that ledger. The original round-three
endpoint remains intact alongside the post-hoc, three-round descriptive summary.
"""
import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

from prepare_main_statistics import cluster_bootstrap, paired_tests, task_bootstrap

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/claw_eval_continuation.json'
PROVENANCE = ROOT / 'data/claw_eval_continuation_provenance.json'
SUMMARY = ROOT / 'data/claw_eval_continuation_statistics.json'
SEED = 20261005


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def validate_rows(rows, original):
    """Reject incomplete/mixed pools and score-based recombinations."""
    old = {r['task']: r for r in original}
    assert len(old) == 104 and len(rows) == 312
    assert {(r['task'], r['round']) for r in rows} == {(t, n) for t in old for n in (1, 2, 3)}
    for row in rows:
        assert row['selected_arm'] in 'ABC' and len(row['selected_arm']) == 1
        for key in ('A', 'B', 'C', 'selected_score', 'A_mean_round1'):
            v = row[key]
            assert type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 100
        assert math.isclose(row['selected_score'], row[row['selected_arm']], abs_tol=1e-8)
        assert math.isclose(row['A_mean_round1'], old[row['task']]['A_mean'], abs_tol=1e-8)
        if row['round'] == 1:
            for key in ('A', 'B', 'C', 'selected_score'):
                assert math.isclose(row[key], old[row['task']]['A_run' if key == 'A' else key], abs_tol=1e-8)
            assert row['selected_arm'] == old[row['task']]['selected_arm']


def freeze(source):
    report = source / 'reports/claw_eval/claw-eval-report/results.json'
    doc = read(report)
    assert doc['status'] == 'completed' and doc['completed_rounds'] == 312 and doc['valid_scores'] == 936
    assert doc['primary_result'] == 'round_3_ABC_and_runtime_selection'
    hashes = {}

    def bind(path, expected=None):
        path = Path(path)
        value = digest(path)
        if expected is not None:
            assert value == expected, str(path)
        hashes[str(path.relative_to(source))] = value
        return value

    bind(report)
    for name, value in doc['source_files_sha256'].items():
        bind(name, value)
    original = read(ROOT / 'data/claw_eval.json')['tasks']
    validate_rows(doc['tasks'], original)
    rows, evidence, packs = [], [], {str(n): {'packs': 0, 'criteria': 0, 'tests': 0} for n in (1, 2, 3)}
    for row in sorted(doc['tasks'], key=lambda r: (r['task'], r['round'])):
        phase = Path(row['checkpoint']).parent
        selection = read(phase / 'selected_delivery.json')
        assert selection['arm'] == row['selected_arm']
        bind(phase / 'selected_delivery.json')
        bind(phase / 'summary.private.json')
        pack = read(phase / 'pack.json')
        bind(phase / 'pack.json')
        counts = packs[str(row['round'])]
        counts['packs'] += 1
        for key in ('criteria', 'tests'):
            counts[key] += len(pack[key])
        for arm in 'ABC':
            e = row['evidence'][arm]
            receipt = read(Path(e['receipt']))
            actor = read(Path(e['actor']))
            assert receipt['status'] == 'graded' and actor['status'] == 'completed'
            assert receipt['protocol'] == 'native-claw-eval-custom-judge'
            assert receipt['judge_model'] == 'deepseek-v4-flash'
            assert math.isclose(receipt['task_score'] * 100, row[arm], abs_tol=1e-8)
            assert receipt['trace_id'] == actor['candidate']['candidate_id'] == e['candidate_id']
            trial = Path(actor['native_grader_trial'])
            trace_hash = bind(trial / 'trace.jsonl', receipt['binding']['trace'])
            snapshot_hash = bind(trial / 'environment-snapshot.json', receipt['binding']['snapshot'])
            evidence.append(dict(task=row['task'], round=row['round'], arm=arm,
                                 candidate_id=e['candidate_id'], receipt=str(Path(e['receipt']).relative_to(source)),
                                 receipt_sha256=bind(e['receipt']), actor_sha256=bind(e['actor']),
                                 trace_sha256=trace_hash, snapshot_sha256=snapshot_hash))
        fields = ('task', 'round', 'A', 'B', 'C', 'selected_arm', 'selected_score', 'A_mean_round1')
        rows.append({**{k: row[k] for k in fields}, 'cluster': re.sub(r'^T\d+(?:zh)?_', '', row['task'])})
    for update in doc['round_updates']:
        for key in ('context', 'jury', 'pack', 'hypothesis', 'checkpoint'):
            bind(update[key])
    for campaign in sorted({r['source_campaign'] for r in doc['tasks']}):
        bind(Path(campaign) / 'config.json')
    for replacement in doc['actor_replacement_attempts']:
        bind(replacement['receipt'])
    for amendment in doc['role_output_budget_amendments']:
        bind(Path(amendment['source_campaign']) / 'role_limits.json', amendment['limits_sha256'])
        for name, value in amendment['applied_requests'].items():
            bind(name, value)
    for name in ('continuation_report.py', 'continue_rounds.py', 'continuation_pipeline.py'):
        bind(source / 'experiments/claw_eval/opencode' / name)
    write(DATA, dict(schema='claw_eval_continuation.v1', primary_round=3, n_tasks=104,
                     rounds=3, valid_scores=936, n_scenario_clusters=68, tasks=rows,
                     pack_inventory=packs, actor_replacement_attempts=doc['actor_replacement_count'],
                     role_output_budget_amendments=len(doc['role_output_budget_amendments']),
                     protocol='Sequential same-task continuation, not independent trials or unseen-task transfer'))
    write(PROVENANCE, dict(snapshot_sha256=digest(DATA), original_snapshot_sha256=digest(ROOT / 'data/claw_eval.json'),
                           report_sha256=digest(report), native_protocol='native-claw-eval-custom-judge',
                           judge_model='deepseek-v4-flash', source_files_sha256=hashes, evidence=evidence))
    print(f'Verified {len(evidence)} native receipts and {len(hashes)} source bindings.')


def comparison(x, y, clusters):
    delta = x - y
    return dict(mean_difference=float(delta.mean()),
                task_bootstrap_ci95=task_bootstrap(delta, SEED),
                scenario_cluster_bootstrap_ci95=cluster_bootstrap(delta, clusters, SEED),
                **paired_tests(x, y))


ARMS = ('A', 'B', 'C', 'selected_score')


def three_round_arrays(rows):
    """Align by task/round, retaining whole three-round blocks for each arm."""
    grouped = {}
    for row in rows:
        task = grouped.setdefault(row['task'], {})
        assert row['round'] in (1, 2, 3) and row['round'] not in task
        assert row['selected_arm'] in ('A', 'B', 'C')
        assert math.isclose(row['selected_score'], row[row['selected_arm']], abs_tol=1e-8)
        task[row['round']] = row
    assert grouped
    tasks = sorted(grouped)
    for task in tasks:
        assert set(grouped[task]) == {1, 2, 3}
        assert len({r['cluster'] for r in grouped[task].values()}) == 1
    clusters = [grouped[t][1]['cluster'] for t in tasks]
    arrays = {k: np.array([[grouped[t][n][k] for n in (1, 2, 3)] for t in tasks]) for k in ARMS}
    assert all(np.isfinite(a).all() and ((a >= 0) & (a <= 100)).all() for a in arrays.values())
    return tasks, clusters, arrays


def three_round_metrics(arrays, threshold=75):
    """Pass indicators are computed before averaging; all-pass is not p cubed."""
    return {k: dict(mean_score=a.mean(axis=1), average_pass_rate=(a >= threshold).mean(axis=1),
                   pass_at_3=(a >= threshold).any(axis=1).astype(float),
                   pass_power_3=(a >= threshold).all(axis=1).astype(float)) for k, a in arrays.items()}


def aggregate_three_rounds(rows, threshold=75, cluster_resampling=True, seed=SEED):
    tasks, clusters, arrays = three_round_arrays(rows)
    metrics = three_round_metrics(arrays, threshold)

    interval = 'scenario_cluster_bootstrap_ci95' if cluster_resampling else 'task_bootstrap_ci95'

    def paired(x, y):
        delta = x - y
        difference = float(delta.mean())
        return dict(mean_difference=0.0 if abs(difference) < 1e-12 else difference,
                    **{interval: (cluster_bootstrap(delta, clusters, seed) if cluster_resampling
                                  else task_bootstrap(delta, seed))})

    pairs = [(k, 'A') for k in ARMS[1:]] + [('selected_score', 'B')]
    uniform = np.mean([arrays[k] for k in 'ABC'], axis=0).mean(axis=1)
    oracle = np.maximum.reduce([arrays[k] for k in 'ABC']).mean(axis=1)
    selected = metrics['selected_score']['mean_score']
    result = dict(n_tasks=len(tasks), n_rounds=3, n_runs_per_arm=3*len(tasks),
        n_scenario_clusters=len(set(clusters)), threshold=threshold,
        bootstrap=dict(iterations=200000, seed=seed, unit="scenario" if cluster_resampling else "task"),
        interpretation='Post-hoc descriptive aggregation of dependent sequential rounds, not independent trials or leaderboard Pass^3.',
        definitions=dict(mean_score='Mean of per-task three-round mean scores; equal task weights.',
            average_pass_rate='Successful runs / (3 * n_tasks); threshold each run before averaging.',
            pass_at_3='Tasks with at least one successful sequential round / n_tasks.',
            pass_power_3='Tasks successful in all three sequential rounds / n_tasks; not marginal p cubed.',
            selected_score='Recorded runtime selection within each round; no post-hoc candidate choice.',
            bootstrap=('Resample scenario clusters, retaining all task versions, rounds and arms together; task-version weighted.'
                       if cluster_resampling else 'Resample tasks, retaining all three rounds and arms together.'),
            oracle='Mean of within-task, within-round max(A,B,C), never a best-round choice.'),
        arms={k: dict(mean_score=float(m['mean_score'].mean()),
                     passed_runs=int((arrays[k] >= threshold).sum()),
                     average_pass_rate=float(m['average_pass_rate'].mean()),
                     pass_at_3_tasks=int(m['pass_at_3'].sum()), pass_at_3=float(m['pass_at_3'].mean()),
                     pass_power_3_tasks=int(m['pass_power_3'].sum()), pass_power_3=float(m['pass_power_3'].mean()))
              for k, m in metrics.items()},
        comparisons={f'{x}_vs_{y}': {metric: paired(metrics[x][metric], metrics[y][metric])
                                   for metric in metrics[x]} for x, y in pairs},
        utility=dict(uniform_mean=float(uniform.mean()), oracle_mean=float(oracle.mean()),
                     selected_vs_uniform=paired(selected, uniform),
                     headroom_recovered=float((selected-uniform).mean()/(oracle-uniform).mean()),
                     best_fixed_arm=max('ABC', key=lambda k: arrays[k].mean())))
    for x, y in pairs:
        result['comparisons'][f'{x}_vs_{y}']['mean_score'].update(
            paired_tests(metrics[x]['mean_score'], metrics[y]['mean_score']))
    round_oracle = np.maximum.reduce([arrays[k] for k in 'ABC'])
    result['utility']['mean_regret'] = {k: float((round_oracle-arrays[k]).mean()) for k in ARMS}
    result['utility']['best_hits'] = {k: int(np.isclose(arrays[k], round_oracle, atol=1e-8, rtol=0).sum()) for k in ARMS}
    result['utility']['best_hit_rate'] = {k: v / (3*len(tasks)) for k, v in result['utility']['best_hits'].items()}
    result['selected_arm_counts'] = {k: sum(r['selected_arm'] == k for r in rows) for k in 'ABC'}
    ledger = []
    for i, task in enumerate(tasks):
        row = dict(task=task, cluster=clusters[i])
        for k in ARMS:
            row.update({f'{k}_mean': float(metrics[k]['mean_score'][i]),
                        f'{k}_pass_count': int((arrays[k][i] >= threshold).sum())})
        ledger.append(row)
    return result, ledger


def write_aggregate_tables(result):
    from prepare_main_statistics import fmt_ci, fmt_delta
    headers = [r'\begin{tabular}{lrrrr}', r'\toprule',
        r'Comparison & $\Delta$ mean & $\Delta$ avg. pass (pp) & $\Delta\mathrm{pass@3}$ (pp) & $\Delta\mathrm{Pass}^{3}$ (pp) \\', r'\midrule']
    # The hypothesis-guided arm is reported separately (prepare_three_round_statistics.py).
    names = {'B_vs_A': r'Revision $-$ baseline', 'selected_score_vs_A': r'Selected $-$ baseline',
             'selected_score_vs_B': r'Selected $-$ revision'}
    for key, values in result['comparisons'].items():
        if key not in names:
            continue
        cells = []
        for metric, scale in [('mean_score', 1), ('average_pass_rate', 100), ('pass_at_3', 100), ('pass_power_3', 100)]:
            v = values[metric]
            cells.append(f"{fmt_delta(scale*v['mean_difference'])} {fmt_ci([scale*x for x in v['scenario_cluster_bootstrap_ci95']])}")
        headers.append(names[key]+' & '+' & '.join(cells)+r' \\')
    (ROOT/'tables/claw_eval_three_round_comparisons.tex').write_text('\n'.join(headers+[r'\bottomrule', r'\end{tabular}'])+'\n')
    sensitivity = result['known_exposure_sensitivity']
    lines = [r'\begin{tabular}{lrrrr}', r'\toprule',
        r'Arm & Mean & Avg. pass (\%) & $\mathrm{pass@3}$ (\%) & $\mathrm{Pass}^{3}$ (\%) \\', r'\midrule']
    for key in ARMS:
        v = sensitivity['arms'][key]
        name = {'A': 'Baseline', 'B': 'Revision', 'C': 'Hypothesis', 'selected_score': 'Selected by reward'}[key]
        cells = [f"{v['mean_score']:.2f}"]
        for rate, count, n in [('average_pass_rate', 'passed_runs', sensitivity['n_runs_per_arm']),
                             ('pass_at_3', 'pass_at_3_tasks', sensitivity['n_tasks']),
                             ('pass_power_3', 'pass_power_3_tasks', sensitivity['n_tasks'])]:
            cells.append(f"{100*v[rate]:.2f} ({v[count]}/{n})")
        lines.append(name+' & '+' & '.join(cells)+r' \\')
    (ROOT/'tables/claw_eval_three_round_sensitivity.tex').write_text('\n'.join(lines+[r'\bottomrule', r'\end{tabular}'])+'\n')


def summary():
    doc, prov = read(DATA), read(PROVENANCE)
    assert digest(DATA) == prov['snapshot_sha256']
    assert digest(ROOT / 'data/claw_eval.json') == prov['original_snapshot_sha256']
    validate_rows(doc['tasks'], read(ROOT / 'data/claw_eval.json')['tasks'])
    result = dict(primary_round=3, snapshot_sha256=digest(DATA), bootstrap=dict(iterations=200000, seed=SEED,
                  primary_unit='68 scenario clusters, equally weighted sampled task versions',
                  secondary_unit='104 paired task versions', multiplicity='nominal, exploratory'), rounds={})
    arrays = {}
    for number in (1, 2, 3):
        rows = [r for r in doc['tasks'] if r['round'] == number]
        clusters = [r['cluster'] for r in rows]
        assert len(set(clusters)) == 68
        a = {key: np.array([r[key] for r in rows]) for key in ('A', 'B', 'C', 'selected_score', 'A_mean_round1')}
        a['uniform'] = (a['A'] + a['B'] + a['C']) / 3
        a['oracle'] = np.maximum.reduce([a[k] for k in 'ABC'])
        arrays[number] = a
        result['rounds'][str(number)] = dict(n_tasks=104, n_scenario_clusters=68,
            means={k: float(v.mean()) for k, v in a.items()},
            pass_at_75={k: int((a[k] >= 75).sum()) for k in ('A', 'B', 'C', 'selected_score')},
            comparisons={f'{k}_vs_A': comparison(a[k], a['A'], clusters) for k in ('B', 'C', 'selected_score')},
            selected_vs_uniform=comparison(a['selected_score'], a['uniform'], clusters),
            selected_vs_B=comparison(a['selected_score'], a['B'], clusters),
            selected_vs_original_A_mean=comparison(a['selected_score'], a['A_mean_round1'], clusters),
            headroom_recovered=float((a['selected_score']-a['uniform']).mean()/(a['oracle']-a['uniform']).mean()),
            best_fixed_arm=max('ABC', key=lambda k: a[k].mean()),
            mean_regret={k: float((a['oracle']-a[k]).mean()) for k in ('A', 'B', 'C', 'selected_score')},
            best_hits={k: int(np.isclose(a[k], a['oracle'], atol=1e-8, rtol=0).sum()) for k in ('A', 'B', 'C', 'selected_score')},
            selected_arm_counts={k: sum(r['selected_arm'] == k for r in rows) for k in 'ABC'})
    result['round3_minus_round1'] = {k: comparison(arrays[3][k], arrays[1][k], clusters)
                                   for k in ('A', 'B', 'C', 'selected_score')}
    exposure_path = ROOT / 'data/claw_eval_reference_exposure.json'
    exposure = read(exposure_path)
    affected = {r['task'] for r in exposure['affected_tasks']}
    assert affected <= {r['task'] for r in doc['tasks']}
    kept = [r for r in doc['tasks'] if r['round'] == 3 and r['task'] not in affected]
    kept_clusters = [r['cluster'] for r in kept]
    ka, ks = (np.array([r[k] for r in kept]) for k in ('A', 'selected_score'))
    result['known_exposure_sensitivity'] = dict(excluded_tasks=sorted(affected),
        exposure_index_sha256=digest(exposure_path), n_tasks=len(kept), n_scenario_clusters=len(set(kept_clusters)),
        task_only_mean=float(ka.mean()), selected_mean=float(ks.mean()),
        comparison=comparison(ks, ka, kept_clusters),
        interpretation='Excludes confirmed exposure only; remaining tasks are not certified clean.')
    aggregate, ledger = aggregate_three_rounds(doc['tasks'])
    sensitivity, _ = aggregate_three_rounds([r for r in doc['tasks'] if r['task'] not in affected])
    sensitivity.update(excluded_tasks=sorted(affected), exposure_index_sha256=digest(exposure_path),
                       interpretation='Excludes confirmed exposure only; remaining tasks are not certified clean.')
    aggregate['known_exposure_sensitivity'] = sensitivity
    result['presentation_summary'] = 'three_round_mean'
    result['three_round_summary'] = aggregate
    with (ROOT/'data/claw_eval_three_round_summary.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ledger[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(ledger)
    write_aggregate_tables(aggregate)
    write(SUMMARY, result)
    with (ROOT / 'data/claw_eval_rounds.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(doc['tasks'][0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(doc['tasks'])
    lines = [r'\begin{tabular}{lrrrrrl}', r'\toprule',
             r'Round & Baseline & Revision & Hypothesis & Selected & Pass (\%) & Selected $-$ baseline [95\% CI] \\', r'\midrule']
    for n, r in result['rounds'].items():
        m, c = r['means'], r['comparisons']['selected_score_vs_A']
        lo, hi = c['scenario_cluster_bootstrap_ci95']
        lines.append(f"{n} & {m['A']:.2f} & {m['B']:.2f} & {m['C']:.2f} & {m['selected_score']:.2f} & "
                     f"{100*r['pass_at_75']['selected_score']/104:.2f} & {c['mean_difference']:+.2f} [{lo:.2f}, {hi:.2f}] " + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (ROOT / 'tables/claw_eval_rounds.tex').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'rounds': {n: r['means'] for n, r in result['rounds'].items()},
                      'final': result['rounds']['3']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    args = parser.parse_args()
    if args.source:
        freeze(args.source.resolve())
    summary()
