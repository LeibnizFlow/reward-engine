"""Summarize original-pool utility (claw-eval round one), not the continuation.

The separate continuation summary supplies final-round metrics. No model calls.
"""
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from prepare_reward_agreement import ratio_summary, table

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT / 'data' / name).read_text())


def build():
    known = read('known_reward_metrics/summary.json')
    with (ROOT/'data/known_reward_metrics/candidate_scores.csv').open() as handle:
        known_candidates = list(csv.DictReader(handle))
    coverage = {b: [r for r in known_candidates if r['benchmark'] == b] for b in ('workbuddy', 'claw_eval')}
    for source, expected in known['protocol']['sources'].items():
        assert hashlib.sha256((ROOT / source).read_bytes()).hexdigest() == expected
    runtime, ledger = {}, []
    for name, filename, baseline, seed in [('workbuddy', 'office50.json', 'A', 20260915),
                                          ('claw_eval', 'claw_eval.json', 'A_run', 20260920)]:
        rows = read(filename)['tasks']
        scores = np.array([[r[baseline], r['B'], r['C']] for r in rows])
        chosen = np.array([r['selected_score'] for r in rows])
        assert np.isfinite(scores).all() and ((scores >= 0) & (scores <= 100)).all()
        assert all(r['selected_arm'] in 'ABC' and abs(chosen[i]-scores[i, 'ABC'.index(r['selected_arm'])]) < 1e-8
                   for i, r in enumerate(rows))
        clusters = [r['task'] if name == 'workbuddy' else re.sub(r'^T\d+(?:zh)?_', '', r['task']) for r in rows]
        uniform, oracle = scores.mean(axis=1), scores.max(axis=1)
        gain = ratio_summary(chosen-uniform, np.ones(len(rows)), clusters, 200000, seed)
        runtime[name] = dict(tasks=len(rows), clusters=len(set(clusters)), selected=float(chosen.mean()),
                             uniform=float(uniform.mean()), oracle=float(oracle.mean()), gain=gain,
                             headroom_recovered=float((chosen-uniform).mean()/(oracle-uniform).mean()), seed=seed)
        for i, row in enumerate(rows):
            ledger.append(dict(benchmark=name, task=row['task'], cluster=clusters[i], A=float(scores[i, 0]),
                               B=float(scores[i, 1]), C=float(scores[i, 2]), selected=float(chosen[i]),
                               uniform=float(uniform[i]), oracle=float(oracle[i]), gain=float(chosen[i]-uniform[i])))
    # Existing point estimates must agree with the independently stored main analysis.
    main = read('main_statistics.json')
    for name, values in runtime.items():
        assert abs(values['selected'] - main[name]['means']['reward_engine']) < 1e-10
        assert abs(values['headroom_recovered'] - main[name]['headroom_recovered_over_uniform']) < 1e-10

    def ci(value, places=2, scale=1, signed=False):
        v = value.get('value', value.get('mean')) * scale
        lo, hi = [x*scale for x in value['ci95']]
        return f"{v:+.{places}f} [{lo:.{places}f}, {hi:.{places}f}]" if signed else f"{v:.{places}f} [{lo:.{places}f}, {hi:.{places}f}]"

    benchmarks = ('workbuddy', 'claw_eval')
    rows = [r'\multicolumn{3}{l}{\textit{Runtime candidate pools: recorded Jury selection}}',
            r'Uniform $\rightarrow$ selected score & ' + ' & '.join(f"{runtime[b]['uniform']:.2f} $\\rightarrow$ {runtime[b]['selected']:.2f}" for b in benchmarks),
            r'Gain over uniform [95\% CI] & ' + ' & '.join(ci(runtime[b]['gain'], signed=True) for b in benchmarks),
            r'\midrule',
            r'\multicolumn{3}{l}{\textit{Separate retrospective pools: known-criterion scalar}}',
            'Candidates with a scalar & ' + ' & '.join(f"{sum(bool(r['known_reward']) for r in coverage[b])}/{len(coverage[b])}" for b in benchmarks),
            'Tasks with an externally ordered pair & ' + ' & '.join(f"{known[b]['known_all']['pair_evaluable_tasks']}/{known[b]['known_all']['tasks']}" for b in benchmarks),
            r'Pair agreement (\%) [95\% CI] & ' + ' & '.join(ci(known[b]['known_all']['half_accuracy'], places=1, scale=100) for b in benchmarks),
            r'Argmax gain over uniform [95\% CI] & ' + ' & '.join(ci(known[b]['known_all']['gain'], signed=True) for b in benchmarks)]
    # table() terminates each row; a rule is emitted separately to avoid a spurious line break.
    table('reward_utility_summary', 'lrr', 'Quantity & WorkBuddy & claw-eval', [r for r in rows if r != r'\midrule'])
    path = ROOT / 'tables/reward_utility_summary.tex'
    path.write_text(path.read_text().replace(r'\multicolumn{3}{l}{\textit{Separate', '\\midrule\n'+r'\multicolumn{3}{l}{\textit{Separate'))
    sources = ['office50.json', 'claw_eval.json', 'main_statistics.json', 'known_reward_metrics/summary.json', 'known_reward_metrics/candidate_scores.csv']
    result = dict(runtime=runtime, known={b: known[b]['known_all'] for b in benchmarks},
                  sources={f'data/{p}': hashlib.sha256((ROOT/'data'/p).read_bytes()).hexdigest() for p in sources},
                  definitions={'runtime': 'Task-paired selected minus analytic uniform score in the recorded A/B/C pool; 200000 percentile bootstrap draws, WorkBuddy task and claw-eval scenario clusters, task-weighted.',
                               'joint': 'Separate native-score-stratified retrospective candidates, P/(P+F) excluding unknowns; 20000 clustered bootstrap draws from existing summary.',
                               'scope': 'Exploratory fixed-pool utility, not a causal module ablation, direct judge comparison, or equal-total-budget experiment.',
                               'workbuddy_interval': 'Lower bound is approximately 0.009: marginal and not evidence of a robust effect.'})
    (ROOT/'data/reward_utility_summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    with (ROOT/'data/runtime_selection_utility.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ledger[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(ledger)
    print(json.dumps(runtime, indent=2))


if __name__ == '__main__':
    build()
