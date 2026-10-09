"""Offline reward agreement and candidate-selection analysis; no model calls.

Dependencies: numpy and scipy. Frozen inputs are never modified.
"""
from collections import defaultdict
from pathlib import Path
from fractions import Fraction
import csv
import hashlib
import json
import re
import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
SEED, REPLICATES, TOL = 20260928, 20000, 1e-8


def read(name):
    return json.loads((ROOT / 'data' / name).read_text())


def checked(values, maximum):
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values):
        raise ValueError('Scores must be numeric and nonmissing')
    a = np.asarray(values, dtype=float)
    if not np.isfinite(a).all() or (a < 0).any() or (a > maximum).any():
        raise ValueError('Scores outside finite declared scale')
    return a


def paired_rewards(tasks, records):
    ids = [t['task'] for t in tasks]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate task')
    groups, unique, packs = defaultdict(list), set(), defaultdict(set)
    for r in records:
        key = r['task'], r['arm'], r['test_id']
        if key in unique or r['task'] not in ids or r['arm'] not in ('A', 'B', 'C'):
            raise ValueError('Duplicate or unmatched judgment')
        unique.add(key)
        checked([r['score']], 1)
        groups[r['task'], r['arm']].append(r)
        packs[r['task']].add(r['pack_id'])
    ledger = []
    for t in tasks:
        sets = [{r['test_id'] for r in groups[t['task'], a]} for a in 'ABC']
        if not sets[0] or not sets[0] == sets[1] == sets[2] or len(packs[t['task']]) != 1:
            raise ValueError('Incomplete or mismatched task/pack/arm pairing')
        checked([t[a] for a in 'ABC'], 100)
        for a in 'ABC':
            rs = groups[t['task'], a]
            exact = sum(Fraction(str(r['score'])) for r in rs) / len(rs)
            ledger.append(dict(task=t['task'], arm=a, pack_id=rs[0]['pack_id'],
                               n_tests=len(rs), reward=float(exact), reward_fraction=str(exact),
                               external_score=t[a],
                               empty_artifact=any(not r['artifact_chars'] for r in rs),
                               missing_inputs=any(not r['include_inputs'] for r in rs)))
    x = np.array([r['reward'] for r in ledger]).reshape(-1, 3)
    y = np.array([r['external_score'] for r in ledger]).reshape(-1, 3)
    return x, y, ledger


def corr(x, y, ranked=False):
    """Last-axis correlation; constants remain undefined (NaN)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if ranked:
        x, y = rankdata(x, axis=-1), rankdata(y, axis=-1)
    x, y = x - x.mean(axis=-1, keepdims=True), y - y.mean(axis=-1, keepdims=True)
    denom = np.sqrt((x*x).sum(axis=-1) * (y*y).sum(axis=-1))
    with np.errstate(divide='ignore', invalid='ignore'):
        return (x*y).sum(axis=-1) / denom


def interval(values):
    a = np.asarray(values)
    valid = a[np.isfinite(a)]
    return dict(ci95=np.quantile(valid, [.025, .975]).tolist() if len(valid) else None,
                valid_replicates=len(valid), undefined_replicates=len(a)-len(valid))


def correlations(x, y, replicates=REPLICATES):
    if x.shape != y.shape or x.ndim != 2 or not len(x):
        raise ValueError('Expected paired nonempty task by arm matrices')
    rng, boots = np.random.default_rng(SEED), {False: [], True: []}
    for start in range(0, replicates, 500):
        ix = rng.integers(len(x), size=(min(500, replicates-start), len(x)))
        bx, by = x[ix].reshape(len(ix), -1), y[ix].reshape(len(ix), -1)
        for ranked in boots:
            boots[ranked].extend(corr(bx, by, ranked).tolist())
    out = dict(n_tasks=len(x), n_artifacts=x.size)
    for ranked, name in [(False, 'pearson'), (True, 'spearman')]:
        value = float(corr(x.ravel(), y.ravel(), ranked))
        out[name] = dict(value=value if np.isfinite(value) else None, **interval(boots[ranked]))
    return out


def ratio_summary(numerators, denominators, clusters, replicates=REPLICATES, seed=SEED):
    """Task-weighted ratio, resampling whole clusters including all their rows."""
    grouped = defaultdict(lambda: [0., 0.])
    for n, d, key in zip(numerators, denominators, clusters, strict=True):
        grouped[key][0] += n
        grouped[key][1] += d
    values = np.array(list(grouped.values()))
    rng, boots = np.random.default_rng(seed), []
    for start in range(0, replicates, 500):
        ix = rng.integers(len(values), size=(min(500, replicates-start), len(values)))
        sums = values[ix].sum(axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            boots.extend((sums[:, 0] / sums[:, 1]).tolist())
    den = values[:, 1].sum()
    return dict(value=float(values[:, 0].sum()/den) if den else None,
                n_clusters=len(values), **interval(boots))


def sign(v):
    return 0 if abs(v) <= TOL else 1 if v > 0 else -1


def pairwise(tasks, x, y):
    ledger = []
    for t, rx, ry in zip(tasks, x, y, strict=True):
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            rs, es = sign(rx[i]-rx[j]), sign(ry[i]-ry[j])
            label = ('external_tie' if not es else 'reward_tie' if not rs
                     else 'concordant' if rs == es else 'discordant')
            ledger.append(dict(task=t['task'], arms='ABC'[i]+'ABC'[j],
                               reward_difference=float(rx[i]-rx[j]),
                               external_difference=float(ry[i]-ry[j]),
                               reward_sign=rs, external_sign=es, outcome=label))
    counts = {k: sum(r['outcome'] == k for r in ledger)
              for k in ['concordant', 'discordant', 'external_tie', 'reward_tie']}
    counts['strict_agreement'] = ratio_summary(
        [r['outcome'] == 'concordant' for r in ledger],
        [r['external_sign'] != 0 for r in ledger], [r['task'] for r in ledger])
    counts['both_nontied_agreement'] = ratio_summary(
        [r['outcome'] == 'concordant' for r in ledger],
        [r['external_sign'] != 0 and r['reward_sign'] != 0 for r in ledger],
        [r['task'] for r in ledger])
    return counts, ledger


def selection(rows, baseline, scenario=False, replicates=200000):
    ids = [r['task'] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate selection task')
    v = np.array([checked([r[baseline], r['B'], r['C']], 100) for r in rows])
    chosen = checked([r['selected_score'] for r in rows], 100)
    for r, score in zip(rows, chosen, strict=True):
        if r['selected_arm'] not in ('A', 'B', 'C') or abs(score-r[baseline if r['selected_arm']=='A' else r['selected_arm']]) > TOL:
            raise ValueError('Selection is not its recorded candidate')
    clusters = [re.sub(r'^T\d+(?:zh)?_', '', task) if scenario else task for task in ids]
    best = v.max(axis=1)
    policies = dict(always_A=v[:, 0], always_B=v[:, 1], always_C=v[:, 2],
                    uniform=v.mean(axis=1), selected=chosen, oracle=best)
    def estimate(values):
        return ratio_summary(values, np.ones(len(rows)), clusters, replicates,
                             seed=20260920 if scenario else 20260915)
    result, ledger = {}, []
    for policy, scores in policies.items():
        hits = (np.abs(v-best[:, None]) <= TOL).mean(axis=1) if policy == 'uniform' else (np.abs(scores-best) <= TOL).astype(float)
        result[policy] = dict(mean=float(scores.mean()),
                              delta_vs_B=estimate(scores-v[:, 1]),
                              best_hit_count=float(hits.sum()), best_hit_rate=float(hits.mean()),
                              regret=estimate(best-scores))
        for r, s, b, h, key in zip(rows, scores, best, hits, clusters, strict=True):
            ledger.append(dict(task=r['task'], cluster=key, policy=policy, score=float(s),
                               selected_arm=r['selected_arm'], oracle_score=float(b), regret=float(b-s), best_hit=float(h)))
    return dict(n_tasks=len(rows), n_clusters=len(set(clusters)), policies=result), ledger


def table(name, spec, heading, rows):
    (ROOT/'tables'/f'{name}.tex').write_text('\n'.join([
        r'\begin{tabular}{'+spec+'}', r'\toprule', heading+r' \\', r'\midrule',
        *[r+r' \\' for r in rows], r'\bottomrule', r'\end{tabular}'])+'\n')


def fmt(s, places=3):
    if s['value'] is None:
        return 'Undefined'
    return f"{s['value']:.{places}f} [{s['ci95'][0]:.{places}f}, {s['ci95'][1]:.{places}f}]"


def save_csv(name, rows):
    with (ROOT/'data'/name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n"); w.writeheader(); w.writerows(rows)


def main():
    sources = ['office50.json', 'claw_eval.json', 'semantic_archive.json',
               'semantic_archive_statistics.json', 'semantic_input_statistics.json']
    tasks = read('office50.json')['tasks']
    x, y, ledger = paired_rewards(tasks, read('semantic_archive.json')['records'])
    assert x.shape == (50, 3)
    groups = {'All A/B/C': correlations(x, y)}
    for i, a in enumerate('ABC'):
        groups[a] = correlations(x[:, i:i+1], y[:, i:i+1])
    sens = {}
    exclusion = {
        'Input availability': read('semantic_archive_statistics.json')['excluded_input_scope_tasks'],
        'Input routing': read('semantic_input_statistics.json')['routing_excluded_tasks'],
        'Grader exception': ['priority-sync-notification-pipeline']}
    for label, excluded in exclusion.items():
        mask = [t['task'] not in excluded for t in tasks]
        sens[label] = {**correlations(x[mask], y[mask]), 'excluded_tasks': excluded}
    exact_x = [[Fraction(r['reward_fraction']) for r in ledger[i:i+3]] for i in range(0, len(ledger), 3)]
    exact_y = [[Fraction(str(t[a])) for a in 'ABC'] for t in tasks]
    def center(rows):
        return np.array([[float(v-sum(row)/len(row)) for v in row] for row in rows])
    centered = correlations(center(exact_x), center(exact_y))
    pairs, pair_ledger = pairwise(tasks, x, y)
    sels, sel_ledger = {}, []
    for name, rs, base, scenario in [('WorkBuddy', tasks, 'A', False),
                                    ('claw-eval', read('claw_eval.json')['tasks'], 'A_run', True)]:
        sels[name], ls = selection(rs, base, scenario)
        sel_ledger.extend([dict(benchmark=name, **r) for r in ls])
    assert sels['claw-eval']['n_clusters'] == 68
    result = dict(
        scope='Post hoc offline archive analysis; text-only reward is not a full executable Reward Engine scalar.',
        definitions=dict(reward='Unweighted mean of cached decimal test scores per task/arm, scale 0--1; exact rational aggregation before float conversion preserves genuine ties.',
                         external='Accepted third-iteration WorkBuddy benchmark composite, scale 0--100.',
                         correlation_unit='Artifact; resampling unit is task, retaining all included arms.',
                         centered='Subtract the three-arm task mean independently from reward and external score.',
                         pairwise='Strict correct order / external-nontied pairs; reward ties count as not correct. External ties reported separately.',
                         tolerance=TOL, selection='Recorded A/B/C pool; claw A is A_run, never A_mean.',
                         uniform='Analytic expectation of uniform random choice; hit probability includes all tied maxima.',
                         oracle='Post hoc candidate maximum using external scores, not deployable.',
                         bootstrap=dict(seed=SEED, replicates=REPLICATES, interval='percentile',
                                        selection='200000 replicates; WorkBuddy seed 20260915/task clusters; claw seed 20260920/68 scenario clusters; task-weighted ratio of resampled sums.')),
        sources={f'data/{s}': hashlib.sha256((ROOT/'data'/s).read_bytes()).hexdigest() for s in sources},
        correlations=groups, centered=centered, sensitivities=sens, pairwise=pairs, selection=sels)
    (ROOT/'data/reward_agreement.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    save_csv('reward_agreement_pairs.csv', ledger)
    save_csv('reward_pairwise_order.csv', pair_ledger)
    save_csv('selection_quality.csv', sel_ledger)
    table('reward_agreement', 'lrrll', r'Artifact pool & Tasks & Artifacts & Pearson $r$ [95\% CI] & Spearman $\rho$ [95\% CI]',
          [f"{dict(A='Baseline', B='Reward-guided revision', C='Hypothesis guidance').get(label, 'All candidates')} & {s['n_tasks']} & {s['n_artifacts']} & {fmt(s['pearson'])} & {fmt(s['spearman'])}" for label, s in groups.items()])
    table('reward_agreement_sensitivity', 'lrrll', r'Whole-task exclusion & Tasks & Artifacts & Pearson $r$ [95\% CI] & Spearman $\rho$ [95\% CI]',
          [f"{label} & {s['n_tasks']} & {s['n_artifacts']} & {fmt(s['pearson'])} & {fmt(s['spearman'])}" for label, s in sens.items()])
    labels = dict(always_A='Always baseline', always_B='Always reward-guided revision',
                  always_C='Always hypothesis guidance', uniform='Uniform expectation',
                  selected='Selected by reward', oracle='Post hoc best candidate')
    rows = []
    for name, s in sels.items():
        rows.append(r'\multicolumn{6}{l}{\textbf{'+name+f" ({s['n_tasks']} tasks)"+r'}}')
        for policy, p in s['policies'].items():
            hits = f"{p['best_hit_count']:.2f}" if policy == 'uniform' else str(int(p['best_hit_count']))
            rows.append(f"{labels[policy]} & {p['mean']:.2f} & {fmt(p['delta_vs_B'], 2)} & {hits}/{s['n_tasks']} & {100*p['best_hit_rate']:.2f} & {p['regret']['value']:.2f}")
    table('selection_quality', 'lr lrrr', r'Policy & Score & $\Delta$ vs. reward-guided revision [95\% CI] & Best hits & Hit rate (\%) & Regret', rows)
    print(json.dumps(dict(correlations=groups, centered=centered, pairwise=pairs, selection=sels), indent=2))


if __name__ == '__main__':
    main()
