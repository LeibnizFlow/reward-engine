"""Reproduce benchmark pass rates and the compact main table from frozen scores.

No actor or grader calls. --verify-receipts also checks all 150 WorkBuddy and
624 claw-eval source receipts against the existing provenance snapshots.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[1]
BASELINE_IDS = ('A_run', 'seed_1', 'seed_2', 'seed_3')
THRESHOLDS = {'workbuddy': 85.0, 'claw_eval': 75.0}


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def passes(score, threshold):
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 100:
        raise ValueError(f'Invalid score: {score!r}')
    return score >= threshold


def baseline_passes(row):
    scores = row['baseline_scores']
    assert set(scores) == set(BASELINE_IDS)
    assert math.isclose(sum(scores.values()) / 4, row['A_mean'], abs_tol=1e-8)
    return [passes(scores[k], THRESHOLDS['claw_eval']) for k in BASELINE_IDS]


def summarize(values):
    return {'passes': sum(values), 'runs': len(values), 'rate_percent': 100 * sum(values) / len(values)}


def verify_receipts(runner, wb, claw):
    wb_index = {(r['task'], r['arm']): r for r in read(ROOT / 'data/workbuddy_full_archive_audit.json')['candidates']}
    evidence = {r['task']: r for r in read(ROOT / 'data/claw_eval_provenance.json')['evidence']}
    verified = []
    for row in wb:
        for arm in 'ABC':
            rec = wb_index[row['task'], arm]
            receipt = runner / rec['official_receipt']
            assert digest(receipt) == rec['official_receipt_sha256'], receipt
            path = runner / rec['native_score_file']
            native = read(path)
            score = next(native[k] for k in ('reward', 'overall', 'test_pass_rate') if isinstance(native.get(k), (int, float)))
            assert math.isclose(100 * score, row[arm], abs_tol=1e-7), (row['task'], arm, score, row[arm])
            native_pass = score >= THRESHOLDS['workbuddy'] / 100 and native.get('test_status') != 'build_error'
            assert native_pass == passes(row[arm], THRESHOLDS['workbuddy']), (row['task'], arm)
            verified.append({'benchmark': 'workbuddy', 'task': row['task'], 'arm': arm,
                             'receipt': str(receipt.relative_to(runner)), 'receipt_sha256': digest(receipt),
                             'native_score_file': str(path.relative_to(runner)), 'native_score_sha256': digest(path), 'pass': native_pass})
    for row in claw:
        refs = {r['id']: r for r in evidence[row['task']]['candidates']}
        for arm in (*BASELINE_IDS, 'B', 'C'):
            if arm.startswith('seed_'):
                path = runner / f"local/claw_eval/historical_run/tasks/{row['task']}/seeds/{arm[-1]}/OFFICIAL.private.json"
                expected = row['baseline_scores'][arm]
            else:
                actor = 'A' if arm == 'A_run' else arm
                path = runner / f"local/claw_eval/office104_web19_composite_20260924/tasks/{row['task']}/round_1/actors/{actor}/OFFICIAL.private.json"
                expected = row[arm]
            native = read(path)
            assert digest(path) == refs[arm]['receipt_sha256'], path
            assert native['status'] == 'graded'
            assert math.isclose(100 * native['task_score'], expected, abs_tol=1e-8)
            assert native['score_threshold_met'] == passes(expected, 75), path
            verified.append({'benchmark': 'claw_eval', 'task': row['task'], 'arm': arm,
                             'receipt': str(path.relative_to(runner)), 'receipt_sha256': digest(path), 'pass': native['score_threshold_met']})
    assert len(verified) == 774
    (ROOT / 'data/pass_rate_receipt_audit.json').write_text(json.dumps({'source_root': str(runner), 'verified_receipts': len(verified), 'records': verified}, indent=2) + '\n')
    return len(verified)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-receipts', type=Path, metavar='RUNNER')
    args = parser.parse_args()
    wb = read(ROOT / 'data/office50.json')['tasks']
    claw = read(ROOT / 'data/claw_eval.json')['tasks']
    assert len(wb) == len({r['task'] for r in wb}) == 50
    assert len(claw) == len({r['task'] for r in claw}) == 104
    ledger = []
    stats = {}
    for name, rows, arms in [('workbuddy', wb, ('A', 'B', 'C', 'selected_score')),
                             ('claw_eval', claw, ('B', 'C', 'selected_score'))]:
        threshold = THRESHOLDS[name]
        stats[name] = {}
        for arm in arms:
            values = [passes(r[arm], threshold) for r in rows]
            stats[name][arm] = summarize(values)
            for r, passed in zip(rows, values):
                assert math.isclose(r['selected_score'], r['A_run' if name == 'claw_eval' and r['selected_arm'] == 'A' else r['selected_arm']], abs_tol=1e-8)
                ledger.append({'benchmark': name, 'task': r['task'], 'arm': arm,
                               'score': r[arm], 'threshold': threshold, 'pass': int(passed),
                               'selected_arm': r['selected_arm']})
    four = [p for r in claw for p in baseline_passes(r)]
    stats['claw_eval']['A_four_run'] = summarize(four)
    for name, base in [('workbuddy', 'A'), ('claw_eval', 'A_four_run')]:
        stats[name]['difference_pp'] = {arm: stats[name][arm]['rate_percent'] - stats[name][base]['rate_percent']
                                        for arm in ('B', 'C', 'selected_score')}
    output = {'definitions': {'thresholds_on_100_scale': THRESHOLDS,
              'workbuddy': 'Official inclusive pass threshold 85/100 on final composite score, third iteration, all 50 tasks.',
              'claw_eval': 'Native single-trajectory threshold; custom judge. Not leaderboard Pass^3.',
              'A_four_run': 'Mean of four binary pass indicators per task, then mean over 104 tasks; not threshold of mean score.',
              'selected_score': 'Archived runtime choice among task-only, reward-guided revision, and Hypothesis guidance; never the post hoc highest score.'},
              'sources': {name: digest(ROOT / 'data' / name) for name in ('office50.json', 'claw_eval.json')},
              'statistics': stats}
    (ROOT / 'data/benchmark_pass_rates.json').write_text(json.dumps(output, indent=2) + '\n')
    with (ROOT / 'data/benchmark_pass_rates.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ledger[0])); writer.writeheader(); writer.writerows(ledger)
    def rate(name, arm):
        s = stats[name][arm]
        return f"{s['rate_percent']:.2f} ({s['passes']}/{s['runs']})"
    wb_stats = read(ROOT / 'data/office50_statistics.json')
    claw_stats = read(ROOT / 'data/claw_eval_statistics.json')
    lines = [r'\begin{tabular}{lrrr}', r'\toprule',
             r'\multicolumn{4}{l}{\textbf{(a) WorkBuddy Bench Office: 50 tasks}} \\',
             r'Method & Mean & Pass $\geq85$ (\%) & $\Delta$ vs. task only [95\% interval] \\', r'\midrule']
    for arm, label in [('A', 'Task only'), ('B', 'Reward-guided revision'), ('C', 'Hypothesis guidance'), ('selected_score', 'Selected by reward')]:
        mean = sum(r[arm] for r in wb) / 50
        contrast = '---'
        if arm != 'A':
            st = wb_stats['selected' if arm == 'selected_score' else arm]
            contrast = f"{st['delta']:+.2f} [{st['ci'][0]:.2f}, {st['ci'][1]:.2f}]"
        lines.append(f"{label} & {mean:.2f} & {rate('workbuddy', arm)} & {contrast}" + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', '',
              r'\begin{tabular}{lrrr}', r'\toprule',
              r'\multicolumn{4}{l}{\textbf{(b) claw-eval: 104 task versions}} \\',
              r'Method & Mean score & Pass rate (\%) & $\Delta$ vs. task-only mean \\', r'\midrule']
    means = {arm: sum(r[arm] for r in claw) / 104 for arm in ('A_mean', 'B', 'C', 'selected_score')}
    for arm, label in [('A_mean', 'Task-only mean'), ('B', 'Reward-guided revision'), ('C', 'Hypothesis guidance'), ('selected_score', 'Selected by reward')]:
        mean = means[arm]
        delta_mean = '---' if arm == 'A_mean' else f"{mean-means['A_mean']:+.2f}"
        if arm in ('B', 'C'):
            delta_mean = f"{claw_stats['vs_A_mean'][arm]['delta']:+.2f}"
        lines.append(f"{label} & {mean:.2f} & {rate('claw_eval', 'A_four_run' if arm == 'A_mean' else arm)} & {delta_mean}" + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (ROOT / 'tables/benchmark_candidates.tex').write_text('\n'.join(lines) + '\n')
    main_lines = [line for line in lines if not line.startswith(('Reward-guided revision &', 'Hypothesis guidance &'))]
    (ROOT / 'tables/benchmark_main.tex').write_text('\n'.join(main_lines) + '\n')
    count = verify_receipts(args.verify_receipts, wb, claw) if args.verify_receipts else None
    print(json.dumps({'status': 'passed', 'statistics': stats, 'verified_receipts': count}, indent=2))


if __name__ == '__main__':
    main()
