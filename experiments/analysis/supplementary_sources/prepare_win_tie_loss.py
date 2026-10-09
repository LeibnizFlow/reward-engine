"""Reproduce task-level score comparisons from frozen benchmark snapshots."""
from pathlib import Path
import csv
import json
import math

ROOT = Path(__file__).resolve().parents[1]
TIE_TOLERANCE = 1e-8
ARMS = ('B', 'C', 'selected_score')
LABELS = {'B': 'Reward-guided revision', 'C': 'Hypothesis guidance',
          'selected_score': 'Selected by reward'}


def outcome(score, baseline):
    for value in (score, baseline):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 100:
            raise ValueError(f'Invalid score: {value!r}')
    delta = score - baseline
    return 'win' if delta > TIE_TOLERANCE else 'loss' if delta < -TIE_TOLERANCE else 'tie'


def compare(rows, arm, baseline):
    counts = dict.fromkeys(('win', 'tie', 'loss'), 0)
    for row in rows:
        counts[outcome(row[arm], row[baseline])] += 1
    n = len(rows)
    assert n and sum(counts.values()) == n
    return {**counts, 'n': n, 'win_or_tie': counts['win'] + counts['tie'],
            'win_or_tie_percent': 100 * (counts['win'] + counts['tie']) / n}


def main():
    wb = json.loads((ROOT / 'data/office50.json').read_text())['tasks']
    claw = json.loads((ROOT / 'data/claw_eval.json').read_text())['tasks']
    for rows, n, baseline in ((wb, 50, 'A'), (claw, 104, 'A_run')):
        assert len(rows) == len({r['task'] for r in rows}) == n
        for row in rows:
            chosen = baseline if row['selected_arm'] == 'A' else row['selected_arm']
            assert outcome(row['selected_score'], row[chosen]) == 'tie'
    groups = [('workbuddy', wb, 'A', '(a) WorkBuddy: 50 tasks, vs. task only'),
              ('claw_eval', claw, 'A_mean', '(b) claw-eval: 104 tasks, vs. task-only mean')]
    summaries, ledger = [], []
    lines = [r'\begin{tabular}{lrr}', r'\toprule',
             r'Method & W/T/L & W + T (\%) \\']
    for name, rows, baseline, title in groups:
        lines += [r'\midrule', r'\multicolumn{3}{l}{\textbf{' + title + r'}} \\']
        for arm in ARMS:
            stats = compare(rows, arm, baseline)
            summaries.append({'benchmark': name, 'baseline': baseline, 'arm': arm, **stats})
            for row in rows:
                ledger.append({'benchmark': name, 'task': row['task'], 'arm': arm,
                               'baseline': baseline, 'score': row[arm],
                               'baseline_score': row[baseline], 'delta': row[arm] - row[baseline],
                               'outcome': outcome(row[arm], row[baseline]), 'selected_arm': row['selected_arm']})
            cells = [LABELS[arm], f"{stats['win']}/{stats['tie']}/{stats['loss']}",
                     f"{stats['win_or_tie']}/{stats['n']} ({stats['win_or_tie_percent']:.2f})"]
            if arm == 'selected_score':
                cells = [r'\textbf{' + cell + '}' for cell in cells]
            lines.append(' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    selection_ties = {name: {'baseline': baseline,
                            'selected_A': sum(r['selected_arm'] == 'A' for r in rows),
                            'ties': compare(rows, 'selected_score', baseline)['tie']}
                      for name, rows, baseline in [('workbuddy', wb, 'A')]}
    output = {'definitions': {'tie_tolerance': TIE_TOLERANCE, 'score_scale': '0--100',
              'comparison': 'Unrounded task-level score difference; tie when absolute difference <= tolerance.',
              'selection': 'Archived runtime choice, not post hoc best score.',
              'win_or_tie': 'Descriptive observed score comparison, not statistical non-inferiority.'},
              'sources': ['data/office50.json', 'data/claw_eval.json'],
              'comparisons': summaries, 'selection_ties': selection_ties}
    (ROOT / 'data/benchmark_win_tie_loss.json').write_text(json.dumps(output, indent=2) + '\n')
    with (ROOT / 'data/benchmark_win_tie_loss.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ledger[0]))
        writer.writeheader()
        writer.writerows(ledger)
    (ROOT / 'tables/candidate_win_tie_loss.tex').write_text('\n'.join(lines) + '\n')
    main_lines = [line for line in lines if not line.startswith(('Reward-guided revision &', 'Hypothesis guidance &'))]
    (ROOT / 'tables/benchmark_win_tie_loss.tex').write_text('\n'.join(main_lines) + '\n')
    print(json.dumps({'status': 'passed', 'comparisons': len(summaries), 'task_comparisons': len(ledger)}, indent=2))


if __name__ == '__main__':
    main()
