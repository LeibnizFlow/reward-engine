"""Offline three-round reporting; preserve raw scores and historical endpoints."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('A', 'B', 'C', 'selected_score')
NAMES = ('Baseline', 'Revision', 'Hypothesis', 'Selected by reward')
# The main text reports the baseline, reward-guided revision, and final selection; the
# hypothesis-guided arm is reported in its own appendix table (hypothesis_arm.tex).
MAIN_ARMS = ('A', 'B', 'selected_score')
LABEL = dict(zip(ARMS, NAMES))
INVALID = ('analyst-forecast-extract-L3-018', 1, 'A')


def workbuddy_rows():
    path = ROOT / 'data/office50.json'
    doc = json.loads(path.read_text())
    prov = json.loads((ROOT / 'data/office50_provenance.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == prov['snapshot_sha256']
    rows, imputed = [], []
    for task in doc['rounds']:
        for raw in task['rounds']:
            row = dict(raw, task=task['task'], cluster=task['task'])
            for arm in filter(None, raw['invalid_scoring_arms'].split(',')):
                assert (task['task'], raw['round'], arm) == INVALID
                imputed.append(dict(task=task['task'], round=raw['round'], arm=arm,
                                    archived_score=raw[arm], analysis_score=0.0))
                row[arm] = 0.0
            row['selected_score'] = row[row['selected_arm']]
            rows.append(row)
    assert len(rows) == 150 and len(imputed) == 1
    final = {r['task']: r for r in rows if r['round'] == 3}
    for old in doc['tasks']:
        assert all(final[old['task']][k] == old[k] for k in (*ARMS, 'selected_arm'))
    return rows, imputed


def workbuddy_summary():
    from prepare_claw_continuation import aggregate_three_rounds
    rows, imputed = workbuddy_rows()
    result, ledger = aggregate_three_rounds(rows, threshold=80, cluster_resampling=False, seed=20260915)
    result['imputations'] = imputed
    result['snapshot_sha256'] = hashlib.sha256((ROOT / 'data/office50.json').read_bytes()).hexdigest()
    result['definitions']['invalid_scores'] = 'The sole invalid first-round A score is zero in this analysis only; source snapshot unchanged.'
    (ROOT / 'data/workbuddy_three_round_statistics.json').write_text(json.dumps(result, indent=2)+'\n')
    for name, records in [('workbuddy_three_round_summary.csv', ledger), ('workbuddy_three_round_analysis.csv', rows)]:
        with (ROOT / 'data' / name).open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(records)
    return result


def write_tables(out):
    from prepare_main_statistics import fmt_ci, fmt_delta
    summaries = [('WorkBuddy Bench Office', out['workbuddy_three_round'], 'task_bootstrap_ci95'),
                 ('claw-eval', out['claw_eval_three_round'], 'scenario_cluster_bootstrap_ci95')]

    def table(name, columns, header, rows):
        text = '\n'.join([r'\begin{tabular}{'+columns+'}', r'\toprule', header+r' \\',
                          r'\midrule', *rows, r'\bottomrule', r'\end{tabular}'])+'\n'
        (ROOT / 'tables' / name).write_text(text)

    def delta_cells(comp, interval):
        c = comp['mean_score']
        return [f"{fmt_delta(c['mean_difference'])} {fmt_ci(c[interval])}",
                f"{c['wins']}/{c['ties']}/{c['losses']}", f"{c['sign_test_two_sided_p']:.4f}"]

    main, success, diagnostics, utility, hypothesis = [], [], [], [], []
    for label, summary, interval in summaries:
        n, runs = summary['n_tasks'], summary['n_runs_per_arm']
        main.append(r'\multicolumn{6}{l}{\textit{'+f"{label}: three-round summary, {n} tasks, threshold {summary['threshold']}"+r'}} \\')
        diagnostics.append(r'\multicolumn{6}{l}{\textit{'+label+r'}} \\')
        hypothesis.append(r'\multicolumn{9}{l}{\textit{'+f"{label}: {n} tasks, threshold {summary['threshold']}"+r'}} \\')
        u = summary['utility']
        utility.append(f"{label} & {u['uniform_mean']:.2f} & {u['oracle_mean']:.2f} & {100*u['headroom_recovered']:.2f} & "
                       f"{u['mean_regret']['selected_score']:.2f} & {100*u['best_hit_rate']['selected_score']:.2f} ({u['best_hits']['selected_score']}/{runs})"+r' \\')
        best_mean = max(v['mean_score'] for v in summary['arms'].values())
        best_pass = max(v['passed_runs'] for v in summary['arms'].values())
        for key in MAIN_ARMS:
            v = summary['arms'][key]
            m, p = f"{v['mean_score']:.2f}", f"{100*v['average_pass_rate']:.2f}"
            if abs(v['mean_score']-best_mean) < 1e-8: m = r'\textbf{'+m+'}'
            if v['passed_runs'] == best_pass: p = r'\textbf{'+p+'}'
            cells = [LABEL[key], m, p]
            if key == 'A': cells += ['---', '---', '---']
            else:
                comp = summary['comparisons'][key+'_vs_A']
                c, rate = comp['mean_score'], comp['average_pass_rate']
                cells += delta_cells(comp, interval)
                diagnostics.append(' & '.join([LABEL[key], f"{c['median_difference']:+.2f}",
                    f"{c['sign_test_two_sided_p']:.4f}", f"{c['wilcoxon_two_sided_p']:.5f}",
                    f"{fmt_delta(100*rate['mean_difference'])} {fmt_ci([100*x for x in rate[interval]])}",
                    f"{v['passed_runs']}/{runs}"])+r' \\')
            main.append(' & '.join(cells)+r' \\')
        for key in ('A', 'C'):
            v = summary['arms'][key]
            cells = [LABEL[key], f"{v['mean_score']:.2f}", f"{100*v['average_pass_rate']:.2f} ({v['passed_runs']}/{runs})",
                     f"{100*v['pass_at_3']:.2f} ({v['pass_at_3_tasks']}/{n})",
                     f"{100*v['pass_power_3']:.2f} ({v['pass_power_3_tasks']}/{n})"]
            if key == 'A': cells += ['---', '---', '---', '---']
            else:
                comp = summary['comparisons']['C_vs_A']
                cells += delta_cells(comp, interval)+[f"{comp['mean_score']['wilcoxon_two_sided_p']:.5f}"]
            hypothesis.append(' & '.join(cells)+r' \\')
        main.append(r'\midrule'); diagnostics.append(r'\midrule'); hypothesis.append(r'\midrule')
    table('main_results.tex', 'lrrlrr', r'Method & Mean & Avg. pass (\%) & $\Delta$ mean [95\% CI] & W/T/L & Sign $p$', main[:-1])
    compact = []
    for key in MAIN_ARMS:
        cells = [LABEL[key]]
        for _, summary, _ in summaries:
            v, n = summary['arms'][key], summary['n_tasks']
            cells += [f"{100*v['pass_at_3']:.2f} ({v['pass_at_3_tasks']}/{n})",
                      f"{100*v['pass_power_3']:.2f} ({v['pass_power_3_tasks']}/{n})"]
        compact.append(' & '.join(cells)+r' \\')
    table('three_round_success.tex', 'lrrrr',
          r' & \multicolumn{2}{c}{WorkBuddy} & \multicolumn{2}{c}{claw-eval} \\'+ '\n' +
          r'Method & $\mathrm{pass@3}$ (\%) & $\mathrm{Pass}^{3}$ (\%) & $\mathrm{pass@3}$ (\%) & $\mathrm{Pass}^{3}$ (\%)', compact)
    table('three_round_diagnostics.tex', 'lrrrlr', r'Versus baseline & Median $\Delta$ & Sign $p$ & Wilcoxon $p$ & $\Delta$ avg. pass (pp) [95\% CI] & Pass runs', diagnostics[:-1])
    table('three_round_utility.tex', 'lrrrrr', r'Benchmark & Uniform & Oracle & Headroom (\%) & Regret & Best hit (\%)', utility)
    table('hypothesis_arm.tex', 'lrrrrlrrr',
          r'Method & Mean & Avg. pass (\%) & $\mathrm{pass@3}$ (\%) & $\mathrm{Pass}^{3}$ (\%) & $\Delta$ mean [95\% CI] & W/T/L & Sign $p$ & Wilcoxon $p$',
          hypothesis[:-1])
