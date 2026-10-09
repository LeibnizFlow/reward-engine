"""Offline Jury agreement on native-score gaps; no model or grader calls.

The restricted tau-b-style statistic counts only eligible pairs. It is not
standard Kendall tau-b on an induced candidate subset. Partial pools contribute
only pairs whose two Jury ranks exist; official eligibility is counted even when
ranks are missing. All estimates retain the original task/scenario sampling frame.
"""
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from compute_joint_offline_metrics import bootstrap_mean, BOOTSTRAPS, SEED
from prepare_six_baseline import inputs
from prepare_expanded_ranking import jury_scores

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'jury_inputs'
OUT = ROOT / 'data/jury_gap_analysis'
THRESHOLDS = (2, 3, 5, 10)
TOLERANCE = 1e-8
METRICS = ('tau_gap', 'order_agreement', 'strict_accuracy', 'tie_rate')
COUNTS = ('eligible_pairs', 'usable_pairs', 'missing_rank_pairs', 'concordant', 'discordant', 'tied')
SCOPES = {'current_round': '当前轮三个候选', 'accumulated': '累计候选池（含历史三条轨迹）'}
BENCHMARKS = {'workbuddy': 'WorkBuddy', 'claw_eval': 'claw-eval'}
EXCLUSIONS = {'workbuddy': 'analyst-forecast-extract-L3-018', 'claw_eval': 'T101_wal_recovery'}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def eligible(gap, threshold):
    return gap >= threshold - TOLERANCE


def pair_metrics(pairs, threshold):
    kept = [p for p in pairs if eligible(p['official_gap'], threshold)]
    counts = Counter(p['outcome'] for p in kept)
    c, d, t = (counts[k] for k in ('concordant', 'discordant', 'tied'))
    n = c + d + t
    result = dict(eligible_pairs=len(kept), usable_pairs=n, missing_rank_pairs=len(kept)-n,
                  concordant=c, discordant=d, tied=t,
                  tau_gap=(c-d)/math.sqrt(n*(c+d)) if c+d else None,
                  order_agreement=(c+t)/n if n else None,
                  strict_accuracy=c/n if n else None, tie_rate=t/n if n else None)
    missing_reason = 'no_eligible_pairs' if not kept else 'no_available_rank_pairs'
    for metric in METRICS:
        result[metric+'_undefined_reason'] = (None if result[metric] is not None else
            'all_jury_ties' if metric == 'tau_gap' and n else missing_reason)
    return result


def aggregate_rounds(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['task']].append(row)
    result = []
    for task, group in sorted(grouped.items()):
        assert {r['round'] for r in group} == {1, 2, 3} and len(group) == 3
        row = dict(task=task, scenario=group[0]['scenario'])
        row.update({k: sum(r[k] for r in group) for k in COUNTS})
        for metric in METRICS:
            defined = [r[metric] for r in group if r[metric] is not None]
            row[metric] = sum(defined)/len(defined) if defined else None
            row[metric+'_defined_rounds'] = len(defined)
            row[metric+'_undefined_reason'] = None if defined else 'no_defined_round'
        result.append(row)
    return result


def summarize(rows, benchmark):
    return dict(total_tasks=len(rows),
        sampling_units=len({r['task'] if benchmark == 'workbuddy' else r['scenario'] for r in rows}),
        pair_counts={k: sum(r[k] for r in rows) for k in COUNTS},
        metrics={m: dict(**bootstrap_mean(rows, m, benchmark),
            defined_tasks=sum(r[m] is not None for r in rows),
            complete_three_round_tasks=(sum(r.get(m+'_defined_rounds') == 3 for r in rows)
                                        if rows and m+'_defined_rounds' in rows[0] else None),
            undefined_reasons=dict(Counter(r[m+'_undefined_reason'] for r in rows if r[m] is None)))
            for m in METRICS})


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def value_ci(value, percent=False):
    if value['mean'] is None:
        return '未定义'
    scale, digits = (100, 2) if percent else (1, 3)
    lo, hi = value['ci95']
    return f"{scale*value['mean']:.{digits}f} [{scale*lo:.{digits}f}, {scale*hi:.{digits}f}]"


def report(summary):
    lines = ['# Official 分差限定的 Jury 排序一致性', '',
        '仅使用归档 Jury，不调用模型、原生 grader，也不恢复补评。', '',
        '对同一任务、同一轮的候选 pairs，独立筛选 official 分差 ≥2、≥3、≥5、≥10 分（含边界，容差 1e-8）。使用两端都有排名的 pairs，不要求全池排名完整。累计候选池在各轮分别含 6、9、12 条轨迹，包括三条历史轨迹；Selected 不重复计数，六条 baseline 的平均分不是候选。', '',
        'C/D/T 分别为同序/逆序/Jury 平局，N=C+D+T。分差限定 τ=(C−D)/√[N(C+D)]，不是全候选排序的标准 Kendall τ-b。Order agreement=(C+T)/N，严格同序率=C/N，平局率=T/N。全部 Jury 平局时 τ 未定义，但 order agreement 为 100%；没有可用 pair 时四项指标均未定义。', '',
        '单轮先算每个任务的指标再等权平均。三轮汇总先平均各任务内该指标有定义的轮次，再对任务等权平均；各指标的有定义任务集合可能不同。不会按 pair 数量加权，也不把未定义值填零。', '',
        '95% CI：20,000 次 percentile bootstrap，NumPy default RNG，seed=20260929；WorkBuddy 按 50 个任务，claw-eval 按 68 个场景簇，保留双语版本和轮次关联。所有原始抽样单位均保留，无有效任务的重采样不产生估计。', '',
        '## 三轮任务等权汇总', '',
        '| Benchmark | 范围 | 分差 ≥ | 限定 τ [95% CI] | Order agreement % [95% CI] | 严格同序 % | 平局 % | τ 有定义任务 | Agreement 有定义任务 | 可用/合格 pairs |',
        '|---|---|---:|---|---|---:|---:|---|---|---|']
    for benchmark, scopes in summary['benchmarks'].items():
        for scope, gaps in scopes.items():
            for gap, phases in gaps.items():
                s = phases['task_equal_aggregate']; m = s['metrics']; p = s['pair_counts']
                def pct(key):
                    return '未定义' if m[key]['mean'] is None else f"{100*m[key]['mean']:.2f}"
                lines.append(f"| {BENCHMARKS[benchmark]} | {SCOPES[scope]} | {gap} | {value_ci(m['tau_gap'])} | {value_ci(m['order_agreement'], True)} | {pct('strict_accuracy')} | {pct('tie_rate')} | {m['tau_gap']['defined_tasks']}/{s['total_tasks']} | {m['order_agreement']['defined_tasks']}/{s['total_tasks']} | {p['usable_pairs']}/{p['eligible_pairs']} |")
    lines += ['', '可用/合格 pairs 为三个轮次的 pair 槽位总数；同一历史 pair 可在不同轮次出现，因为对应的 Jury 排名不同。区间只描述各分差阈值下的可观察 pairs；阈值会改变任务覆盖率，数值升高不等于在同一任务集合上得到改进，也不证明整体排序能力提高。', '', '## 各轮明细', '',
        '| Benchmark | 范围 | 分差 ≥ | 轮次 | 限定 τ [95% CI] | Agreement % [95% CI] | τ/Agreement 有定义任务 | 可用/合格 pairs | C/D/T |',
        '|---|---|---:|---:|---|---|---|---|---|']
    for row in summary_rows(summary):
        if row['phase'] == 'task_equal_aggregate':
            continue
        lines.append(f"| {BENCHMARKS[row['benchmark']]} | {SCOPES[row['scope']]} | {row['threshold']} | {row['phase'][-1]} | {row['tau_display']} | {row['agreement_display']} | {row['tau_defined_tasks']}/{row['agreement_defined_tasks']}（共 {row['total_tasks']}） | {row['usable_pairs']}/{row['eligible_pairs']} | {row['concordant']}/{row['discordant']}/{row['tied']} |")
    lines += ['', '## 排除敏感性：三轮汇总', '',
        'WorkBuddy 排除唯一第一轮无效基线所属任务；主分析仍将该原生分数计为 0。claw-eval 排除已知暴露任务 T101_wal_recovery。各轮敏感性和完整覆盖率见 summary.json。', '',
        '| Benchmark | 范围 | 分差 ≥ | 限定 τ [95% CI] | Agreement % [95% CI] | τ/Agreement 有定义任务 |',
        '|---|---|---:|---|---|---|']
    for b, scopes in summary['benchmarks'].items():
        for scope, gaps in scopes.items():
            for gap, phases in gaps.items():
                s = phases['task_equal_aggregate']['sensitivity']; m = s['metrics']
                lines.append(f"| {BENCHMARKS[b]} | {SCOPES[scope]} | {gap} | {value_ci(m['tau_gap'])} | {value_ci(m['order_agreement'], True)} | {m['tau_gap']['defined_tasks']}/{m['order_agreement']['defined_tasks']}（共 {s['total_tasks']}） |")
    lines += ['', '完整计数、各指标独立覆盖率、未定义原因、三轮均有定义任务数、有效 bootstrap 次数和输入 SHA-256 均记录在 summary.json。pairs.csv 包含所有候选 pairs、两端分数和排名、可用状态、四个阈值标记；task_statistics.csv 和 task_aggregates.csv 保留逐任务计算过程。论文与 PDF 保持原样。']
    return '\n'.join(lines)+'\n'


def summary_rows(summary):
    rows = []
    for b, scopes in summary['benchmarks'].items():
        for scope, gaps in scopes.items():
            for gap, phases in gaps.items():
                for phase, s in phases.items():
                    m = s['metrics']
                    row = dict(benchmark=b, scope=scope, threshold=int(gap), phase=phase, total_tasks=s['total_tasks'], **s['pair_counts'],
                        tau_display=value_ci(m['tau_gap']), agreement_display=value_ci(m['order_agreement'], True),
                        tau_defined_tasks=m['tau_gap']['defined_tasks'], agreement_defined_tasks=m['order_agreement']['defined_tasks'])
                    for metric, value in m.items():
                        row[metric] = value['mean']
                        row[metric+'_ci_low'] = value['ci95'][0] if value['ci95'] else None
                        row[metric+'_ci_high'] = value['ci95'][1] if value['ci95'] else None
                    rows.append(row)
    return rows


def prepare():
    sources = ['data/office50.json', 'data/workbuddy_seed_scores.json', 'data/claw_eval.json',
               'data/claw_eval_continuation.json', 'jury_inputs/candidates.json',
               'jury_inputs/jury.json', 'jury_inputs/binding_audit.json']
    preserved = sources + ['main.tex', 'appendix.tex', 'main.pdf', 'data/main_statistics.json',
                          'data/known_reward_kendall.json', 'data/ranking_expansion/summary.json']
    before = {p: sha(ROOT/p) for p in preserved}
    audit = read(EXP/'binding_audit.json')
    for group in ('native_sources', 'preserved_legacy_sources'):
        for p, digest in audit[group].items():
            assert sha(ROOT/p) == digest, p
    for p in ('candidates.json', 'jury.json'):
        assert sha(EXP/p) == audit['experiment_sources'][p]
    candidates = read(EXP/'candidates.json'); juries = read(EXP/'jury.json')
    by_task = defaultdict(dict)
    for c in candidates:
        key = c['benchmark'], c['task']
        assert c['candidate_id'] not in by_task[key]
        by_task[key][c['candidate_id']] = c
    grades, scenarios = {}, {}
    for b in BENCHMARKS:
        tasks, clusters, arms, seeds = inputs(b)
        for i, task in enumerate(tasks):
            scenarios[b, task] = clusters[i]
            for r in (1, 2, 3):
                grades[b, task, f'seed_{r}'] = float(seeds[i, r-1])
                for arm in 'ABC':
                    grades[b, task, f'round_{r}_{arm}'] = float(arms[arm][i, r-1])
    pair_rows, task_rows = [], []
    seen = set()
    for item in sorted(juries, key=lambda x: (x['benchmark'], x['task'])):
        b, task = item['benchmark'], item['task']; candidates = by_task[b, task]
        assert len(candidates) == 12
        for phase in item['phases']:
            r = phase['round']; ids = phase['candidate_ids']
            assert len(ids) == len(set(ids)) == 3+3*r and set(ids) <= candidates.keys()
            assert set(phase['product_to_candidate'].values()) == set(ids)
            for scope in SCOPES:
                selected = sorted((candidates[c] for c in ids if scope == 'accumulated' or
                                   candidates[c]['slot'].startswith(f'round_{r}_')), key=lambda c: c['slot'])
                assert len(selected) == (3 if scope == 'current_round' else 3+3*r)
                ranks, unavailable = jury_scores(phase, [c['candidate_id'] for c in selected])
                pairs = []
                for i, j in combinations(range(len(selected)), 2):
                    a, z = selected[i], selected[j]
                    ga, gz = grades[b, task, a['slot']], grades[b, task, z['slot']]
                    ra, rz = ranks[i], ranks[j]
                    outcome = ('jury_unavailable' if unavailable else 'rank_unknown') if ra is None or rz is None else (
                        'tied' if ra == rz else 'native_tie' if abs(ga-gz) <= TOLERANCE else
                        'concordant' if (ra-rz)*(ga-gz) > 0 else 'discordant')
                    pair = dict(benchmark=b, task=task, scenario=scenarios[b, task], scope=scope, round=r,
                        candidate_1=a['candidate_id'], slot_1=a['slot'], official_1=ga, jury_rank_1=ra,
                        candidate_2=z['candidate_id'], slot_2=z['slot'], official_2=gz, jury_rank_2=rz,
                        official_gap=abs(ga-gz), outcome=outcome, rank_available=ra is not None and rz is not None)
                    pair.update({f'eligible_gap_{k}': eligible(pair['official_gap'], k) for k in THRESHOLDS})
                    key = (b, task, scope, r, a['candidate_id'], z['candidate_id'])
                    assert key not in seen; seen.add(key)
                    flags = [pair[f'eligible_gap_{k}'] for k in THRESHOLDS]
                    assert flags == sorted(flags, reverse=True)
                    pairs.append(pair); pair_rows.append(pair)
                for gap in THRESHOLDS:
                    row = dict(benchmark=b, task=task, scenario=scenarios[b, task], scope=scope, round=r,
                               threshold=gap, **pair_metrics(pairs, gap))
                    assert row['concordant']+row['discordant']+row['tied'] == row['usable_pairs']
                    task_rows.append(row)
    results, aggregate_rows = {}, []
    for b in BENCHMARKS:
        results[b] = {}
        for scope in SCOPES:
            results[b][scope] = {}
            for gap in THRESHOLDS:
                rows = [x for x in task_rows if (x['benchmark'], x['scope'], x['threshold']) == (b, scope, gap)]
                aggregate = aggregate_rounds(rows)
                aggregate_rows.extend(dict(benchmark=b, scope=scope, threshold=gap, **x) for x in aggregate)
                views = {f'round_{r}': [x for x in rows if x['round'] == r] for r in (1, 2, 3)}
                views['task_equal_aggregate'] = aggregate
                results[b][scope][str(gap)] = {}
                for phase, view in views.items():
                    s = summarize(view, b)
                    s['sensitivity'] = dict(excluded_task=EXCLUSIONS[b], **summarize([x for x in view if x['task'] != EXCLUSIONS[b]], b))
                    results[b][scope][str(gap)][phase] = s
    after = {p: sha(ROOT/p) for p in preserved}
    assert before == after
    summary = dict(protocol=dict(thresholds=list(THRESHOLDS),comparison='official_gap >= threshold - 1e-8',
        tau_gap='(C-D)/sqrt((C+D+T)*(C+D)); undefined if C+D=0; gap-restricted tau-b-style, not standard full-pool Kendall tau-b',
        order_agreement='(C+T)/(C+D+T); Jury ties receive full credit',strict_accuracy='C/(C+D+T)',tie_rate='T/(C+D+T)',
        missing='Use pairs with two available Jury ranks; no complete-pool requirement; never zero-fill.',
        aggregation='Per-task metric, then equal task mean; aggregate rounds within each task before across tasks, independently for each metric.',
        counts='Aggregate pair counts sum task-round pair slots, not unique cross-round trajectory pairs.',
        bootstrap=dict(iterations=BOOTSTRAPS, seed=SEED, units='WorkBuddy task; claw-eval scenario clusters; all original units retained'),
        native_imputation='WorkBuddy analyst-forecast-extract-L3-018 round_1_A remains zero; raw source unchanged.',
        model_calls=0, sources={p: before[p] for p in sources},
        code_sources={p: sha(ROOT/p) for p in ['scripts/prepare_jury_gap_analysis.py','scripts/prepare_six_baseline.py','scripts/prepare_three_round_statistics.py','scripts/prepare_claw_continuation.py','scripts/prepare_expanded_ranking.py','scripts/compute_joint_offline_metrics.py']}),
        validation=dict(preserved_file_hashes=after, input_hashes_verified=True, nested_thresholds_verified=True,
                        unique_pair_slots=len(seen), task_round_threshold_rows=len(task_rows)), benchmarks=results)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in [('pairs.csv', pair_rows), ('task_statistics.csv', task_rows),
                       ('task_aggregates.csv', aggregate_rows), ('summary.csv', summary_rows(summary))]:
        write_csv(OUT/name, rows)
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    (OUT/'REPORT.md').write_text(report(summary))
    print(json.dumps(dict(status='complete',pair_slots=len(pair_rows),task_rows=len(task_rows),model_calls=0,report=str(OUT/'REPORT.md'))))
    return summary


if __name__ == '__main__':
    prepare()
