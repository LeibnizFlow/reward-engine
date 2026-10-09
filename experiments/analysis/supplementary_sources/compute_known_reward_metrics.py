"""Known-criterion scoring, tie-half agreement, and uniform top-tie selection. Offline only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr

from compute_joint_offline_metrics import BOOTSTRAPS, SEED, bootstrap_mean, read_csv, write_csv

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/known_reward_metrics'
SOURCE=ROOT/'data/rewardpack_joint_repair'


def score_counts(criteria):
    counts=Counter(c['status'] for c in criteria)
    if not criteria or set(counts)-{'pass','fail','unknown'}:
        raise ValueError('Empty or invalid criterion set')
    p,f,u=counts['pass'],counts['fail'],counts['unknown'];k=p+f+u
    return {'passed':p,'failed':f,'unknown':u,'criterion_count':k,'known_count':p+f,
            'known_fraction':(p+f)/k,'known_reward':float(Fraction(p,p+f)) if p+f else None,
            'score_status':'available' if p+f else 'all_unknown'}


def task_metrics(rows, mode):
    """Exact rational reward ties; native ties use archived identical numeric scores."""
    scores=[Fraction(r['passed'],r['known_count']) if r['known_count'] and (mode=='known' or not r['unknown']) else None for r in rows]
    grades=[r['official_score'] for r in rows]
    result={'candidate_count':len(rows),'score_available':all(s is not None for s in scores),
            'official_all_tied':len(set(grades))==1,'official_tie_pairs':0,'correct_pairs':0,
            'reward_tie_pairs':0,'reversed_pairs':0,'strict_pairs':0,
            **dict.fromkeys(('strict_accuracy','half_accuracy','full_credit','selected','uniform','oracle','gain','regret',
                            'best_hit','top_size','top_tied','tau_b','spearman_rho','reward_all_tied'))}
    result['official_tie_pairs']=sum(a==b for a,b in itertools.combinations(grades,2))
    if not result['score_available']:
        result['undefined_reason']='missing_reward';return result
    for i,j in itertools.combinations(range(len(rows)),2):
        if grades[i]==grades[j]:continue
        result['strict_pairs']+=1
        if scores[i]==scores[j]:result['reward_tie_pairs']+=1
        elif (scores[i]>scores[j])==(grades[i]>grades[j]):result['correct_pairs']+=1
        else:result['reversed_pairs']+=1
    if result['strict_pairs']:
        result['strict_accuracy']=result['correct_pairs']/result['strict_pairs']
        result['half_accuracy']=(result['correct_pairs']+.5*result['reward_tie_pairs'])/result['strict_pairs']
        result['full_credit']=(result['correct_pairs']+result['reward_tie_pairs'])/result['strict_pairs']
    maximum=max(scores);winners=[i for i,s in enumerate(scores) if s==maximum]
    selected=float(np.mean([grades[i] for i in winners]));uniform=float(np.mean(grades));oracle=max(grades)
    result.update(selected=selected,uniform=uniform,oracle=oracle,gain=selected-uniform,regret=oracle-selected,
                  best_hit=sum(grades[i]==oracle for i in winners)/len(winners),top_size=len(winners),
                  top_tied=int(len(winners)>1),reward_all_tied=len(set(scores))==1)
    result['undefined_reason']='both_constant' if result['reward_all_tied'] and result['official_all_tied'] else 'reward_constant' if result['reward_all_tied'] else 'official_constant' if result['official_all_tied'] else None
    if result['undefined_reason'] is None:
        floats=[float(s) for s in scores]
        result['tau_b']=float(kendalltau(floats,grades,variant='b').statistic)
        result['spearman_rho']=float(spearmanr(floats,grades).statistic)
    return result


def pooled_half_bootstrap(tasks, benchmark, tie_credit=.5):
    groups=defaultdict(list)
    for t in tasks:groups[t['task'] if benchmark=='workbuddy' else t['scenario']].append(t)
    keys=sorted(groups)
    num=np.array([sum(t['correct_pairs']+tie_credit*t['reward_tie_pairs'] for t in groups[k]) for k in keys])
    den=np.array([sum(t['strict_pairs'] for t in groups[k]) for k in keys])
    if not keys or not den.sum():return {'mean':None,'ci95':None,'defined_replicates':0}
    sampled=np.random.default_rng(SEED).integers(len(keys),size=(BOOTSTRAPS,len(keys)))
    d=den[sampled].sum(axis=1);good=d>0;est=num[sampled].sum(axis=1)[good]/d[good]
    return {'mean':float(num.sum()/den.sum()),'ci95':np.quantile(est,[.025,.975]).tolist(),'defined_replicates':int(good.sum())}


def summarize(tasks, benchmark):
    counts={name:sum(t[name] for t in tasks) for name in ('correct_pairs','reward_tie_pairs','reversed_pairs','strict_pairs','official_tie_pairs')}
    result={'tasks':len(tasks),'selection_tasks':sum(t['score_available'] for t in tasks),
            'pair_evaluable_tasks':sum(t['half_accuracy'] is not None for t in tasks),
            'official_all_tied_tasks':sum(t['official_all_tied'] for t in tasks),
            'top_tie_tasks':sum(t['top_tied'] or 0 for t in tasks),
            'reward_all_tied_tasks':sum(t['reward_all_tied'] is True for t in tasks),
            'defined_correlation_tasks':sum(t['tau_b'] is not None for t in tasks), 'pair_counts':counts,
            'pooled_half_accuracy':pooled_half_bootstrap(tasks,benchmark),
            'pooled_full_credit':pooled_half_bootstrap(tasks,benchmark,tie_credit=1)}
    for field in ('strict_accuracy','half_accuracy','full_credit','selected','uniform','oracle','gain','regret','best_hit','top_size','tau_b','spearman_rho'):
        result[field]=bootstrap_mean(tasks,field,benchmark)
    return result


def fmt(v,percent=False):return '---' if v is None else f'{v*(100 if percent else 1):.2f}'

def value_ci(result,percent=False):
    if result['mean'] is None:return '---'
    return f"{fmt(result['mean'],percent)} [{fmt(result['ci95'][0],percent)}, {fmt(result['ci95'][1],percent)}]"


def make_tables(summary):
    # On complete tasks the known-criterion reward equals the original score, so the
    # identical same-task rows are kept in summary.json but omitted from the tables.
    names={'old_complete':'Complete only','known_all':'Known, all tasks'}
    pair=[r'\begin{tabular}{llrrrrr}',r'\toprule',r'Benchmark & Rule & Pair tasks & Strict (\%) & Half (\%) & Full credit (\%) [95\% CI] & Tied/strict pairs \\',r'\midrule']
    utility=[r'\begin{tabular}{llrrrrr}',r'\toprule',r'Benchmark & Rule & Tasks & Selected & Uniform & Gain [95\% CI] & Regret \\',r'\midrule']
    for b,name in (('workbuddy','WorkBuddy'),('claw_eval','claw-eval')):
        for mode,label in names.items():
            s=summary[b][mode];c=s['pair_counts']
            pair.append(f"{name} & {label} & {s['pair_evaluable_tasks']}/{s['tasks']} & {fmt(s['strict_accuracy']['mean'],True)} & {fmt(s['half_accuracy']['mean'],True)} & {value_ci(s['full_credit'],True)} & {c['reward_tie_pairs']}/{c['strict_pairs']} \\\\")
            utility.append(f"{name} & {label} & {s['selection_tasks']}/{s['tasks']} & {fmt(s['selected']['mean'])} & {fmt(s['uniform']['mean'])} & {value_ci(s['gain'])} & {fmt(s['regret']['mean'])} \\\\")
    for table in (pair,utility):table.extend([r'\bottomrule',r'\end{tabular}'])
    return '\n'.join(pair)+'\n','\n'.join(utility)+'\n'


def order_agreement_table(summary):
    """Frozen known-criterion pool: non-reversal, tau-b, and tau coverage."""
    rows=[r'\begin{tabular}{lllr}',r'\toprule',
          r'Benchmark & Order agreement (\%) [95\% CI] & Mean $\tau_b$ [95\% CI] & Defined tasks \\',r'\midrule']
    for b,name in (('workbuddy','WorkBuddy'),('claw_eval','claw-eval')):
        s=summary[b]['known_all']; tau=s['tau_b']
        value=f"{tau['mean']:.3f} [{tau['ci95'][0]:.3f}, {tau['ci95'][1]:.3f}]"
        rows.append(f"{name} & {value_ci(s['full_credit'],True)} & {value} & {s['defined_correlation_tasks']}/{s['tasks']}"+r' \\')
    rows.extend([r'\bottomrule',r'\end{tabular}'])
    return '\n'.join(rows)+'\n'


def report(summary):
    lines=['# 忽略unknown与平局一分：离线对照','',
           '新分数 R_known=P/(P+F)。未知项不进入分母；所有原始criterion判断和官方标签保持不变。462条候选均有已知项，全部154个任务可进入选优分析。已知项比例逐候选报告，不用它打破平局。',
           '', '## 全部任务结果','',
           '| 数据集 | 选优任务 | 配对可评任务 | 任务等权平局一分指标 [95% CI] | reward选优 | 随机 | 提升 [95% CI] | regret |',
           '|---|---:|---:|---|---:|---:|---|---:|']
    for b,name in (('workbuddy','WorkBuddy'),('claw_eval','claw-eval')):
        s=summary[b]['known_all']
        lines.append(f"| {name} | {s['selection_tasks']}/{s['tasks']} | {s['pair_evaluable_tasks']}/{s['tasks']} | {value_ci(s['full_credit'],True)}% | {fmt(s['selected']['mean'])} | {fmt(s['uniform']['mean'])} | {value_ci(s['gain'])} | {fmt(s['regret']['mean'])} |")
    lines+=['','## 相同完整任务对照','',
            '旧规则在任一候选含unknown时无法选优。新规则在相同完整任务上与旧规则的分数/排序完全相同；指标值变得更高来自平局计分的定义变化，不是模型改善。',
            '', '| 数据集 | 完整任务 | 严格一致率 | 半分一致率 | 平局一分指标 | 新规则同任务选优提升 |', '|---|---:|---:|---:|---:|---:|']
    for b,name in (('workbuddy','WorkBuddy'),('claw_eval','claw-eval')):
        s=summary[b]['old_complete'];n=summary[b]['known_same']
        lines.append(f"| {name} | {s['selection_tasks']} | {fmt(s['strict_accuracy']['mean'],True)}% | {fmt(s['half_accuracy']['mean'],True)}% | {fmt(s['full_credit']['mean'],True)}% | {fmt(n['gain']['mean'])} |")
    lines+=['','## 口径与限制','',
            '- 官方有严格高低的候选对：排对=1、reward平局=1、排反=0；官方平局单列。任务全部官方同分时配对指标缺失，但仍计入选优，regret为0。',
            '- 主配对指标先按任务计算、再任务等权汇总；总体配对汇总及计数另列在summary.json。二者分母不同，不与之前的总体严格准确率混用。',
            '- 最高reward并列时均匀选择，按期望官方分计算；从不拿官方分解开reward平局。平局半分相当于随机打破该对平局的期望正确率，不代表模型识别了高低。',
            '- 平局一分指标=(排对对数+reward平局对数)/官方严格对数，亦即不逆序率。全部reward同分也能得100%，不能解释为正确区分候选的准确率；半分指标继续保留。',
            '- 原有相关性未定义值不填0/1；新分数也仍可能全同分。相关性是辅助指标。',
            '- 忽略unknown会让少量通过项且其余未知的候选获得高分。分数必须连同known_fraction阅读，不能称为完整质量分。',
            '- 20,000次bootstrap，seed=20260929；WorkBuddy按任务、claw-eval按scenario聚类并保留双语。原始与新规则相同完整任务对照使用同一组任务及相同重采样规则。',
            '- 全部结果条件于既有候选的选取方式和有已知局限的归档官方标签。没有新增LLM、actor、grader、网络或校准训练调用。',
            '- 无已知criterion的异常候选记为missing，不补分；本数据不存在此情况。',
            '', '复现：`python scripts/compute_known_reward_metrics.py`。新增数据在本目录，原始及修复版数据仍原样保留。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')


def publish(summary,pair,utility):
    """Refresh the generated tables only.

    Since the 2026-10-05 concision revision, the surrounding prose lives in
    appendix.tex (Appendix E.2 and F.7) and main.tex (Section 4.4) and is edited
    by hand; run scripts/sync_manuscript_tables.py afterwards to refresh the
    embedded table copies.
    """
    (ROOT/'tables/known_reward_selection.tex').write_text(utility)
    # Keep the current manuscript presentation distinct from historical Kendall data.
    from publish_tie_credit_ranking import publish_tables
    publish_tables()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--publish',action='store_true');args=parser.parse_args()
    files=[SOURCE/'trajectory_scores.csv',SOURCE/'criterion_receipts.csv']; hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    base=read_csv(files[0]);cs=defaultdict(list)
    for c in read_csv(files[1]):cs[c['identity']].append(c)
    rows=[];groups=defaultdict(list)
    for r in base:
        assert len({c['criterion_id'] for c in cs[r['identity']]})==len(cs[r['identity']])
        row={k:r[k] for k in ('identity','benchmark','task','scenario','slot')}
        row.update(official_score=float(r['official_score']),old_reward=float(r['reward']) if r['reward'] else None,**score_counts(cs[r['identity']]))
        if row['old_reward'] is not None:assert not row['unknown'] and abs(row['old_reward']-row['known_reward'])<1e-12
        rows.append(row);groups[(row['benchmark'],row['task'])].append(row)
    assert len(rows)==462 and len(groups)==154 and Counter(r['benchmark'] for r in rows)=={'workbuddy':150,'claw_eval':312}
    tasks=[];summary={}
    for b in ('workbuddy','claw_eval'):
        by_mode=defaultdict(list)
        for (bench,task),rs in sorted(groups.items()):
            if bench!=b:continue
            assert len(rs)==3
            old=task_metrics(rs,'old');known=task_metrics(rs,'known')
            modes={'old_complete':old,'known_all':known}
            if old['score_available']:
                assert old==known
                modes['known_same']=known
            for mode,metrics in modes.items():
                item={'benchmark':b,'task':task,'scenario':rs[0]['scenario'],'mode':mode,**metrics}
                by_mode[mode].append(item);tasks.append(item)
        # Old complete comparison excludes unknown tasks; paired new subset is identical.
        by_mode['old_complete']=[t for t in by_mode['old_complete'] if t['score_available']]
        summary[b]={mode:summarize(tr,b) for mode,tr in by_mode.items()}
        fractions=[r['known_fraction'] for r in rows if r['benchmark']==b]
        summary[b]['known_fraction']={'mean':float(np.mean(fractions)),'min':min(fractions),'median':float(np.median(fractions)),'max':max(fractions)}
    OUT.mkdir(exist_ok=True);write_csv(OUT/'candidate_scores.csv',rows);write_csv(OUT/'task_metrics.csv',tasks)
    summary['protocol']={'name':'known_criterion_tie_full_v2','pair_tie_credit':1,'retained_pair_tie_credits':[0,.5],'sources':hashes,'seed':SEED,'bootstrap':BOOTSTRAPS,'llm_calls':0,'network_calls':0,
                         'score':'P/(P+F); all-unknown stays missing','ties':'exact rational reward ties; uniform selection among maxima',
                         'native_ties':'excluded from strict pairs; all-native-tied tasks remain in utility',
                         'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n');report(summary)
    pair,utility=make_tables(summary)
    if args.publish:publish(summary,pair,utility)
    for p in files:assert hashes[str(p.relative_to(ROOT))]==hashlib.sha256(p.read_bytes()).hexdigest()
    print((OUT/'REPORT.md').read_text())


if __name__=='__main__':main()
