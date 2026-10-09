"""Offline selection utility and grouped calibration; no network or model clients."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/joint_offline_metrics'
SEED = 20260929
BOOTSTRAPS = 20000
FEATURES = ['pass_fraction', 'fail_fraction', 'evidence_unknown_fraction', 'technical_unknown_fraction', 'log_criterion_count']


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def select_metrics(grades, values):
    grades, values = np.asarray(grades, float), np.asarray(values, float)
    winners = np.isclose(values, values.max(), atol=1e-12, rtol=0)
    selected = float(grades[winners].mean()); uniform = float(grades.mean()); oracle = float(grades.max())
    return {'selected': selected, 'uniform': uniform, 'oracle': oracle, 'gain': selected-uniform,
            'regret': oracle-selected, 'best_hit': float(np.isclose(grades[winners], oracle, atol=1e-8, rtol=0).mean()),
            'reward_top_tie': int(winners.sum()>1), 'winners': int(winners.sum())}


def bootstrap_mean(records, field, benchmark):
    clusters = defaultdict(list)
    for r in records:
        clusters[r['task'] if benchmark=='workbuddy' else r['scenario']].append(r)
    keys=sorted(clusters)
    if not keys:return {'mean':None,'ci95':None,'defined_replicates':0}
    sums=np.array([sum(r[field] for r in clusters[k] if r.get(field) is not None) for k in keys],float)
    counts=np.array([sum(r.get(field) is not None for r in clusters[k]) for k in keys],int)
    if counts.sum()==0:return {'mean':None,'ci95':None,'defined_replicates':0}
    draws=np.random.default_rng(SEED).integers(len(keys),size=(BOOTSTRAPS,len(keys)))
    denom=counts[draws].sum(axis=1); good=denom>0
    estimates=sums[draws].sum(axis=1)[good]/denom[good]
    return {'mean':float(sums.sum()/counts.sum()),'ci95':np.quantile(estimates,[.025,.975]).tolist(),
            'defined_replicates':int(good.sum()),'clusters':len(keys)}


def summarize_selection(tasks, benchmark):
    result={'tasks':len(tasks),'eligible_tasks':sum(r.get('selected') is not None for r in tasks),
            'reward_top_tie_tasks':sum(r.get('reward_top_tie') or 0 for r in tasks)}
    for field in ('selected','uniform','oracle','gain','regret','best_hit'):
        result[field]=bootstrap_mean(tasks,field,benchmark)
    return result


def load(version):
    folder=ROOT/'data'/('rewardpack_joint' if version=='original' else 'rewardpack_joint_repair')
    rows=read_csv(folder/'trajectory_scores.csv'); judgments=read_csv(folder/'criterion_receipts.csv')
    by_id=defaultdict(list)
    for c in judgments:by_id[c['identity']].append(c)
    for r in rows:
        for field in ('reward','r_min','r_max','official_score'):
            r[field]=float(r[field]) if r[field] else None
        cs=by_id[r['identity']]; k=len(cs)
        r['features']=[sum(c['status']=='pass' for c in cs)/k,sum(c['status']=='fail' for c in cs)/k,
                       sum(c['status']=='unknown' and c['protocol_error']!='True' for c in cs)/k,
                       sum(c['status']=='unknown' and c['protocol_error']=='True' for c in cs)/k,float(np.log(k))]
        assert abs(sum(r['features'][:4])-1)<1e-12
    return rows


def selection_analysis(rows, version):
    groups=defaultdict(list)
    for r in rows:groups[(r['benchmark'],r['task'])].append(r)
    tasks=[]; certain=[]
    for (benchmark,task),rs in sorted(groups.items()):
        assert len(rs)==3 and all(r['official_score'] is not None for r in rs)
        base={'version':version,'benchmark':benchmark,'task':task,'scenario':rs[0]['scenario']}
        grades=[r['official_score'] for r in rs]
        row={**base,**dict.fromkeys(('selected','uniform','oracle','gain','regret','best_hit','reward_top_tie','winners'))}
        complete=all(r['reward'] is not None for r in rs)
        if complete:row.update(select_metrics(grades,[r['reward'] for r in rs]))
        row['all_three_complete']=complete; tasks.append(row)
        winners=[i for i,r in enumerate(rs) if all(i==j or r['r_min']>other['r_max'] for j,other in enumerate(rs))]
        cert={**base,**dict.fromkeys(('selected','uniform','oracle','gain','regret','best_hit','reward_top_tie','winners')),'incomplete_task':not complete}
        if len(winners)==1:
            scores=[int(i==winners[0]) for i in range(3)];cert.update(select_metrics(grades,scores))
        certain.append(cert)
    summary={}
    for b in ('workbuddy','claw_eval'):
        sub=[r for r in tasks if r['benchmark']==b]; cert=[r for r in certain if r['benchmark']==b]
        summary[b]={'complete_task_selection':summarize_selection(sub,b),
                    'strict_interval_winner_selection':summarize_selection(cert,b),
                    'strict_interval_winners_in_incomplete_tasks':sum(r['incomplete_task'] and r['selected'] is not None for r in cert)}
    return summary,tasks,certain


def safe_corr(x,y,kind):
    if len(set(x))<2 or len(set(y))<2:return None
    return float((pearsonr if kind=='pearson' else spearmanr)(x,y).statistic)


def calibration(rows):
    summary={}; predictions=[]; task_rows=[]; folds=[]
    for b in ('workbuddy','claw_eval'):
        rs=[r for r in rows if r['benchmark']==b]
        group=lambda r:r['task'] if b=='workbuddy' else r['scenario']
        keys=sorted({group(r) for r in rs},key=lambda k:hashlib.sha256(f'{SEED}:{b}:{k}'.encode()).hexdigest())
        assignment={key:i%5 for i,key in enumerate(keys)}
        x=np.asarray([r['features'] for r in rs]); y=np.asarray([r['official_score'] for r in rs]); pred=np.zeros(len(rs)); baseline=np.zeros(len(rs))
        for fold in range(5):
            test=np.array([assignment[group(r)]==fold for r in rs]); train=~test
            assert {group(r) for r,t in zip(rs,test) if t}.isdisjoint({group(r) for r,t in zip(rs,train) if t})
            model=make_pipeline(StandardScaler(),Ridge(alpha=10.0));model.fit(x[train],y[train]);pred[test]=np.clip(model.predict(x[test]),0,100);baseline[test]=y[train].mean()
            folds.append({'benchmark':b,'fold':fold,'train_candidates':int(train.sum()),'test_candidates':int(test.sum()),
                          'test_groups':[k for k in keys if assignment[k]==fold], 'training_mean':float(y[train].mean()),
                          'coefficients':model[-1].coef_.tolist(),'intercept':float(model[-1].intercept_)})
        groups=defaultdict(list)
        for i,r in enumerate(rs):
            record={k:r[k] for k in ('identity','benchmark','task','scenario','slot','official_score','reward')}
            record.update(prediction=float(pred[i]),training_mean_prediction=float(baseline[i]),fold=assignment[group(r)])
            predictions.append(record);groups[r['task']].append(record)
        tr=[]
        for task,items in sorted(groups.items()):
            grades=[r['official_score'] for r in items]; scores=[r['prediction'] for r in items]
            tau=float(kendalltau(grades,scores,variant='b').statistic) if len(set(grades))>1 and len(set(scores))>1 else None
            row={'benchmark':b,'task':task,'scenario':items[0]['scenario'],'tau_b':tau,
                 'same_complete_cohort':all(r['reward'] is not None for r in items),**select_metrics(grades,scores)}
            row['same_top_set_as_reward'] = (
                np.flatnonzero(np.isclose(scores, max(scores), atol=1e-12, rtol=0)).tolist() ==
                np.flatnonzero(np.isclose([r['reward'] for r in items], max(r['reward'] for r in items), atol=1e-12, rtol=0)).tolist()
            ) if row['same_complete_cohort'] else None
            tr.append(row);task_rows.append(row)
        summary[b]={'candidate_count':len(rs),'groups':len(keys),'folds':5,'ridge_alpha':10.0,
                    'features':FEATURES,'mae':float(np.abs(pred-y).mean()),'rmse':float(np.sqrt(((pred-y)**2).mean())),
                    'training_mean_baseline_mae':float(np.abs(baseline-y).mean()),
                    'training_mean_baseline_rmse':float(np.sqrt(((baseline-y)**2).mean())),
                    'pooled_pearson':safe_corr(pred,y,'pearson'),'pooled_spearman':safe_corr(pred,y,'spearman'),
                    'task_tau':bootstrap_mean(tr,'tau_b',b),'defined_tau_tasks':sum(r['tau_b'] is not None for r in tr),
                    'complete_tasks_with_same_top_set':sum(r['same_top_set_as_reward'] is True for r in tr),
                    'selection_all_tasks':summarize_selection(tr,b),
                    'selection_same_complete_cohort':summarize_selection([r for r in tr if r['same_complete_cohort']],b)}
    return summary,predictions,task_rows,folds


def fmt(v):return '不可计算' if v is None else f'{v:.2f}'

def report(summary):
    lines=['# 现有联合评分的离线选优与校准分析','',
           '全程仅本地计算，无 LLM、网络、actor 或 grader 调用。以技术修复后的分数为主，原始版本作对照；原始评分、官方标签和462条候选不变。','',
           '## 原 reward 的选优效果（每任务三条分数均完整）','',
           'reward 顶部分数平局时等概率选取，直接计算期望官方分；不使用官方分打破平局。随机基线是在同一任务三个候选中均匀选择。regret = 该任务最高官方分 − 选择的期望官方分。单位为0–100分。','',
           '| 数据集 | 可评任务/全部 | reward选优 | 随机选择 | 最优候选 | 提升 [95% CI] | regret | 命中最优概率 | 顶部平局任务 |',
           '|---|---:|---:|---:|---:|---|---:|---:|---:|']
    for b,name in (('workbuddy','WorkBuddy'),('claw_eval','claw-eval')):
        s=summary['repaired'][b]['complete_task_selection'];ci=s['gain']['ci95']
        lines.append(f"| {name} | {s['eligible_tasks']}/{s['tasks']} | {fmt(s['selected']['mean'])} | {fmt(s['uniform']['mean'])} | {fmt(s['oracle']['mean'])} | {fmt(s['gain']['mean'])} [{fmt(ci[0])}, {fmt(ci[1])}] | {fmt(s['regret']['mean'])} | {100*s['best_hit']['mean']:.1f}% | {s['reward_top_tie_tasks']} |")
    lines+=['','## 分数区间能确定唯一赢家的任务','','要求某候选的下界严格高于另外两条的上界；其余任务弃权，不把unknown当零。此子集由区间决定，不由官方分数决定。','',
            '| 数据集 | 确定赢家任务/全部 | 其中含unknown的任务 | 选优分 | 随机分 | 提升 [95% CI] | regret |','|---|---:|---:|---:|---:|---|---:|']
    for b,name in (('workbuddy','WorkBuddy'),('claw_eval','claw-eval')):
        bundle=summary['repaired'][b];s=bundle['strict_interval_winner_selection'];ci=s['gain']['ci95']
        ci_text='不可计算' if ci is None else f'[{fmt(ci[0])}, {fmt(ci[1])}]'
        lines.append(f"| {name} | {s['eligible_tasks']}/{s['tasks']} | {bundle['strict_interval_winners_in_incomplete_tasks']} | {fmt(s['selected']['mean'])} | {fmt(s['uniform']['mean'])} | {fmt(s['gain']['mean'])} {ci_text} | {fmt(s['regret']['mean'])} |")
    lines+=['','## 监督校准：留出任务预测','','固定5折Ridge（alpha=10），无调参。输入仅为pass/fail/证据unknown/技术unknown比例及log criterion数；不包含任务ID、轮次、官方分或原生检查细节。官方分仅在训练折作为标签。WorkBuddy按任务、claw-eval按scenario分组，双语版本不跨折。标准化只拟合训练折；输出截到[0,100]。这是用官方标签训练的新预测器，不是原RewardPack评分，也不是未见数据的独立确认实验。','',
            '| 数据集 | OOF MAE | 训练均值基线MAE | OOF RMSE | 均值基线RMSE | 任务平均tau | 有定义任务 | 全任务选优相对随机提升 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for b,name in (('workbuddy','WorkBuddy'),('claw_eval','claw-eval')):
        s=summary['calibration'][b]
        lines.append(f"| {name} | {fmt(s['mae'])} | {fmt(s['training_mean_baseline_mae'])} | {fmt(s['rmse'])} | {fmt(s['training_mean_baseline_rmse'])} | {fmt(s['task_tau']['mean'])} | {s['defined_tau_tasks']} | {fmt(s['selection_all_tasks']['gain']['mean'])} |")
    lines+=['','在同一完整任务子集上，校准模型与原 reward 的最高分候选集合完全一致：WorkBuddy 20/20、claw-eval 89/89，因而选优收益没有变化。对于每任务criterion数固定且无unknown的候选，这些比例特征将校准限制为原通过率的线性变换；本次各折没有改变其顶部选择。',
            '', '全任务校准选优提升的95%区间：WorkBuddy [-3.26,5.07]，claw-eval [-0.69,2.63]，均包含零。',
            '', '同一完整任务子集上的校准选优比较、OOF逐候选预测、全部置信区间见 summary.json 和 calibration_predictions.csv。',
            '', '## 逐要求一致率', '',
            '未生成全量逐要求一致率：现有导出没有冻结的Pack criterion↔官方原子检查映射，也没有这套共同要求的成对pass/fail标签。不能把任务级官方分阈值化、复制给每个criterion，或把8个有意挑选的分歧案例当作总体一致率。现有客观检查准确率属于其他小型研究，不替代这里的逐要求一致率。',
            '', '## 解释边界', '',
            '1. 20,000次bootstrap，seed=20260929；WorkBuddy按任务、claw-eval按scenario聚类，保留双语关系。选优分、随机分及regret只在相同可评任务上比较。',
            '2. 校准区间条件于固定OOF预测，bootstrap不重新拟合模型，未涵盖完整训练不确定性。所有区间均未校正多重比较。',
            '3. 候选原本按可用性与官方低/中/高分选择，且官方评分存在已记录的路径、参考日期和运行错误等问题。结果只适用于这组历史候选及归档官方标签。',
            '4. OOF预测可利用unknown模式，但原reward的unknown仍不计零；校准得到的数字不回写原评分。',
            '', '复现：`python scripts/compute_joint_offline_metrics.py`。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    OUT.mkdir(exist_ok=True)
    summary={}; sources={}
    for version in ('original','repaired'):
        rows=load(version); summary[version],tasks,certain=selection_analysis(rows,version)
        write_csv(OUT/f'{version}_task_selection.csv',tasks);write_csv(OUT/f'{version}_interval_selection.csv',certain)
        folder='rewardpack_joint' if version=='original' else 'rewardpack_joint_repair'
        for file in ('trajectory_scores.csv','criterion_receipts.csv'):
            p=ROOT/'data'/folder/file;sources[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    summary['calibration'],predictions,tasks,folds=calibration(load('repaired'))
    write_csv(OUT/'calibration_predictions.csv',predictions);write_csv(OUT/'calibration_task_selection.csv',tasks)
    (OUT/'calibration_folds.json').write_text(json.dumps(folds,indent=2)+'\n')
    summary['protocol']={'llm_calls':0,'network_calls':0,'seed':SEED,'bootstrap_draws':BOOTSTRAPS,'sources':sources,
                         'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                         'requirement_agreement':'not_identifiable_without_fixed_criterion_to_native_check_mapping'}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n');report(summary)
    print((OUT/'REPORT.md').read_text())


if __name__=='__main__':main()
