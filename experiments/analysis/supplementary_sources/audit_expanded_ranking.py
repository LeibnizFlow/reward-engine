"""Audit all archived candidate/pack/Jury bindings before new model calls."""
import argparse
import hashlib
import json
from pathlib import Path
from run_ranking_expansion import OUT,ROOT,read,write,modules

def audit(runner):
    inv,joint,_,build_context=modules(runner)
    rows=read(OUT/'manifest.public.json');cs=read(OUT/'candidates.json');js=read(OUT/'jury.json')
    assert len(rows)==2772 and len(cs)==1848 and len(js)==154
    lookup={r['identity']:r for r in rows};assert len(lookup)==2772
    phases={(p['benchmark'],p['task']):p['phases'] for p in js}
    cached={};terminal_sources={}
    for c in cs:
        row=lookup[c['evaluations']['3']];key=c['benchmark'],c['task']
        pp=runner/row['paths']['pack'];cpfile=pp.parent/'checkpoint.json'
        if key not in cached:cached[key]={x['candidate_id']:x for x in read(cpfile)['pool']}
        assert cached[key][c['candidate_id']]==read(runner/row['paths']['result'])['candidate'],(key,c['slot'])
        for r,eid in c['evaluations'].items():
            assert lookup[eid]['candidate_id']==c['candidate_id'] and lookup[eid]['pack_round']==int(r)
        assert set(c['evaluations'])==({'3',c['slot'].split('_')[1]} if c['slot'].startswith('round') else {'3'})
    for item in js:
        key=item['benchmark'],item['task']
        for phase in item['phases']:
            pack=runner/phase['pack'];base=pack.parent;cp=read(base/'checkpoint.json');context,_=build_context(cp['task'],cp['pool'])
            roots=[base/'terminal']
            if item['benchmark']=='claw_eval' and phase['round']==1:
                roots.append(Path(read(pack.parents[3]/'config.json')['source_campaign'])/'tasks'/item['task']/'round_1/terminal')
            matching=[p for root in roots for p in root.rglob('request.json') if read(p).get('context',{}).get('products')==context['products']]
            assert matching or phase['ranking'].get('status')!='compared'
            for p in matching:terminal_sources[str(p.relative_to(runner))]=inv.sha(p)
            assert phase['product_to_candidate']=={p['product_id']:c['candidate_id'] for p,c in zip(context['products'],cp['pool'])}
            assert len(set(phase['candidate_ids']))==3+3*phase['round']
    native=['data/office50.json','data/workbuddy_seed_scores.json','data/claw_eval.json','data/claw_eval_continuation.json']
    legacy=['data/known_reward_kendall.json','data/known_reward_metrics/summary.json','data/rewardpack_joint_repair/trajectory_scores.csv','data/rewardpack_joint_repair/criterion_receipts.csv']
    result=dict(status='passed',candidates=len(cs),evaluations=len(rows),tasks=len(js),
        terminal_request_sources=terminal_sources,native_sources={p:inv.sha(ROOT/p) for p in native},
        preserved_legacy_sources={p:inv.sha(ROOT/p) for p in legacy},
        experiment_sources={p:inv.sha(OUT/p) for p in ['protocol.json','manifest.public.json','candidates.json','jury.json']})
    packs={}
    for row in rows:
        digest=row['hashes']['pack']
        if digest not in packs:
            pack=read(runner/row['paths']['pack'])
            packs[digest]=dict(pack_id=pack['pack_id'],criterion_ids=[c['id'] for c in pack['criteria']])
    catalog=OUT/'pack_criteria.json'
    if catalog.exists():assert read(catalog)==packs
    else:write(catalog,packs)
    target=OUT/'binding_audit.json'
    if target.exists():assert read(target)==result
    else:write(target,result)
    print(json.dumps({k:v for k,v in result.items() if k in ['status','candidates','evaluations','tasks']}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runner',type=Path,default=Path('machine_path_78fc56f4d8a9ea39'));a=p.parse_args();audit(a.runner)
