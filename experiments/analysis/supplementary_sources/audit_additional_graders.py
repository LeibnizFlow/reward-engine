"""Inspect native checks after freezing independent public-source references.

Replay archived native checks without modifying artifacts, gold, or graders.
Compare their intermediate representations with the separately frozen source
calculation. No accepted scores change and no actors or model judges are called.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import types

ROOT=Path(__file__).resolve().parents[1]
SNAPSHOT=ROOT/'data/additional_grader_audit.json'


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def collect(source):
    independent=read(ROOT/'data/additional_source_checks.json')
    evidence=read(ROOT/'data/semantic_input_evidence.json')['artifacts']
    cohort=source/'local/workbuddy/outcome/office50/sources/historical_run/main28'
    hashes={};tasks=[]
    for task in independent['tasks']:
        name=task['task'];dataset=cohort/'dataset/wb-bench-office-v1.0/tasks'/name
        code=dataset/'tests/grading/eval_core.py';gold_file=dataset/'tests/gold/gold_answer.json';gold=read(gold_file)
        module=types.ModuleType('archived_grader')
        exec(compile(code.read_text(),str(code),'exec'),module.__dict__)
        module.load_gold=lambda:gold
        for p in [code,gold_file,dataset/'instruction.md']:hashes[str(p.relative_to(source))]=sha(p)
        arms=[]
        for arm in 'ABC':
            record=next(r for r in evidence if r['task']==name and r['arm']==arm)
            f=next(f for f in record['listed_files'] if f['path'].endswith('.xlsx'))
            book=source/f['bound_path'];assert sha(book)==f['sha256']
            actor=cohort/'tasks'/name/'round_3/actors'/arm
            verifier=next((actor/'harbor').glob('*/verifier'))
            # Bind to the actual workbook retained by the native verifier.
            copies=[p for p in (verifier/'raw_artifacts').rglob('*.xlsx') if sha(p)==sha(book)]
            assert copies,(name,arm)
            native=module.evaluate(str(book));logged=read(verifier/'reward.json')
            assert native['passed_count']==logged['tests_passed'] and native['total_count']==logged['tests_total']
            statuses={m.group(1):m.group(2)=='PASSED' for m in re.finditer(r'test_atomic_check\[\d+_([^\]]+)\]\s+(PASSED|FAILED)',(verifier/'test_output.txt').read_text())}
            assert statuses=={c['name']:c['passed'] for c in native['checks']}
            for p in [book,*copies,verifier/'test_output.txt',verifier/'reward.json']:hashes[str(p.relative_to(source))]=sha(p)
            result=dict(arm=arm,workbook=str(book.relative_to(source)),native=native)
            if name.startswith('procurement'):
                sheets=module._read_xlsx_sheets([book]);classified=module._classify_sheets(sheets,gold)
                result['native_sheet_classification']={k:[dict(sheet=x[0],ids=sorted(x[3])) for x in v] for k,v in classified.items()}
                amount=next(x for x in sheets if x[1]=='金额差异')
                ids=module._extract_ids_from_sheet(amount[3],module._OID_PAT)
                result['amount_sheet_jaccard']={k:module._jaccard(ids,{m['order_id'] for m in gold[k+'_mismatches']}) for k in ['qty','amt']}
            elif name.startswith('ticket'):
                headers,rows,_=module._read_output(str(book))
                pairs,status=module._match_agent_rows_to_gold(rows,headers,gold['weekly_summary'])
                result['native_rows']=rows;result['native_match_status']=status;result['matched_weeks']=len(pairs)
            arms.append(result)
        comparisons={}
        ref=task['arms'][0]
        if name.startswith('health'):
            comparisons=dict(source_category_labels=ref['reference']['category_label_mapping'],native_category_labels=gold['modules'],source_subtotals=ref['reference']['category_subtotals'],native_subtotals=gold['module_subtotals'])
        elif name.startswith('procurement'):
            source_qty=set(next(x for x in ref['sets'] if x['key']=='数量差异')['expected'])
            source_amt=set(next(x for x in ref['sets'] if x['key']=='金额差异')['expected'])
            gold_qty={m['order_id'] for m in gold['qty_mismatches']};gold_amt={m['order_id'] for m in gold['amt_mismatches']}
            comparisons=dict(source_quantity_count=len(source_qty),source_amount_count=len(source_amt),gold_quantity_count=gold['qty_mismatch_count'],gold_amount_count=gold['amt_mismatch_count'],source_amount_omitted_by_gold=sorted(source_amt-gold_amt),omitted_amount_ids_equal_quantity_ids=(source_amt-gold_amt)==source_qty,source_amount_ids_equal_union_of_gold_categories=source_amt==gold_qty|gold_amt)
        tasks.append(dict(task=name,reference_comparison=comparisons,arms=arms))
    SNAPSHOT.write_text(json.dumps(dict(scope='Post-reference diagnosis of the fixed three-task sample; original native checks only, no replacement scores.',independent_snapshot_sha256=sha(ROOT/'data/additional_source_checks.json'),source_hashes=hashes,tasks=tasks),ensure_ascii=False,indent=2)+'\n')


def build():
    d=read(SNAPSHOT)
    assert d['independent_snapshot_sha256']==sha(ROOT/'data/additional_source_checks.json')
    summary=[]
    for t in d['tasks']:
        for a in t['arms']:
            n=a['native'];assert n['passed_count']==sum(c['passed'] for c in n['checks'])
            summary.append(dict(task=t['task'],arm=a['arm'],native=f"{n['passed_count']}/{n['total_count']}",failed=[c['name'] for c in n['checks'] if not c['passed']]))
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path);args=p.parse_args()
    if args.source:collect(args.source.resolve())
    build()
