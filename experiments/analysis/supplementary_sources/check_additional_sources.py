"""Prespecified purposive checks on three additional archived tasks.

Only public inputs and delivered workbooks are used for reference arithmetic;
no native gold or graders. --source records comparisons, default summarizes.
"""
import argparse
from collections import defaultdict
import csv
import datetime as dt
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import openpyxl

ROOT=Path(__file__).resolve().parents[1]
SNAPSHOT=ROOT/'data/additional_source_checks.json'
D=Decimal


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def rows(ws):return list(ws.values)
def cmp(key,expected,observed,tolerance='0.005'):
    try:error=abs(D(str(expected))-D(str(observed)));matches=error<=D(tolerance)
    except Exception:error=None;matches=False
    return dict(key=key,expected=str(expected),observed=str(observed),absolute_error=str(error) if error is not None else None,tolerance=tolerance,matches=matches)

def setcmp(key,expected,observed):
    return dict(key=key,expected=sorted(expected),observed=sorted(observed),matches=set(expected)==set(observed) and len(observed)==len(set(observed)))

def health(inputs,book):
    amounts={};categories=defaultdict(D);category=None;category_index=-1;category_labels={}
    for line in inputs['报价内容.md'].read_text().splitlines():
        cells=[c.strip() for c in line.strip().strip('|').split('|')]
        if len(cells)!=5 or not re.fullmatch('HQ-\\d+',cells[1]):continue
        if cells[0]:
            category_index+=1;category='一二三四五'[category_index]
            text=cells[0].strip('#* ');category_labels[text]=category
            category_labels[re.sub(r'^[一二三四五][、.． ]*','',text)]=category
            category_labels[category]=category
        value=D(cells[4]);amounts[cells[1]]=value;categories[category]+=value
    assert len(amounts)==80 and len(categories)==5
    main=rows(book.worksheets[0]);headers=list(main[0]);code=headers.index('编号');price=headers.index('报价(万元)')
    output={r[code]:r[price] for r in main[1:] if r[code] in amounts}
    checks=[cmp(k,v,output.get(k)) for k,v in amounts.items()]
    summary=rows(book['类别汇总']);out={}
    for r in summary[1:]:
        label=str(r[0]).strip('#* ');key=category_labels.get(label,category_labels.get(re.sub(r'^[一二三四五][、.． ]*','',label)))
        if key is not None:
            assert key not in out
            out[key]=r[1]
    checks += [cmp('category_'+k,v,out.get(k)) for k,v in categories.items()]
    return dict(numeric=checks,sets=[setcmp('item_ids',amounts,[r[code] for r in main[1:] if r[code] is not None]),setcmp('categories',categories,list(out))],reference=dict(items=len(amounts),category_subtotals={k:str(v) for k,v in categories.items()},category_label_mapping=category_labels))

def procurement(inputs,book):
    with inputs['采购订单.csv'].open(encoding='utf-8-sig') as f:orders=list(csv.DictReader(f))
    with inputs['入库记录.csv'].open(encoding='utf-8-sig') as f:receipts=list(csv.DictReader(f))
    order={r['ORDER_ID']:r for r in orders};assert len(order)==len(orders)
    qty=defaultdict(D);amount=defaultdict(D);receipt_ids=defaultdict(list)
    for r in receipts:
        key=r['RECEIPT_NO'].rsplit('-',1)[0];qty[key]+=D(r['RECV_QTY']);amount[key]+=D(r['RECV_AMOUNT']);receipt_ids[key].append(r['RECEIPT_NO'])
    matched=set(order)&set(qty);missing=set(order)-set(qty);orphans={x for k,ids in receipt_ids.items() if k not in order for x in ids}
    sets=[];checks=[];counts={}
    for sheet,field,received,tol in [('数量差异','ORDER_QTY',qty,D(0)),('金额差异','TOTAL_AMOUNT',amount,D('0.01'))]:
        expected={k:(D(order[k][field]),received[k],D(order[k][field])-received[k]) for k in matched if abs(D(order[k][field])-received[k])>tol}
        rr=rows(book[sheet]);out={r[0]:r for r in rr[1:] if r[0] is not None};sets.append(setcmp(sheet,expected,[r[0] for r in rr[1:] if r[0] is not None]));counts[sheet]=len(expected)
        for key,values in sorted(expected.items()):
            for j,value in enumerate(values,1):checks.append(cmp(f'{sheet}/{key}/{j}',value,out.get(key,[None]*4)[j], '0' if field=='ORDER_QTY' else '0.01'))
    sets += [setcmp('未入库订单',missing,[r[0] for r in rows(book['未入库订单'])[1:] if r[0] is not None]),setcmp('无对应订单入库',orphans,[r[0] for r in rows(book['无对应订单入库'])[1:] if r[0] is not None])]
    # Compare the six explicitly labeled summary counts, without accepting aliases.
    summary={str(r[0]):r[1] for r in rows(book['汇总'])[1:] if r[0] is not None}
    expected_summary={'总采购订单数':len(orders),'总入库数':len(receipts)}
    for label,n in counts.items():
        keys=[k for k in summary if label in k];assert len(keys)==1
        expected_summary[keys[0]]=n
    for label,n in [('未入库',len(missing)),('无对应订单',len(orphans))]:
        keys=[k for k in summary if label in k];assert len(keys)==1
        expected_summary[keys[0]]=n
    checks += [cmp(k,v,summary.get(k),'0') for k,v in expected_summary.items()]
    return dict(numeric=checks,sets=sets,reference=dict(orders=len(orders),receipts=len(receipts),quantity_differences=counts['数量差异'],amount_differences=counts['金额差异'],missing_orders=len(missing),orphan_receipts=len(orphans)))

def date(value):
    if isinstance(value,dt.datetime):return value.date()
    if re.fullmatch(r'\d{8}',str(value)):return dt.datetime.strptime(str(value),'%Y%m%d').date()
    m=re.search(r'(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})',str(value));assert m,value
    return dt.date(*map(int,m.groups()))

def weekly(inputs,book):
    source=openpyxl.load_workbook(inputs['客服工单处理记录.xlsx'],read_only=True,data_only=True);rr=rows(source.active);source.close();h=list(rr[0]);source_rows=len(rr)-1;groups=defaultdict(lambda:[D(0),D(0)])
    for row in rr[1:]:
        day=date(row[h.index('TICKET_DATE')]);monday=day-dt.timedelta(days=day.weekday())
        groups[str(monday)][0]+=D(str(row[h.index('PLAN_COUNT')]));groups[str(monday)][1]+=D(str(row[h.index('RESOLVED_COUNT')]))
    rr=rows(book.active);h=list(rr[0]);datecol=next(i for i,x in enumerate(h) if '日期区间'==x or '周起始' in str(x));plan=h.index('计划处理量');actual=h.index('实际解决量');rate=next(i for i,x in enumerate(h) if '完成率' in str(x));output={};notes=[]
    for row_number,row in enumerate(rr[1:],2):
        if any(str(x).strip() in ['合计','总计'] for x in row):continue
        if not re.search(r'\d{4}[-/年]\d',str(row[datecol])):continue
        key=str(date(row[datecol]));assert key not in output
        output[key]=(row,row_number)
    checks=[]
    for key,(p,a) in sorted(groups.items()):
        row,row_number=output.get(key,(None,None));checks += [cmp(key+'/plan',p,row[plan] if row else None,'0'),cmp(key+'/actual',a,row[actual] if row else None,'0')]
        value=row[rate] if row else None
        number_format=book.active.cell(row_number,rate+1).number_format if row else None
        # Explicit percentage headers/strings or a literal percent in Excel's
        # number format indicate percentage-valued data. Standard 0.00% formatting
        # multiplies fractional cell values for display, so those remain fractions.
        if value is not None:
            value=D(str(value).rstrip('%'))
            if '%' in str(h[rate]) or '%' in str(row[rate]) or '"%"' in number_format or '\\%' in number_format:value/=100
        notes.append(dict(week=key,header=h[rate],stored_value=row[rate] if row else None,number_format=number_format,normalized_fraction=str(value)))
        checks.append(cmp(key+'/rate',a/p,value,'0.0005'))
    return dict(numeric=checks,sets=[setcmp('week_starts',groups,list(output))],reference=dict(source_rows=source_rows,weekly_totals={k:list(map(str,v)) for k,v in sorted(groups.items())}),rate_cells=notes,rate_policy='Normalize explicit percentage headers/strings or literal-percent Excel number formats to fractions; ordinary percent number formats already store fractions. Tolerance 0.0005 permits rounding to one decimal percent.')


def collect(source):
    protocol=read(ROOT/'review-stage/ADDITIONAL_ARITHMETIC_PROTOCOL.json');e=read(ROOT/'data/semantic_input_evidence.json')['artifacts'];tasks=[];hashes={}
    for task in protocol['tasks']:
        records=[next(r for r in e if r['task']==task and r['arm']==a) for a in 'ABC'];reference_ws=source/records[0]['historical_root'];actor=reference_ws.parents[2];initial=actor/'snapshots/initial/workspace/input'
        inputs={p.name:p for p in initial.iterdir() if p.is_file()};assert inputs
        identity=[]
        for r in records:
            ws=source/r['historical_root']
            for name,p in inputs.items():
                assert sha(ws.parents[2]/'snapshots/initial/workspace/input'/name)==sha(p)
                identity.append(dict(arm=r['arm'],file=name,final_input_unchanged=sha(ws/'input'/name)==sha(p)))
                hashes[str(p.relative_to(source))]=sha(p)
        checks=[]
        for r in records:
            files=[f for f in r['listed_files'] if f['path'].endswith('.xlsx')];assert len(files)==1
            p=source/files[0]['bound_path'];assert sha(p)==files[0]['sha256'];hashes[str(p.relative_to(source))]=sha(p)
            wb=openpyxl.load_workbook(p,read_only=True,data_only=True)
            fn=health if task.startswith('health') else procurement if task.startswith('procurement') else weekly
            result=fn(inputs,wb);wb.close();checks.append(dict(arm=r['arm'],**result))
        tasks.append(dict(task=task,input_identity=identity,arms=checks))
    SNAPSHOT.write_text(json.dumps(dict(protocol_sha256=sha(ROOT/'review-stage/ADDITIONAL_ARITHMETIC_PROTOCOL.json'),scope='Scoped public-source arithmetic and membership checks on a purposive sample; no full-task certification, native gold, or grading code.',source_hashes=hashes,tasks=tasks),ensure_ascii=False,indent=2)+'\n')


def build():
    d=read(SNAPSHOT);lines=[r'\begin{tabular}{llrr}',r'\toprule',r'Task & Candidate & Numeric matches & Membership matches \\',r'\midrule'];out=[]
    for t in d['tasks']:
        for a in t['arms']:
            n=a['numeric'];s=a['sets']
            for c in n:
                assert c == cmp(c['key'],c['expected'],c['observed'],c['tolerance'])
            for c in s:
                assert c == setcmp(c['key'],c['expected'],c['observed'])
            passed=sum(c['matches'] for c in n);sp=sum(c['matches'] for c in s)
            label={'health-quote-L2-003':'Quote subtotals','procurement-reconcile-L3-015':'Procurement','ticket-weekly-L3-010':'Weekly tickets'}[t['task']]
            candidate={'A':'Task only','B':'Reward-guided revision','C':'Hypothesis guidance'}[a['arm']]
            lines.append(f"{label} & {candidate} & {passed}/{len(n)} & {sp}/{len(s)}"+r' \\')
            out.append(dict(task=t['task'],arm=a['arm'],numeric=f'{passed}/{len(n)}',membership=f'{sp}/{len(s)}',failures=[c for c in n+s if not c['matches']]))
    lines += [r'\bottomrule',r'\end{tabular}'];(ROOT/'tables/workbuddy_additional_checks.tex').write_text('\n'.join(lines)+'\n');print(json.dumps(out,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path);args=parser.parse_args()
    if args.source:collect(args.source.resolve())
    build()
