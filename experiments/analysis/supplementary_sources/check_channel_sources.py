"""Independent arithmetic check of one preselected case against public inputs.

Uses exact-row deduplication, nonempty QUARTER filtering and Decimal sums.
Never reads benchmark gold, native scoring code, or model judgments.
--source freezes per-cell comparisons; default rebuilds the descriptive table.
"""
import argparse
from collections import defaultdict
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import openpyxl

ROOT=Path(__file__).resolve().parents[1]
SNAPSHOT=ROOT/'data/channel_source_checks.json'
TOLERANCE=Decimal('0.005')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def norm(s):return str(s).replace('_','').replace(' ','').strip()


def collect(source):
    actors=source/'data/workbuddy_office/tasks/channel-period-compare-L4-017/round_3/actors'
    definitions=[('settlement','渠道结算明细.xlsx','25Q4',[('GMV','GMV'),('REVENUE','营收'),('ORDER_COUNT','订单量')]),
                 ('payment','渠道回款明细.xlsx','25Q1',[('GMV','GMV'),('REVENUE','营收')])]
    references={};provenance={};cleaning={}
    for dimension,filename,baseline,metrics in definitions:
        input_path=actors/'A/snapshots/initial/workspace/input'/filename
        expected_hash=sha(input_path);provenance[str(input_path.relative_to(source))]=expected_hash
        for arm,delivery in [('A',1),('B',3),('C',1)]:
            for snapshot in ['initial',f'delivery-{delivery}']:
                p=actors/arm/f'snapshots/{snapshot}/workspace/input'/filename
                assert sha(p)==expected_hash,(arm,snapshot,filename)
        wb=openpyxl.load_workbook(input_path,read_only=True,data_only=True)
        rows=list(wb.active.values);wb.close();headers=list(rows[0]);raw=rows[1:]
        unique=list(dict.fromkeys(raw));q=headers.index('QUARTER');channel=headers.index('CHANNEL_NAME')
        cleaned=[r for r in unique if r[q] is not None and str(r[q]).strip()]
        cleaning[dimension]=dict(raw_rows=len(raw),unique_rows=len(unique),nonempty_quarter_rows=len(cleaned))
        sums=defaultdict(Decimal)
        for row in cleaned:
            for field,label in metrics:sums[row[channel],str(row[q]),field]+=Decimal(str(row[headers.index(field)]))
        refs={}
        for name in sorted({r[channel] for r in cleaned}):
            for field,label in metrics:
                low=sums[name,baseline,field];high=sums[name,'26Q1',field]
                refs[name,label,baseline]=low;refs[name,label,'26Q1']=high;refs[name,label,'change']=high-low
        references[dimension]=(baseline,metrics,refs)
    candidates=[]
    for arm,delivery in [('A',1),('B',3),('C',1)]:
        book=actors/arm/f'snapshots/delivery-{delivery}/workspace/output/result.xlsx'
        provenance[str(book.relative_to(source))]=sha(book)
        wb=openpyxl.load_workbook(book,read_only=True,data_only=True);results=[]
        for dimension,(baseline,metrics,refs) in references.items():
            # Select by explicit sheet period and verify unique header/data identities.
            sheets=[ws for ws in wb if baseline in ws.title];assert len(sheets)==1
            ws=sheets[0];rows=list(ws.values)
            indices=[i for i,row in enumerate(rows) if row and row[0]=='渠道商'];assert len(indices)==1
            idx=indices[0];headers=list(rows[idx]);names={key[0] for key in refs};data={}
            for row in rows[idx+1:]:
                if row and row[0] in names:
                    assert row[0] not in data
                    data[row[0]]=row
            assert set(data)==names and len(names)==20
            for (channel,label,period),expected in refs.items():
                names_allowed={label+'变化额'} if period=='change' else {label+period,period+label}
                columns=[j for j,h in enumerate(headers) if norm(h) in names_allowed];assert len(columns)==1,(arm,label,period)
                actual=Decimal(str(data[channel][columns[0]]));error=abs(actual-expected)
                results.append(dict(dimension=dimension,channel=channel,metric=label,period=period,
                    expected=str(expected),observed=str(actual),absolute_error=str(error),matches=error<=TOLERANCE))
        wb.close();candidates.append(dict(arm=arm,checks=results))
    snapshot=dict(task='channel-period-compare-L4-017',selection='Previously selected largest runtime-selection gap; no population inference.',
        excluded_metrics=['Commission rate and denominator','B-side revenue definition','Headcount aggregation','Relative percentage changes'],
        procedure='Deduplicate exact full input rows, discard empty QUARTER rows, sum public metric values by channel and quarter using Decimal, subtract base from current quarter. Match literal metric/period output headers after removing spaces and underscores; no gold or grader used.',
        tolerance=str(TOLERANCE),input_initial_and_final_identity_checked_for_all_arms=True,
        source_hashes=provenance,cleaning=cleaning,arms=candidates)
    SNAPSHOT.write_text(json.dumps(snapshot,ensure_ascii=False,indent=2)+'\n')


def build():
    d=json.loads(SNAPSHOT.read_text());lines=[r'\begin{tabular}{lrrr}',r'\toprule',r'Candidate & Settlement values & Payment values & Maximum absolute error \\',r'\midrule'];summaries=[]
    for a in d['arms']:
        by=defaultdict(list)
        for c in a['checks']:
            error=abs(Decimal(c['expected'])-Decimal(c['observed']))
            assert error==Decimal(c['absolute_error'])
            assert c['matches']==(error<=Decimal(d['tolerance']))
            by[c['dimension']].append(c['matches'])
        maximum=max(Decimal(c['absolute_error']) for c in a['checks'])
        parts=[f"{sum(by[k])}/{len(by[k])}" for k in ['settlement','payment']]
        label={'A':'Task only','B':'Reward-guided revision','C':'Hypothesis guidance'}[a['arm']]
        lines.append(f"{label} & {parts[0]} & {parts[1]} & {float(maximum):.1e}"+r' \\')
        summaries.append(dict(arm=a['arm'],counts=parts,max_error=str(maximum)))
    lines += [r'\bottomrule',r'\end{tabular}']
    (ROOT/'tables/workbuddy_source_checks.tex').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(cleaning=d['cleaning'],summaries=summaries),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path);args=parser.parse_args()
    if args.source:collect(args.source.resolve())
    build()
