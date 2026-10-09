"""Audit cached semantic-judge input routing without running actors or judges.

Explicit --source reads the archive; the default recomputes from frozen metadata.
Input-directory placement does not determine a task file's semantic role.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from analyze_semantic_archive import confusion

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / 'data/semantic_input_evidence.json'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(source):
    report_path = source / 'reports/workbuddy/office50/report_detail.json'
    study = source / 'experiments/workbuddy/studies/reward_pack_semantic_gap_20260918'
    records_path = study / 'output/semantic_records.json'
    report = read(report_path)['tasks']
    grouped = defaultdict(list)
    for r in read(records_path):
        grouped[r['task'], r['arm']].append(r)
    frozen = {(r['task'], r['arm'], r['test_id']): r for r in read(ROOT / 'data/semantic_archive.json')['records']}
    assert len(report) == 50 and len(grouped) == 150
    rows = []
    for task in report:
        rr = next(r for r in task['rounds'] if r['round'] == 3)
        for arm in 'ABC':
            actor = rr['actors'][arm]
            records = grouped[task['task'], arm]
            for r in records:
                old = frozen[r['task'], r['arm'], r['test_id']]
                assert old['score'] == r['result']['parsed']['score']
                assert old['artifact_chars'] == r['artifact_chars']
                assert r['artifact_manifest'] == records[0]['artifact_manifest']
                assert r['input_manifest'] == records[0]['input_manifest']
            # Reproduce the historical first-link resolution; do not silently fix it.
            link = actor['links'][0]['href']
            ws = Path(str((report_path.parent / link).resolve()).replace('workspace.tar.gz', 'workspace'))
            assert ws.is_dir() and ws.is_relative_to(source.resolve())
            inputs = {f['path']: f for f in records[0]['input_manifest']}
            files = []
            for entry in actor.get('files', []):
                rel = Path(entry['path'])
                assert not rel.is_absolute() and '..' not in rel.parts
                path = ws / rel
                candidate = path if path.is_file() else ws / 'workspace' / rel
                assert candidate.is_file(), (task['task'], arm, str(rel))
                actual_hash = sha(candidate)
                assert actual_hash == entry['sha256'], (task['task'], arm, str(rel))
                files.append(dict(path=str(rel), bytes=entry['bytes'], sha256=actual_hash,
                    historical_root_exists=path.is_file(), bound_path=str(candidate.relative_to(source)),
                    filtered_from_artifact_slot='input' in rel.parts[:-1],
                    cached_input_entry=inputs.get(str(rel))))
            noninput = [str(p.relative_to(ws)) for p in ws.rglob('*')
                        if p.is_file() and 'input' not in p.relative_to(ws).parts[:-1]] if not files else None
            rows.append(dict(task=task['task'], arm=arm, public_task=task['task_prompt'],
                historical_root=str(ws.relative_to(source)), first_link=link,
                artifact_chars=records[0]['artifact_chars'], input_chars=records[0]['input_chars'],
                include_inputs=records[0]['include_inputs'], n_judgments=len(records),
                artifact_manifest=records[0]['artifact_manifest'], listed_files=files,
                noninput_inventory_when_unlisted=noninput,
                final_response=actor.get('final_response') if not files else None))
    payload=dict(scope='Read-only routing and file-identity audit; no regrading or independent defect labels.',
        source_files={str(p.relative_to(source)):sha(p) for p in [report_path, records_path, study/'semantic_rescore.py']},
        artifacts=rows)
    SNAPSHOT.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')


def build():
    rows = read(SNAPSHOT)['artifacts']
    categories = Counter(); excluded=set(); zero=[]; listed=0; mismatched_roots=0
    for row in rows:
        fs=row['listed_files']; listed+=len(fs)
        misplaced=[f for f in fs if f['filtered_from_artifact_slot']]
        badroot=[f for f in fs if not f['historical_root_exists']]
        mismatched_roots+=len(badroot)
        if misplaced or badroot:excluded.add(row['task'])
        if row['artifact_chars']:
            label='Nonempty artifact slot'
        elif fs and len(misplaced)==len(fs):
            label='Outputs filtered by input-directory rule'
            assert all(f['cached_input_entry'] is not None for f in fs)
        elif badroot:
            label='Workspace root mismatch'
        else:
            label='No non-input file in selected snapshot'
            assert not fs and not row['noninput_inventory_when_unlisted']
        categories[label]+=1
        if not row['artifact_chars']:
            zero.append(dict(task=row['task'],arm=row['arm'],category=label,
                listed_files=len(fs), listed_files_in_input_text=sum(
                    bool(f['cached_input_entry']) and f['cached_input_entry']['status']=='ok' for f in fs)))
    scores=defaultdict(list)
    for r in read(ROOT/'data/semantic_archive.json')['records']:
        scores[r['task'],r['arm']].append(r['score'])
    scores={key:sum(v)/len(v) for key,v in scores.items()}
    office=read(ROOT/'data/office50.json')['tasks'];kept=[r for r in office if r['task'] not in excluded]
    result=dict(n_artifacts=len(rows), verified_listed_files=listed,
        files_under_misresolved_root=mismatched_roots,
        input_directory_output_artifacts=sum(any(f['filtered_from_artifact_slot'] for f in r['listed_files']) for r in rows),
        categories=dict(categories),zero_slot_cases=zero, routing_excluded_tasks=sorted(excluded),
        routing_sensitivity=confusion(kept,scores),
        interpretation='Path routing is observed; no causal score effect is estimated. Text in the source-material section may still expose output content. No corrected judge score is inferred.')
    (ROOT/'data/semantic_input_statistics.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=[r'\begin{tabular}{lr}',r'\toprule',r'Archived artifact-slot condition & Candidates \\',r'\midrule']
    for label,n in categories.items():lines.append(f'{label} & {n}'+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (ROOT/'tables/workbuddy_semantic_inputs.tex').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='zero_slot_cases'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path);args=parser.parse_args()
    if args.source:freeze(args.source.resolve())
    build()
