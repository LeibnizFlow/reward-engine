"""Freeze supplementary receipts for the WorkBuddy Bench Office analysis.

Explicit, read-only access to the execution archive; writes only a manuscript
snapshot. It never runs actors or graders. Normal reporting uses the snapshot.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
args = parser.parse_args()
archive = args.source / 'local/workbuddy/outcome/office50'
report = args.source / 'reports/workbuddy/office50/report_detail.json'
snapshot = ROOT / 'data/office50.json'
data = json.loads(snapshot.read_text())
detail = {t['task']: t for t in json.loads(report.read_text())['tasks']}
run_sources = {t['task']: t['source'] for t in json.loads((args.source / 'reports/workbuddy/office50/results.json').read_text())['tasks']}
files = {}

def read(path):
    raw = path.read_bytes()
    files[str(path.relative_to(args.source))] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)

audit = []
cases = []
selected_measurements = []
case_ids = ['priority-sync-notification-pipeline', 'channel-period-compare-L4-017']
for task in data['tasks']:
    run = archive / run_sources[task['task']] / 'round_3'
    for arm in 'ABC':
        actor_record = detail[task['task']]['rounds'][-1]['actors'][arm]
        receipt_path = archive / actor_record['native_source']
        receipt = read(receipt_path)
        assert abs(100 * receipt['reward'] - task[arm]) < 1e-6, (task['task'], arm)
        judges = []
        for judge in receipt.get('judges', []):
            meta = judge.get('metadata', {})
            stdout = meta.get('stdout', '')
            judges.append({
                'name': judge['judge_name'], 'scores': judge.get('scores'),
                'status': judge.get('status'), 'ok': judge.get('ok'),
                'tests_passed': meta.get('tests_passed'), 'tests_total': meta.get('tests_total'),
                'script_exit_code': meta.get('script_exit_code'),
                'evaluator_exception': 'evaluator_runtime_error' in stdout or 'evaluator raised:' in stdout,
                'missing_extracted_value': 'agent=None' in stdout,
                'diagnostic_lines': list(dict.fromkeys(line for line in stdout.splitlines()
                    if 'evaluator raised:' in line or ('agent=None' in line and line.startswith('E   AssertionError:')))),
                'errors': judge.get('errors', []),
            })
        entry = {'task': task['task'], 'arm': arm, 'score': task[arm],
                 'source': str(receipt_path.relative_to(args.source)), 'judges': judges}
        if task['task'] in case_ids:
            entry['verdicts'] = [{k: v.get(k) for k in ['item_id', 'status', 'score', 'reason', 'metadata']}
                                 for v in receipt.get('verdicts', [])]
        audit.append(entry)
    selected = read(run / 'selected_delivery.json')
    ranking = read(run / 'ranking.json')
    selected_arm = selected['arm']
    measurement = read(run / selected_arm / 'outcome_estimate.json')
    selected_delivery = read(run / selected_arm / 'delivery.json')
    measurement_rows = measurement.get('results', [])
    status_counts = {status: sum(row.get('status') == status for row in measurement_rows)
                     for status in ['pass', 'fail', 'unknown']}
    selected_measurements.append({
        'task': task['task'],
        'selected_arm': selected_arm,
        'measurement_rows': len(measurement_rows),
        'status_counts': status_counts,
        'execution_error_rows': sum(bool(row.get('error')) for row in measurement_rows),
        'tool_observations': sum(len(row.get('tool_observations', {})) for row in measurement_rows),
        'official_verifier_observed': selected_delivery.get('official_verifier_observed'),
        'ranking_mentions_grader_or_checker': any(term in json.dumps(ranking, ensure_ascii=False).lower()
            for term in ['official checker', 'external checker', 'official grader', 'external grader']),
    })
    if task['task'] in case_ids:
        hypothesis = read(run / 'hypothesis.json')
        pack = read(run / 'pack.json')
        actors = {}
        for arm in 'ABC':
            delivery = read(run / arm / 'delivery.json')
            artifact = delivery['candidate']['artifact']
            actors[arm] = {'final_response': artifact.get('final_response'),
                           'changed_files': artifact.get('changed_files'),
                           'review_status': delivery.get('review_status')}
        case = {'task': task['task'], 'task_prompt': detail[task['task']]['task_prompt'],
                      'criterion_count': detail[task['task']]['criterion_count'],
                      'test_count': detail[task['task']]['test_count'],
                      'hypothesis_applied': detail[task['task']]['hypothesis_applied'],
                      'hypothesis': hypothesis, 'selected_delivery': selected,
                      'ranking': {'groups': ranking['groups'], 'reason': ranking.get('reason'),
                                  'selected_observations': [o for o in ranking.get('observations', [])
                                      if o['product_id'] == selected['product_id']]},
                      'actors': actors}
        if task['task'] == 'channel-period-compare-L4-017':
            criteria = {row['id']: row for row in pack['criteria']}
            tests = {row['test_id']: row for row in pack['tests']}
            b_measurement = read(run / 'B' / 'outcome_estimate.json')
            receipts = {row['test_id']: row for row in b_measurement.get('results', [])}
            case['rewardpack_example'] = {
                'pack_id': pack['pack_id'],
                'criterion': {key: criteria['C4_output_workbook'][key]
                              for key in ['id', 'standard', 'kind', 'origin', 'measurement_question',
                                          'subrequirements', 'anchors']},
                'test': {key: tests['C4_workbook_sheets'][key]
                         for key in ['test_id', 'criterion_id', 'checker', 'question', 'probes',
                                     'binding_status', 'required_tools']},
                'receipts': [{key: receipts[test_id].get(key)
                              for key in ['test_id', 'status', 'error', 'reason', 'gap', 'evidence_refs']}
                             for test_id in ['C4_workbook_sheets', 'C5_metric_traceability',
                                             'C7_reproducibility_reporting']],
                'runtime_selection': selected_arm,
                'selected_arm_measurement_rows': len(measurement_rows),
            }
        cases.append(case)
files[str(report.relative_to(args.source))] = hashlib.sha256(report.read_bytes()).hexdigest()
out = {'source_root': str(args.source), 'score_snapshot_sha256': hashlib.sha256(snapshot.read_bytes()).hexdigest(),
       'audit_scope': 'All 150 third-iteration A/B/C score receipts; scan stored rule stdout for evaluator_runtime_error or evaluator raised:. This is not a complete revalidation of graders or artifacts.',
       'selected_measurement_scope': 'Stored outcome estimates for all 50 selected third-iteration deliveries. Empty results mean that no per-criterion measurement row was retained in that estimate; they do not establish that no check was applicable.',
       'files': files, 'receipt_audit': audit, 'selected_measurements': selected_measurements,
       'cases': cases}
(ROOT / 'data/workbuddy_office_evidence.json').write_text(json.dumps(out, ensure_ascii=False, indent=2)+'\n')
print(f'Collected {len(audit)} receipts and {len(cases)} cases; source hashes recorded.')
