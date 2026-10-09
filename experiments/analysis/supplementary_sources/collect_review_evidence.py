"""Read-only collection of final measurements and original-campaign references.

No actors, graders, or external services are run. Final estimates do not count
all intermediate measurements, and external score explanations are not labels.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def collect(source):
    claw = read(ROOT / 'data/claw_eval.json')['tasks']
    office = read(ROOT / 'data/office50.json')['tasks']
    campaign = source / 'local/claw_eval/office104_web19_composite_20260924'
    run_sources = {t['task']: t['source'] for t in read(source / 'reports/workbuddy/office50/results.json')['tasks']}
    measurements, checkers, scenarios = [], Counter(), {}
    for benchmark, tasks in [('claw-eval', claw), ('WorkBuddy Office', office)]:
        for row in tasks:
            run = (campaign / 'tasks' / row['task'] / 'round_1' if benchmark == 'claw-eval'
                   else source / 'local/workbuddy/outcome/office50' / run_sources[row['task']] / 'round_3')
            selected = read(run / 'selected_delivery.json')['arm']
            assert selected == row['selected_arm']
            for arm in 'ABC':
                path = run / arm / 'outcome_estimate.json'
                estimate = read(path)
                delivery = read(run / arm / 'delivery.json')
                records = estimate.get('results', [])
                measurements.append(dict(benchmark=benchmark, task=row['task'], arm=arm,
                    selected=arm == selected, source=str(path.relative_to(source)),
                    review_status=delivery.get('review_status'), hypothesis_applied=delivery.get('hypothesis_applied'),
                    results=[{k: item.get(k) for k in ('test_id', 'criterion_id', 'checker',
                        'status', 'error', 'reason', 'diagnostic')} for item in records]))
            if benchmark == 'claw-eval':
                checkers.update(t['checker'] for t in read(run / 'pack.json')['tests'])
                scenarios.setdefault(re.sub(r'^T\d+(?:zh)?_', '', row['task']), []).append(row['task'])
    # Verify pair membership against the frozen task definitions, not names alone.
    import yaml
    scenario_checks = []
    for key, ids in scenarios.items():
        if len(ids) != 2:
            continue
        definitions = [yaml.safe_load((campaign / 'benchmark/tasks' / task / 'task.yaml').read_text()) for task in ids]
        assert {d['prompt']['language'] for d in definitions} == {'zh', 'en'}
        assert [x['name'] for x in definitions[0].get('services', [])] == [x['name'] for x in definitions[1].get('services', [])]
        assert [x['name'] for x in definitions[0].get('tools', [])] == [x['name'] for x in definitions[1].get('tools', [])]
        scenario_checks.append(dict(scenario=key, tasks=ids, same_service_and_tool_names=True))
    assert len(scenario_checks) == 36
    # Reproduce the deterministic early exit behind the largest C loss without
    # rerunning the native scorer or changing its accepted score.
    worst = min(claw, key=lambda r: r['C'] - r['A_mean'])
    assert worst['task'] == 'T040_onboarding_coordinator'
    actor = campaign / 'tasks' / worst['task'] / 'round_1/actors/C'
    receipt = read(actor / 'OFFICIAL.private.json')
    sends = [event['request_body'] for event in map(json.loads, (actor / 'trace.jsonl').read_text().splitlines())
             if event['type'] == 'tool_dispatch' and event.get('tool_name') == 'gmail_send_message']
    case = dict(task=worst['task'], sampling_rule='largest C minus A_mean loss',
                score=receipt['task_score'] * 100, dimensions=receipt['scores'],
                judge_call_count=len(receipt['judge_calls']), sent_messages=sends,
                trace=str((actor / 'trace.jsonl').relative_to(source)),
                grader='benchmark/tasks/T040_onboarding_coordinator/grader.py',
                base_grader='benchmark/tasks/T039zh_onboarding_coordinator/grader.py',
                interpretation='Native safety early exit; no corrected score assigned. This is not an independently adjudicated artifact failure.')
    # Fixed descriptive sample: three largest C losses and two closest-to-median
    # cases, with task-ID tie breaking. Sampling does not create defect labels.
    import statistics
    median = statistics.median(r['C'] - r['A_mean'] for r in claw)
    sample = sorted(claw, key=lambda r:(r['C']-r['A_mean'], r['task']))[:3]
    sample += sorted(claw, key=lambda r:(abs(r['C']-r['A_mean']-median), r['task']))[:2]
    cases=[]
    for row in sample:
        run=campaign/'tasks'/row['task']/'round_1'
        delivery=read(run/'C/delivery.json');receipt=read(run/'actors/C/OFFICIAL.private.json')
        task=yaml.safe_load((campaign/'benchmark/tasks'/row['task']/'task.yaml').read_text())
        cases.append(dict(task=row['task'],public_prompt=task['prompt']['text'],
                          hypothesis=read(run/'hypothesis.json'),
                          hypothesis_applied=delivery.get('hypothesis_applied'),
                          final_response=delivery['candidate']['artifact']['final_response'],
                          native_scores=receipt['scores'],judge_calls=receipt['judge_calls'],
                          source_directory=str(run.relative_to(source))))
    out = dict(scope='All final A/B/C outcome estimates at WorkBuddy iteration 3 and claw-eval iteration 1; not all historical measurement calls.',
               measurements=measurements, claw_test_dispatchers=dict(checkers),
               verified_scenario_pairs=scenario_checks,
               largest_C_loss_case=case, fixed_diagnostic_cases=cases)
    path = ROOT / 'data/archive_review_evidence.json'
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
    print(f'Collected {len(measurements)} final estimates, {sum(checkers.values())} tests.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    collect(parser.parse_args().source)
