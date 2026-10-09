"""Recompute task-local ranking statistics for both frozen joint conditions."""
import csv
import json
from collections import defaultdict
from pathlib import Path

from . import analyze
from .inputs import read
from .midpoint import load_version


def run(data_root: Path, output: Path, *, iterations=20000):
    if output.exists():
        raise FileExistsError('Use a new output directory')
    analyze.ITERATIONS = iterations
    summary = {}
    task_rows = []
    for version in ('joint_original', 'joint_repaired'):
        rows = load_version(data_root / version, {})
        groups = defaultdict(list)
        for row in rows:
            groups[(row['benchmark'], row['task'])].append(row)
        computed = []
        for (benchmark, task), records in sorted(groups.items()):
            computed.append({'benchmark': benchmark, 'task': task, 'scenario': records[0]['scenario'],
                             **analyze.task_statistics(records)})
        # Compare to frozen task statistics before claiming numerical agreement.
        with (data_root / version / 'task_statistics.csv').open() as stream:
            expected = {(r['benchmark'], r['task']): r for r in csv.DictReader(stream)}
        for row in computed:
            original = expected[(row['benchmark'], row['task'])]
            for field in ('tau_b', 'spearman_rho'):
                wanted = float(original[field]) if original[field] else None
                if row[field] != wanted:
                    raise ValueError('Joint task statistics changed: ' + row['task'])
        summary[version] = {}
        for benchmark in ('workbuddy', 'claw_eval'):
            selected = [r for r in computed if r['benchmark'] == benchmark]
            summary[version][benchmark] = {'tasks': len(selected), 'metrics': {}}
            for field in ('tau_b', 'spearman_rho'):
                values = [r[field] for r in selected if r[field] is not None]
                summary[version][benchmark]['metrics'][field] = {
                    'value': sum(values)/len(values) if values else None,
                    'defined_tasks': len(values), **analyze.bootstrap(selected, benchmark, field)}
        task_rows.extend({'version': version, **r} for r in computed)
    output.mkdir(parents=True)
    analyze.write_csv(output/'task_statistics.csv', task_rows)
    (output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    (output/'verification.json').write_text(json.dumps({'task_statistics_match': True, 'model_calls': 0})+'\n')
    return summary
