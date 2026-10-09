"""Verify the published evidence, score bindings, and report links offline."""
import argparse
import html.parser
import json
import math
from pathlib import Path
from urllib.parse import unquote, urlparse

from .evidence import Evidence, sha256
from .report import EXPERIMENTS, validate_rows


class Links(html.parser.HTMLParser):
    def __init__(self):
        super().__init__(); self.links = []

    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if name in ('href', 'src') and value:
                self.links.append(value)


def verify_seed_grades(evidence, task_ids):
    """Bind later native seed measurements without altering the main experiment."""
    document = json.loads(evidence.resolve('data/analysis/supplementary/inputs/workbuddy_seed_scores.json').read_text())
    manifest = json.loads(evidence.resolve('data/workbuddy_office/seed_grading/manifest.json').read_text())
    records = {(r['task'], r['seed']): r for r in manifest['records']}
    expected = {(task, seed) for task in task_ids for seed in (1, 2, 3)}
    rows = document['scores']
    if len(rows) != 150 or len(records) != 150 or set(records) != expected or document['actor_calls'] != 0:
        raise ValueError('Incomplete supplementary seed pool')
    seen = set()
    for row in rows:
        key = row['task'], row['seed']
        if key in seen or key not in expected:
            raise ValueError('Duplicate or unknown supplementary seed')
        seen.add(key)
        receipt_path = evidence.resolve(row['receipt'])
        if evidence.original_hash(receipt_path) != row['receipt_sha256']:
            raise ValueError('Seed receipt identity changed')
        receipt = json.loads(receipt_path.read_text())
        if (receipt['task'], receipt['seed']) != key or receipt['candidate_id'] != row['candidate_id'] or row['candidate_id'] != records[key]['candidate_id']:
            raise ValueError('Seed candidate identity changed')
        if row['status'] != 'scored' or receipt['status'] != 'scored' or receipt['errors'] or receipt['actor_calls'] != 0:
            raise ValueError('Invalid supplementary seed measurement')
        score = row['score']
        if not isinstance(score, (float, int)) or not math.isfinite(score) or not 0 <= score <= 100:
            raise ValueError('Invalid supplementary seed score')
        if abs(score - receipt['score']) > 1e-7 or abs(score - 100 * receipt['raw_reward']['overall']) > 1e-7:
            raise ValueError('Supplementary seed score differs from native receipt')
        for name, expected_hash in records[key]['source_files_sha256'].items():
            if evidence.original_hash(evidence.resolve(name)) != expected_hash:
                raise ValueError('Seed grading source binding changed: ' + name)
    return {'tasks': len(task_ids), 'valid_scores': len(rows), 'actor_calls': 0,
            'condition': 'later_native_grading_of_frozen_seed_deliveries'}


def verify(data_root: Path, reports: Path | None = None, full=True):
    evidence = Evidence(data_root)
    checked = evidence.verify() if full else 0
    derived = json.loads(Path(__file__).with_name('derived_files.json').read_text())
    for entry in derived:
        path = evidence.resolve(entry['path'])
        if not path.is_file() or path.stat().st_size != entry['bytes'] or sha256(path) != entry['sha256']:
            raise ValueError('Generated metadata missing or changed: ' + entry['path'])
    counts = {}
    for name in EXPERIMENTS:
        document = json.loads((data_root / name / 'experiment.json').read_text())
        counts[name] = validate_rows(document)
        for row in document['rounds']:
            for arm in 'ABC':
                ref = row['evidence'][arm]
                path = evidence.resolve(ref['receipt'])
                if evidence.original_hash(path) != ref['score_sha256']:
                    raise ValueError('Original native receipt binding changed')
                receipt = json.loads(path.read_text())
                if name == 'claw_eval':
                    if receipt['status'] != 'graded':
                        raise ValueError('Invalid native grade')
                    score = receipt['task_score']
                else:
                    score = receipt.get('overall', receipt.get('rewards', {}).get('overall'))
                if score is None or abs(float(score) * 100 - row[arm]) > 1e-7:
                    raise ValueError('Report differs from accepted native score')
    links = 0
    supplementary = {}
    if (data_root/'workbuddy_office/seed_grading/manifest.json').exists():
        task_ids = json.loads((data_root/'workbuddy_office/experiment.json').read_text())['task_ids']
        supplementary['workbuddy_seed_grading'] = verify_seed_grades(evidence, task_ids)
    if reports and reports.exists():
        for path in reports.rglob('*.html'):
            parser = Links(); parser.feed(path.read_text())
            for link in parser.links:
                parsed = urlparse(link)
                if parsed.scheme or parsed.netloc or not parsed.path:
                    continue
                target = (path.parent / unquote(parsed.path)).resolve()
                if not target.is_relative_to(data_root.parent.resolve()) or not target.exists():
                    raise ValueError('Broken or escaping report link: ' + str(path) + ': ' + link)
                links += 1
    return {'status': 'passed', 'files_verified': checked, 'derived_files_verified': len(derived), 'counts': counts, 'report_links_verified': links,
            'supplementary': supplementary, 'model_calls': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument('--reports', type=Path, default=Path('reports'))
    parser.add_argument('--scores-only', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = verify(args.data_root.resolve(), args.reports.resolve(), not args.scores_only)
    if args.output:
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
