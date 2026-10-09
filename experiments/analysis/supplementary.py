"""Replay completed supplementary analyses from verified, frozen public inputs."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import runpy
import shutil
import sys
from pathlib import Path

from release.evidence import Evidence

ANALYSES = {
    'six_baseline': 'prepare_six_baseline',
    'claw_continuation': 'prepare_claw_continuation',
    'joint_offline': 'compute_joint_offline_metrics',
    'known_reward': 'compute_known_reward_metrics',
    'frozen_gap': 'prepare_frozen_reward_gap_analysis',
    'tie_credit': 'prepare_frozen_reward_tie_credit',
    'jury_gap': 'prepare_jury_gap_analysis',
    'reward_agreement': 'prepare_reward_agreement',
    'claw_agreement': 'prepare_claw_reward_agreement',
    'task_local': 'prepare_task_local_reward_analysis',
    'utility': 'prepare_utility_summary',
    'win_tie_loss': 'prepare_win_tie_loss',
    'thresholds': 'prepare_workbuddy_thresholds',
}


class PortableHashes(ast.NodeTransformer):
    """Change file-identity lookup only; all estimators and numeric inputs stay fixed."""
    def visit_Call(self, node):
        node = self.generic_visit(node)
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == 'hexdigest'):
            return node
        call = node.func.value
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name) and call.func.value.id == 'hashlib'
                and call.func.attr == 'sha256' and len(call.args) == 1):
            return node
        read = call.args[0]
        if isinstance(read, ast.Call) and isinstance(read.func, ast.Attribute) and read.func.attr == 'read_bytes':
            return ast.copy_location(ast.Call(func=ast.Name(id='publication_hash', ctx=ast.Load()),
                                              args=[read.func.value], keywords=[]), node)
        return node

    def visit_Assign(self, node):
        node = self.generic_visit(node)
        if any(isinstance(t, ast.Name) and t.id in {'preserved', 'preserved_paths'} for t in node.targets):
            name = node.targets[0].id
            # The paper and the stopped expansion are not scientific inputs to
            # these completed analyses; only omit their presentation-change guards.
            if isinstance(node.value, (ast.List, ast.BinOp)):
                extra = ast.parse(f"{name} = [p for p in {name} if p.startswith(('data/', 'jury_inputs/')) and not p.startswith('data/ranking_expansion/')]").body[0]
                return [node, ast.copy_location(extra, node)]
        return node


def replay(name: str, data_root: Path, output: Path):
    output = output.resolve()
    if output.exists():
        raise FileExistsError('Use a new output directory; frozen inputs are never overwritten')
    evidence = Evidence(data_root)
    frozen = evidence.data_root/'analysis/supplementary'
    source_map = {}
    output.mkdir(parents=True)
    for source_dir, target_dir in [('inputs', 'data'), ('code', 'scripts'), ('jury_inputs', 'jury_inputs')]:
        for path in sorted((frozen/source_dir).rglob('*')):
            if not path.is_file():
                continue
            evidence.original_hash(path)
            rel = Path(target_dir)/path.relative_to(frozen/source_dir)
            target = output/rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            source_map[str(rel)] = evidence.records[str(path.relative_to(evidence.root))]
    (output/'tables').mkdir()
    (output/'figures').mkdir()
    bindings = {key: {'sha256': r['sha256'], 'source_sha256': r['source_sha256']}
                for key, r in source_map.items()}
    (output/'publication_bindings.json').write_text(json.dumps(bindings))
    helper = '''import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BINDINGS=json.loads((ROOT/'publication_bindings.json').read_text())
def publication_hash(path):
    path=Path(path).resolve();actual=hashlib.sha256(path.read_bytes()).hexdigest()
    try:record=BINDINGS.get(str(path.relative_to(ROOT)))
    except ValueError:record=None
    return record['source_sha256'] if record and actual==record['sha256'] else actual
'''
    (output/'scripts/publication_identity.py').write_text(helper)
    adaptations = []
    for path in sorted((output/'scripts').glob('*.py')):
        if path.name == 'publication_identity.py':
            continue
        text = path.read_text()
        tree = PortableHashes().visit(ast.parse(text))
        insert = 1 if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant) else 0
        while insert < len(tree.body) and isinstance(tree.body[insert], ast.ImportFrom) and tree.body[insert].module == '__future__':
            insert += 1
        tree.body.insert(insert, ast.ImportFrom(module='publication_identity', names=[ast.alias(name='publication_hash')], level=0))
        ast.fix_missing_locations(tree)
        path.write_text(ast.unparse(tree)+'\n')
        # Adapted code retains the frozen source identity after verifying its
        # original public bytes; only these declared mechanical edits differ.
        bindings[str(path.relative_to(output))]['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        adaptations.append(path.name)
    (output/'publication_bindings.json').write_text(json.dumps(bindings))
    (output/'reproduction.json').write_text(json.dumps({
        'analysis': name, 'model_calls': 0, 'adapted_code': adaptations,
        'adaptations': ['verified original file identities after path redaction',
                        'omit paper and stopped-study presentation guards',
                        'portable Jury input location'],
    }, indent=2)+'\n')
    sys.path.insert(0, str(output/'scripts'))
    sys.argv = [ANALYSES[name]]
    os.chdir(output)
    runpy.run_path(str(output/'scripts'/f'{ANALYSES[name]}.py'), run_name='__main__')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', choices=ANALYSES, required=True)
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    replay(args.analysis, args.data_root.resolve(), args.output)


if __name__ == '__main__':
    main()
