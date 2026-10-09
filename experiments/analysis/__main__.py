"""Recompute the frozen score-gap and midpoint analyses without model calls."""
import argparse
import json
from pathlib import Path

from .rewardpack import inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--analysis', choices=('score_gap', 'midpoint', 'joint'), required=True)
    parser.add_argument('--bootstrap-iterations', type=int, default=20000)
    args = parser.parse_args()
    inputs.configure(args.data_root)
    from .rewardpack import subset_analysis, midpoint, joint
    # Modules also expose their original pure functions for synthetic tests.
    midpoint.REPO = inputs.REPO
    if args.analysis == 'score_gap':
        result = subset_analysis.run(inputs.OUTPUT / 'joint_original', inputs.OUTPUT / 'joint_repaired',
                                     args.output, iterations=args.bootstrap_iterations)
    elif args.analysis == 'midpoint':
        result = midpoint.run(inputs.OUTPUT / 'joint_original', inputs.OUTPUT / 'joint_repaired',
                              args.output, reference_summary=inputs.OUTPUT / 'score_gap/summary.json',
                              iterations=args.bootstrap_iterations)
    else:
        result = joint.run(inputs.OUTPUT, args.output, iterations=args.bootstrap_iterations)
    print(json.dumps({'output': str(args.output), 'model_calls': 0, 'analysis': args.analysis}))


if __name__ == '__main__':
    main()
