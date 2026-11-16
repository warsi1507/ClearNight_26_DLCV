"""Train several configs back to back, then score each best checkpoint on the test set.

    python run_comparison.py --configs official baseline dwt --epochs 40 --samples_per_branch 1500

Accepts every train.py option. Results go to results/<config>_seed<seed>.json
and results/comparison.md.
"""
import subprocess
import sys

from configs import CONFIGS
from train import get_parser, train


def main():
    parser = get_parser()
    parser.description = __doc__
    parser.add_argument('--configs', nargs='+', choices=list(CONFIGS), default=['official', 'baseline', 'dwt'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[20])
    args = parser.parse_args()
    if args.resume:
        parser.error('resume single runs with train.py --resume')

    results = []
    for seed in args.seeds:
        for name in args.configs:
            args.config, args.seed = name, seed
            summary = train(args)
            out_json = f'results/{name}_seed{seed}.json'
            subprocess.run([sys.executable, 'evaluate.py', '--checkpoint', summary['checkpoint'],
                            '--data_root', args.data_root, '--protocol', args.protocol,
                            '--out_json', out_json], check=True)
            results.append(out_json)
    subprocess.run([sys.executable, 'compare.py', *results, '--out', 'results/comparison.md'], check=True)


if __name__ == '__main__':
    main()
