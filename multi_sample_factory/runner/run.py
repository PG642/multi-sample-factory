import importlib
import sys
import argparse

from multi_sample_factory.algorithms.utils.algo_utils import ExperimentStatus
from multi_sample_factory.runner.run_slurm import add_slurm_args
from multi_sample_factory.utils.utils import log


def runner_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', default='./train_dir', type=str, help='Directory for sub-experiments')
    parser.add_argument('--run', default=None, type=str,
                        help='Name of the python module that describes the run, e.g. sample_factory.runner.runs.doom_battle_hybrid')
    parser.add_argument('--runner', default='processes', choices=['processes', 'slurm'])
    parser.add_argument('--pause_between', default=10, type=int, help='Pause in seconds between processes')
    parser.add_argument('--num_gpus', default=1, type=int, help='How many GPUs to use')
    parser.add_argument('--experiments_per_gpu', default=-1, type=int, help='How many experiments can we squeeze on a single GPU (-1 for not altering CUDA_VISIBLE_DEVICES at all)')
    parser.add_argument('--max_parallel', default=4, type=int, help='Maximum simultaneous experiments')
    parser.add_argument('--experiment_suffix', default='', type=str, help='Append this to the name of the experiment dir')

    parser = add_slurm_args(parser)

    return parser


def parse_args():
    args = runner_argparser().parse_args(sys.argv[1:])
    return args


def main():
    args = parse_args()

    try:
        # assuming we're given the full name of the module
        run_module = importlib.import_module(f'{args.run}')
    except ImportError:
        try:
            run_module = importlib.import_module(f'multi_sample_factory.runner.runs.{args.run}')
        except ImportError:
            log.error('Could not import the run module')
            return ExperimentStatus.FAILURE

    run_description = run_module.RUN_DESCRIPTION
    run_description.experiment_suffix = args.experiment_suffix

    if args.runner == 'processes':
        from multi_sample_factory.runner.run_processes import run
        run(run_description, args)
    elif args.runner == 'slurm':
        from multi_sample_factory.runner.run_slurm import run_slurm
        run_slurm(run_description, args)

    return ExperimentStatus.SUCCESS


if __name__ == '__main__':
    sys.exit(main())
