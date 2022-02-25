"""
Execute with: python -m multi_sample_factory.slurm.run --grid=performance_grid --time_limit=30

To cancel all jobs (Don't forget to replace the name):
for entry in `squeue -u smkoramt | awk '{print $1}' | tail -n +2`;
do
	scancel ${entry}
done
"""
import argparse
import csv
import importlib
import itertools
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Tuple, Optional

from multi_sample_factory.algorithms.utils.algo_utils import ExperimentStatus
from multi_sample_factory.slurm.grids.grid import Grid
from multi_sample_factory.utils.utils import str2bool


def runner_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', default='/work/grudelpg/Trainingsergebnisse', type=str,
                        help='Directory for sub-experiments')
    parser.add_argument('--grid', default=None, type=str,
                        help='Name of the python module that describes the run, e.g. sample_factory.runner.runs.doom_battle_hybrid')
    parser.add_argument('--info_file',
                        default='info',
                        type=str,
                        help='Name of the CSV file containing the parameter.')
    parser.add_argument('--time_limit',
                        default=10,
                        type=int,
                        help='The training time for each job in minutes.')
    parser.add_argument('--time_buffer',
                        default=5,
                        type=int,
                        help='The difference between the slurm job time limit and the training time for MSF.')
    parser.add_argument('--repeat',
                        default=3,
                        type=int,
                        help='How many times each job should be repeated.')
    parser.add_argument('--msf_dir',
                        default='/work/grudelpg/multi-sample-factory/',
                        type=str,
                        help='Absolute path to the multi sample factory repo.')
    parser.add_argument('--destination',
                        default='/work/grudelpg/jobs/',
                        type=str,
                        help="Destination of the jobs folder.")
    parser.add_argument('--only_files',
                        default=True,
                        type=str2bool,
                        help="If True, the bash files are only created, but not scheduled. Currently we get an error when executing the jobs using sbatch in this script. Use the run_all.sh in the job folder to run all scripts.")
    parser.add_argument('--log_dir',
                        default='/work/grudelpg/logs',
                        type=str,
                        help="Directory for logs files. This path will be extended with the name of the grid.")
    parser.add_argument('--slurm_partition',
                        default=None,
                        type=str,
                        help="Optional custom slurm partition. If not specified, the partition is evaluated based on the time_limit parameter, resulting in either short, med or long.")
    return parser


def parse_time_limit(time_limit: int, custom_partition: Optional[str]) -> Tuple[str, str]:
    if time_limit <= 120:
        partition = 'short'
    elif time_limit <= 480:
        partition = 'med'
    elif time_limit <= 2880:
        partition = 'long'
    else:
        if custom_partition is None:
            raise ValueError("Jobs longer than 48 hours can not be run on GPU nodes.")

    if custom_partition is not None:
        partition = custom_partition

    time_str = time.strftime('%H:%M:%S', time.gmtime(time_limit * 60))
    return time_str, partition


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")


def main():
    args = runner_argparser().parse_args(sys.argv[1:])

    # Import the module
    try:
        # assuming we're given the full name of the module
        run_module = importlib.import_module(f'{args.grid}')
    except ImportError:
        try:
            run_module = importlib.import_module(f'multi_sample_factory.slurm.grids.{args.grid}')
        except ImportError:
            print('Could not import the run module')
            return ExperimentStatus.FAILURE
    grid: Grid = run_module.GRID

    params = grid.params.values()
    keys = list(grid.params.keys())
    info = []

    # Remove old log and job dirs
    jobs_directory = os.path.join(args.destination, grid.name)
    logs_directory = os.path.join(args.log_dir, grid.name)
    if os.path.isdir(jobs_directory):
        if query_yes_no('A jobs directory for this grid already exists. Do you want to delete it? If you answer with no, this script will abord.'):
            shutil.rmtree(jobs_directory)
        else:
            return ExperimentStatus.INTERRUPTED
    if os.path.isdir(logs_directory):
        if query_yes_no('A log directory for this grid already exists. Do you want to delete it? If you answer with no, this script will abord.'):
            shutil.rmtree(logs_directory)
        else:
            return ExperimentStatus.INTERRUPTED
    Path(jobs_directory).mkdir(parents=True, exist_ok=True)
    Path(logs_directory).mkdir(parents=True, exist_ok=True)

    for i, combination in enumerate(itertools.product(*params)):
        job_name = "{0}_{1:03d}".format(grid.name, i)
        info.append([job_name] + list(combination))
        time_limit_str, partition = parse_time_limit(args.time_limit, args.slurm_partition)
        NUM_NODES_PARAM = 'N'
        if NUM_NODES_PARAM in keys:
            n = combination[keys.index(NUM_NODES_PARAM)]
        else:
            n = 1
        for repetition in range(args.repeat):
            full_job_name = job_name + "_{0:03d}".format(repetition)
            bash_script = grid.setup.format(partition, time_limit_str, n, full_job_name, grid.env, args.msf_dir,
                                            os.path.join(logs_directory, full_job_name) + '.log')
            if grid.base_parameters != "":
                bash_script = bash_script + " " + grid.base_parameters
            for parameter, value in zip(keys, combination):
                if parameter == NUM_NODES_PARAM:
                    continue
                else:
                    bash_script = bash_script + " --{0}={1}".format(parameter, value)
            # Add experiment and env
            bash_script = bash_script + " --env={0} --experiment={1} --train_for_seconds={2} --train_dir={3}".format(
                grid.env, full_job_name, (args.time_limit - args.time_buffer) * 60, os.path.join(args.train_dir, grid.name))

            # Write the file
            file_path = os.path.join(jobs_directory, '{0}.sh'.format(full_job_name))
            with open(file_path, 'w') as file:
                file.write(bash_script)

            if not args.only_files:
                bashCommand = "sbatch {0}".format(file_path)
                os.system(bashCommand)

    with open(os.path.join(jobs_directory, '{0}.csv'.format(args.info_file)), 'w', newline='') as csv_file:
        fieldnames = ['job_name'] + keys
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, delimiter=';')
        writer.writeheader()
        for line in info:
            writer.writerow(dict(zip(fieldnames, line)))

    if args.only_files:
        with open(os.path.join(jobs_directory, 'run_all.sh'), 'w') as run_all_file:
            run_all_file.write("#!/bin/sh\nfor entry in ./{0}*;\ndo\n   sbatch ${{entry}}\ndone".format(grid.name))


if __name__ == '__main__':
    sys.exit(main())
