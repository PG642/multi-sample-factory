import argparse
import csv
import importlib
import itertools
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Tuple

from multi_sample_factory.algorithms.utils.algo_utils import ExperimentStatus
from multi_sample_factory.runner.run import runner_argparser
from multi_sample_factory.slurm.grids.grid import Grid
from multi_sample_factory.utils.utils import log, str2bool


def runner_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', default='/work/grudelpg/Trainingsergebnisse', type=str,
                        help='Directory for sub-experiments')
    parser.add_argument('--grid', default=None, type=str,
                        help='Name of the python module that describes the run, e.g. sample_factory.runner.runs.doom_battle_hybrid')
    parser.add_argument('--info_file',
                        default='info',
                        type=str,
                        help='name of the CSV file containing the parameter.')
    parser.add_argument('--experiment_suffix', default='', type=str,
                        help='Append this to the name of the experiment dir')
    parser.add_argument('--time_limit',
                        default=10,
                        type=int,
                        help='The training time for each job in minutes.')
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
                        default=False,
                        type=str2bool,
                        help="If True, the bash files are only created, but not scheduled.")
    return parser


def parse_time_limit(time_limit: int) -> Tuple[str, str]:
    if time_limit <= 120:
        partition = 'short'
    elif time_limit <= 480:
        partition = 'med'
    elif time_limit <= 2880:
        partition = 'long'
    else:
        raise ValueError("Jobs longer than 48 hours can not be run on GPU nodes.")

    time_str = time.strftime('%H:%M:%S', time.gmtime(time_limit * 60))
    return time_str, partition


def main():
    args = runner_argparser().parse_args(sys.argv[1:])

    try:
        # assuming we're given the full name of the module
        run_module = importlib.import_module(f'{args.grid}')
    except ImportError:
        try:
            run_module = importlib.import_module(f'multi_sample_factory.slurm.grids.{args.grid}')
        except ImportError:
            log.error('Could not import the run module')
            return ExperimentStatus.FAILURE

    grid: Grid = run_module.GRID

    params = grid.params.values()
    keys = list(grid.params.keys())
    info = []

    # Create directory for jobs
    directory = os.path.join(args.destination, grid.name)
    try:
        shutil.rmtree(directory)
    except OSError as e:
        print("Path did not exist previously, creating a new one.")
    Path(directory).mkdir(parents=True, exist_ok=True)

    for i, combination in enumerate(itertools.product(*params)):
        job_name = "{0}_{1:03d}".format(grid.name, i)
        info.append([job_name] + list(combination))
        time_limit, partition = parse_time_limit(args.time_limit)
        NUM_NODES_PARAM = 'N'
        if NUM_NODES_PARAM in keys:
            n = combination[keys.index(NUM_NODES_PARAM)]
        else:
            n = 1
        for repetition in range(args.repeat):
            full_job_name = job_name + "_{0:03d}".format(repetition)
            bash_script = grid.setup.format(partition, time_limit, n, full_job_name, grid.env, args.msf_dir, grid.name)
            if grid.base_parameters != "":
                bash_script = bash_script + " " + grid.base_parameters
            for parameter, value in zip(keys, combination):
                if parameter == NUM_NODES_PARAM:
                    continue
                else:
                    bash_script = bash_script + " --{0}={1}".format(parameter, value)

            # Write the file
            file_path = os.path.join(directory, '{0}.sh'.format(full_job_name))
            with open(file_path, 'w') as file:
                file.write(bash_script)

            if not args.only_files:
                import subprocess
                bashCommand = "sbatch {0}".format(file_path)
                process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
                output, error = process.communicate()

    with open(os.path.join(directory, '{0}.csv'.format(args.info_file)), 'w', newline='') as csv_file:
        fieldnames = ['job_name'] + keys
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, delimiter=';')
        writer.writeheader()
        for line in info:
            writer.writerow(dict(zip(fieldnames, line)))


if __name__ == '__main__':
    sys.exit(main())
