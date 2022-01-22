from dataclasses import dataclass
from typing import Dict, List
import re as regex


class Grid:

    DEFAULT_SETUP = """#!/bin/bash -l
#SBATCH -C cgpu01
#SBATCH -c 20
#SBATCH --mem=60G
#SBATCH --gres=gpu:2
#SBATCH --partition={0}
#SBATCH --time={1}
#SBATCH -N {2}
#SBATCH --job-name={3}
#SBATCH --output=/work/grudelpg/logs/{3}.log
#SBATCH --signal=B:SIGQUIT@120
#----------------------------------------------------------------------------------------
#Variables
EXEC_NAME="{4}"
PY_ENV_NAME="multi-sample-factory"
EXECUTABLES_DIR_NAME="executables"
EXPERIMENT="{3}"
#----------------------------------------------------------------------------------------
#Prepare python environment
module purge
module load nvidia/cuda/11.1.1
module load gcc/9.2.0
module load git/2.30.2
conda activate /work/grudelpg/envs/multi-sample-factory-env
#----------------------------------------------------------------------------------------
#Unzip executable into job folder
MYHOSTLIST=$( srun hostname | sort | uniq -c | awk '{{print $2 "*" $1}}' | paste -sd, )
export MYHOSTLIST=$MYHOSTLIST
ulimit -u 131072
#----------------------------------------------------------------------------------------
#Start learning
cd {5}
srun -X --propagate=NPROC  python -m multi_sample_factory.algorithms.appo.train_appo --algo=APPO"""
    def __init__(self, name: str, params: Dict[str, List], base_parameters: str, env: str, setup: str = DEFAULT_SETUP):

        sub_strings = base_parameters.split('--')[1:]
        for sub_string in sub_strings:
            param = sub_string.split("=")[0]
            if param in params:
                raise ValueError("The parameter {0} is inlcuded in the grid search, but is also set in the base parameters.".format(param))
        self.name = name
        self.params = params
        self.base_parameters = base_parameters
        self.env = env
        self.setup = setup



