from multi_sample_factory.slurm.grids.grid import Grid

_setup = """#!/bin/bash -l
#SBATCH -c 64
#SBATCH --mem=460G
#SBATCH --gres=gpu:2
#SBATCH --partition={0}
#SBATCH --time={1}
#SBATCH -N {2}
#SBATCH --job-name={3}
#SBATCH --output={6}
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


_params = {'N': [1,3]}
_name = 'grid_striker_performance'
_base_parameters = '--hidden_size=512 --num_envs_per_worker=8 --batch_size=8192 --experiment_summaries_interval=5'
_env = 'unity_striker'

GRID = Grid(name=_name, params=_params, base_parameters=_base_parameters, env=_env, setup=_setup)