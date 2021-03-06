#!/bin/bash

FILE=/work/${USER}/torch-1.8.0a0+unknown-cp39-cp39-linux_x86_64.whl
if [ -f "$FILE" ]; then
    echo "$FILE exists!"
else 
    echo "$FILE does not exist. Please download the torch wheel file!"
    exit
fi


wget https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh -P /work/${USER}/
bash /work/${USER}/Anaconda3-2021.05-Linux-x86_64.sh -b -u -p /work/${USER}/anaconda3
source /work/${USER}/anaconda3/bin/activate
conda init
cd /work/${USER}/anaconda3
source ~/.bashrc
cd ..
rm Anaconda3-2021.05-Linux-x86_64.sh


module purge
module load cmake/3.8.2
module load gcc/9.2.0
module load git/2.30.2

cd /work/${USER}/multi-sample-factory

conda env create -f pip-environment-simple.yml
conda activate multi-sample-factory
cd ..

#module load nvidia/cuda/11.1.1

conda install astunparse numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses
conda install -c pytorch magma-cuda111

pip install $FILE
echo "export LD_LIBRARY_PATH=/work/${USER}/anaconda3/envs/multi-sample-factory/lib" >> ~/.bashrc
