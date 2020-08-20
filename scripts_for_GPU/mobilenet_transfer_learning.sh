#!/bin/bash
# request resources:
#PBS -j oe
#PBS -l walltime=02:00:00
#PBS -l select=2:ngpus=2:mem=24gb

module add lang/python/anaconda/3.8-2020.07
module load lang/python/anaconda/3.8-2020.07

module add load lang/cuda
module load lang/cuda

python
import tensorflow

# on compute node, change directory to 'submission directory':
cd $PBS_O_WORKDIR

SECONDS=0
# run program
python ./scripts_for_GPU/mobilenet_transfer_learning.py

duration=$SECONDS
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."
