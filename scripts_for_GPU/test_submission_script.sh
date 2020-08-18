#!/bin/bash
# request resources:
#PBS -l nodes=1:ppn=1
#PBS -l walltime=02:00:00

module add lang/python/anaconda/3.7-2020.02-tensorflow-2.1.0

module load lang/python/anaconda/3.7-2020.02-tensorflow-2.1.0

# on compute node, change directory to 'submission directory':
cd $PBS_O_WORKDIR
# run program
python ./scripts_for_GPU/test_training_script.py
