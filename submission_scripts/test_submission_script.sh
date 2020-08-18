#!/bin/bash
# request resources:
#PBS -l nodes=1:ppn=1
#PBS -l walltime=02:00:00
# on compute node, change directory to 'submission directory':
cd $PBS_O_WORKDIR
# run program, timing it for good measure:
time ../training_scripts/test_training_script.py