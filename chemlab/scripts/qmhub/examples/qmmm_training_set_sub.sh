#!/bin/bash
#SBATCH -J "amber"
#SBATCH -p batch
#SBATCH --time=2-00:30:00
#SBATCH --nodes=2
#SBATCH --tasks-per-node=32
#SBATCH -o ./logs/OUTPUT_%x_%j.log
#SBATCH -e ./logs/ERRORS_%x_%j.log

win=$1
qcscratch="/scratch/moriya/cache/calc3/"

chemlab qmhub qmmm_job_manager \
--qmmmpath="/scratch/moriya/md/chorismate_mutase/calc3/windows/" \
--window=$win \
--ncore=64 \
--njob=16 \
--QCSCRATCH=$qcscratch

