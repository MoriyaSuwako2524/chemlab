#!/bin/bash
#SBATCH -J "amber"
#SBATCH -p express
#SBATCH --time=00:60:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=16


out="./windows/"
mkdir -p $out
chemlab qmhub qmmm_training_set_dft \
--qmmmpath="/scratch/samitha/chorismate_mutase/CHO_d3/" \
--outpath=$out \
--windows=40 \
--method="gas"
