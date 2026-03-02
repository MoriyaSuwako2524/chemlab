#!/bin/bash
#SBATCH -J "amber"
#SBATCH -p batch
#SBATCH --time=2-00:30:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=16


out="./npys/"
mkdir -p $out
chemlab qmhub qmmm_training_set_data \
--qmmmpath="/scratch/moriya/md/chorismate_mutase/calc3/windows/" \
--outpath=$out \
--windows=40 \
--cache_path="/scratch/moriya/cache/calc3/" \
--nframes=500 \
--method="gas"
