#!/bin/bash
#SBATCH -J "amber"
#SBATCH -p express
#SBATCH --time=00:60:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=16

num=$1
chemlab ml_data soap \
--npy_path="/scratch/moriya/md/chorismate_mutase/calc3/npys/" \
--n_select=$1 \
--test_set="test_split.npz" \
--method="random" \
