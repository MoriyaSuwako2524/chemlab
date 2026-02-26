#!/bin/bash
#SBATCH -J "amber"
#SBATCH -p express
#SBATCH --time=00:60:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=16


chemlab ml_data order_test \
--npy_path="/scratch/moriya/md/chorismate_mutase/calc3/npys/" \
