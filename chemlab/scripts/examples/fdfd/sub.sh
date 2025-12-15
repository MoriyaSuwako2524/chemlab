#!/bin/bash
#SBATCH -J "FD"
#SBATCH -p batch
#SBATCH --time=1-00:60:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=32
#SBATCH -o ./logs/OUTPUT_%J.log
#SBATCH -e ./logs/ERRORS_%J.log

export QC=/scratch/moriya/software/soc
export QCAUX=/home/chance/software/qcaux
source $QC/bin/qchem.setup.sh
export QCSCRATCH=/scratch/$USER/cache

module purge
module load intel/2021.2.0

filename=$1

chemlab qchem fpfd --project="soc" --out="./out2/"
