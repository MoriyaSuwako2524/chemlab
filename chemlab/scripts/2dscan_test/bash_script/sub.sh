#!/bin/bash
#SBATCH -J "ts-ZX"
#SBATCH -p batch
#SBATCH --time=24:00:00
#SBATCH --ntasks=8
#SBATCH --tasks-per-node=8
#SBATCH --nodes=1
#SBATCH -o ./logs/OUTPUT_%J.log
#SBATCH -e ./logs/ERRORS_%J.log

export QC=/scratch/chance/update_restraints/rest_opt
export QCAUX=/home/chance/software/qcaux
source $QC/bin/qchem.setup.sh
export QCSCRATCH=/scratch/$USER

module purge
module load intel/2021.2.0

filename=$1



qchem -nt 8 "$filename" "${filename%}.out"


