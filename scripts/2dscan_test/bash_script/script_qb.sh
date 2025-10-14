#!/bin/bash
#SBATCH -J "ID377-ZX"
#SBATCH -p batch
#SBATCH --time=3-00:00:00
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
path=$2



echo "script_q working"
input_file="${path}${filename}.inp"
qchem -nt 8 "$input_file" "${input_file%}.out"

