#!/bin/bash
#SBATCH -J "Fe-mecp"
#SBATCH -p batch
#SBATCH --time=5-00:00:00
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

mkdir -p ${filename%.inp}

run_mecp --file $filename \
	  --path "/scratch/moriya/calculation/Fe/path/mecp/" \
	  --jobtype mecp \
         --out /scratch/moriya/calculation/Fe/path/mecp/${filename%.inp}/ \
         --spin1 5 --spin2 3 --nthreads 32 --max-steps 80


