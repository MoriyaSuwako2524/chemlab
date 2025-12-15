#!/bin/bash
#SBATCH -J "mecp"
#SBATCH -p batch
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=16
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

chemlab qchem run_mecp --file $filename \
	  --path "./" \
	  --jobtype mecp \
         --out ./${filename%.inp}/ \
         --spin1 2 --spin2 4 --nthreads 16 --max_steps 160 --step_size 0.5 --max_stepsize 0.01


