#!/bin/bash
#SBATCH -J "phbdi_tddft"
#SBATCH -p express
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=16
#SBATCH -o ./logs/OUTPUT_%J.log
#SBATCH -e ./logs/ERRORS_%J.log


mkdir logs
out="/raw_data/"
mkdir ".$out"
prepare_tddft_inp --path="/scratch/moriya/md/h2obpc/" \
--file="namd_single_nvt.out,namd_single_nvt_2.out" \
--out="$out" \
--charge="0" \
--spin="1"



