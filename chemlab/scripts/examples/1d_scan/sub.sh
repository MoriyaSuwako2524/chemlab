#!/bin/bash
#SBATCH -J qchem-1d-scan
#SBATCH -N 2
#SBATCH -p batch
#SBATCH --ntasks-per-node=2        # 每节点开 2 个 srun
#SBATCH --cpus-per-task=16          # 每个 srun 16 核
#SBATCH --time=5-00:00:00
#SBATCH --exclusive
#SBATCH -o ./logs/OUTPUT_%J.log
#SBATCH -e ./logs/ERRORS_%J.log

export OMP_NUM_THREADS=8
module purge
module load impi/2021.2.0
module load intel/2021.2.0

filename=$1
mkdir -p ${filename%.inp}
chemlab scan 1d_scan \
    --path "./" \
    --out "./${filename%.inp}/" \
    --prefix "${filename%.inp}" \
    --ref "$1" \
    --row_max 40 \
    --row_start 1.25 \
    --row_distance 0.05 \
    --ncore 16 \
    --njob 4 \
    --scan_limit_init 10 \
    --scan_limit_progress 10 \
    --poll_interval 60 \
    --launcher "srun" \
    --result_saver_prefix "${filename%.inp}"
