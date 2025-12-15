#!/bin/bash
#SBATCH -J qchem-1d-scan
#SBATCH -N 1
#SBATCH -p batch
##SBATCH -p express
#SBATCH --ntasks-per-node=4        # 每节点开 2 个 srun 
#SBATCH --cpus-per-task=8          # 每个 srun 16 核
#SBATCH --time=3-00:60:00
#SBATCH --exclusive
#SBATCH -o ./logs/OUTPUT_%J.log
#SBATCH -e ./logs/ERRORS_%J.log

export OMP_NUM_THREADS=8
module purge
module load impi/2021.2.0
module load intel/2021.2.0

filename=$1
mkdir -p ${filename%.inp}
chemlab scan mecp_scan \
    --path "./" \
    --ref_file $filename \
    --out_path "./${filename%.inp}" \
    --prefix ${filename%.inp} \
    --spin1 1 --spin2 3 \
    --restrain_atom_i 4 --restrain_atom_j 3 \
    --distance_start 1.25 \
    --distance_end 3.2 \
    --distance_step 0.05 \
    --mecp_max_steps 200 \
    --njob 4 \
    --nthreads 8 \
    --jobtype "mecp_soc"

