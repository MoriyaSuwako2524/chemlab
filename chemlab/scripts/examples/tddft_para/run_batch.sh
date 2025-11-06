#!/bin/bash
source /home/van/modulefiles/modules.sh
cd raw_data
jobname="qchem_batch"
cores=4      # ?? qchem ?? 4 ?
jobs=8       # ??????? = 32/4
total=3000   # ??????
per_batch=200
batches=$((total / per_batch))
prefix="train"
mkdir -p logs
# ?? runqchem.sh:?????????
cat > runqchem.sh <<EOF
#!/bin/bash
date
source /home/van/modulefiles/modules.sh
module load parallel
export QC=/scratch/moriya/software/qchem_default
export QCAUX=/home/chance/software/qcaux
source /scratch/moriya/software/qchem_default/bin/qchem.setup.sh
export QCSCRATCH=/scratch/moriya/cache

module purge
module load intel/2021.2.0

export MKL_NUM_THREADS=${cores}
export OMP_NUM_THREADS=${cores}

file=\$1
out=\${file%.inp}.out

# ?????? 4 ?
qchem -nt ${cores} \$file \$out
date
EOF
chmod +x runqchem.sh

# ?????
for batch in $(seq 0 $((batches-1))); do
    start=$((batch * per_batch+1))
    end=$((start + per_batch-1))

    sbatch <<_EOF
#!/bin/bash
#SBATCH -p batch
#SBATCH -t 5-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --output=./logs/%j.out
#SBATCH --error=./logs/%j.err
#SBATCH --job-name=${jobname}

date
source /home/van/modulefiles/modules.sh
module load parallel

# ? parallel ??? ${jobs} ? qchem ??
parallel --jobs ${jobs} bash runqchem.sh {} ::: \$(seq -f "${prefix}_%04g.inp" ${start} ${end})

date
_EOF

done
