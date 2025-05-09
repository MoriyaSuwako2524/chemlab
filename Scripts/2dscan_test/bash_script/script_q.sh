#!/bin/bash
#SBATCH -J "test-ZX"
#SBATCH -p batch
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=4
#SBATCH --tasks-per-node=4
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
row=$3
current_col=$4


former_col=$((current_col -1))
echo "script_q working"
if [ ${current_col} -eq 1 ]; then
input_file="${path}${filename}.inp"
qchem -nt 16 "$input_file" "${input_file%}.out"
echo "col1 submitted"

else
echo "submitting col${current_col}"
eval $(python locate_row_col.py $row $former_col $path)
eval $(python trans_out_into_inp.py "$path" "${filename}.inp" 0.1)
new_input_file="${path}${new_filename}"
echo "name of new input file $new_input_file"
qchem -nt 4 "$new_input_file" "${new_input_file%}.out"
echo "col${current_col} submitted"
fi