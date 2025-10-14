#!/bin/bash

path="./ID377/"
eval  $(python row_col_max.py $path)

# 使用变量
echo "最大行数: $max_row"
echo "最大列数: $max_col"

# 第一阶段：提交第一列（col=1）的所有行任务，并记录作业ID
echo "提交第一列任务..."
declare -a col1_job_ids

for col in $(seq 1 "$max_col"); do
for row in $(seq 6 12); do


if [[ ${row} -eq 6 ]]; then

    eval $(python locate_row_col.py $row $col $path)
    input_file="${path}${filename}.inp"
	echo $input_file
    job_id=$(sbatch --parsable -J "row${row}_col${col}" script_qb.sh "$filename" "$path"  )
sleep 0.1
echo $job_id
    col1_job_ids[$row]=$job_id
    echo "行 ${row} 列 ${col} 任务已提交，作业ID: $job_id"

else

    eval $(python locate_row_col.py $row $col $path)
    input_file="${path}${filename}.inp"
	echo $input_file
    new_job_id=$(sbatch --parsable -J "ID377row${row}_col${col}" --dependency=afterany:"$job_id" script_qb.sh "$filename" "$path" )
sleep 0.1
    job_id=$new_job_id
    col1_job_ids[$row]=$job_id
    echo "行 ${row} 列 ${col} 任务已提交，作业ID: $job_id"
fi

done

done
