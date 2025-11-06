#!/bin/bash

mkdir -p logs

for file in *.inp
do
echo $file
sbatch sub.sh $file
done
