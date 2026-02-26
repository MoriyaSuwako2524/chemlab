#!/bin/bash
endwindow=39
startwindow=0
for i in $(seq ${startwindow} ${endwindow}); do
    sbatch qmmm_training_set_sub.sh $i
done
