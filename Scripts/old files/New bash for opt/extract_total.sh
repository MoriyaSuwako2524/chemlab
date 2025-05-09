#!/bin/bash
source ~/.bashrc
rm -f tem_ene_out.txt
rm -f tem_opt_out.txt

for inf in *_ene.log
do
echo $inf >> tem_ene_out.txt
tac $inf | grep -m 1 "SCF Done" | tr '\n' ' ' >> tem_ene_out.txt
done

for inf in *_opt.log
do
echo $inf >> tem_opt_out.txt
tac $inf | grep -m 1 "Thermal correction to Gibbs Free Energy=" | tr '\n' ' ' >> tem_opt_out.txt
done


python extract.py

echo "extract opt and ene finish"
