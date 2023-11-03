#!/bin/bash
source ~/.bashrc

chkpath=/home/zxwei/cal/chk
mem=16
nproc=16

# Never edit code below unless you know what you are doing

#generate tem1 input
generate_tem1_input(){
echo "now begin to echo gaussian input opt files"
for inf in *.gjf
do
python3 judge.py $inf
jobtype=$?
echo $jobtype
if [ "$jobtype" == "0" ];
then
jobname="tem1"
python3 modify.py $inf $jobname
fi
done
echo "all .gjf files have generated ."
}


#run tem1 jobs
run_tem1_jobs(){
echo " Now begin to run tem1 jobs"

for optinf in *.gjf
do
python3 judge.py $optinf
jobtype=$?
if [ "$jobtype" == "1" ];
then
echo "running $optinf"
g16 $optinf
wait
echo "$optinf has finished"
fi
done
rm -f *_tem1.gjf
echo "all tem1 jobs have done."
}


#handle_tem1out
handle_tem1out(){
icc=0
nfile=`ls *.log|wc -l`
for inf in *_tem1.log
do
((icc++))
echo Converting ${inf} to ${inf//log/gjf} ... \($icc of $nfile\)
Multiwfn ${inf} << EOF > /dev/null
100
2
10
${inf//log/gjf}-out
0
q
EOF
done
}

generate_ts_input(){
icc=0
nfile=`ls *.log|wc -l`
for inf in *.log
do
((icc++))
echo Converting ${inf} to ${inf//log/gjf} ... \($icc of $nfile\)
Multiwfn ${inf} << EOF > /dev/null
100
2
10
${inf//log/gjf}-out
0
q
EOF
done
}


#generate tem2 input

generate_tem2_input(){
echo "now begin to generate tem2 files"
for inf in *_tem1.gjf-out
do
python3 judge.py $inf
jobtype=$?
echo $jobtype
if [ "$jobtype" == "1" ];
then
jobname="tem2"
python3 modify.py $inf $jobname
fi
done
echo "all .gjf files have generated ."
}


#run tem2 jobs
run_tem2_jobs(){
echo " Now begin to run tem2 jobs"

for optinf in *.gjf
do
python3 judge.py $optinf
jobtype=$?
if [ "$jobtype" == "2" ];
then
echo "running $optinf"
g16 $optinf
wait
echo "$optinf has finished"
fi
done
rm -f *_tem2.gjf
echo "all tem2 jobs have done."
}

#generate orca input
generate_orca_input(){
echo "now begin to generate orca files"
for inf in *.gjf-out
do
jobname="orca"
python3 modify_orca.py $inf $jobname
done
echo "all orca inp files have generated ."
}


#run oraca jobs
run_orca_jobs(){
echo " Now begin to run orca jobs"

for optinf in *.inp
do
echo "running $optinf"
/home/zxwei/data/orca504/orca $optinf
wait
echo "$optinf has finished"
done
#rm -f *_orca.inp
echo "all orca jobs have done."
}























run_confi_search_jobs(){
echo " Now begin to run configuration search jobs, these jobs will takes large amount of time!"
bash gjf2xyz.sh
for csinf in *cs.xyz
do
python3 judge.py $csinf
jobtype=$?
if [ "$jobtype" == "5" ];
then
echo "running $csinf configuration search"
bash xtb_configuration_search.sh << EOF > /dev/null
$csinf
EOF
wait
echo "$csinf configuration search have done"
fi
done
echo "all configuration search jobs have done."
}


# All functions list generate_opt_input() ,  run_opt_jobs() , generate_ene_input() , run_ene_jobs(), transform_chk_to_fchk() bash extract_total.sh(this is a bash file))
#,generate_frez_input() , run_frez_jobs() , run_ts_jobs() ,  generate_ts_input() , generate_ts_input_without_frez()
#run_confi_search_jobs(),generate_tdsp_input(), run_tdsp_jobs()


#handle_tem1out
#generate_orca_input
#run_orca_jobs


#generate_tem1_input
#run_tem1_jobs
#handle_tem1out
#generate_tem2_input
#run_tem2_jobs

run_confi_search_jobs

echo "all jobs have done"





