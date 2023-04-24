#!/bin/bash
source ~/.bashrc

optsolvent="dichloroethane"
enesolvent="dichloroethane"
chkpath=/data/zxwei
mem=12
nproc=18
OptBasisSet="def2svp"
EneBasisSet="def2tzvp"
theory="UM062x"

# Never edit code below unless you know what you are doing

#generate opt input
generate_opt_input(){
echo "now begin to echo gaussian input opt files"
for inf in *.gjf
do
python3 judge.py $inf
jobtype=$?
echo $jobtype
if [ "$jobtype" == "0" ];
then
jobname="opt"
python3 modify.py $inf $optsolvent $jobname $chkpath $mem $nproc $OptBasisSet $theory
fi
done
echo "all .gjf files have generated ."
}


#run opt jobs
run_opt_jobs(){
echo " Now begin to run opt jobs"

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
rm -f *_opt.gjf
echo "all opt jobs have done."
}


#generate ene input
generate_ene_input(){
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

for outinf in *.gjf-out
do
jobname="ene"
python3 modify.py $outinf $enesolvent $jobname $chkpath $mem $nproc $EneBasisSet $theory
done
rm -f *-out
echo "all opt job have deleted and generated new ene jobs input"
}


#run ene job
run_ene_jobs(){
for eneinf in *.gjf
do
python3 judge.py $eneinf
jobtype=$?
if [ "$jobtype" == "2" ];
then
echo "running $eneinf"
g16 $eneinf
wait
echo "$eneinf has finished"
fi
done
rm -f _ene.gjf
echo "all ene jobs have done"
}
echo "all typical jobs have done . Please check the results"

transform_chk_to_fchk(){
for inf in *.gjf
do
python3 judge.py $inf
jobtype=$?
echo $jobtype
if [ "$jobtype" == "0" ];
then
cp $chkpath/${inf:0:-4}.chk $(pwd)/
fi
done

for chkinf in *.chk
do
bash Conversion_Scripts.sh <<EOF /dev/null
4
1
1
EOF
rm -f *.chk
done
}


generate_frez_input(){
echo "now begin to echo gaussian input frez files"
for inf in *.gjf
do
python3 judge.py $inf
jobtype=$?
echo $jobtype
if [ "$jobtype" == "0" ];
then
jobname="frez"
python3 modify.py $inf $optsolvent $jobname $chkpath $mem $nproc $OptBasisSet $theory
fi
done
echo "all .gjf files have generated ."
}



#run frez jobs
run_frez_jobs(){
echo " Now begin to run opt jobs"

for frezinf in *.gjf
do
python3 judge.py $frezinf
jobtype=$?
if [ "$jobtype" == "4" ];
then
echo "running $frezinf"
g16 $frezinf
wait
echo "$frezinf has finished"
fi
done
rm -f *_frez.gjf
echo "all frez jobs have done."
}



#generate ts input
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

for outinf in *.gjf-out
do
jobname="ts"
python3 modify.py $outinf $enesolvent $jobname $chkpath $mem $nproc $EneBasisSet $theory
done
rm -f *-out
echo "all frez job have deleted and generated new ts jobs input"
}


generate_ts_input_without_frez(){
echo "now begin to echo gaussian input ts files"
for inf in *.gjf
do
python3 judge.py $inf
jobtype=$?
echo $jobtype
if [ "$jobtype" == "0" ];
then
jobname="ts"
python3 modify.py $inf $optsolvent $jobname $chkpath $mem $nproc $OptBasisSet $theory
fi
done
echo "all .gjf files have generated ."
}



#run ts jobs
run_ts_jobs(){
echo " Now begin to run ts jobs"

for tsinf in *.gjf
do
python3 judge.py $tsinf
jobtype=$?
if [ "$jobtype" == "3" ];
then
echo "running $tsinf"
g16 $tsinf
wait
echo "$tsinf has finished"
fi
done
rm -f *_ts.gjf
echo "all ts jobs have done."
}




# All functions list generate_opt_input() ,  run_opt_jobs() , generate_ene_input() , run_ene_jobs(), transform_chk_to_fchk() bash extract_total.sh(this is a bash file))
#,generate_frez_input() , run_frez_jobs() , run_ts_jobs() ,  generate_ts_input() , generate_ts_input_without_frez()



generate_ts_input_without_frez
run_ts_jobs
generate_ene_input
run_ene_jobs

echo "all jobs have done"





