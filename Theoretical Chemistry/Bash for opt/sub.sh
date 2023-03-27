#!/bin/bash
source ~/.bashrc

optsolvent="gas"
enesolvent="gas"
chkpath=/data/zxwei
mem=12
nproc=18
OptBasisSet="def2svp"
EneBasisSet="def2tzvp"
theory="M062x"

# Never edit code below unless you know what are you doing

#generate opt input
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

echo "all .gjf files have generated . Now begin to run opt jobs"

#run opt jobs
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

echo "all opt jobs have generated . Now begin to run ene jobs"

#generate ene input
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

#run ene job

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

echo "all typical jobs have done . Please check the results"


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

echo "All jobs have done"



