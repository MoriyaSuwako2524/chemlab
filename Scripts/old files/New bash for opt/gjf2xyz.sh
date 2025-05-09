#Convert all .gjf files to .xyz file by Multiwfn
#!/bin/bash
icc=0
nfile=`ls *cs.gjf|wc -l`
for inf in *cs.gjf
do
((icc++))
echo Converting ${inf} to ${inf//gjf/xyz} ... \($icc of $nfile\)
Multiwfn ${inf} << EOF > /dev/null
100
2
2
${inf//gjf/xyz}
0
q
EOF
done
