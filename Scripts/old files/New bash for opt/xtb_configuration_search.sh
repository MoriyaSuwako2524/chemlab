#!/bin/bash
source ~/.bashrc

# variables below is what you need to edit
molclus_path=/home/zxwei/data/molclus_1.9.9.9_Linux

# Never edit function below unless you know what you are doing!

current_path=`pwd`

# run md
run_md(){
mkdir temmd

read inf
wait
cp $inf ./temmd/
cp md.inp ./temmd/
cd temmd
xtb $inf --input md.inp --omd --gfn 0
wait
cp xtb.trj ../traj.xyz
cd ../
echo "md work has done"
}



#run molclus xtb GFN0-XTB
run_molclus_1(){
cp -f traj.xyz $molclus_path/
cp -f ./molclusinp/settings_xtb_1.ini $molclus_path/settings.ini 

cd $molclus_path
./molclus
wait
./isostat << EOF > /dev/null

0.5
0.5

EOF
cd $current_path
cp -f $molclus_path/cluster.xyz ./traj.xyz
cp -f ./traj.xyz $molclus_path/traj.xyz
}


#run molclus xtb GFN2-XTB
run_molclus_2(){
cp -f traj.xyz $molclus_path/
cp -f ./molclusinp/settings_GFN2xtb.ini $molclus_path/settings.ini 

cd $molclus_path
./molclus
wait
./isostat << EOF > /dev/null

0.5
0.5

EOF
cd $current_path
cp -f $molclus_path/cluster.xyz ./traj.xyz
cp -f ./traj.xyz $molclus_path/traj.xyz
}

#run molclus gaussian+orca
run_molclus_3(){
cp -f traj.xyz $molclus_path/
cp -f ./molclusinp/settings_gaussian.ini $molclus_path/settings.ini 
cp -f ./molclusinp/template.gjf $molclus_path/
cp -f ./molclusinp/template_SP.inp $molclus_path/

cd $molclus_path
./molclus |tee out.txt
wait
./isostat << EOF > /dev/null

0.25
0.25
y
298.15
EOF
cd $current_path
cp -f $molclus_path/cluster.xyz ./traj.xyz

}

run_molclus_3

