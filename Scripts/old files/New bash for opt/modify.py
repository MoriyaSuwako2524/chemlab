import sys
def main(filename,temname):
    if ".gjf-out" in filename:
        file_head = filename[:-13]
    elif "_ts.gjf-out" in filename:
        file_head = filename[:-11]
    elif "_frez.gjf-out" in filename:
        file_head = filename[:-13]
    else:
        file_head = filename[:-4]
    file = open(filename, "r").read().split("\n")
    for i in range(len(file)):
        if file[i] == '':
            line = i+1
            break
    if temname == "orca":
        tem_file = open("./tem/" + temname + ".inp", "r")
    else:
        tem_file = open("./tem/"+temname+".gjf","r")
    saved_para = tem_file.read()
    if temname == "orca":
        file2 = open(file_head+"_orca.inp","w")
    else:
        file2 = open(file_head+"_{}.gjf".format(temname),"w")
    saved_para = saved_para.split("\n")
    N = 0
    for k in saved_para:
        if N == 0 :
            file2.write("%chk=/home/zxwei/cal/chk/{}.chk".format(file_head))
            N+=1
        file2.write(k.replace(" ","\t")+"\n")
    for j in range(len(file)):
        if j <= line-2:
            continue
        else:
            file2.write(file[j]+"\n")


main(sys.argv[1],sys.argv[2])



