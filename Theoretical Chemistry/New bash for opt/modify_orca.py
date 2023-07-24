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
    tem_file = open("./tem/" + temname + ".inp", "r")
    saved_para = tem_file.read()
    file2 = open(file_head+"_orca.inp","w")
    saved_para = saved_para.split("\n")
    N = 0
    count= 0
    count2 = 0
    for k in saved_para:
        if N == 0 :
            file2.write("%chk=/home/zxwei/cal/chk/{}.chk".format(file_head))
            N+=1
        file2.write(k.replace(" ","\t")+"\n")
    for j in range(len(file)):
        if j <= line +1:
            continue
        if j == line +2:
            file2.write("* xyz "+ file[j] + "\n")
        else:
            print(file[j])
            if len(file[j]) >10 and count == 0:
                file2.write(file[j])
                count +=1
            elif len(file[j]) >10 and count !=0:
                file2.write("\n")
                file2.write(file[j])
            elif len(file[j]) <10 and count2 ==0:
                file2.write("\n"+"*"+"\n")
                count2 += 1
            else:
                file2.write(file[j])



main(sys.argv[1],sys.argv[2])



