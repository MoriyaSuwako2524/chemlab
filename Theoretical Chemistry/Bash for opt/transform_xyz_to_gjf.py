name = "ZNX-1"
inp_file = open(name+".xyz","r").read().split("\n")
tem_stru = []
cluster = 0
for i in range(len(inp_file)):
    if "Cluster" in inp_file[i]:
        tem_stru = []
        cluster+=1
        j = i
        while "Cluster" not in inp_file[j+1]:
            tem_stru.append(inp_file[j+1])
            j+=1
            if j+1 ==len(inp_file):
                break
        out = open(name+"cluster"+str(cluster)+".gjf","w")
        out.write("%chk=/home/zxwei/cal/chk/trans.chk\n%mem=16GB\n%nprocshared=16\n#opt=(ts,calcfc,noeigen,maxstep=10)	scrf=(solvent=tetrahydrofuran)	nosymm	UM062X	def2SVP	guess=(mix,always)	fre\n\nTitle Card Required\n\n0 1\n")
        for k in tem_stru:
            if len(k) <= 30:
                continue
            out.write(k+"\n")



