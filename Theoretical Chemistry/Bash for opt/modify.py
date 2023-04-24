import sys
optpara="%chk={}/{}.chk\n%mem={}GB\n%nprocshared={}\n#opt fre scrf=(solvent={}) nosymm {} {}"
enepara="%chk={}/{}.chk\n%mem={}GB\n%nprocshared={}\n#scrf=(SMD,solvent={}) nosymm {} {} guess=read"
tspara ="%chk={}/{}.chk\n%mem={}GB\n%nprocshared={}\n#opt=(ts,noeigen,calcfc,maxstep=10) scrf=(SMD,solvent={}) nosymm {} {} guess=(mix,always)"
frezpara ="%chk={}/{}.chk\n%mem={}GB\n%nprocshared={}\n#opt=(modredundant,loose) scrf=(SMD,solvent={}) nosymm {} {} guess=(mix,always)"
gas_optpara = "%chk={}/{}.chk\n%mem={}GB\n%nprocshared={}\n#opt fre  nosymm {} {}"
gas_enepara="%chk={}/{}.chk\n%mem={}GB\n%nprocshared={}\n# nosymm {} {} guess=read"
def main(filename,solvent,jobname,chk_site,mem,nproc,basis_set,theory):
    if "_opt.gjf-out" in filename:
        file_head = filename[:-12]
    if "_ts.gjf-out" in filename:
        file_head = filename[:-11]
    else:
        file_head = filename[:-4]
    file = open(filename, "r").read().split("\n")
    for i in range(len(file)):
        if file[i] == '':
            line = i+1
            break
    if solvent == "gas":
        if jobname == "opt":
            saved_para = gas_optpara.format(chk_site,file_head,mem,nproc,basis_set,theory)
        elif jobname == "ene":
            saved_para = gas_enepara.format(chk_site,file_head,mem,nproc,basis_set,theory)
    else:
        if jobname == "opt":
            saved_para = optpara.format(chk_site,file_head,mem,nproc,solvent,basis_set,theory)
        elif jobname == "ene":
            saved_para = enepara.format(chk_site,file_head,mem,nproc,solvent,basis_set,theory)
        elif jobname == "ts":
            saved_para = tspara.format(chk_site, file_head, mem, nproc, solvent, basis_set, theory)
        elif jobname == "frez":
            saved_para = frezpara.format(chk_site, file_head, mem, nproc, solvent, basis_set, theory)

    file2 = open(file_head+"_{}.gjf".format(jobname),"w")
    saved_para = saved_para.split("\n")
    for k in saved_para:
        file2.write(k.replace(" ","\t")+"\n")
    for j in range(len(file)):
        if j <= line-2:
            continue
        else:
            file2.write(file[j]+"\n")


main(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6],sys.argv[7],sys.argv[8])



