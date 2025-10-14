from modify import qchem_file,qchem_out_file



def trans_out_into_inp(path,filename,col_add,col_change=1):
    col_add = float(col_add)
    qf = qchem_file()
    qof = qchem_out_file()
    qf.molecule.check = True
    qf.opt2.check = True
    qf.read_from_file(path+filename)
    qof.read_opt_from_file(path+filename+".out")
    qf.molecule.carti = qof.return_final_molecule_carti()
    name = filename.split("_")
    row_num = float(name[1])
    col_num = float(name[2])+col_add
    row = name[3]
    col_digit = int(name[4][3:-4])+col_change
    new_inp_name = name[0]+"_"+str(row_num)+"_"+str(col_num)+"_"+row+"_col"+str(col_digit)+".inp"
    qf.opt2.r12mr34[0][4] = col_num
    out_file = open(path+new_inp_name,"w")
    out_file.write(qf.molecule.return_output_format() + qf.remain_texts+qf.opt2.return_output_format())
    return new_inp_name

import sys
def main(path,filename,col_add):
    new_inp_name = trans_out_into_inp(path, filename, col_add, col_change=1)
    print("new_filename={}".format(new_inp_name))
main(sys.argv[1],sys.argv[2],sys.argv[3])


