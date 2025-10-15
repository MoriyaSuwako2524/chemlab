import sys
from chemlab.util.modify_inp import conver_opt_out_to_inp
def main(filename):
    ref = "ref.in"
    inp = conver_opt_out_to_inp()
    inp.ref_name = ref
    inp.convert(new_out_file_name=filename)
main(sys.argv[1])
