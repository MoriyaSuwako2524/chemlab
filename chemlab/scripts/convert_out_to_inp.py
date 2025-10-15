import sys
from chemlab.util.modify_inp import conver_opt_out_to_inp
def main(filename):
    ref = "ref.in"
    int = conver_opt_out_to_inp()
    int.ref_name = ref
    int.convert(new_out_file_name=filename)
main(sys.argv[1])
