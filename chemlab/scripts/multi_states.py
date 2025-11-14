import sys
from chemlab.util.modify_inp import single_spin_job
def main(filename):
	ref = "ref.in"
	int = single_spin_job()
	int.spins = 1
	int.charge = -1
	int.ref_name = ref
	int.xyz_name = filename
	int.generate_outputs()
main(sys.argv[1])
	
