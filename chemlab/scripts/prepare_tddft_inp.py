import os
from chemlab.util.modify_inp import qchem_out_excite_multi
import argparse
from chemlab.util.ml_data import MLData
from chemlab.util.modify_inp import qchem_out_aimd_multi
import numpy as np
from pathlib import Path
from chemlab.util.modify_inp import single_spin_job
def export_aimd(args):
    path = args.path
    files = list(args.file)
    print(files)
    for i in range(len(files)):
        files[i] = f"{path}{files[i]}"

    multi = qchem_out_aimd_multi()
    multi.read_files(files)
    out = f"{path}{args.out}"
    # 转换成 kcal/mol, Angstrom, kcal/mol/Angstrom
    multi.export_numpy(prefix=f"{out}/tmp_",
                       energy_unit=args.energy_unit,
                       distance_unit=args.distance_unit,
                       force_unit=args.force_unit)
    dataset = MLData(prefix=f"{out}/tmp_",files=["coord", "energy", "grad", "type"])
    dataset.save_split(n_train=args.dataset_size,n_val=0,n_test=0,prefix=out)
    dataset.export_xyz_from_split(split_file=f"{out}/split.npz", outdir=f"{out}", prefix_map=None)
    for xyz_file in Path(f"{out}").glob('*.xyz'):
        tem = single_spin_job()
        tem.spins = args.charge
        tem.charge = args.spin
        tem.ref_name = f"{path}{args.ref}"
        tem.xyz_name = str(xyz_file)
        tem.generate_outputs(prefix=f"{out}",new_file_name=xyz_file.name)
    #print(xyz_file)
def main():
    parser = argparse.ArgumentParser(description="Export tddft calculation inputs from md simulation frames")
    parser.add_argument("--path", required=True, help="Path to md output file.")
    parser.add_argument("--file", required=True,type=list, help="Md output files.")
    parser.add_argument("--out",type=str,default="/raw_data/", help="Output directory.")
    parser.add_argument("--charge", type=int,default=0, help="Charge of the molecule")
    parser.add_argument("--spin", type=int, default=1, help="Spin of the molecule")
    parser.add_argument("--ref", type=str, default="ref.in", help="Reference file")
    parser.add_argument("--dataset_size",type=int,default=3000, help="Size of dataset")
    parser.add_argument("--energy_unit", type=str, default="kcal/mol", help="Unit of energy")
    parser.add_argument("--ex_energy_unit", type=str, default="ev", help="Unit of excitation energy")
    parser.add_argument("--distance_unit", type=str, default="ang", help="Unit of coordinates")
    parser.add_argument("--grad_unit", type=tuple, default=("kcal/mol", "ang"), help="Unit of gradient")
    parser.add_argument("--force_unit", type=tuple, default=("kcal/mol", "ang"), help="Unit of force")
    args = parser.parse_args()
    export_aimd(args)

if __name__ == "__main__":
    main()
