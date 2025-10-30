#!/usr/bin/env python3
import os
import numpy as np
from chemlab.util.modify_inp import qchem_out_excite_multi
import argparse
'''
This scrirpt export npy files from TDDFT calculation folder
'''

def export_numpy(args):
    # ======== Settings ========
    path = args. data       # Directory containing Q-Chem output files
    out_path = args.out
    prefix = args.prefix            # Output file prefix
    state_idx = args.state_idx                 # Excited state index (e.g. S1)
    energy_unit = args.energy_unit       # Energy output unit
    ex_energy_unit = args.ex_energy_unit
    distance_unit = args.distance_unit          # Coordinate output unit
    grad_unit = args.grad_unit  # For gradient conversion
    force_unit = args.force_unit # For force conversion

    # ======== Split files into train/val/test groups ========
    groups = {"train": [], "val": [], "test": []}
    for fn in sorted(os.listdir(path)):
        if fn.endswith(".out"):
            if fn.startswith("train"):
                groups["train"].append(fn)
            elif fn.startswith("val"):
                groups["val"].append(fn)
            elif fn.startswith("test"):
                groups["test"].append(fn)

    # ======== Prepare accumulators ========
    split_idx = {}
    idx_offset = 0

    all_coords, all_gs_energy, all_ex_energy,all_ex_state_energy = [], [], [], []
    all_grad, all_force, all_transmom, all_dipolemom,all_transition_density = [], [], [], [] ,[]
    qm_type = None

    # ======== Process each subset ========
    for grp, files in groups.items():
        if not files:
            continue

        print(f"Processing {grp} ({len(files)} files)...")

        multi = qchem_out_excite_multi()
        multi.read_files(files, path=path)

        # === Use the new modular export system ===
        results = {
            "coords": multi.export_coords(prefix=grp, distance_unit=distance_unit),
            "gs_energy": multi.export_gs_energy(prefix=grp, energy_unit=energy_unit, state_idx=0),
            "ex_energy": multi.export_ex_energy(prefix=grp, energy_unit=ex_energy_unit, state_idx=state_idx),
            "ex_state_energy": multi.export_gs_energy(prefix=grp, energy_unit=energy_unit, state_idx=state_idx),
            "grad": multi.export_gradients(prefix=grp, grad_unit=grad_unit, state_idx=state_idx),
            "force": multi.export_forces(prefix=grp, grad_unit=force_unit, state_idx=state_idx),
            "transmom": multi.export_transmom(prefix=grp, unit="au", state_idx=state_idx),
            "dipolemom": multi.export_dipolemom(prefix=grp, unit="au", state_idx=0),
            "transition_density": multi.export_transition_density(prefix=grp, unit="e", state_idx=state_idx),
	}
        # === Append arrays ===
        all_coords.append(results["coords"])
        all_gs_energy.append(results["gs_energy"])
        all_ex_energy.append(results["ex_energy"])
        all_ex_state_energy.append(results["ex_state_energy"])
        all_grad.append(results["grad"])
        all_force.append(results["force"])
        all_transmom.append(results["transmom"])
        all_dipolemom.append(results["dipolemom"])
        all_transition_density.append(results["transition_density"])

        # === Atom types (from first file only) ===
        if qm_type is None:
            from chemlab.util.file_system import atom_charge_dict
            atom_symbols = [atm[0] for atm in multi.tasks[0].molecule.carti]
            qm_type = np.array([atom_charge_dict[sym] for sym in atom_symbols])

        # === Record split indices ===
        split_idx[f"idx_{grp}"] = np.arange(idx_offset, idx_offset + len(files))
        idx_offset += len(files)

    # ======== Merge all datasets ========
    coords = np.concatenate(all_coords, axis=0)
    gs_energy = np.concatenate(all_gs_energy, axis=0)
    ex_energy = np.concatenate(all_ex_energy, axis=0)
    ex_state_energy = np.concatenate(all_ex_state_energy, axis=0)
    grad = np.concatenate(all_grad, axis=0)
    force = np.concatenate(all_force, axis=0)
    transmom = np.concatenate(all_transmom, axis=0)
    dipolemom = np.concatenate(all_dipolemom, axis=0)
    transition_dentsity = np.concatenate(all_transition_density,axis=0)

    ref_mom = transmom[0]
    ref_norm = np.linalg.norm(ref_mom)

    if ref_norm == 0:
        raise ValueError("Reference dipole moment is zero; cannot determine direction alignment.")

    aligned_mom = []
    aligned_transition_density = []
    for i in range(len(transmom)):
        mom = transmom[i]
        td = transition_dentsity[i]
        #  cosθ = (a·b)/(|a||b|)
        dot = np.dot(ref_mom, mom)
        mom_norm = np.linalg.norm(mom)
        if mom_norm == 0:
            aligned_mom.append(mom)
            continue
        cos_val = dot / (ref_norm * mom_norm)

        if cos_val < 0:
            mom = -mom
            td = -td

        aligned_transition_density.append(td)
        aligned_mom.append(mom)


    transmom_aligned = np.array(aligned_mom)
    prefix = f"{out_path}{prefix}"
    np.save(prefix + "coord.npy", coords)
    np.save(prefix + "gs_energy.npy", gs_energy)
    np.save(prefix + "ex_energy.npy", ex_energy)
    np.save(prefix + "ex_state_energy.npy", ex_state_energy)
    np.save(prefix + "grad.npy", grad)
    np.save(prefix + "force.npy", force)
    np.save(prefix + "transmom.npy", transmom_aligned)
    np.save(prefix + "dipolemom.npy", dipolemom)
    np.save(prefix + "transition_density.npy", transition_dentsity)
    np.save(prefix + "aligned_td.npy", aligned_transition_density)
    np.save(prefix + "qm_type.npy", qm_type)
    np.savez(prefix + "split.npz", **split_idx)

    print("\n? Export Completed Successfully")
    print("   coords:", coords.shape)
    print("   gs_energy:", gs_energy.shape)
    print("   ex_energy:", ex_energy.shape)
    print("   ex_state_energy:", ex_state_energy.shape)
    print("   grad:", grad.shape)
    print("   force:", force.shape)
    print("   transmom:", transmom_aligned.shape)
    print("   dipolemom:", dipolemom.shape)
    print("   qm_type:", qm_type.shape)
    print(" transition_density:", transition_dentsity.shape)
    print("   split:", {k: v.shape for k, v in split_idx.items()})

def main():
    parser = argparse.ArgumentParser(description="Export tddft calculation using Q-Chem.")
    parser.add_argument("--data", required=True,default="./raw_data/" ,help="Path to reference Q-Chem input file.")
    parser.add_argument("--out", required=True,default="./" ,help="Output directory.")
    parser.add_argument("--prefix", type=str, default="full_", help="prefix of files")
    parser.add_argument("--state_idx", type=int, default=1, help="excited state index")
    parser.add_argument("--energy_unit", type=str, default="kcal/mol", help="Unit of energy")
    parser.add_argument("--ex_energy_unit", type=str, default="ev", help="Unit of excitation energy")
    parser.add_argument("--distance_unit", type=str, default="ang", help="Unit of coordinates")
    parser.add_argument("--grad_unit", type=tuple, default=("kcal/mol", "ang"), help="Unit of gradient")
    parser.add_argument("--force_unit", type=tuple, default=("kcal/mol", "ang"), help="Unit of force")
    args = parser.parse_args()
    export_numpy(args)
if __name__ == "__main__":
    main()
