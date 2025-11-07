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
    ref_transition_density = transition_dentsity[0]


    if args.align_ref == "dipole":
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
    elif args.align_ref == "transition_density":
        ref_norm = np.linalg.norm(ref_transition_density)
        if ref_norm == 0:
            raise ValueError("Reference dipole moment is zero; cannot determine direction alignment.")
        aligned_mom = []
        aligned_transition_density = []
        for i in range(len(transition_dentsity)):
            mom = transmom[i]
            td = transition_dentsity[i]
            #  cosθ = (a·b)/(|a||b|)
            dot = np.dot(ref_transition_density, td)
            td_norm = np.linalg.norm(td)
            if td_norm == 0:
                aligned_mom.append(mom)
                aligned_transition_density.append(td)
                continue
            cos_val = dot / (ref_norm * td_norm)

            if cos_val < 0:
                mom = -mom
                td = -td

            aligned_transition_density.append(td)
            aligned_mom.append(mom)
    else:
        aligned_mom = transmom
        aligned_transition_density = transition_dentsity


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
	
    train_splits = args.train_splits
    val_splits = args.val_splits
    test_splits = args.test_splits
    save_multiple_splits_same_test(coords.shape[0], train_splits, val_splits, test_splits,prefix=out_path)

    return None


def save_multiple_splits_same_test(n_total,train_sizes, n_val, n_test, prefix="./", seed=42):
    """
    Generate multiple dataset splits with the same test set.
    Different train sizes but fixed test indices.

    Parameters
    ----------
    n_total: int
        Total number of samples.
    train_sizes : list[int]
        List of training set sizes to generate (e.g., [2000, 1000, 500, 250]).
    n_val : int
        Validation set size for all splits.
    n_test : int
        Fixed test set size (shared across all splits).
    prefix : str
        Output directory or filename prefix (e.g., "./" or "./phbdi_").
    seed : int
        Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    all_indices = np.arange(n_total)

    # Step 1: create fixed test set
    idx_test = rng.choice(all_indices, size=n_test, replace=False)
    remaining = np.setdiff1d(all_indices, idx_test)

    print(f"Fixed test set selected: {len(idx_test)} samples.")
    # Step 2: loop over multiple train sizes
    for n_train in train_sizes:
        n_train=int(n_train)
        # avoid exceeding remaining count
        if n_train + n_val > len(remaining):
            print(f"⚠️ Skipping n_train={n_train} (too large for available data)")
            continue

        # create new RNG for reproducibility per split (optional)
        rng_split = np.random.default_rng(seed + n_train)

        idx_train = rng_split.choice(remaining, size=n_train, replace=False)
        idx_val_candidates = np.setdiff1d(remaining, idx_train)
        idx_val = rng_split.choice(idx_val_candidates, size=n_val, replace=False)

        # Step 3: save split
        split_dict = {
            "idx_train": idx_train,
            "idx_val": idx_val,
            "idx_test": idx_test,
        }

        out_file = os.path.join(prefix, f"{n_train}_split.npz")
        np.savez(out_file, **split_dict)

        print(
            f"✅ Saved {out_file} "
            f"(train={len(idx_train)}, val={len(idx_val)}, test={len(idx_test)})"
        )

import ast, json, re

def smart_parse_list(s):
    s = s.strip()
    if s.startswith("["):
        return ast.literal_eval(s)
    elif "," in s:
        return [int(x) for x in re.split(r"[,\s]+", s.strip("[] ")) if x]
    else:
        return [int(x) for x in s.split()]

#


def main():
    parser = argparse.ArgumentParser(description="Export tddft calculation using Q-Chem.")
    parser.add_argument("--data", required=True,default="./raw_data/" ,help="Path to reference Q-Chem input file.")
    parser.add_argument("--out", required=True,default="./" ,help="Output directory.")
    parser.add_argument("--prefix", type=str, default="full_", help="prefix of files")
    parser.add_argument("--align_ref", type=str, default="transition_density", help="Align vector reference")
    parser.add_argument("--state_idx", type=int, default=1, help="excited state index")
    parser.add_argument("--energy_unit", type=str, default="kcal/mol", help="Unit of energy")
    parser.add_argument("--ex_energy_unit", type=str, default="ev", help="Unit of excitation energy")
    parser.add_argument("--distance_unit", type=str, default="ang", help="Unit of coordinates")
    parser.add_argument("--grad_unit", type=tuple, default=("kcal/mol", "ang"), help="Unit of gradient")
    parser.add_argument("--force_unit", type=tuple, default=("kcal/mol", "ang"), help="Unit of force")
    parser.add_argument("--train_splits", type=smart_parse_list, default="[1000,500,2000]" ,help="To export splits. This should be a list")
    parser.add_argument("--val_splits", type=int, default=400,
                        help="To export val splits. This should be a list")
    parser.add_argument("--test_splits", type=int, default=400,
                        help="To export test splits. This should be a list")
    args = parser.parse_args()
    export_numpy(args)
if __name__ == "__main__":
    main()
