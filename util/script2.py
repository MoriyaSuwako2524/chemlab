import os
import numpy as np
from util.modify_inp import qchem_out_excite_multi

def main():
    path = "./xyz_splits/"
    state_idx = 1  # S1

    groups = {"train": [], "val": [], "test": []}
    for fn in sorted(os.listdir(path)):
        if fn.endswith(".out"):
            if fn.startswith("train"):
                groups["train"].append(fn)
            elif fn.startswith("val"):
                groups["val"].append(fn)
            elif fn.startswith("test"):
                groups["test"].append(fn)

    split_idx = {}
    idx_offset = 0

    all_coords, all_energy, all_ex_energy, all_grad, all_transmom = [], [], [], [], []
    qm_type = None

    for grp, files in groups.items():
        if not files:
            continue
        print(f"Processing {grp} ({len(files)} files)")
        multi = qchem_out_excite_multi()
        multi.read_files(files, path=path)

        coords, energy, ex_energy, grad, transmom, qm_type = multi.export_numpy(
            state_idx=state_idx,
            prefix="",  # ?????
        )

        all_coords.append(coords)
        all_energy.append(energy)
        all_ex_energy.append(ex_energy)
        all_grad.append(grad)
        all_transmom.append(transmom)

        split_idx[f"idx_{grp}"] = np.arange(idx_offset, idx_offset + len(files))
        idx_offset += len(files)

    # ??
    coords = np.concatenate(all_coords, axis=0)
    energy = np.concatenate(all_energy, axis=0)         # S1 ???
    ex_energy = np.concatenate(all_ex_energy, axis=0)   # ???? (eV)
    grad = np.concatenate(all_grad, axis=0)
    transmom = np.concatenate(all_transmom, axis=0)

    prefix = "./full_"
    np.save(prefix + "coord.npy", coords)
    np.save(prefix + "S1_energy.npy", energy)    
    np.save(prefix + "ex_energy.npy", ex_energy)  
    np.save(prefix + "grad.npy", grad)
    np.save(prefix + "transmom.npy", transmom)
    np.save(prefix + "qm_type.npy", qm_type)
    np.savez(prefix + "split.npz", **split_idx)

    print("? Done")
    print("   coords:", coords.shape)
    print("   S1_energy:", energy.shape)
    print("   ex_energy:", ex_energy.shape)
    print("   grad:", grad.shape)
    print("   transmom:", transmom.shape)
    print("   qm_type:", qm_type.shape)
    print("   split:", split_idx)

if __name__ == "__main__":
    main()
