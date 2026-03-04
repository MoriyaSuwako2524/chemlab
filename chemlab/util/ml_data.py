import numpy as np
import os
from chemlab.util.file_system import atom_charge_dict
def load_npy_files(prefix, files):

    data = {}
    for f in files:
        path = prefix + f + ".npy"
        arr = np.load(path, allow_pickle=True)
        data[f] = arr
    return data



class MLData:
    def __init__(self, prefix="./", files=["coord", "energy", "grad", "type"], energy_key="energy", qm_types_key="type", coords_key="coord",grad_key = "grad",type="npy"):
        if type == "npy":
            self.data = {f: np.load(prefix + f + ".npy", allow_pickle=True) for f in files}

            self.coords = self.data.get(coords_key, None)
            self.energies = self.data.get(energy_key, None)
            self.grads = self.data.get(grad_key, None)
            self.qm_types = self.data.get(qm_types_key, None)

            self.nframes = len(self.coords) if self.coords is not None else 0
            print(f"Loaded dataset with {self.nframes} frames, "
                  f"{self.coords.shape[1] if self.coords is not None else '?'} atoms")
        elif type == "xyz":
            self.read_xtb_traj_ase(prefix+"traj.xyz")
            self.nframes = len(self.coords) if self.coords is not None else 0
            print(f"Loaded dataset with {self.nframes} frames, "
                  f"{self.coords.shape[1] if self.coords is not None else '?'} atoms")

    def split_dataset(self, n_train, n_val, n_test, seed=42):
        import numpy as np
        total = n_train + n_val + n_test
        assert total <= self.nframes

        indices = np.linspace(0, self.nframes - 1, total, dtype=np.int64)

        rng = np.random.default_rng(seed)
        perm = rng.permutation(total)
        idx_train = indices[perm[:n_train]]
        idx_val = indices[perm[n_train:n_train + n_val]]
        idx_test = indices[perm[n_train + n_val:]]

        assert len(idx_train) == n_train and len(idx_val) == n_val and len(idx_test) == n_test

        return {
            "idx_train": idx_train,
            "idx_val": idx_val,
            "idx_test": idx_test,
        }

    def save_split(self, n_train, n_val, n_test, prefix="./", seed=42):
        print(f"Split:n_train:{n_train}, n_val:{n_val}, n_test:{n_test},Total_frames:{self.nframes}")
        split_dict = self.split_dataset(n_train, n_val, n_test, seed=seed)
        np.savez(prefix + "split.npz",
                 idx_train=split_dict["idx_train"],
                 idx_val=split_dict["idx_val"],
                 idx_test=split_dict["idx_test"])
        print(f"Saved split to {prefix}split.npz "
              f"(idx_train={len(split_dict['idx_train'])}, "
              f"idx_val={len(split_dict['idx_val'])}, "
              f"idx_test={len(split_dict['idx_test'])})")

    def save_multiple_splits_same_test(self, train_sizes, n_val, n_test, prefix="./", seed=42):
        """
        Generate multiple dataset splits with the same test set.
        Different train sizes but fixed test indices.

        Parameters
        ----------
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
        n_total = self.nframes
        all_indices = np.arange(n_total)

        # Step 1: create fixed test set
        idx_test = rng.choice(all_indices, size=n_test, replace=False)
        remaining = np.setdiff1d(all_indices, idx_test)

        print(f"Fixed test set selected: {len(idx_test)} samples.")

        # Step 2: loop over multiple train sizes
        for n_train in train_sizes:
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
    def get_split_data(self, split, part="train"):

        if isinstance(split, str):
            split = np.load(split)

        if hasattr(split, "files"):
            idx = split[part]
        elif isinstance(split, dict):
            idx = split[part]
        else:
            raise TypeError("split must be file path (.npz), dict, or NpzFile")

        subset = {}
        for k, v in self.data.items():
            if v.ndim > 0 and len(v) == self.nframes:
                subset[k] = v[idx]
            else:
                subset[k] = v
        return subset
    def export_xyz_from_split(self, split_file, outdir="./raw_data", prefix_map=None):
        import os
        if isinstance(split_file, str):
            split = np.load(split_file, allow_pickle=True)
        else:
            split = split_file
        if prefix_map is None:
            prefix_map = {"idx_train": "train", "idx_val": "val", "idx_test": "test"}
    
        if not os.path.exists(outdir):
            os.makedirs(outdir)


        inv_dict = {v: k for k, v in atom_charge_dict.items()}
    
        for key, prefix in prefix_map.items():
            if key not in split:
                continue
            indices = split[key]
            for i, idx in enumerate(indices, 0):
                coords = self.coords[idx]
                types = self.qm_types  
                natoms = len(types)
                fname = f"{prefix}_{i:04d}.xyz"
                path = os.path.join(outdir, fname)
                with open(path, "w") as f:
                    f.write(f"{natoms}\n")
                    f.write(f"{prefix} frame {idx}\n")
                    for sym, (x, y, z) in zip(types, coords):
                        if isinstance(sym, (int, np.integer)):
                            sym = inv_dict.get(int(sym), str(sym))
                        f.write(f"{sym} {x:.8f} {y:.8f} {z:.8f}\n")
        print(f"XYZ files exported to {outdir}")
    def read_xtb_traj_ase(self,xyz):
        from ase.io import read

        atoms_list = read(xyz, index=":")
        self.coords = np.array([a.positions for a in atoms_list])
        energies = []
        for atoms in atoms_list:
            info_keys = list(atoms.info.keys())
            for i, key in enumerate(info_keys):
                if "energy" in key:
                    try:
                        e = float(info_keys[i + 1])
                        energies.append(e)
                    except (IndexError, ValueError):
                        energies.append(np.nan)
                    break
            else:
                energies.append(np.nan)
        self.energies = np.array(energies)
        self.qm_types = np.array(atoms_list[0].get_chemical_symbols())  # assume same atoms each frame

'''
Example:
'''
#test = MLData("D:\calculate\github\chemlab\examples\rhodamin\\xtb.traj")
