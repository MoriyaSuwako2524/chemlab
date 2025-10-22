import numpy as np
import os
def load_npy_files(prefix, files):
    """
    Load multiple .npy files from prefix.
    Args:
        prefix (str): folder prefix (e.g. "./examples/")
        files (list[str]): list of base filenames without .npy suffix
    Returns:
        dict: {name: np.ndarray}
    """
    data = {}
    for f in files:
        path = prefix + f + ".npy"
        arr = np.load(path, allow_pickle=True)
        data[f] = arr
    return data



class MLData:
    def __init__(self, prefix="./", files=["coord", "energy", "grad", "type"], energy_key="energy",type="npy"):
        if type == "npy":
            self.data = {f: np.load(prefix + f + ".npy", allow_pickle=True) for f in files}

            self.coords = self.data.get("coord", None)
            self.energies = self.data.get(energy_key, None)
            self.grads = self.data.get("grad", None)
            self.qm_types = self.data.get("type", None)

            self.nframes = len(self.energies) if self.energies is not None else 0
            print(f"Loaded dataset with {self.nframes} frames, "
                  f"{self.coords.shape[1] if self.coords is not None else '?'} atoms")
        elif type == "xyz":
            self.read_xtb_traj_ase(prefix+"traj.xyz")
            self.nframes = len(self.coords) if self.coords is not None else 0
            print(f"Loaded dataset with {self.nframes} frames, "
                  f"{self.coords.shape[1] if self.coords is not None else '?'} atoms")
    def split_dataset(self, n_train, n_val, n_test, seed=42):
        """
        Energy-aware balanced split:
        1) sort frames by energy
        2) take 'total' indices evenly across the sorted list (no duplicates)
        3) randomly assign labels [train/val/test] across those evenly spaced indices
           so each split covers the whole energy range (no "shift")
        """
        import numpy as np
        assert n_train + n_val + n_test <= self.nframes

        rng = np.random.default_rng(seed)

        # 1) sort by energy
        order = np.argsort(self.energies)

        # 2) evenly cover the whole energy range with UNIQUE picks
        total = n_train + n_val + n_test
        # Use array_split: split sorted indices into 'total' chunks, pick 1 per chunk.
        # This guarantees exactly 'total' unique indices spread across the whole range.
        chunks = np.array_split(order, total)
        sampled = np.array([ch[len(ch) // 2] for ch in chunks], dtype=np.int64)  # pick middle of each chunk

        # 3) build randomized labels, *not* sequential slicing
        # labels: 0=train, 1=val, 2=test
        labels = np.array([0] * n_train + [1] * n_val + [2] * n_test, dtype=np.int64)
        rng.shuffle(labels)

        buckets = {0: [], 1: [], 2: []}
        for idx, lab in zip(sampled, labels):
            buckets[lab].append(int(idx))  # ensure python int / int64

        # 4) shuffle inside each split (optional but recommended)
        idx_train = rng.permutation(np.array(buckets[0], dtype=np.int64))
        idx_val = rng.permutation(np.array(buckets[1], dtype=np.int64))
        idx_test = rng.permutation(np.array(buckets[2], dtype=np.int64))

        # 5) sanity checks (sizes & dtypes)
        assert len(idx_train) == n_train and len(idx_val) == n_val and len(idx_test) == n_test
        assert idx_train.dtype.kind in "iu" and idx_val.dtype.kind in "iu" and idx_test.dtype.kind in "iu"

        return {
            "idx_train": idx_train,
            "idx_val": idx_val,
            "idx_test": idx_test
        }

    def save_split(self, n_train, n_val, n_test, prefix="./", seed=42):
        split_dict = self.split_dataset(n_train, n_val, n_test, seed=seed)


        np.savez(prefix + "split.npz",
                 idx_train=split_dict["idx_train"],
                 idx_val=split_dict["idx_val"],
                 idx_test=split_dict["idx_test"])
        print(f"Saved split to {prefix}split.npz "
              f"(idx_train={len(split_dict['idx_train'])}, "
              f"idx_val={len(split_dict['idx_val'])}, "
              f"idx_test={len(split_dict['idx_test'])})")
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
    
        # 原子序号→元素符号
        from .file_system import atom_charge_dict
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
