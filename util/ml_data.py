import numpy as np

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
    def __init__(self, prefix="./", files=None):
        if files is None:
            files = ["coord", "energy", "grad", "type"]

        self.data = {f: np.load(prefix + f + ".npy", allow_pickle=True) for f in files}

        self.coords = self.data.get("coord", None)
        self.energies = self.data.get("energy", None)
        self.grads = self.data.get("grad", None)
        self.qm_types = self.data.get("type", None)

        self.nframes = len(self.energies) if self.energies is not None else 0
        print(f"Loaded dataset with {self.nframes} frames, "
              f"{self.coords.shape[1] if self.coords is not None else '?'} atoms")

    def split_dataset(self, n_train, n_val, n_test, seed=42):
        assert n_train + n_val + n_test <= self.nframes

        order = np.argsort(self.energies)
        total_needed = n_train + n_val + n_test
        indices = np.linspace(0, self.nframes - 1, total_needed, dtype=int)
        sampled = order[indices]

        idx_train = sampled[:n_train]
        idx_val = sampled[n_train:n_train + n_val]
        idx_test = sampled[n_train + n_val:]

        rng = np.random.default_rng(seed)
        idx_train = idx_train[rng.permutation(len(idx_train))]
        idx_val = idx_val[rng.permutation(len(idx_val))]
        idx_test = idx_test[rng.permutation(len(idx_test))]

        return {"idx_train": idx_train, "idx_val": idx_val, "idx_test": idx_test}

    def save_split(self, n_train, n_val, n_test, prefix="./", seed=42):
        split_dict = self.split_dataset(n_train, n_val, n_test, seed)


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

