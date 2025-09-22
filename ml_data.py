import numpy as np

class MLData:
    def __init__(self, prefix="./"):
        self.coords = np.load(prefix + "coord.npy")
        self.energies = np.load(prefix + "energy.npy")
        self.grads = np.load(prefix + "grad.npy")
        self.qm_types = np.load(prefix + "qm_type.npy")

        self.nframes = len(self.energies)
        print(f"Loaded dataset with {self.nframes} frames, {self.coords.shape[1]} atoms")

    def split_dataset(self, n_train, n_val, n_test, seed=42):

        assert n_train + n_val + n_test <= self.nframes, "数据不足以分配"

        order = np.argsort(self.energies)


        total_needed = n_train + n_val + n_test
        indices = np.linspace(0, self.nframes - 1, total_needed, dtype=int)
        sampled = order[indices]

        # 3) 划分
        idx_train = sampled[:n_train]
        idx_val = sampled[n_train:n_train+n_val]
        idx_test = sampled[n_train+n_val:]

        # 4) 打乱
        rng = np.random.default_rng(seed)
        idx_train = idx_train[rng.permutation(len(idx_train))]
        idx_val = idx_val[rng.permutation(len(idx_val))]
        idx_test = idx_test[rng.permutation(len(idx_test))]

        return {"train": idx_train, "val": idx_val, "test": idx_test}

    def save_split(self, n_train, n_val, n_test, prefix="./", seed=42):
        split_dict = self.split_dataset(n_train, n_val, n_test, seed)
        np.save(prefix + "split.npy", split_dict)
        print(f"Saved split to {prefix}split.npy "
              f"(train={len(split_dict['train'])}, val={len(split_dict['val'])}, test={len(split_dict['test'])})")





