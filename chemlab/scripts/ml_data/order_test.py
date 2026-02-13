import numpy as np
from chemlab.scripts.base import Script
from chemlab.config.config_loader import ConfigBase
from chemlab.scripts.ml_data.soap import build_global_to_window_mapping

class OrderTestConfig(ConfigBase):
    section_name = "order_test"
class OrderTest(Script):
    name = "order_test"
    config = OrderTestConfig
    def run(self,cfg):
        windows = cfg.windows
        npy_path = cfg.npy_path
        out_path=cfg.out_path

        n_test = cfg.n_test
        n_val = cfg.n_val
        p,window_sizes = build_global_to_window_mapping(npy_path, windows)
        total = sum(window_sizes)
        all_test_global = np.array([], dtype=int)
        all_val_global = np.array([], dtype=int)

        for i in range(windows):
            test_local, val_local = select_evenly_spaced_separate(
                window_sizes[i], n_test, n_val
            )
            window_start = sum(window_sizes[:i])
            all_test_global.extend(test_local + window_start)
            all_val_global.extend(val_local + window_start)
        all_test_global = np.array(all_test_global, dtype=int)
        all_val_global = np.array(all_val_global, dtype=int)
        all_train_global = np.array([], dtype=int)
        print(f"Test: {len(all_test_global)} ({len(all_test_global) / total * 100:.1f}%)")
        print(f"Val:  {len(all_val_global)} ({len(all_val_global) / total * 100:.1f}%)")
        print(f"可用于train: {total - len(all_test_global) - len(all_val_global)} "
              f"({(total - len(all_test_global) - len(all_val_global)) / total * 100:.1f}%)")


        os.makedirs(out_path, exist_ok=True)
        output_file = os.path.join(out_path, "test_split.npz")

        np.savez(
            output_file,
            idx_train=all_train_global,
            idx_val=all_val_global,
            idx_test=all_test_global
        )

        print(f"\n 已保存: {output_file}")
        print(f"   idx_train: {all_train_global.shape} (空)")
        print(f"   idx_val:   {all_val_global.shape}")
        print(f"   idx_test:  {all_test_global.shape}")


def select_evenly_spaced_separate(n_total, n_test, n_val):
    total_select = n_test + n_val
    if total_select >= n_total:

        test_indices = np.arange(0, n_total, 2)[:n_test]
        val_indices = np.arange(1, n_total, 2)[:n_val]
        return test_indices, val_indices
    step = n_total / total_select

    positions = np.floor(np.arange(total_select) * step).astype(int)

    test_indices = positions[:n_test]
    val_indices = positions[n_test:n_test + n_val]

    return test_indices, val_indices