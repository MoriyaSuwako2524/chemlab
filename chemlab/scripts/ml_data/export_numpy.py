import os
import numpy as np
from chemlab.util.modify_inp import qchem_out_excite_multi
from chemlab.scripts.base import Script
from chemlab.config import ExportNumpyConfig


class ExportNumpy(Script):
    """
    Export TDDFT dataset into numpy arrays.

    CLI will load this class automatically and call `run(cfg)`.
    """
    name = "export_numpy"
    config = ExportNumpyConfig   # link to config section in config.toml

    # -------------------------------------------------------
    # Main execution
    # -------------------------------------------------------
    def run(self, cfg):

        # ======== Settings ========
        path = cfg.data
        out_path = cfg.out
        prefix = cfg.prefix
        state_idx = cfg.state_idx
        energy_unit = cfg.energy_unit
        ex_energy_unit = cfg.ex_energy_unit
        distance_unit = cfg.distance_unit
        grad_unit = cfg.grad_unit
        force_unit = cfg.force_unit

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

        all_coords = []
        all_gs_energy = []
        all_ex_energy = []
        all_ex_state_energy = []
        all_grad = []
        all_force = []
        all_transmom = []
        all_dipolemom = []
        all_transition_density = []
        qm_type = None

        # ======== Process each subset ========
        for grp, files in groups.items():
            if not files:
                continue

            print(f"Processing {grp} ({len(files)} files)...")

            multi = qchem_out_excite_multi()
            multi.read_files(files, path=path)

            # === Modular export system ===
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

            # append arrays
            all_coords.append(results["coords"])
            all_gs_energy.append(results["gs_energy"])
            all_ex_energy.append(results["ex_energy"])
            all_ex_state_energy.append(results["ex_state_energy"])
            all_grad.append(results["grad"])
            all_force.append(results["force"])
            all_transmom.append(results["transmom"])
            all_dipolemom.append(results["dipolemom"])
            all_transition_density.append(results["transition_density"])

            # atom types only first time
            if qm_type is None:
                from chemlab.util.file_system import atom_charge_dict
                atom_symbols = [atm[0] for atm in multi.tasks[0].molecule.carti]
                qm_type = np.array([atom_charge_dict[sym] for sym in atom_symbols])

            # record indices
            split_idx[f"idx_{grp}"] = np.arange(idx_offset, idx_offset + len(files))
            idx_offset += len(files)

        # ======== Merge datasets ========
        coords = np.concatenate(all_coords)
        gs_energy = np.concatenate(all_gs_energy)
        ex_energy = np.concatenate(all_ex_energy)
        ex_state_energy = np.concatenate(all_ex_state_energy)
        grad = np.concatenate(all_grad)
        force = np.concatenate(all_force)
        transmom = np.concatenate(all_transmom)
        dipolemom = np.concatenate(all_dipolemom)
        transition_density = np.concatenate(all_transition_density)

        # ======== Alignment ========
        aligned_mom, aligned_transition_density = self._align_data(
            transmom,
            transition_density,
            cfg.align_ref
        )

        # ======== Save outputs ========
        prefix = f"{out_path}{prefix}"
        np.save(prefix + "coord.npy", coords)
        np.save(prefix + "gs_energy.npy", gs_energy)
        np.save(prefix + "ex_energy.npy", ex_energy)
        np.save(prefix + "ex_state_energy.npy", ex_state_energy)
        np.save(prefix + "grad.npy", grad)
        np.save(prefix + "force.npy", force)
        np.save(prefix + "transmom.npy", aligned_mom)
        np.save(prefix + "dipolemom.npy", dipolemom)
        np.save(prefix + "transition_density.npy", transition_density)
        np.save(prefix + "aligned_td.npy", aligned_transition_density)
        np.save(prefix + "qm_type.npy", qm_type)
        np.savez(prefix + "split.npz", **split_idx)

        print("Export completed.")

        # ===== multiple splits =====
        self._save_splits(
            coords.shape[0],
            cfg.train_splits,
            cfg.val_splits,
            cfg.test_splits,
            prefix=out_path
        )

    # -------------------------------------------------------
    # Helper: alignment
    # -------------------------------------------------------
    def _align_data(self, transmom, transition_density, mode):

        ref_mom = transmom[0]
        ref_td = transition_density[0]

        if mode == "dipole":
            base_vec = ref_mom
        elif mode == "transition_density":
            base_vec = ref_td
        else:
            return np.array(transmom), np.array(transition_density)

        base_norm = np.linalg.norm(base_vec)
        if base_norm == 0:
            raise ValueError("Reference vector is zero; cannot align.")

        aligned_mom = []
        aligned_td = []

        for m, td in zip(transmom, transition_density):
            dot = np.dot(base_vec, m if mode == "dipole" else td)
            target_norm = np.linalg.norm(m if mode == "dipole" else td)

            # skip zero vector
            if target_norm == 0:
                aligned_mom.append(m)
                aligned_td.append(td)
                continue

            cos = dot / (base_norm * target_norm)
            if cos < 0:
                m = -m
                td = -td

            aligned_mom.append(m)
            aligned_td.append(td)

        return np.array(aligned_mom), np.array(aligned_td)

    # -------------------------------------------------------
    # Helper: dataset splits
    # -------------------------------------------------------
    def _save_splits(self, n_total, train_sizes, n_val, n_test, prefix="./", seed=42):

        rng = np.random.default_rng(seed)
        all_indices = np.arange(n_total)

        # fixed test set
        idx_test = rng.choice(all_indices, size=n_test, replace=False)
        remaining = np.setdiff1d(all_indices, idx_test)

        print(f"Fixed test set: {len(idx_test)}")

        for n_train in train_sizes:
            n_train = int(n_train)
            if n_train + n_val > len(remaining):
                print(f"Skipping train={n_train}: too large")
                continue

            rng_split = np.random.default_rng(seed + n_train)
            idx_train = rng_split.choice(remaining, size=n_train, replace=False)

            idx_val_candidates = np.setdiff1d(remaining, idx_train)
            idx_val = rng_split.choice(idx_val_candidates, size=n_val, replace=False)

            out_file = os.path.join(prefix, f"{n_train}_split.npz")
            np.savez(out_file, idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)

            print(f"Saved {out_file}")

