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
    config = ExportNumpyConfig  # link to config section in config.toml

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
        groups = {"train": [], "val": [], "test": [],"frame":[]}
        for fn in sorted(os.listdir(path)):
            if fn.endswith(".out"):
                if fn.startswith("train"):
                    groups["train"].append(fn)
                elif fn.startswith("val"):
                    groups["val"].append(fn)
                elif fn.startswith("test"):
                    groups["test"].append(fn)
                elif fn.startswith("frame"):
                    groups["frame"].append(fn)

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

        # For all-states export
        all_multi_readers = []

        # ======== Process each subset ========
        for grp, files in groups.items():
            if not files:
                continue

            print(f"Processing {grp} ({len(files)} files)...")

            multi = qchem_out_excite_multi()
            multi.read_files(files, path=path)


            # Keep reference for all-states export
            all_multi_readers.append(multi)

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
        full_prefix = f"{out_path}{prefix}"
        np.save(full_prefix + "coord.npy", coords)
        np.save(full_prefix + "gs_energy.npy", gs_energy)
        np.save(full_prefix + "ex_energy.npy", ex_energy)
        np.save(full_prefix + "ex_state_energy.npy", ex_state_energy)
        np.save(full_prefix + "grad.npy", grad)
        np.save(full_prefix + "force.npy", force)
        np.save(full_prefix + "transmom.npy", aligned_mom)
        np.save(full_prefix + "dipolemom.npy", dipolemom)
        np.save(full_prefix + "transition_density.npy", transition_density)
        np.save(full_prefix + "aligned_td.npy", aligned_transition_density)
        np.save(full_prefix + "qm_type.npy", qm_type)
        np.savez(full_prefix + "split.npz", **split_idx)

        print("Export completed (single-state .npy files).")

        # ===== Export all states to single NPZ =====
        tddft_npz_path = f"{out_path}{prefix}tddft.npz"
        self._export_all_states_npz(
            all_multi_readers,
            tddft_npz_path,
            split_idx,
            qm_type,
            atom_symbols,
        )

        # ===== multiple splits =====
        self._save_splits(
            coords.shape[0],
            cfg.train_splits,
            cfg.val_splits,
            cfg.test_splits,
            prefix=out_path
        )

    # -------------------------------------------------------
    # Helper: Export all excited states to single NPZ
    # -------------------------------------------------------
    def _export_all_states_npz(self, multi_readers, output_file, split_idx, qm_type, atom_symbols):
        """
        Export all excited states data to a single NPZ file.

        This creates a comprehensive file containing:
        - All excited states (not just state_idx)
        - Ground state data
        - ESP charges for all states
        - Transition densities for all states
        - Gradients (where available)

        Args:
            multi_readers: List of qchem_out_excite_multi objects
            output_file: Output NPZ filename
            split_idx: Dictionary of train/val/test indices
            qm_type: Atom type array
            atom_symbols: List of atom symbols
        """
        # Collect all tasks
        all_tasks = []
        for multi in multi_readers:
            all_tasks.extend(multi.tasks)

        if not all_tasks:
            print("Warning: No tasks to export for all-states NPZ")
            return

        nframes = len(all_tasks)
        natoms = len(all_tasks[0].molecule.carti)
        n_excited = len(all_tasks[0].states) - 1  # exclude ground state

        print(f"\nExporting all-states NPZ: {nframes} frames, {natoms} atoms, {n_excited} excited states")

        # ======== Initialize arrays ========
        # Coordinates (raw, in Angstrom)
        coords = np.zeros((nframes, natoms, 3))

        # Ground state
        gs_energies = np.zeros(nframes)  # Hartree
        gs_dipoles = np.zeros((nframes, 3))  # Debye
        gs_esp_charges = np.zeros((nframes, natoms))

        # Excited states - all states
        ex_energies = np.zeros((nframes, n_excited))  # eV
        total_energies = np.zeros((nframes, n_excited))  # Hartree
        osc_strengths = np.zeros((nframes, n_excited))
        trans_moms = np.zeros((nframes, n_excited, 3))
        esp_charges_ex = np.zeros((nframes, n_excited, natoms))
        esp_trans_density = np.zeros((nframes, n_excited, natoms))
        gradients = np.full((nframes, n_excited, natoms, 3), np.nan)  # Hartree/Bohr

        # ======== Extract data ========
        for i, task in enumerate(all_tasks):
            # Coordinates
            coords[i] = np.array(task.molecule.carti)[:, 1:].astype(float)

            # Ground state (index 0)
            gs = task.states[0]
            gs_energies[i] = gs.total_energy if gs.total_energy else np.nan

            if gs.dipole_mom:
                gs_dipoles[i] = gs.dipole_mom

            if gs.esp_charges is not None:
                gs_esp_charges[i] = gs.esp_charges

            # Excited states (index 1, 2, 3, ...)
            for j, st in enumerate(task.states[1:]):
                if j >= n_excited:
                    break

                ex_energies[i, j] = st.excitation_energy if st.excitation_energy else np.nan
                total_energies[i, j] = st.total_energy if st.total_energy else np.nan
                osc_strengths[i, j] = st.osc_strength if st.osc_strength else 0.0

                if st.trans_mom:
                    trans_moms[i, j] = st.trans_mom

                if st.esp_charges is not None:
                    esp_charges_ex[i, j] = st.esp_charges

                if st.esp_transition_density is not None:
                    esp_trans_density[i, j] = st.esp_transition_density

                if st.gradient is not None:
                    gradients[i, j] = st.gradient

        # ======== Build output dict ========
        data = {
            # Coordinates and atoms
            'coords': coords,  # (nframes, natoms, 3) Angstrom
            'atom_symbols': np.array(atom_symbols, dtype='U2'),
            'qm_type': qm_type,

            # Ground state
            'gs_energies': gs_energies,  # (nframes,) Hartree
            'gs_dipoles': gs_dipoles,  # (nframes, 3) Debye
            'gs_esp_charges': gs_esp_charges,  # (nframes, natoms)

            # Excited states - ALL states
            'excitation_energies': ex_energies,  # (nframes, n_excited) eV
            'total_energies': total_energies,  # (nframes, n_excited) Hartree
            'osc_strengths': osc_strengths,  # (nframes, n_excited)
            'trans_moms': trans_moms,  # (nframes, n_excited, 3)
            'esp_charges_excited': esp_charges_ex,  # (nframes, n_excited, natoms)
            'esp_trans_density': esp_trans_density,  # (nframes, n_excited, natoms)
            'gradients': gradients,  # (nframes, n_excited, natoms, 3) Hartree/Bohr

            # Metadata
            'n_frames': nframes,
            'n_atoms': natoms,
            'n_excited': n_excited,
            'state_indices': np.arange(1, n_excited + 1),
        }

        # Add split indices
        data.update(split_idx)

        # ======== Save ========
        np.savez(output_file, **data)

        print(f"Saved all-states NPZ: {output_file}")
        print(f"  coords: {coords.shape}")
        print(f"  gs_energies: {gs_energies.shape}")
        print(f"  excitation_energies: {ex_energies.shape}")
        print(f"  osc_strengths: {osc_strengths.shape}")
        print(f"  trans_moms: {trans_moms.shape}")
        print(f"  esp_charges_excited: {esp_charges_ex.shape}")
        print(f"  esp_trans_density: {esp_trans_density.shape}")
        print(f"  gradients: {gradients.shape}")

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