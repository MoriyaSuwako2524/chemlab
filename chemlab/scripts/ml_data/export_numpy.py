import os
from pathlib import Path
import warnings
import numpy as np
from chemlab.util.modify_inp import qchem_out_excite_multi
from chemlab.scripts.base import Script
from chemlab.config import ExportNumpyConfig


class ExportNumpy(Script):
    """
    Export TDDFT dataset into numpy arrays.
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
        source_files = []
        reference_symbols = None
        reference_states = None

        # For all-states export
        all_multi_readers = []

        for grp, files in groups.items():
            if not files:
                continue

            print(f"Processing {grp} ({len(files)} files)...")

            multi = qchem_out_excite_multi()
            multi.read_files(files, path=path)

            if not multi.tasks:
                continue
            for task in multi.tasks:
                symbols = [atom[0] for atom in task.molecule.carti]
                states = [state.state_idx for state in task.states]
                if not symbols or len(states) < 2 or state_idx not in states:
                    raise ValueError(f"Missing geometry or requested excited state: {task.filename}")
                if reference_symbols is None:
                    reference_symbols, reference_states = symbols, states
                if symbols != reference_symbols or states != reference_states:
                    raise ValueError(f"Inconsistent atoms or state indices: {task.filename}")
                source_files.append(str(Path(task.filename).resolve()))


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
            count = len(multi.tasks)
            split_idx[f"idx_{grp}"] = np.arange(idx_offset, idx_offset + count)
            idx_offset += count

        # ======== Merge datasets ========
        if not all_coords:
            raise ValueError("No valid TDDFT outputs found (expected train/val/test/frame*.out)")
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
        Path(out_path).mkdir(parents=True, exist_ok=True)
        full_prefix = str(Path(out_path) / prefix)
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
        np.save(full_prefix + "source_files.npy", np.asarray(source_files, dtype=str))
        np.savez(full_prefix + "split_uma.npz", **split_idx)

        print("Export completed (single-state .npy files).")

        tddft_npz_path = full_prefix + "tddft.npz"
        self._export_all_states_npz(
            all_multi_readers,
            tddft_npz_path,
            split_idx,
            qm_type,
            atom_symbols,
        )

        self._save_splits(
            coords.shape[0],
            cfg.train_splits,
            cfg.val_splits,
            cfg.test_splits,
            prefix=out_path
        )

    def _export_all_states_npz(self, multi_readers, output_file, split_idx, qm_type, atom_symbols):

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
        gs_dipoles = np.full((nframes, 3), np.nan)  # Debye
        gs_esp_charges = np.full((nframes, natoms), np.nan)

        # Excited states - all states
        ex_energies = np.full((nframes, n_excited), np.nan)  # eV
        total_energies = np.full((nframes, n_excited), np.nan)  # Hartree
        osc_strengths = np.full((nframes, n_excited), np.nan)
        trans_moms = np.full((nframes, n_excited, 3), np.nan)
        esp_charges_ex = np.full((nframes, n_excited, natoms), np.nan)
        esp_trans_density = np.full((nframes, n_excited, natoms), np.nan)
        gradients = np.full((nframes, n_excited, natoms, 3), np.nan)  # Hartree/Bohr

        # ======== Extract data ========
        for i, task in enumerate(all_tasks):
            # Coordinates
            coords[i] = np.array(task.molecule.carti)[:, 1:].astype(float)

            # Ground state (index 0)
            gs = task.states[0]
            gs_energies[i] = gs.total_energy if gs.total_energy is not None else np.nan

            if gs.dipole_mom is not None:
                gs_dipoles[i] = gs.dipole_mom

            if gs.esp_charges is not None:
                gs_esp_charges[i] = gs.esp_charges

            # Excited states (index 1, 2, 3, ...)
            for j, st in enumerate(task.states[1:]):
                if j >= n_excited:
                    break

                ex_energies[i, j] = st.excitation_energy if st.excitation_energy is not None else np.nan
                total_energies[i, j] = st.total_energy if st.total_energy is not None else np.nan
                osc_strengths[i, j] = st.osc_strength if st.osc_strength is not None else np.nan

                if st.trans_mom is not None:
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
            'state_indices': np.array([st.state_idx for st in all_tasks[0].states[1:]]),
            'source_files': np.array([str(Path(t.filename).resolve()) for t in all_tasks]),
            'schema_version': 2,
            'coordinate_unit': 'angstrom',
            'excitation_energy_unit': 'eV',
            'total_energy_unit': 'hartree',
            'gs_dipole_unit': 'Debye',
            'transition_dipole_unit': 'e*bohr',
            'gradient_unit': 'hartree/bohr',
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

        if mode == "dipole":
            vectors = np.asarray(transmom)
        elif mode == "transition_density":
            vectors = np.asarray(transition_density)
        else:
            return np.array(transmom), np.array(transition_density)

        valid = np.isfinite(vectors).all(axis=1) & (np.linalg.norm(vectors, axis=1) > 0)
        if not valid.any():
            warnings.warn("No finite nonzero reference vector; phase alignment skipped", UserWarning)
            return np.array(transmom), np.array(transition_density)
        base_vec = vectors[np.flatnonzero(valid)[0]]

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

        if not train_sizes:
            return
        if n_val < 0 or n_test < 0 or any(int(n) < 0 for n in train_sizes):
            raise ValueError("Split sizes must be nonnegative")
        if n_test + n_val > n_total:
            warnings.warn("Requested validation/test sets exceed valid frames; additional splits skipped", UserWarning)
            return

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
