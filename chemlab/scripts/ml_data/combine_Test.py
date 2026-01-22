import os
import numpy as np
from pathlib import Path
from chemlab.scripts.base import Script
from chemlab.config import ConfigBase


class MergeTestDataConfig(ConfigBase):
    section_name = "merge_test_data"


class MergeTestData(Script):
    """
    Merge test data from multiple exported numpy datasets.

    This script:
    1. Reads multiple datasets (each with split.npz)
    2. Extracts the test portion from each dataset
    3. Merges all test data into a single dataset
    4. Optionally merges tddft.npz files
    """
    name = "merge_test_data"
    config = MergeTestDataConfig

    def run(self, cfg):
        """Main execution"""

        print("=" * 60)
        print("Merging Test Datasets")
        print("=" * 60)

        dataset_dirs = cfg.dataset_dirs
        prefix = cfg.prefix
        out_dir = cfg.out_dir
        out_prefix = cfg.out_prefix
        merge_tddft = getattr(cfg, 'merge_tddft', True)  # 默认合并 tddft

        if not dataset_dirs:
            raise ValueError("No dataset directories specified!")

        # Create output directory
        os.makedirs(out_dir, exist_ok=True)

        # Data fields to merge
        data_fields = [
            "coord",
            "gs_energy",
            "ex_energy",
            "ex_state_energy",
            "grad",
            "force",
            "transmom",
            "dipolemom",
            "transition_density",
            "aligned_td"
        ]

        # Accumulators
        merged_data = {field: [] for field in data_fields}
        qm_type = None
        total_frames = 0

        # Process each dataset
        for i, dataset_dir in enumerate(dataset_dirs):
            print(f"\n[{i + 1}/{len(dataset_dirs)}] Processing: {dataset_dir}")

            # Load split indices
            split_file = os.path.join(dataset_dir, f"1000_split.npz")
            if not os.path.exists(split_file):
                print(f"  WARNING: split file not found: {split_file}")
                continue

            split_data = np.load(split_file)
            idx_test = split_data['idx_test']
            print(f"  Found {len(idx_test)} test samples")

            # Load and extract test data for each field
            for field in data_fields:
                filepath = os.path.join(dataset_dir, f"{prefix}{field}.npy")

                if not os.path.exists(filepath):
                    print(f"  WARNING: {field}.npy not found, skipping")
                    continue

                data = np.load(filepath)
                test_data = data[idx_test]
                merged_data[field].append(test_data)

                print(f"  Loaded {field}: {test_data.shape}")

            # Load qm_type (only once, should be the same for all datasets)
            if qm_type is None:
                qm_type_file = os.path.join(dataset_dir, f"{prefix}qm_type.npy")
                if os.path.exists(qm_type_file):
                    qm_type = np.load(qm_type_file)
                    print(f"  Loaded qm_type: {qm_type.shape}")

            total_frames += len(idx_test)

        # Merge all data
        print("\n" + "=" * 60)
        print("Merging data...")
        print("=" * 60)

        final_data = {}
        for field in data_fields:
            if merged_data[field]:
                final_data[field] = np.concatenate(merged_data[field], axis=0)
                print(f"  {field}: {final_data[field].shape}")
            else:
                print(f"  WARNING: No data for {field}")

        # Save merged data
        print("\n" + "=" * 60)
        print("Saving merged test dataset...")
        print("=" * 60)

        full_prefix = os.path.join(out_dir, out_prefix)

        for field, data in final_data.items():
            outfile = f"{full_prefix}{field}.npy"
            np.save(outfile, data)
            print(f"  Saved: {outfile}")

        # Save qm_type
        if qm_type is not None:
            np.save(f"{full_prefix}qm_type.npy", qm_type)
            print(f"  Saved: {full_prefix}qm_type.npy")

        # Create split file (all data is test)
        n_total = total_frames
        split_dict = {
            'idx_train': np.array([], dtype=int),
            'idx_val': np.array([], dtype=int),
            'idx_test': np.arange(n_total)
        }
        np.savez(f"{full_prefix}split.npz", **split_dict)
        print(f"  Saved: {full_prefix}split.npz")

        print("\n" + "=" * 60)
        print(f"Merge completed! Total test frames: {total_frames}")
        print(f"Output directory: {out_dir}")
        print("=" * 60)

        # ===== Merge TDDFT files if requested =====
        if merge_tddft:
            self._merge_tddft_files(dataset_dirs, prefix, out_dir, out_prefix)

    def _merge_tddft_files(self, dataset_dirs, prefix, out_dir, out_prefix):
        """Merge TDDFT NPZ files from multiple datasets"""

        print("\n" + "=" * 60)
        print("Merging TDDFT Test Datasets")
        print("=" * 60)

        # Fields that are per-frame and need to be merged
        frame_fields = [
            "coords",
            "gs_energies",
            "gs_dipoles",
            "gs_esp_charges",
            "excitation_energies",
            "total_energies",
            "osc_strengths",
            "trans_moms",
            "esp_charges_excited",
            "esp_trans_density",
            "gradients"
        ]

        # Accumulators
        merged_data = {field: [] for field in frame_fields}

        # Reference data (should be same across all files)
        ref_atom_symbols = None
        ref_qm_type = None
        ref_n_atoms = None
        ref_n_excited = None
        ref_state_indices = None

        total_frames = 0

        # Process each TDDFT file
        for i, dataset_dir in enumerate(dataset_dirs):
            tddft_file = os.path.join(dataset_dir, f"{prefix}tddft.npz")

            print(f"\n[{i + 1}/{len(dataset_dirs)}] Processing: {tddft_file}")

            if not os.path.exists(tddft_file):
                print(f"  WARNING: File not found, skipping")
                continue

            # Load data
            data = np.load(tddft_file)

            # Extract test indices
            if 'idx_test' not in data:
                print(f"  WARNING: No idx_test found")
                continue

            idx_test = data['idx_test']
            print(f"  Found {len(idx_test)} test samples")

            # Verify consistency of reference data
            if ref_atom_symbols is None:
                ref_atom_symbols = data['atom_symbols']
                ref_qm_type = data['qm_type']
                ref_n_atoms = int(data['n_atoms'])
                ref_n_excited = int(data['n_excited'])
                ref_state_indices = data['state_indices']
                print(f"  Reference: {ref_n_atoms} atoms, {ref_n_excited} excited states")
            else:
                # Check consistency
                if not np.array_equal(data['atom_symbols'], ref_atom_symbols):
                    raise ValueError(f"Inconsistent atom_symbols in {tddft_file}")
                if not np.array_equal(data['qm_type'], ref_qm_type):
                    raise ValueError(f"Inconsistent qm_type in {tddft_file}")
                if int(data['n_atoms']) != ref_n_atoms:
                    raise ValueError(f"Inconsistent n_atoms in {tddft_file}")
                if int(data['n_excited']) != ref_n_excited:
                    raise ValueError(f"Inconsistent n_excited in {tddft_file}")

            # Extract test data for each field
            for field in frame_fields:
                if field not in data:
                    print(f"  WARNING: {field} not found, skipping")
                    continue

                full_data = data[field]
                test_data = full_data[idx_test]
                merged_data[field].append(test_data)

                print(f"  Loaded {field}: {test_data.shape}")

            total_frames += len(idx_test)
            data.close()

        if total_frames == 0:
            print("  WARNING: No TDDFT data to merge")
            return

        # Merge all data
        print("\n" + "=" * 60)
        print("Merging TDDFT data...")
        print("=" * 60)

        final_data = {}
        for field in frame_fields:
            if merged_data[field]:
                final_data[field] = np.concatenate(merged_data[field], axis=0)
                print(f"  {field}: {final_data[field].shape}")
            else:
                print(f"  WARNING: No data for {field}")

        # Add reference data
        final_data['atom_symbols'] = ref_atom_symbols
        final_data['qm_type'] = ref_qm_type
        final_data['n_atoms'] = ref_n_atoms
        final_data['n_excited'] = ref_n_excited
        final_data['n_frames'] = total_frames
        final_data['state_indices'] = ref_state_indices

        # Create split (all data is test)
        final_data['idx_train'] = np.array([], dtype=int)
        final_data['idx_val'] = np.array([], dtype=int)
        final_data['idx_test'] = np.arange(total_frames)

        # Save merged data
        print("\n" + "=" * 60)
        print("Saving merged TDDFT dataset...")
        print("=" * 60)

        out_file = os.path.join(out_dir, f"{out_prefix}tddft.npz")
        np.savez(out_file, **final_data)
        print(f"  Saved: {out_file}")

        print("\n" + "=" * 60)
        print(f"TDDFT merge completed!")
        print(f"  Total test frames: {total_frames}")
        print(f"  Atoms: {ref_n_atoms}")
        print(f"  Excited states: {ref_n_excited}")
        print("=" * 60)