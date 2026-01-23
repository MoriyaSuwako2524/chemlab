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
    4. Merges tddft.npz files using the same split
    5. Tracks source file information for each frame
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
        merge_tddft = getattr(cfg, 'merge_tddft', True)
        source_field_name = getattr(cfg, 'source_field_name', 'source_files')  # 可配置的字段名

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

        # TDDFT fields to merge
        tddft_fields = [
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
        merged_data = {field: [] for field in data_fields}
        merged_tddft = {field: [] for field in tddft_fields}
        merged_source_files = []  # 新增：存储源文件信息
        merged_dataset_indices = []  # 新增：存储数据集索引

        qm_type = None
        total_frames = 0

        # TDDFT reference data
        ref_atom_symbols = None
        ref_qm_type = None
        ref_n_atoms = None
        ref_n_excited = None
        ref_state_indices = None

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

            # ===== Load source file information =====
            source_file_path = os.path.join(dataset_dir, f"{prefix}{source_field_name}.npy")

            if os.path.exists(source_file_path):
                source_files = np.load(source_file_path, allow_pickle=True)
                test_source_files = source_files[idx_test]
                merged_source_files.append(test_source_files)
                print(f"  Loaded source files: {len(test_source_files)} entries")
            else:
                # 如果没有source_files字段，创建默认值
                print(f"  WARNING: {source_field_name}.npy not found, using dataset path as identifier")
                default_sources = np.array([f"{Path(dataset_dir).name}_frame_{j}" for j in idx_test])
                merged_source_files.append(default_sources)

            # 记录每个帧来自哪个数据集
            dataset_indices = np.full(len(idx_test), i, dtype=int)
            merged_dataset_indices.append(dataset_indices)

            # ===== Process .npy files =====
            for field in data_fields:
                filepath = os.path.join(dataset_dir, f"{prefix}{field}.npy")

                if not os.path.exists(filepath):
                    print(f"  WARNING: {field}.npy not found, skipping")
                    continue

                data = np.load(filepath)
                test_data = data[idx_test]
                merged_data[field].append(test_data)

                print(f"  Loaded {field}: {test_data.shape}")

            # Load qm_type (only once)
            if qm_type is None:
                qm_type_file = os.path.join(dataset_dir, f"{prefix}qm_type.npy")
                if os.path.exists(qm_type_file):
                    qm_type = np.load(qm_type_file)
                    print(f"  Loaded qm_type: {qm_type.shape}")

            # ===== Process tddft.npz file =====
            if merge_tddft:
                tddft_file = os.path.join(dataset_dir, f"{prefix}tddft.npz")

                if os.path.exists(tddft_file):
                    print(f"  Loading TDDFT data...")
                    tddft_data = np.load(tddft_file)

                    # Verify consistency of reference data
                    if ref_atom_symbols is None:
                        ref_atom_symbols = tddft_data['atom_symbols']
                        ref_qm_type = tddft_data['qm_type']
                        ref_n_atoms = int(tddft_data['n_atoms'])
                        ref_n_excited = int(tddft_data['n_excited'])
                        ref_state_indices = tddft_data['state_indices']
                        print(f"  TDDFT reference: {ref_n_atoms} atoms, {ref_n_excited} excited states")

                    # Extract test data using the same idx_test
                    for field in tddft_fields:
                        if field not in tddft_data:
                            print(f"  WARNING: {field} not in tddft.npz, skipping")
                            continue

                        full_data = tddft_data[field]
                        test_data = full_data[idx_test]
                        merged_tddft[field].append(test_data)

                    tddft_data.close()
                    print(f"  Loaded TDDFT data")
                else:
                    print(f"  WARNING: {tddft_file} not found")

            total_frames += len(idx_test)

        # ===== Merge all .npy data =====
        print("\n" + "=" * 60)
        print("Merging .npy data...")
        print("=" * 60)

        final_data = {}
        for field in data_fields:
            if merged_data[field]:
                final_data[field] = np.concatenate(merged_data[field], axis=0)
                print(f"  {field}: {final_data[field].shape}")
            else:
                print(f"  WARNING: No data for {field}")

        # Merge source file information
        if merged_source_files:
            final_source_files = np.concatenate(merged_source_files, axis=0)
            final_dataset_indices = np.concatenate(merged_dataset_indices, axis=0)
            print(f"  source_files: {final_source_files.shape}")
            print(f"  dataset_indices: {final_dataset_indices.shape}")

        # Save merged .npy data
        print("\n" + "=" * 60)
        print("Saving merged test dataset...")
        print("=" * 60)

        full_prefix = os.path.join(out_dir, out_prefix)

        for field, data in final_data.items():
            outfile = f"{full_prefix}{field}.npy"
            np.save(outfile, data)
            print(f"  Saved: {outfile}")

        # Save source file information
        if merged_source_files:
            np.save(f"{full_prefix}source_files.npy", final_source_files)
            print(f"  Saved: {full_prefix}source_files.npy")

            np.save(f"{full_prefix}dataset_indices.npy", final_dataset_indices)
            print(f"  Saved: {full_prefix}dataset_indices.npy")

            # 同时保存一个文本文件方便查看
            with open(f"{full_prefix}source_mapping.txt", 'w') as f:
                f.write("Frame_Index\tDataset_Index\tSource_File\n")
                for idx, (ds_idx, src_file) in enumerate(zip(final_dataset_indices, final_source_files)):
                    f.write(f"{idx}\t{ds_idx}\t{src_file}\n")
            print(f"  Saved: {full_prefix}source_mapping.txt")

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

        # ===== Merge TDDFT data =====
        if merge_tddft and any(merged_tddft.values()):
            print("\n" + "=" * 60)
            print("Merging TDDFT data...")
            print("=" * 60)

            final_tddft = {}
            for field in tddft_fields:
                if merged_tddft[field]:
                    final_tddft[field] = np.concatenate(merged_tddft[field], axis=0)
                    print(f"  {field}: {final_tddft[field].shape}")

            # Add reference data
            final_tddft['atom_symbols'] = ref_atom_symbols
            final_tddft['qm_type'] = ref_qm_type
            final_tddft['n_atoms'] = ref_n_atoms
            final_tddft['n_excited'] = ref_n_excited
            final_tddft['n_frames'] = total_frames
            final_tddft['state_indices'] = ref_state_indices

            # Create split (all data is test)
            final_tddft['idx_train'] = np.array([], dtype=int)
            final_tddft['idx_val'] = np.array([], dtype=int)
            final_tddft['idx_test'] = np.arange(total_frames)

            # Save
            out_file = os.path.join(out_dir, f"{out_prefix}tddft.npz")
            np.savez(out_file, **final_tddft)
            print(f"\n  Saved: {out_file}")

        print("\n" + "=" * 60)
        print(f"Merge completed! Total test frames: {total_frames}")
        print(f"Output directory: {out_dir}")
        print("=" * 60)