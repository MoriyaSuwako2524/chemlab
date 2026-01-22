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
            split_file = os.path.join(dataset_dir, f"{prefix}split.npz")
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