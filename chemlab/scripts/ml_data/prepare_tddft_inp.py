"""Prepare reproducible Q-Chem inputs from XYZ, AIMD and NumPy coordinates."""
import json
from pathlib import Path

import numpy as np

from chemlab.scripts.base import Script
from chemlab.config.config_loader import ConfigBase
from chemlab.util.modify_inp import single_spin_job
from chemlab.util.tddft_trajectory import read_trajectory


class PrepareTddftConfig(ConfigBase):
    section_name = "prepare_tddft"

    @classmethod
    def section_dict(cls):
        # Optional fields stay available with older, user-customized config files.
        return {"types": "auto", "seed": 42, "allow_incomplete": False,
                "input_distance_unit": "ang", **super().section_dict()}


class PrepareTddftInp(Script):
    name = "prepare_tddft_inp"
    config = PrepareTddftConfig

    def run(self, cfg):
        source = Path(cfg.file)
        coords, symbols, source_indices = read_trajectory(
            source, types=getattr(cfg, "types", "auto"),
            allow_incomplete=getattr(cfg, "allow_incomplete", False),
            input_distance_unit=getattr(cfg, "input_distance_unit", "ang"),
        )
        start = cfg.start
        if start < 0 or start >= len(coords):
            raise ValueError(f"start must be between 0 and {len(coords) - 1}")
        if cfg.mode not in ("all", "custom"):
            raise ValueError("mode must be 'all' or 'custom'")
        if cfg.dataset_size < 0:
            raise ValueError("dataset_size must be nonnegative (0 means all)")
        indices = np.arange(start, len(coords))
        seed = getattr(cfg, "seed", 42)
        if cfg.mode != "all" and 0 < cfg.dataset_size < len(indices):
            indices = np.sort(np.random.default_rng(seed).choice(
                indices, size=cfg.dataset_size, replace=False))

        out_dir = Path(cfg.out)
        if out_dir.exists() and any(out_dir.glob("train_*")):
            raise FileExistsError(f"Existing train_* files in {out_dir}; use a new output directory")
        if not Path(cfg.ref).is_file():
            raise FileNotFoundError(cfg.ref)
        out_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for number, idx in enumerate(indices):
            name = f"train_{number:04d}"
            xyz = out_dir / f"{name}.xyz"
            rows = [str(len(symbols)), f"source_frame={source_indices[idx]}"]
            rows.extend(f"{s} {x:.10f} {y:.10f} {z:.10f}"
                        for s, (x, y, z) in zip(symbols, coords[idx]))
            xyz.write_text("\n".join(rows) + "\n")
            job = single_spin_job()
            job.charge, job.spin = cfg.charge, cfg.spin
            job.ref_name, job.xyz_name = str(cfg.ref), str(xyz)
            job.generate_outputs(new_file_name=xyz.name, prefix=str(out_dir) + "/")
            records.append({"input": f"{name}.inp", "source_frame": int(source_indices[idx])})
        manifest = {"source": str(source.resolve()), "seed": seed,
                    "coordinate_unit": "angstrom", "frames": records}
        (out_dir / "frames.json").write_text(json.dumps(manifest, indent=2) + "\n")
        np.save(out_dir / "source_indices.npy", source_indices[indices])
        print(f"Prepared {len(records)} inputs in {out_dir}")
