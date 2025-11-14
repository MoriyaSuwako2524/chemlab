from chemlab.config.config_loader import ConfigBase


from chemlab.scripts.base import Script
from chemlab.util.ml_data import MLData


class TrajSplitConfig(ConfigBase):
    section_name = "traj_split"
class TrajSplit(Script):
    """
    Split a molecular trajectory dataset into train/val/test
    and export xyz files for each split.
    """

    name = "traj_split"
    config = TrajSplitConfig

    def run(self, cfg):

        print(f"[traj_split] Loading trajectory from: {cfg.input_dir}")

        data = MLData(cfg.input_dir, type="xyz")

        print(f"[traj_split] Creating split: "
              f"train={cfg.train}, val={cfg.val}, test={cfg.test}")

        data.save_split(cfg.train, cfg.val, cfg.test)

        print(f"[traj_split] Exporting xyz from split: {cfg.split_file}")

        data.export_xyz_from_split(cfg.split_file, outdir=cfg.outdir)

        print("[traj_split] Done.")
