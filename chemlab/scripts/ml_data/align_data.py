import numpy as np
from chemlab.scripts.base import Script
from chemlab.config.config_loader import ConfigBase


class AlignDataConfig(ConfigBase):
    section_name = "align_data"


class AlignData(Script):
    name = "align_data"
    config = AlignDataConfig

    def run(self, cfg):
        ref = cfg.ref
        coord = cfg.coord
        gradient = cfg.gradient
        dipole = cfg.dipole

        rotate_matrix = np.array()