from .base import Script
from chemlab.config.config_loader import QchemEnvConfig

class ExampleConfig(QchemEnvConfig):
    section_name = "example_config" #and then add configs to config.toml in chemlab.config.config.toml


class ExampleScript(Script):
    name = "example_script"
    config = AlignDataConfig

    def run(self, cfg):
        ref = cfg.ref
        coord = cfg.coord
        gradient = cfg.gradient
        dipole = cfg.dipole

        rotate_matrix = np.array()