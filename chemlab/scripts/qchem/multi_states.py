from chemlab.scripts.base import Script
from chemlab.util.modify_inp import single_spin_job
from chemlab.config.config_loader import ConfigBase


class MultiStatesConfig(ConfigBase):
    """
    Config for multi_states script.
    """
    section_name = "multi_states"   # You must add this section in config.toml


class MultiStates(Script):
    """
    Generate multi-state input files by wrapping single_spin_job().
    """
    name = "multi_states"
    config = MultiStatesConfig

    def run(self, cfg):
        job = single_spin_job()
        job.spins = cfg.spins
        job.charge = cfg.charge
        job.ref_name = cfg.ref
        job.xyz_name = cfg.file   # from CLI --file argument

        print(f"[multi_states] Processing {cfg.file} ...")
        job.generate_outputs()
        print("[multi_states] Done.")