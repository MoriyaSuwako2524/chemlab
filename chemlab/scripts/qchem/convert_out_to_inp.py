from chemlab.util.modify_inp import conver_opt_out_to_inp
from chemlab.scripts.base import Script
from chemlab.config.config_loader import ConfigBase

class ConvertOutToInpConfig(ConfigBase):
    section_name = "convert_out_to_inp"

class ConvertOutToInp(Script):
    name = "convert_out_to_inp"

    config = ConvertOutToInpConfig

    def run(self, cfg):
        ref = cfg.ref
        inp = conver_opt_out_to_inp()
        inp.ref_name = ref
        inp.convert(new_out_file_name=cfg.file)


