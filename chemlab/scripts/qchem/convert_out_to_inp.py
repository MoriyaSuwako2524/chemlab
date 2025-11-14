from chemlab.util.modify_inp import conver_opt_out_to_inp
from chemlab.scripts.base import Script
from chemlab.config.config_loader import ConfigBase

class ConvertOutToInpConfig(ConfigBase):
    section_name = "convert_out_to_inp"

class ConvertOutToInp(Script):
    name = "convert_out_to_inp"

    # 这个 script 不使用 config
    config = None

    def run(self, filename):
        ref = "ref.in"
        inp = conver_opt_out_to_inp()
        inp.ref_name = ref
        inp.convert(new_out_file_name=filename)


