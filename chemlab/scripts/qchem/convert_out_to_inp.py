from chemlab.util.modify_inp import qchem_out_opt, single_spin_job
from chemlab.scripts.base import Script
from chemlab.config.config_loader import ConfigBase

class ConvertOutToInpConfig(ConfigBase):
    section_name = "convert_out_to_inp"

class ConvertOutToInp(Script):
    """
    Convert Q-Chem optimization OUT → single-point INP
    """

    name = "convert_out_to_inp"
    config = ConvertOutToInpConfig

    def run(self, cfg):
        """
        cfg.file : OUT file name
        cfg.ref  : reference file (from config)
        """

        out_file = cfg.file
        ref = cfg.ref

        if not out_file:
            raise ValueError(
                "[convert_out_to_inp] Missing required argument: --file"
            )

        print(f"[convert_out_to_inp] Reading OUT: {out_file}")
        print(f"[convert_out_to_inp] Using ref:  {ref}")

        # ---- Read optimization output ----
        opt = qchem_out_opt()
        opt.read_file(out_file)

        # ---- Determine output INP filename ----
        inp_filename = out_file.replace(".out", ".inp")

        # ---- Construct new input ----
        inp = single_spin_job()
        inp.spin = opt.spin         # from OUT
        inp.charge = opt.charge     # from OUT
        inp._xyz.carti = opt.final_geom
        inp.ref_name = ref

        print(f"[convert_out_to_inp] Writing INP: {inp_filename}")
        inp.generate_outputs(inp_filename)

        print("[convert_out_to_inp] Done.")



