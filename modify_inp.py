from file_system import molecule,qchem_file,qchem_out_opt,multiple_qchem_jobs
from file_system import SPIN_REF
class check_list(object):
    def __init__(self):
        self.molecule_check = True
        self.rem_check = True
        self.opt_check = False
        self.pcm_check = False

def ex_match_check(fjob,tjob):
    if fjob.molecule_check:
        tjob.molecule_check = True
    else:
        tjob.molecule_check = False
    if fjob.rem_check:
        tjob.rem_check = True
    else:
        tjob.rem_check = False
    if fjob.opt_check:
        tjob.opt_check = True
    else:
        tjob.opt_check = False
    if fjob.pcm_check:
        tjob.pcm_check = True
    else:
        tjob.pcm_check = False
    if fjob.opt_check:
        tjob.opt_check = True
    else:
        tjob.opt_check = False




class conver_opt_out_to_inp(object):
    def __init__(self):
        self.out_job_name = ""
        self.inp_file_name = ""
        self.ref_file_name = ""
    def convert(self,new_out_file_name="",new_inp_file_name=""):
        opt = qchem_out_opt()
        if new_out_file_name == "":
            filename = self.out_job_name
        else:
            filename = new_out_file_name
        opt.read_file(filename)
        if new_out_file_name == "":
            inp_filename = self.inp_file_name
        else:
            inp_filename = new_inp_file_name
        inp = qchem_file()
        inp.molecule.check = True
        inp.read_from_file(self.ref_file_name)
        molecule_carti = opt.final_geom
        inp.molecule.carti = molecule_carti
        inp.generate_inp(inp_filename)


class base_job(object):
    def __init__(self):
        self.charge = -100
        self.ref_name = ""
        self.xyz_name = ""
        self.xyz_check = True
        self.check_list = check_list()
        self.spin = 1

    @property
    def ref(self):
        ref = multiple_qchem_jobs()
        ex_match_check(self.check_list, ref)
        ref.read_from_file(self.ref_name)
        return ref

    @property
    def xyz(self):
        xyz = molecule()
        xyz.check = True
        xyz.read_xyz(self.xyz_name)
        return xyz

    def _prepare_ref(self):
        """Prepare reference input file with updated geometry/charge."""
        ref = self.ref
        xyz = self.xyz
        ref.jobs[0].molecule.carti = xyz.carti
        ref.jobs[0].molecule.read = False
        if self.charge > -50:
            ref.jobs[0].molecule.charge = self.charge
        return ref

    def generate_outputs(self, new_file_name=""):
        """To be implemented in subclasses"""
        raise NotImplementedError


class multi_spin_job(base_job):
    def __init__(self):
        super().__init__()
        self.spins = []

    def generate_outputs(self, new_file_name=""):
        filename = self.xyz_name if new_file_name == "" else new_file_name
        ref = self._prepare_ref()

        for spin in self.spins:
            ref.jobs[0].molecule.multistate = spin
            file_name = f"{filename[:-4]}_{SPIN_REF[spin]}.inp"
            ref.generate_inp(file_name)


class single_spin_job(base_job):
    def __init__(self):
        super().__init__()
        self.spin = 1


    def generate_outputs(self, new_file_name=""):
        filename = self.xyz_name if new_file_name == "" else new_file_name
        ref = self._prepare_ref()

        ref.jobs[0].molecule.multistate = self.spin
        file_name = f"{filename[:-4]}.inp"
        ref.generate_inp(file_name)

class multiple_out_jobs(object):
    def __init__(self):
        self.path = "./"
        self.outs = []
class aimd_outs(multiple_out_jobs):
    def __init__(self):
        super().__init__()

from file_system import qchem_out_aimd
out = qchem_out_aimd()
out.read_file(filename="./examples/aimd_bodipy_nvt_1.out")   # 读 AIMD 的 Q-Chem 输出文件




print("总步数:", out.aimd_steps)
print("前 5 步能量:", out.get_energies()[:5])
print("第 1 步几何结构:", out.aimd_geoms[0].carti)
print("第 1 步梯度 shape:", out.aimd_geoms[0].grad.shape if hasattr(out.aimd_geoms[0], "grad") else None)
print("第 1 步温度 (K):", out.aimd_geoms[0].temperature_K)

