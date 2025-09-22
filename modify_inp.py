from file_system import multiple_qchem_jobs,molecule,qchem_file,qchem_out_file
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


class custum_job(multiple_qchem_jobs):
    def __init__(self):
        super(custum_job, self).__init__()
        self.single_ref = qchem_file()
        self.multi_ref = multiple_qchem_jobs()
    def read_single_job_ref(self,ref_name):
        self.match_check(self.single_ref)
        self.single_ref.read_from_file(ref_name)


class opt_fre(multiple_qchem_jobs):
    def __init__(self):
        super(opt_fre, self).__init__()


class conver_out_to_inp(object):
    def __init__(self):
        self.out_job_name = ""
        self.inp_file_name = ""
        self.ref_file_name = ""
    def convert(self,new_out_file_name="",new_inp_file_name=""):
        opt = qchem_out_file()
        if new_out_file_name == "":
            filename = self.out_job_name
        else:
            filename = new_out_file_name
        opt.read_opt_from_file(filename)
        if new_out_file_name == "":
            inp_filename = self.inp_file_name
        else:
            inp_filename = new_inp_file_name
        inp = qchem_file()
        inp.molecule.check = True
        inp.read_from_file(self.ref_file_name)
        molecule_carti = opt.return_final_molecule_carti()
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