from file_system import molecule,qchem_file,qchem_out_opt,multiple_qchem_jobs,qchem_out_multi
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

class qchem_out_aimd_multi(qchem_out_multi):
    """
    管理多个 AIMD 输出文件
    """
    def __init__(self):
        super().__init__()

    def read_files(self, filenames):
        from file_system import qchem_out_aimd
        super().read_files(filenames, qchem_out_aimd)

    @property
    def total_steps(self):
        return sum(task.aimd_steps for task in self.tasks)

    def merge_trajectories(self):
        traj = []
        for task in self.tasks:
            traj.extend(task.aimd_geoms)
        return traj

    def get_all_energies(self):
        return [g.energy for g in self.merge_trajectories() if hasattr(g, "energy")]

    def summary(self):
        print(f"共读取 {self.ntasks} 个 AIMD 输出文件, 总步数={self.total_steps}")
        for i, task in enumerate(self.tasks):
            last_ene = task.aimd_geoms[-1].energy if task.aimd_geoms else None
            print(f"  [{i}] 文件={task.filename}, 步数={task.aimd_steps}, 最终能量={last_ene}")
    def export_numpy(self, prefix=""):

        import numpy as np

        traj = self.merge_trajectories()
        if not traj:
            raise ValueError("没有 AIMD 数据可导出")

        natoms = len(traj[0].carti)
        nframes = len(traj)

        coords = np.zeros((nframes, natoms, 3), dtype=float)
        energies = np.zeros((nframes,), dtype=float)
        grads = np.zeros((nframes, 3, natoms), dtype=float)
        qm_types = np.zeros((natoms,), dtype=int)

        for i, g in enumerate(traj):
            coords[i] = np.array(g.carti)[:, 1:].astype(float)
            energies[i] = g.energy if hasattr(g, "energy") else np.nan
            if hasattr(g, "grad"):
                grads[i] = g.grad
            else:
                grads[i] = 0.0


        atom_symbols = [atm[0] for atm in traj[0].carti]
        from file_system import atom_charge_dict
        qm_types = np.array([atom_charge_dict[sym] for sym in atom_symbols])

        np.save(prefix + "coord.npy", coords)
        np.save(prefix + "energy.npy", energies)
        np.save(prefix + "grad.npy", grads)
        np.save(prefix + "type.npy", qm_types)

        return coords, energies, grads, qm_types


multi = qchem_out_aimd_multi()
multi.read_files([
    "./examples/aimd_bodipy_nvt_1.out",
    "./examples/aimd_bodipy_nvt_2.out"
])

multi.export_numpy(prefix="./full_")







