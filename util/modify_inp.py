from .file_system import molecule,qchem_file,qchem_out_opt,multiple_qchem_jobs,qchem_out_multi
from .file_system import SPIN_REF
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
        from .file_system import qchem_out_aimd
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

    def export_numpy(
        self,
        prefix="",
        energy_unit="hartree",
        distance_unit="ang",
        force_unit=("hartree", "bohr"),
        save_as_force=True,
    ):
        """
        Args:
            prefix (str):
            energy_unit (str): "hartree" / "kcal" / "ev" / "kj"
            distance_unit (str): "ang" / "bohr" / "nm"
            force_unit (tuple): (energy_unit, distance_unit)，如 ("hartree","bohr")
            save_as_force (bool): True=-∇E，False=∇E
        """
        import numpy as np
        from .file_system import ENERGY, DISTANCE, FORCE, GRADIENT, atom_charge_dict

        traj = self.merge_trajectories()
        if not traj:
            raise ValueError("没有 AIMD 数据可导出")

        natoms = len(traj[0].carti)
        nframes = len(traj)

        coords = np.zeros((nframes, natoms, 3), dtype=float)
        energies = np.zeros((nframes,), dtype=float)
        grads = np.zeros((nframes, natoms, 3), dtype=float)


        for i, g in enumerate(traj):
            coords[i] = np.array(g.carti)[:, 1:].astype(float)
            energies[i] = g.energy if hasattr(g, "energy") else np.nan
            if hasattr(g, "grad"):
                grads[i] = g.grad.T
            else:
                grads[i] = 0.0


        energies = ENERGY(energies, "hartree").convert_to(energy_unit)
        coords = DISTANCE(coords, "ang").convert_to(distance_unit)
        if save_as_force:
            grads = FORCE(grads, energy_unit="hartree", distance_unit="bohr").convert_to(
                {"energy": (force_unit[0], 1), "distance": (force_unit[1], -1)}
            )
        else:
            grads = GRADIENT(grads, energy_unit="hartree", distance_unit="bohr").convert_to(
                {"energy": (force_unit[0], 1), "distance": (force_unit[1], -1)}
            )


        atom_symbols = [atm[0] for atm in traj[0].carti]
        qm_types = np.array([atom_charge_dict[sym] for sym in atom_symbols])


        np.save(prefix + "coord.npy", coords)
        np.save(prefix + "energy.npy", energies)
        np.save(prefix + "grad.npy", grads)
        np.save(prefix + "type.npy", qm_types)

        return coords, energies, grads, qm_types

class qchem_out_excite_multi(qchem_out_multi):
    """
    管理多个激发态输出文件
    """
    def __init__(self):
        super().__init__()

    def read_files(self, filenames, path=""):
        from .file_system import qchem_out_excite
        import os
        fullpaths = [os.path.join(path, fn) if path else fn for fn in filenames]
        super().read_files(fullpaths, qchem_out_excite)

    def export_numpy(
            self,
            state_idx=1,
            prefix="",
            energy_unit="hartree",
            distance_unit="ang",
            grad_unit=("hartree", "bohr"),
            force_unit=("hartree", "bohr"),
            transmom_unit="au",  # "au" = e·bohr; "Debye" 需外部转
    ):

        import numpy as np
        from .file_system import ENERGY, DISTANCE, GRADIENT, FORCE, atom_charge_dict

        if not self.tasks:
            raise ValueError("没有读取任何激发态输出文件")

        natoms = len(self.tasks[0].molecule.carti)
        nframes = len(self.tasks)

        coords = np.zeros((nframes, natoms, 3), dtype=float)
        energies = np.zeros((nframes,), dtype=float)
        ex_energies = np.zeros((nframes,), dtype=float)
        grads = np.zeros((nframes, natoms, 3), dtype=float)
        transmom = np.zeros((nframes, 3), dtype=float)

        for i, task in enumerate(self.tasks):
            st = None
            for s in task.states:
                if s.state_idx == state_idx:
                    st = s
                    break
            if st is None:
                raise ValueError(f"文件 {task.filename} 没有找到 state {state_idx}")

            coords[i] = np.array(task.molecule.carti)[:, 1:].astype(float)
            energies[i] = st.total_energy if st.total_energy else np.nan
            ex_energies[i] = st.excitation_energy if st.excitation_energy else np.nan

            if st.gradient is not None:
                grads[i] = np.array(st.gradient, dtype=float)  # 原始梯度
            else:
                grads[i] = 0.0

            if st.trans_mom is not None:
                transmom[i] = np.array(st.trans_mom, dtype=float)
            else:
                transmom[i] = 0.0

        # 单位转换
        energies = ENERGY(energies, "hartree").convert_to(energy_unit)
        coords = DISTANCE(coords, "ang").convert_to(distance_unit)
        grads = GRADIENT(grads, energy_unit="hartree", distance_unit="bohr").convert_to(
            {"energy": (grad_unit[0], 1), "distance": (grad_unit[1], -1)}
        )
        forces = FORCE(grads, energy_unit=grad_unit[0], distance_unit=grad_unit[1]).convert_to(
            {"energy": (force_unit[0], 1), "distance": (force_unit[1], -1)}
        )

        atom_symbols = [atm[0] for atm in self.tasks[0].molecule.carti]
        qm_types = np.array([atom_charge_dict[sym] for sym in atom_symbols])

        # 保存
        np.save(prefix + "coord.npy", coords)
        np.save(prefix + "energy.npy", energies)
        np.save(prefix + "ex_energy.npy", ex_energies)
        np.save(prefix + "grad.npy", grads)
        np.save(prefix + "force.npy", forces)
        np.save(prefix + "transmom.npy", transmom)
        np.save(prefix + "type.npy", qm_types)

        return coords, energies, ex_energies, grads, forces, transmom, qm_types










