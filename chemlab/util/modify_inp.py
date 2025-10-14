from .file_system import molecule,qchem_file,qchem_out_opt,multiple_qchem_jobs,qchem_out_multi
from .file_system import SPIN_REF
import numpy as np
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
    Manage multiple Q-Chem excited-state output files,
    and provide modular export utilities for specific attributes (energy, gradient, etc.)
    """

    def __init__(self):
        super().__init__()
        self._exporters = {}
        self._register_default_exporters()

    def read_files(self, filenames, path=""):
        """
        Read multiple Q-Chem excited-state output files.

        Args:
            filenames (list[str]): List of output filenames.
            path (str): Optional path prefix for all files.
        """
        from .file_system import qchem_out_excite
        import os
        fullpaths = [os.path.join(path, fn) if path else fn for fn in filenames]
        super().read_files(fullpaths, qchem_out_excite)

    # =============================================================
    #  Generic export function (core abstraction)
    # =============================================================
    def export_attr(
        self,
        extractor,
        shape_func=None,
        dtype=float,
        state_idx=1,
    ):
        """
        Generic method to extract and export any attribute from each excited-state output.

        Args:
            attr_name (str): The attribute name, used for filename and identification.
            extractor (Callable): A lambda function `extractor(st, task)` returning the value to export.
            shape_func (Callable): Optional lambda `(natoms, nframes) -> shape` to allocate array shape.
            dtype (type): Numpy dtype for the exported array (default: float).
            prefix (str): File name prefix for output `.npy` files.
            filename (str): Custom file name (if None, use prefix + attr_name + ".npy").
            state_idx (int): Index of the excited state to extract (default: 1).

        Returns:
            np.ndarray: The extracted and exported data array.
        """

        if not self.tasks:
            raise ValueError("No excited-state output files have been read.")

        nframes = len(self.tasks)
        natoms = len(self.tasks[0].molecule.carti)
        shape = shape_func(natoms, nframes) if shape_func else (nframes,)
        arr = np.zeros(shape, dtype=dtype)

        for i, task in enumerate(self.tasks):
            # Find the target excited state
            st = None
            for s in task.states:
                if s.state_idx == state_idx:
                    st = s
                    break
            if st is None:
                raise ValueError(f"File {task.filename} does not contain state {state_idx}")

            # Apply extractor to obtain attribute value
            val = extractor(st, task)
            arr[i] = val if val is not None else np.nan

        return arr

    def export_gs_energy(self, prefix="gs", energy_unit="kcal", state_idx=0):
        """
        Export total energy for each frame.
        """
        from .unit import ENERGY
        energies = self.export_attr(
            extractor=lambda st, task: st.total_energy,
            state_idx=state_idx,
        )
        energies = ENERGY(energies, "hartree").convert_to(energy_unit)
        np.save(f"{prefix}_energies.npy", energies)
        return energies

    def export_ex_energy(self, prefix="ex", energy_unit="kcal", state_idx=1):
        """
        Export excitation energy for each frame.
        """
        from .unit import ENERGY
        ex_energies = self.export_attr(
            extractor=lambda st, task: st.excitation_energy,
            state_idx=state_idx,
        )
        ex_energies = ENERGY(ex_energies, "hartree").convert_to(energy_unit)
        np.save(f"{prefix}{state_idx}_energies.npy", ex_energies)
        return ex_energies
    def export_transmom(self, prefix="S", unit="au", state_idx=1):
        """
        Export transition dipole moment vectors for each frame.
        """
        from .unit import DIPOLE
        transmom = self.export_attr(
            extractor=lambda st, task: np.array(st.trans_mom, dtype=float)
            if st.trans_mom is not None else None,
            shape_func=lambda natoms, nframes: (nframes, 3),
            state_idx=state_idx,
        )
        transmom = DIPOLE(transmom, "au").convert_to(unit)
        np.save(f"{prefix}{state_idx}_transmom.npy", transmom)
        return transmom
    def export_dipolemom(self, prefix="", unit="au", state_idx=0):
        """
        Export ground state dipole moment vectors for each frame.
        """
        from .unit import DIPOLE
        dipolemom = self.export_attr(
            extractor=lambda st, task: np.array(st.trans_mom, dtype=float)
            if st.trans_mom is not None else None,
            shape_func=lambda natoms, nframes: (nframes, 3),
            state_idx=state_idx,
        )
        dipolemom = DIPOLE(dipolemom, "au").convert_to(unit)
        np.save(f"{prefix}_dipolemom.npy", dipolemom)
        return dipolemom

    def export_coords(self, prefix="", distance_unit="ang"):
        """
        Export molecular Cartesian coordinates for each frame.
        """
        from .unit import DISTANCE
        coords = np.array([np.array(t.molecule.carti)[:, 1:].astype(float) for t in self.tasks])
        coords = DISTANCE(coords, "ang").convert_to(distance_unit)
        np.save(f"{prefix}_coord.npy", coords)
        return coords
    def export_forces(self, prefix="", grad_unit=("hartree", "bohr"),state_idx=1):
        """
        Export Forces of state_idx state energy for each frame.
        """
        from .unit import FORCE
        gradients = self.export_attr(
            extractor=lambda st, task: np.array(st.gradient, dtype=float)
            if st.gradient is not None else np.zeros((len(task.molecule.carti), 3)),
            shape_func=lambda natoms, nframes: (nframes, natoms, 3),
            state_idx=state_idx,
        )

        forces = FORCE(gradients, energy_unit="hartree", distance_unit="bohr").convert_to(
        {"energy": (grad_unit[0], 1), "distance": (grad_unit[1], -1)}
    )
        np.save(f"{prefix}_forces.npy", forces)
        return forces

    def export_gradients(self, prefix="", grad_unit=("hartree", "bohr"),state_idx=1):
        """
         Export gradients of state_idx state energy for each frame.
         """
        from .unit import GRADIENT
        gradients = self.export_attr(
            extractor=lambda st, task: np.array(st.gradient, dtype=float)
            if st.gradient is not None else np.zeros((len(task.molecule.carti), 3)),
            shape_func=lambda natoms, nframes: (nframes, natoms, 3),
            state_idx=state_idx,
        )

        gradients = GRADIENT(gradients, energy_unit="hartree", distance_unit="bohr").convert_to(
        {"energy": (grad_unit[0], 1), "distance": (grad_unit[1], -1)}
    )
        np.save(f"{prefix}_gradients.npy", gradients)
        return gradients

    def register_exporter(self, name, func):
        """
        Register a new exporter.

        Args:
            name (str): Exporter name (e.g. 'dipole', 'osc_strength')
            func (callable): Function with signature f(self, **kwargs)
        """
        self._exporters[name] = func

    def _register_default_exporters(self):
        """
        Register the built-in exporters.
        """
        self.register_exporter("coords", self.export_coords)
        self.register_exporter("energy", self.export_gs_energy)
        self.register_exporter("ex_energy", self.export_ex_energy)
        self.register_exporter("gradient", self.export_gradients)
        self.register_exporter("force", self.export_forces)
        self.register_exporter("transmom", self.export_transmom)
        self.register_exporter("dipolemom", self.export_dipolemom)

    def export_all(self, prefix="", **kwargs):
        """
        Run all registered exporters sequentially.

        Args:
            prefix (str): Prefix for output files.
            kwargs: Additional arguments passed to each exporter.
        """
        results = {}
        for name, func in self._exporters.items():
            print(f"[Export] {name}")
            results[name] = func(prefix=prefix, **kwargs)
        return results










