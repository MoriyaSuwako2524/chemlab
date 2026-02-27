import sys
import os
import shutil
import uuid
import subprocess as sp
from multiprocessing import Pool
import numpy as np

from chemlab.util.file_system import ELEMENT_DICT,qchem_out_force
from chemlab.scripts.base import QchemBaseScript
from chemlab.config.config_loader import ConfigBase


class QMMMTrainSetDataConfig(ConfigBase):
    section_name = "qmmm_training_set_data"


class QMMMTrainSetData(QchemBaseScript):
    name = "example_script"
    config = QMMMTrainSetDataConfig

    def run(self, cfg):
        if cfg.method == "gas":
            self.run_gas(cfg)
            return 0
        elif cfg.method == "qmmm":
            self.run_qmmm(cfg)
            return 0
        else:
            return KeyError(f"Method {cfg.method} Not Supported")
    def run_qmmm(self,cfg):
        qmmmpath = cfg.qmmmpath
        cache_path = cfg.cache_path
        outpath = cfg.outpath
        prefix = cfg.prefix
        windows = cfg.windows
        nframes = cfg.nframes
        os.makedirs(outpath, exist_ok=True)

        full_energy = []
        full_qm_grad = []
        full_mm_esp = []
        full_mm_esp_grad = []
        full_qm_coords = []
        full_qm_type = None

        for i in range(windows):
            window = "{:02d}".format(i)
            tem_qmmm_path = f"{qmmmpath}/{window}/"

            win_energy = []
            win_qm_grad = []
            win_mm_esp = []
            win_mm_esp_grad = []
            win_qm_coord = []
            for j in range(nframes):
                frame = "{:04d}".format(j)
                tem_cache_path = f"{cache_path}/{window}/{frame}/"
                tem_input = f"{tem_qmmm_path}/{frame}/{prefix}{frame}.out"
                tem_qmout = qchem_out_force()
                if self.check_qchem_error(tem_input) == -1:
                    print("File not found:", tem_input)
                    continue
                elif self.check_qchem_error(tem_input) == 1:
                    print("Qchem Job Fail:", tem_input)
                    continue
                tem_qmout.read_file(tem_input)


                tem_qm_coord = tem_qmout.molecule.xyz
                tem_energy = np.fromfile(tem_cache_path + "99.0", dtype="f8", count=2)[1]
                tem_qm_grad = np.fromfile(tem_cache_path + "131.0", dtype="f8").reshape(-1, 3)
                tem_mm_esp = np.loadtxt( f"{tem_qmmm_path}/{frame}/{prefix}{frame}.out.esp", dtype=float, skiprows=3)
                try:
                    tem_mm_esp_grad = -np.loadtxt(f"{tem_qmmm_path}/{frame}/{prefix}{frame}.out.efld", max_rows=len(tem_mm_esp), dtype=float)
                except:
                    print(f"{tem_qmmm_path}/{frame}/{prefix}{frame}.out.efld error")
                    continue

                if len(win_mm_esp) == 0:
                    ref_n = tem_mm_esp.shape[0]
                elif tem_mm_esp_grad.shape[0] != ref_n:
                    print("Skip frame due to inconsistent MM ESP size:", tem_input)
                    continue
                elif tem_mm_esp.shape[0] != ref_n:
                    print("Skip frame due to inconsistent MM ESP size:", tem_input)
                    continue
                win_energy.append(tem_energy)
                win_qm_grad.append(tem_qm_grad)
                win_mm_esp.append(tem_mm_esp)
                win_mm_esp_grad.append(tem_mm_esp_grad)
                win_qm_coord.append(tem_qm_coord)


            win_energy = np.asarray(win_energy)
            win_qm_grad = np.asarray(win_qm_grad)
            win_mm_esp = np.asarray(win_mm_esp)
            win_mm_esp_grad = np.asarray(win_mm_esp_grad)
            win_qm_coord = np.asarray(win_qm_coord)

            np.save(f"{outpath}/energy_w{window}.npy", win_energy)
            np.save(f"{outpath}/qm_grad_w{window}.npy", win_qm_grad)
            np.save(f"{outpath}/mm_esp_w{window}.npy", win_mm_esp)
            np.save(f"{outpath}/mm_esp_grad_w{window}.npy", win_mm_esp_grad)
            np.save(f'{outpath}/qm_coord_w{window}.npy', win_qm_coord)
            full_energy.append(win_energy)
            full_qm_grad.append(win_qm_grad)
            full_mm_esp.append(win_mm_esp)
            full_mm_esp_grad.append(win_mm_esp_grad)
            full_qm_coords.append(win_qm_coord)

        full_energy = np.concatenate(full_energy, axis=0)
        np.save(f"{outpath}/full_energy.npy", full_energy)
        full_qm_grad = np.concatenate(full_qm_grad, axis=0)
        np.save(f"{outpath}/full_qm_grad.npy", full_qm_grad)
        full_mm_esp = np.concatenate(full_mm_esp, axis=0)
        np.save(f"{outpath}/full_mm_esp.npy", np.asarray(full_mm_esp))
        full_mm_esp_grad = np.concatenate(full_mm_esp_grad, axis=0)
        np.save(f"{outpath}/full_mm_esp_grad.npy", np.asarray(full_mm_esp_grad))
        full_qm_coords = np.concatenate(full_qm_coords, axis=0)
        np.save(f"{outpath}/full_qm_coords.npy", full_qm_coords)
    def run_gas(self, cfg):
        qmmmpath = cfg.qmmmpath
        cache_path = cfg.cache_path
        outpath = cfg.outpath
        prefix = cfg.prefix
        windows = cfg.windows
        nframes = cfg.nframes
        os.makedirs(outpath, exist_ok=True)

        full_energy = []
        full_qm_grad = []
        full_qm_coords = []
        full_qm_type = None
        for i in range(windows):
            window = "{:02d}".format(i)
            tem_qmmm_path = f"{qmmmpath}/{window}/"

            win_energy = []
            win_qm_grad = []
            win_qm_coord = []

            for j in range(nframes):
                frame = "{:04d}".format(j)
                tem_cache_path = f"{cache_path}/{window}/{frame}/"
                tem_input = f"{tem_qmmm_path}/{frame}/{prefix}{frame}.out"
                tem_qmout = qchem_out_force()
                if self.check_qchem_error(tem_input) == -1:
                    print("File not found:", tem_input)
                    continue
                elif self.check_qchem_error(tem_input) == 1:
                    print("Qchem Job Fail:", tem_input)
                    continue
                tem_qmout.read_file(tem_input)
                if full_qm_type is None:
                    from chemlab.util.file_system import atom_charge_dict
                    atom_symbols = [atm[0] for atm in tem_qmout.molecule.carti]
                    qm_type = np.array([atom_charge_dict[sym] for sym in atom_symbols])
                    full_qm_type = qm_type
                tem_qm_coord = tem_qmout.molecule.xyz
                tem_energy = tem_qmout.ene
                tem_qm_grad = tem_qmout.force

                win_energy.append(tem_energy)
                win_qm_grad.append(tem_qm_grad)

                win_qm_coord.append(tem_qm_coord)

            from chemlab.util.unit import ENERGY,GRADIENT,DISTANCE


            win_qm_coord = np.asarray(win_qm_coord)
            win_energy = np.asarray(win_energy)
            win_qm_grad = np.asarray(win_qm_grad)
            win_energy = ENERGY(win_energy, "hartree").convert_to("kcal")
            win_qm_grad = GRADIENT(win_qm_grad, energy_unit="hartree", distance_unit="bohr").convert_to(
                {"energy": ("kcal", 1), "distance": ("ang", -1)}
            )

            np.save(f"{outpath}/qm_coord_w{window}.npy", win_qm_coord)
            np.save(f"{outpath}/energy_w{window}.npy", win_energy)
            np.save(f"{outpath}/qm_grad_w{window}.npy", win_qm_grad)


            full_energy.append(win_energy)
            full_qm_grad.append(win_qm_grad)
            full_qm_coords.append(win_qm_coord)
        full_energy = np.concatenate(full_energy, axis=0)
        np.save(f"{outpath}/full_energy.npy", full_energy)
        full_qm_grad = np.concatenate(full_qm_grad, axis=0)
        np.save(f"{outpath}/full_qm_grad.npy", full_qm_grad)
        full_qm_coords = np.concatenate(full_qm_coords, axis=0)
        np.save(f"{outpath}/full_qm_coords.npy", full_qm_coords)
        np.save(f"{outpath}/full_qm_type.npy", full_qm_type)