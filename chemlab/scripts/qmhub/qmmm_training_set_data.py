import sys
import os
import shutil
import uuid
import subprocess as sp
from multiprocessing import Pool
import numpy as np

from chemlab.util.file_system import ELEMENT_DICT
from chemlab.scripts.base import QchemBaseScript
from chemlab.config.config_loader import ConfigBase


class QMMMTrainSetDataConfig(ConfigBase):
    section_name = "qmmm_training_set_data"


class QMMMTrainSetData(QchemBaseScript):
    name = "example_script"
    config = QMMMTrainSetDataConfig

    def run(self, cfg):
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
        for i in range(windows):
            window = "{:02d}".format(i)
            tem_qmmm_path = f"{qmmmpath}/{window}/"

            win_energy = []
            win_qm_grad = []
            win_mm_esp = []
            win_mm_esp_grad = []
            for j in range(nframes):
                frame = "{:04d}".format(j)
                tem_cache_path = f"{cache_path}/w{window}/{frame}/"
                tem_input = f"{tem_qmmm_path}/{prefix}{frame}.out"
                if self.check_qchem_error(tem_input) == -1:
                    print("File not found:", tem_input)
                    continue
                elif self.check_qchem_error(tem_input) == 1:
                    print("Qchem Job Fail:", tem_input)
                    continue

                tem_energy = np.fromfile(tem_cache_path + "99.0", dtype="f8", count=2)[1]
                tem_qm_grad = np.fromfile(tem_cache_path + "131.0", dtype="f8").reshape(-1, 3)
                tem_mm_esp = np.loadtxt(tem_qmmm_path + f"{prefix}{frame}.out.esp", dtype=float, skiprows=3)
                try:
                    tem_mm_esp_grad = -np.loadtxt(tem_qmmm_path + f"{prefix}{frame}.out.efld", max_rows=len(tem_mm_esp), dtype=float)
                except:
                    print(tem_qmmm_path + f"{prefix}{frame}.out.efld error")
                    continue
                if tem_mm_esp_grad.shape != tem_mm_esp.shape:
                    print(
                        f"Shape mismatch: {tem_input} "
                        f"esp {tem_mm_esp.shape}, efld {tem_mm_esp_grad.shape}"
                    )
                    continue
                win_energy.append(tem_energy)
                win_qm_grad.append(tem_qm_grad)
                win_mm_esp.append(tem_mm_esp)
                win_mm_esp_grad.append(tem_mm_esp_grad)


            win_energy = np.asarray(win_energy)
            win_qm_grad = np.asarray(win_qm_grad)
            win_mm_esp = np.asarray(win_mm_esp)
            win_mm_esp_grad = np.asarray(win_mm_esp_grad)

            np.save(f"{outpath}/energy_w{window}.npy", win_energy)
            np.save(f"{outpath}/qm_grad_w{window}.npy", win_qm_grad)
            np.save(f"{outpath}/mm_esp_w{window}.npy", win_mm_esp)
            np.save(f"{outpath}/mm_esp_grad_w{window}.npy", win_mm_esp_grad)

            full_energy.append(win_energy)
            full_qm_grad.append(win_qm_grad)
            full_mm_esp.append(win_mm_esp)
            full_mm_esp_grad.append(win_mm_esp_grad)

        np.save(f"{outpath}/full_energy.npy", np.asarray(full_energy))
        np.save(f"{outpath}/full_qm_grad.npy", np.asarray(full_qm_grad))
        np.save(f"{outpath}/full_mm_esp.npy", np.asarray(full_mm_esp))
        np.save(f"{outpath}/full_mm_esp_grad.npy", np.asarray(full_mm_esp_grad))
