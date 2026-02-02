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


class QMMMTrainSetDFT(QchemBaseScript):
    name = "example_script"
    config = QMMMTrainSetDFTConfig

    def run(self, cfg):
        qmmmpath = cfg.qmmmpath
        cache_path = cfg.cache_path
        outpath = cfg.outpath
        prefix = cfg.prefix
        ref = cfg.ref
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
                tem_cache_path = f"{cache_path}/{window}/{frame}/"
                tem_energy = np.fromfile(tem_cache_path + "99.0", dtype="f8", count=2)[1]
                tem_qm_grad = np.fromfile(tem_cache_path + "131.0", dtype="f8").reshape(-1, 3)
                tem_mm_esp = np.loadtxt(tem_qmmm_path + f"{prefix}{frame}.out.esp", dtype=float)
                tem_mm_esp_grad = -np.loadtxt(tem_qmmm_path + f"{prefix}{frame}.out.efld", max_rows=len(mm_esp), dtype=float)
                win_energy.append(energy)
                win_qm_grad.append(qm_grad)
                win_mm_esp.append(mm_esp)
                win_mm_esp_grad.append(mm_esp_grad)


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

def get_qchem_force(qm_pos, qm_elem_num, mm_pos, mm_charge, charge, mult):
    #cwd = "/dev/shm/run_" + str(uuid.uuid4())
    cwd = "dftfmatch/" + str(uuid.uuid4())
    if not os.path.exists(cwd):
        os.makedirs(cwd)
    get_qchem_input(cwd + "/qchem.inp", qm_pos, qm_elem_num, mm_pos, mm_charge, charge, mult)

    cmdline = f"cd {cwd}; "
    cmdline += f"QCSCRATCH=`pwd` qchem qchem.inp qchem.out save > qchem_run.log"
    proc = sp.Popen(args=cmdline, shell=True)
    proc.wait()

    energy = np.fromfile(cwd + "/save/99.0", dtype="f8", count=2)[1]
    qm_grad = np.fromfile(cwd + "/save/131.0", dtype="f8").reshape(-1, 3)
    mm_esp = np.fromfile(cwd + "/save/5001.0", dtype="f8", count=len(mm_charge))
    mm_esp_grad = -np.fromfile(cwd + "/save/5002.0", dtype="f8", count=(len(mm_charge)*3)).reshape(-1, 3)
    mm_esp = np.loadtxt(cwd + "/esp.dat", dtype=float)
    mm_esp_grad = -np.loadtxt(cwd + "/efield.dat", max_rows=len(mm_esp), dtype=float)

    shutil.rmtree(cwd)

    return energy, qm_grad, mm_esp, mm_esp_grad
