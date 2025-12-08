import os
import time
import numpy as np
import subprocess
from dataclasses import dataclass
from typing import Optional, List, Dict

from chemlab.scripts.base import QchemBaseScript, run_qchem_job_async
from chemlab.util.file_system import qchem_file, qchem_out_geomene
from chemlab.config.config_loader import QchemEnvConfig, ConfigBase
from chemlab.util.modify_inp import check_list,ex_match_check

# ========== Config class ==========
class FDConfig(ConfigBase):
    section_name = "FD"

# ========== Internal structure for job tracking ==========
@dataclass
class FDJob:
    idx: int
    iatom: int
    icart: int
    sign: str                # "p","m","p2","m2"
    inp_file: str
    out_file: str
    xyz_perturbed: np.ndarray  # 直接存 xyz，生成 inp 用
    attempts: int = 0
    popen: Optional[subprocess.Popen] = None
    started: bool = False
    finished: bool = False
    converged: bool = False
    start_time: Optional[float] = None



def print_status(jobs: List[FDJob], running: Dict[int, FDJob], njob: int):
    running_list = list(running.values())
    ready_list   = [j for j in jobs if (not j.started and not j.finished)]
    done_ok      = [j for j in jobs if j.finished and j.converged]
    done_fail    = [j for j in jobs if j.finished and not j.converged]

    print("\n" + "=" * 88)
    print(
        f"FD STATUS | RUN={len(running_list)} READY={len(ready_list)} "
        f"DONE_OK={len(done_ok)} DONE_FAIL={len(done_fail)} | LIMIT={njob}"
    )
    print("-" * 88)
    print("[RUN]      " + ", ".join(f"{j.idx}{j.sign}" for j in running_list))
    print("[READY]    " + ", ".join(f"{j.idx}{j.sign}" for j in ready_list))
    print("[DONE_OK]  " + ", ".join(f"{j.idx}{j.sign}" for j in done_ok))
    print("[DONE_FAIL]" + ", ".join(f"{j.idx}{j.sign}" for j in done_fail))
    print("=" * 88 + "\n")


class FDMethod:
    """
    Base class for defining an FD rule.
    Subclasses must implement:
        generate_jobs(mol, cfg)  → return list of FDJob
        assemble_gradient(results, mol, cfg) → return gradient(natom,3)
    """

    def generate_jobs(self, mol, cfg):
        raise NotImplementedError

    def assemble_gradient(self, results, mol, cfg):
        raise NotImplementedError



class FivePointMethod(FDMethod):

    def generate_jobs(self, mol, cfg):
        jobs = []
        xyz0 = mol.xyz.copy()
        natom = xyz0.shape[0]
        h = cfg.distance
        xyz0 = np.ascontiguousarray(xyz0,dtype=np.float64)
        idx_global = 0
        for i in range(natom):
            for j in range(3):

                # +2h
                xyz_p2 = xyz0.copy()
                xyz_p2[i, j] += 2*h
                jobs.append(FDJob(idx_global, i, j, "p2",
                    f"{cfg.outpath}/coord_{idx_global}_p2.inp",
                    f"{cfg.outpath}/coord_{idx_global}_p2.out",
                    xyz_p2
                ))

                # +h
                xyz_p = xyz0.copy()
                xyz_p[i, j] += h
                jobs.append(FDJob(idx_global, i, j, "p",
                    f"{cfg.outpath}/coord_{idx_global}_p.inp",
                    f"{cfg.outpath}/coord_{idx_global}_p.out",
                    xyz_p
                ))

                # -h
                xyz_m = xyz0.copy()
                xyz_m[i, j] -= h
                jobs.append(FDJob(idx_global, i, j, "m",
                    f"{cfg.outpath}/coord_{idx_global}_m.inp",
                    f"{cfg.outpath}/coord_{idx_global}_m.out",
                    xyz_m
                ))

                # -2h
                xyz_m2 = xyz0.copy()
                xyz_m2[i, j] -= 2*h
                jobs.append(FDJob(idx_global, i, j, "m2",
                    f"{cfg.outpath}/coord_{idx_global}_m2.inp",
                    f"{cfg.outpath}/coord_{idx_global}_m2.out",
                    xyz_m2
                ))

                idx_global += 1

        return jobs


    def assemble_gradient(self, results, mol, cfg):
        h = cfg.distance
        natom = mol.xyz.shape[0]
        grad = np.zeros((natom,3))

        for idx in range(3*natom):
            iatom = idx // 3
            icart = idx % 3

            e_p2 = results[(idx,"p2")]
            e_p  = results[(idx,"p")]
            e_m  = results[(idx,"m")]
            e_m2 = results[(idx,"m2")]

            grad[iatom,icart] = (-e_p2 + 8*e_p - 8*e_m + e_m2)/(12*h)

        return grad

class FivePointFiniteDifference(QchemBaseScript):

    name = "FPFD"
    config = FDConfig
    method_class = FivePointMethod

    def run(self, cfg):

        ref = qchem_file()
        ref.molecule.check = True
        print(cfg.path + cfg.ref)
        ref.read_from_file(cfg.path + cfg.ref)
        mol = ref.molecule
        method = self.method_class()
        jobs = method.generate_jobs(mol, cfg)

        os.makedirs(cfg.outpath, exist_ok=True)
        for j in jobs:
            inp = qchem_file()
            ref.molecule.check = True
            inp.read_from_file(cfg.path + cfg.ref)
            inp.molecule.replace_new_xyz(j.xyz_perturbed)
            inp.generate_inp(j.inp_file)

        self.run_jobs(jobs, cfg, print_status_func=print_status)


        results = {}
        for j in jobs:
            out = qchem_out_geomene(j.out_file)
            out.read_file(j.out_file)
            results[(j.idx, j.sign)] = out.ene

        # (5) assemble gradient
        grad = method.assemble_gradient(results, mol, cfg)

        np.save(f"{cfg.path}/fd_gradient.npy", grad)
        print("Gradient saved!")

        return grad

