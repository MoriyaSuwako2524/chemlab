#!/usr/bin/env python3
import os
import time
import argparse
import subprocess
import matplotlib.pyplot as plt
from IPython.display import clear_output

from chemlab.scripts.base import Script
from chemlab.config.config_loader import ConfigBase,QchemEnvConfig
from chemlab.util.mecp import mecp, mecp_soc
from chemlab.util.file_system import Hartree_to_kcal


class RunMecpConfig(ConfigBase):
    section_name = "run_mecp"


class RunMecp(Script):

    name = "run_mecp"
    config = RunMecpConfig

    @staticmethod
    def check_qchem_success(out_file):
        if not os.path.exists(out_file):
            return False
        with open(out_file) as f:
            return "Thank you very much for using Q-Chem" in f.read()

    def wait_for_jobs(self, out_files, log=None, interval=30):
        while True:
            if all(self.check_qchem_success(f) for f in out_files):
                msg = "All Q-Chem jobs completed."
                print(msg)
                if log: log.write(msg + "\n")
                return
            msg = "Waiting for Q-Chem jobs..."
            print(msg)
            if log: log.write(msg + "\n")
            time.sleep(interval)

    def run(self, cfg):

        # =============================
        # Load Q-Chem environment script
        # =============================
        cfg_env = QchemEnvConfig()
        env_script = cfg_env.env_script.strip()
        if not env_script:
            raise ValueError("[run_mecp] No env_script found in [qchem_env].")

        print("[run_mecp] Using Q-Chem environment from config:\n")
        print(env_script)
        print("==============================================")

        # ====================
        # Build MECP object
        # ====================
        if cfg.jobtype == "mecp":
            test_mecp = mecp()
            test_mecp.different_type = cfg.gradient
        else:
            test_mecp = mecp_soc()

        test_mecp.ref_path = os.path.dirname(cfg.path)
        test_mecp.ref_filename = os.path.basename(cfg.file)
        test_mecp.out_path = os.path.dirname(cfg.out)
        os.makedirs(test_mecp.out_path, exist_ok=True)
        test_mecp.step_size = cfg.step_size
        test_mecp.max_stepsize = cfg.max_stepsize
        log_path = os.path.join(test_mecp.out_path, "mecp.log")
        log = open(log_path, "a", buffering=1)

        test_mecp.state_1.spin = cfg.spin1
        test_mecp.state_2.spin = cfg.spin2
        test_mecp.converge_limit = cfg.conv

        test_mecp.read_init_structure()
        test_mecp.generate_new_inp()

        plt.ion()
        fig, ax = plt.subplots(figsize=(6, 4))
        energies, diffs = [], []

        # ======================================
        # Optimization steps
        # ======================================
        for step in range(cfg.max_steps):

            print(f"\n>>> MECP iteration step {step}")
            log.write(f"\n>>> MECP iteration step {step}\n")

            test_mecp.job_num = step
            test_mecp.generate_new_inp()

            processes, out_files = [], []
            for state in [test_mecp.state_1, test_mecp.state_2]:

                inp = os.path.join(test_mecp.out_path, state.job_name)
                out = inp[:-4] + ".out"
                out_files.append(out)

                cmd = f"""{env_script}
qchem -nt {cfg.nthreads // 2} {inp} {out}
"""

                p = subprocess.Popen(cmd, shell=True, executable="/bin/bash")
                processes.append(p)

            for p in processes:
                p.wait()

            self.wait_for_jobs(out_files, log=log)

            # read and compute
            test_mecp.read_output()
            test_mecp.calc_new_gradient()

            e1 = test_mecp.state_1.out.ene
            e2 = test_mecp.state_2.out.ene

            if e1 and e2:
                energies.append((e1, e2))

                # difference in kcal
                diffs.append(abs(e1 - e2) * Hartree_to_kcal)

                # --- reference energy: take the minimum from first step ---
                ref = min(energies[0])
                e1_rel = [(x[0] - ref) * Hartree_to_kcal for x in energies]
                e2_rel = [(x[1] - ref) * Hartree_to_kcal for x in energies]

                clear_output(wait=True)
                ax.clear()

                # ---------- plotting ----------
                ax.plot(e1_rel, "o-", label="State 1 energy")
                ax.plot(e2_rel, "s-", label="State 2 energy")
                ax.plot(diffs, "--", label="|E1 − E2| (gap)")

                ax.set_xlabel("MECP Iteration")
                ax.set_ylabel("Energy (kcal/mol)")
                ax.set_title("MECP Optimization Progress")
                ax.legend()
                ax.grid(True, alpha=0.3)

                plt.pause(0.1)



            test_mecp.update_structure()
            if test_mecp.check_convergence():
                print(f"\n>>> Converged at step {step}")
                log.write(f"\nConverged at step {step}\n")
                break
            plt.savefig(os.path.join(test_mecp.out_path, "mecp_progress.png"))

        log.close()
        print("MECP optimization finished.")
