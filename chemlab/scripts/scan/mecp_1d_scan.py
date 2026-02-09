# MECP soc scan
import os
import shutil
import numpy as np
import time
import subprocess
from subprocess import Popen
from chemlab.util.mecp import mecp
from chemlab.util.file_system import qchem_file
import matplotlib.pyplot as plt

QCHEM_ENV_SETUP = """
module purge
export QC=/scratch/moriya/software/soc
export QCAUX=/scratch/zhengpei/q-chem/qcaux
source $QC/bin/qchem.setup.sh
export QCSCRATCH=/scratch/$USER
module load cmake3/3.24.3
module load impi/2021.2.0
module load intel/2021.2.0
"""


def check_qchem_success(output_file):
    if not os.path.exists(output_file):
        return False
    with open(output_file) as f:
        return "Thank you very much for using Q-Chem" in f.read()


def wait_for_qchem_outputs(out_files, check_interval=30):
    while True:
        if all(check_qchem_success(f) for f in out_files):
            print("Both Q-Chem jobs completed.")
            return
        print("Waiting for Q-Chem jobs...")
        time.sleep(check_interval)


def plot_mecp_scan_progress(dist_list, energy1_list, energy2_list):
    clear_output(wait=True)
    plt.figure(figsize=(6, 4))
    plt.plot(dist_list, energy1_list, marker='o', linestyle='-', color='blue')
    plt.plot(dist_list, energy2_list, marker='o', linestyle='-', color='red')
    plt.xlabel("Restrained Distance (Å)")
    plt.ylabel("Corrected soc Energy (Hartree)")
    plt.title("MECP Scan: Energy vs Restrain")
    plt.grid(True)
    plt.tight_layout()
    display(plt.gcf())
    plt.savefig(f"{scan_dir}/mecp_scan_progress.png")
    plt.close()


def plot_opt_step_progress(energy1_list, energy2_list, gap_list, step_list, dist):
    fig, axs = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

    axs[0].plot(step_list, energy1_list, label="State 1 Energy", marker='o')
    axs[0].plot(step_list, energy2_list, label="State 2 Energy", marker='x')
    axs[0].set_ylabel("Energy (Hartree)")
    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_title(f"Optimization Trajectory at Restrained Distance {dist:.3f} Å")

    axs[1].plot(step_list, gap_list, label="|E1 - E2|", color="purple", marker='s')
    axs[1].set_xlabel("Optimization Step")
    axs[1].set_ylabel("Energy Gap (Hartree)")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(f"{scan_dir}/opt_step_progress_{dist:.3f}.png")
    plt.close()


from chemlab.scripts.base import QchemBaseScript
from chemlab.config.config_loader import ConfigBase
class MECP1DScanConfig(ConfigBase):
    section_name = "mecp_1d_scan_config"

class MECP1DScan(QchemBaseScript):
    name = "mecp_1d_scan"
    def run(self,cfg):
        ref_template = cfg.ref
        scan_dir =cfg.out_path
        atom1 = cfg.atom1
        atom2 = cfg.atom2
        dist_range = cfg.dist_range
        prefix= cfg.prefix
        spin1=cfg.spin1
        spin2=cfg.spin2
        stepsize=cfg.max_stepsize,
        converge_limit=cfg.converge_limit,
        K=cfg.restrain_const,
        ncore = cfg.ncore,
        max_opt_steps=cfg.max_opt_steps,
        os.makedirs(scan_dir, exist_ok=True)

        energy1_list = []
        energy2_list = []
        dist_list = []

        for i, dist in enumerate(dist_range):
            print(f"\n🔹 Step {i}: Restrain bond {atom1}-{atom2} to {dist:.3f} Å")

            inp = qchem_file()
            inp.molecule.check = True
            inp.read_from_file(ref_template if i == 0 else os.path.join(scan_dir, f"{prefix}_step{i - 1}_final.inp"))
            ref_filename = f"{prefix}_step{i}.inp"
            ref_path = os.path.join(scan_dir, ref_filename)
            inp.generate_inp(ref_path)

            test_mecp = mecp()
            test_mecp.ref_path = scan_dir
            test_mecp.ref_filename = ref_filename
            test_mecp.out_path = scan_dir
            test_mecp.prefix = f"{prefix}_step{i}_"
            test_mecp.state_1.spin = spin1
            test_mecp.state_2.spin = spin2
            test_mecp.stepsize = stepsize
            test_mecp.different_type = "soc"
            test_mecp.converge_limit = converge_limit
            test_mecp.read_init_structure()
            test_mecp.initialize_bfgs()
            test_mecp.generate_new_inp()

            for step in range(max_opt_steps):
                print(f"\n>>> MECP optimization step {step}")
                test_mecp.job_num = step
                test_mecp.generate_new_spc_inp()

                processes = []
                out_files = []
                for state in [test_mecp.state_1]:
                    inpfile = os.path.join(test_mecp.out_path, state.job_name)
                    outfile = inpfile + ".out"
                    out_files.append(outfile)
                    cmd = f"""{QCHEM_ENV_SETUP}qchem -nt {ncore//2} {inpfile} {outfile}"""
                    processes.append(Popen(cmd, shell=True, executable="/bin/bash"))

                for p in processes:
                    p.wait()
                wait_for_qchem_outputs(out_files)

                test_mecp.read_soc_output()
                test_mecp.calc_new_gradient()

                test_mecp.restrain_ene(atom1 - 1, atom2 - 1, dist, K=K)
                test_mecp.restrain_force(atom1 - 1, atom2 - 1, dist, K=K)
                test_mecp.parallel_gradient += test_mecp.F_EI

                if test_mecp.check_soc_converge():
                    print(f"✅ Converged at scan step {i}")
                    break
                test_mecp.update_structure()
                energy1 = test_mecp.state_1.out.ene
                energy2 = test_mecp.state_2.out.ene
                gap = abs(energy1 - energy2)
                plot_opt_step_progress(
                    test_mecp.state_1.ene_list,
                    test_mecp.state_2.ene_list,
                    [abs(e1 - e2) for e1, e2 in zip(test_mecp.state_1.ene_list, test_mecp.state_2.ene_list)],
                    list(range(len(test_mecp.state_1.ene_list))),
                    dist
                )

            final_name = os.path.join(scan_dir, f"{prefix}_step{i}_final.inp")
            with open(final_name, "w") as fout:
                fout.write(test_mecp.state_1.inp.molecule.return_output_format())
                fout.write(test_mecp.state_1.inp.remain_texts)

            E1 = test_mecp.state_1.out.ene + test_mecp.EI
            E2 = test_mecp.state_2.out.ene + test_mecp.EI
            energy1_list.append(E1)
            energy2_list.append(E2)
            dist_list.append(dist)
            plot_mecp_scan_progress(dist_list, energy1_list, energy2_list)
            np.save(f"{scan_dir}/E1.npy",energy1_list)
            np.save(f"{scan_dir}/E2.npy",energy2_list)
            np.save(f"{scan_dir}/dist.npy",dist_list)
        print(" MECP scan completed with restrain correction.")
        return dist_list, energy_list

