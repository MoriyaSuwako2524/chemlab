#!/usr/bin/env python3
import os
import time
import argparse
import subprocess
import matplotlib.pyplot as plt
from IPython.display import clear_output
from chemlab.util.mecp import mecp,mecp_soc


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


def wait_for_qchem_outputs(out_files, check_interval=30, log=None):
    """Wait until all output files contain Q-Chem completion line."""
    while True:
        if all(check_qchem_success(f) for f in out_files):
            msg = "All Q-Chem jobs completed."
            print(msg)
            if log: log.write(msg + "\n")
            return
        msg = "Waiting for Q-Chem jobs..."
        print(msg)
        if log: log.write(msg + "\n")
        time.sleep(check_interval)


def run_mecp_optimization(args):
    job_type = args.jobtype
    if job_type == "mecp":
        test_mecp = mecp()
        test_mecp.different_type = args.gradient
    elif job_type == "soc":
        test_mecp = mecp_soc()
    else:
        raise NotImplementedError("Job type %s not implemented,Please use jobtype=mecp or jobtype=soc" % job_type)
    test_mecp.ref_path = os.path.dirname(args.path)
    test_mecp.ref_filename = os.path.basename(args.file)
    test_mecp.out_path = os.path.dirname(args.out)
    os.makedirs(test_mecp.out_path, exist_ok=True)

    # open log file
    log_path = os.path.join(test_mecp.out_path, "mecp.log")
    log = open(log_path, "a", buffering=1)  # line-buffered

    # init
    test_mecp.state_1.spin = args.spin1
    test_mecp.state_2.spin = args.spin2
    test_mecp.stepsize = args.stepsize



    
    test_mecp.converge_limit = args.conv
    test_mecp.read_init_structure()
    test_mecp.generate_new_inp()
    test_mecp.initialize_bfgs()
    #print(test_mecp.out_path)
    #print(os.path.join(test_mecp.out_path,test_mecp.state_1.job_name))
    # prepare plotting
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 4))
    energies, diffs = [], []

    for step in range(args.max_steps):
        msg = f"\n>>> MECP iteration step {step}"
        print(msg)
        log.write(msg + "\n")

        test_mecp.job_num = step
        test_mecp.generate_new_inp()
	
        # run state 1 and 2 jobs
        processes, out_files = [], []
        for state in [test_mecp.state_1, test_mecp.state_2]:
            inp = os.path.join(test_mecp.out_path, state.job_name)
            out = inp + ".out"
            out_files.append(out)
            cmd = f"""{QCHEM_ENV_SETUP}\nqchem -nt {args.nthreads // 2} {inp} {out}"""
            #p = subprocess.Popen(cmd, shell=True, executable="/bin/bash")
            #processes.append(p)

        for p in processes:
            p.wait()

        wait_for_qchem_outputs(out_files, log=log)
	
        # post-processing
        test_mecp.read_output()
        test_mecp.calc_new_gradient()

        # energy record
        e1 = getattr(test_mecp.state_1, "energy", None)
        e2 = getattr(test_mecp.state_2, "energy", None)
        if e1 is not None and e2 is not None:
            energies.append((e1, e2))
            diffs.append(abs(e1 - e2))

        # update plot
        clear_output(wait=True)
        ax.clear()
        ax.plot([i for i in range(len(energies))], [e[0] for e in energies], label="State 1", color="C0")
        ax.plot([i for i in range(len(energies))], [e[1] for e in energies], label="State 2", color="C1")
        ax.plot([i for i in range(len(diffs))], diffs, label="|ΔE|", color="C3", linestyle="--")
        ax.set_xlabel("Step")
        ax.set_ylabel("Energy (Hartree)")
        ax.set_title("MECP Optimization Progress")
        ax.legend()
        plt.tight_layout()
        plt.pause(0.1)

        # check convergence
        if test_mecp.check_convergence():
            msg = f"\n>>> MECP optimization converged at step {step}!"
            print(msg)
            log.write(msg + "\n")
            break

        # update coordinates
        test_mecp.update_structure()
        msg = f"Step {step} updated structure. ΔE = {diffs[-1]:.6f}"
        print(msg)
        log.write(msg + "\n")

    plt.ioff()
    plt.savefig(os.path.join(test_mecp.out_path, "mecp_progress.png"))
    log.write("Optimization finished.\n")
    log.close()
    print("Optimization finished.")
    return test_mecp


def main():
    parser = argparse.ArgumentParser(description="Run MECP optimization using Q-Chem.")
    parser.add_argument("--path", required=True,type=str, help="Path to reference Q-Chem input file.")
    parser.add_argument("--file", required=True,type=str, help="reference Q-Chem input file.")
    parser.add_argument("--out", required=True,type=str, help="Output directory.")
    parser.add_argument("--jobtype", required=True,type=str, help="Default mecp(jobtype=mecp) or spin adiabatic state mecp(soc).")
    parser.add_argument("--gradient",type=str, default="analytical",help="Default analytical gradient(gradient=analytical). Spin adiabatic state mecp use(gradient=soc) as default." )
    parser.add_argument("--spin1", type=int, default=1, help="Spin multiplicity of state 1.")
    parser.add_argument("--spin2", type=int, default=3, help="Spin multiplicity of state 2.")
    parser.add_argument("--nthreads", type=int, default=16, help="Total threads for Q-Chem.")
    parser.add_argument("--max-steps", type=int, default=100, help="Maximum optimization steps.")
    parser.add_argument("--stepsize", type=float, default=0.1, help="Initial step size.")
    parser.add_argument("--conv", type=float, default=1e-5, help="Convergence threshold.")
    args = parser.parse_args()

    run_mecp_optimization(args)


if __name__ == "__main__":
    main()
