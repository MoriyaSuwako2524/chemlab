import os
import time
from chemlab.config.config_loader import QchemEnvConfig
from typing import Dict
import subprocess


class Script:
    """
    Base class for all scripts used by CLI.

    Subclasses must define:
        name   : CLI subcommand name
        config : a ConfigBase subclass(If the script doesn't require config, config=None)
        run(self, cfg)
    """

    name = None
    config = None

    def run(self, cfg):
        raise NotImplementedError


class QchemBaseScript(Script):
    name = "Qchem"
    config = None
    def run(self, cfg):
        raise NotImplementedError
    @staticmethod
    def check_qchem_success(out_file):
        if not os.path.exists(out_file):
            return False
        with open(out_file) as f:
            return "Thank you very much for using Q-Chem" in f.read()
    @staticmethod
    def check_qchem_error(out_file):
        if not os.path.exists(out_file):
            return -1

        with open(out_file) as f:
            content = f.read()

        if "Thank you very much for using Q-Chem" in content:
            return 0
        elif "Q-Chem fatal error" in content:
            return 1
        else:
            return -10

    def generate_qchem_inp(self,molecule,ref):
        from chemlab.util.modify_inp import single_spin_job
        job = single_spin_job()
        job.charge = molecule.charge
        job.spin = molecule.spin
        job.ref_name = ref
        job._xyz = molecule
        job.generate_outputs()



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
    def run_jobs(self, jobs, cfg, print_status_func=None):

        cfg_env = QchemEnvConfig()
        env_script = cfg_env.env_script.strip()
        poll_interval = getattr(cfg, "poll_interval", 20)
        max_attempts  = getattr(cfg, "max_attempts", 2)
        njob          = cfg.njob
        ncore = int(cfg.ncore/njob)
        running: Dict[int, object] = {}
        print("Checking for already finished Q-Chem jobs (resume mode)...")
        for job in jobs:
            if os.path.exists(job.out_file):
                if self.check_qchem_success(job.out_file):
                    # job successfully completed earlier
                    job.started = True
                    job.finished = True
                    print(f"[Resume] Job {job.idx}{job.sign} already finished.")
                else:
                    # output exists but failed or incomplete
                    job.started = False
                    job.finished = False
            else:
                job.started = False
                job.finished = False
        while True:
            # ---- Poll running jobs ----
            completed_ids = []
            for jid, job in running.items():
                ret = job.popen.poll()
                if ret is not None:
                    job.finished = True
                    job.converged = self.check_qchem_success(job.out_file)
                    completed_ids.append(jid)

            # Remove completed entries
            for jid in completed_ids:
                running.pop(jid)

            # Print status table (if provided)
            if print_status_func:
                print_status_func(jobs, running, njob)

            # ---- Launch new jobs up to njob limit ----
            ready_jobs = [j for j in jobs if (not j.started and not j.finished)]
            while len(running) < njob and ready_jobs:
                job = ready_jobs.pop(0)
                job.attempts += 1
                print(f"Launching job {job.idx}{job.sign} attempt={job.attempts}")

                job.popen = run_qchem_job_async(
                    job.inp_file,
                    job.out_file,
                    ncore,
                    env_script,
                    launcher=cfg.launcher,
                    cache=job.cache
                )
                job.started = True
                job.start_time = time.time()
                running[id(job)] = job

            # ---- Check if all done ----
            if all(j.finished for j in jobs):
                print("All Q-Chem jobs completed.")
                break

            # ---- Retry failed ----
            for j in jobs:
                if j.finished and not j.converged:
                    if j.attempts < max_attempts:
                        print(f"Retry job {j.idx}{j.sign}")
                        j.started = False
                        j.finished = False

            time.sleep(poll_interval)

def run_qchem_job_async(inp_file, out_file, ncore, env_script, launcher="srun", cache=""):

    if launcher == "srun":
        cmd = f"""
{env_script}
export OMP_NUM_THREADS={ncore}
srun -n1 -c {ncore} --cpu-bind=cores --hint=nomultithread qchem -nt {ncore} {inp_file} {out_file} {cache}
"""
    else:
        cmd = f"""
{env_script}
export OMP_NUM_THREADS={ncore}
qchem -nt {ncore} {inp_file} {out_file} {cache}
"""

    return subprocess.Popen(
        cmd,
        shell=True,
        executable="/bin/bash",
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT
    )



from dataclasses import dataclass
from typing import Optional, List, Dict
@dataclass
class QMJob:
    inp_file: str
    out_file: str
    attempts: int = 0
    cache: str = "/scratch/moriya/cache/"
    popen: Optional[subprocess.Popen] = None
    started: bool = False
    finished: bool = False
    converged: bool = False
    start_time: Optional[float] = None

def print_status(jobs: List[QMJob], running: Dict[int, QMJob], njob: int):
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
    print("[RUN]      " + ", ".join(f"{os.path.basename(j.inp_file)}" for j in running_list))
    print("[READY]    " + ", ".join(f"{os.path.basename(j.inp_file)}" for j in ready_list))
    print("[DONE_OK]  " + ", ".join(f"{os.path.basename(j.inp_file)}" for j in done_ok))
    print("[DONE_FAIL]" + ", ".join(f"{os.path.basename(j.inp_file)}" for j in done_fail))
    print("=" * 88 + "\n")
