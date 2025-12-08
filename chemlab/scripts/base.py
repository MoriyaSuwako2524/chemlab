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
            return None
        with open(out_file) as f:
            if "Thank you very much for using Q-Chem" in f.read():
                return 0
            elif "Error in gen_scfman" in f.read():
                return 1
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
        ncore = inr(cfg.ncore/njob)
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
                    launcher=cfg.launcher
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

def run_qchem_job_async(inp_file, out_file, ncore, env_script, launcher="srun"):

    if launcher == "srun":
        cmd = f"""
{env_script}
export OMP_NUM_THREADS={ncore}
srun -n1 -c {ncore} --cpu-bind=cores --hint=nomultithread qchem -nt {ncore} {inp_file} {out_file}
"""
    else:
        cmd = f"""
{env_script}
export OMP_NUM_THREADS={ncore}
qchem -nt {ncore} {inp_file} {out_file}
"""

    return subprocess.Popen(
        cmd,
        shell=True,
        executable="/bin/bash",
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT
    )




