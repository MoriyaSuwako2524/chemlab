import os
import time
import subprocess
from dataclasses import dataclass, field
from typing import Dict, Optional, List
from util.file_system import qchem_file, qchem_out_file, molecule
import numpy as np

QCHEM_ENV_SETUP = """
module purge
export QC=/scratch/moriya/software/soc
export QCAUX=/scratch/zhengpei/q-chem/qcaux
source $QC/bin/qchem.setup.sh
export QCSCRATCH=/scratch/$USER
module load impi/2021.2.0
module load intel/2021.2.0
"""

def run_qchem_job_async(input_file: str, output_file: str, nthreads: int, launcher: str = "srun") -> subprocess.Popen:
    """
    Launch Q-Chem as a background job.
    launcher: "srun" (recommended under Slurm multi-node), or "local".
    """
    scratch_tag = os.path.splitext(os.path.basename(input_file))[0]

    if launcher == "srun":
        cmd = f"""
{QCHEM_ENV_SETUP}
mkdir -p /scratch/$USER/$SLURM_JOB_ID/{scratch_tag}
export QCSCRATCH=/scratch/$USER/$SLURM_JOB_ID/{scratch_tag}
export OMP_NUM_THREADS={nthreads}
srun --exclusive -N1 -n1 -c {nthreads} --cpu-bind=cores qchem -nt {nthreads} {input_file} {output_file}
"""
    else:
        cmd = f"""
{QCHEM_ENV_SETUP}
mkdir -p /scratch/$USER/{scratch_tag}
export QCSCRATCH=/scratch/$USER/{scratch_tag}
export OMP_NUM_THREADS={nthreads}
qchem -nt {nthreads} {input_file} {output_file}
"""

    return subprocess.Popen(cmd, shell=True, executable="/bin/bash",
                            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def write_input_from_carti(ref_path: str, ref_filename: str, base_carti: np.ndarray,
                           row_dis: float, inp_file: str) -> None:
    """Build an input by taking the ref file blocks but replacing carti and r12."""
    qf = qchem_file()
    qf.molecule.check = True
    qf.opt2.check = True
    qf.read_from_file(os.path.join(ref_path, ref_filename))
    if base_carti is not None:
        qf.molecule.carti = base_carti
    qf.opt2.modify_r12(0, row_dis)
    with open(inp_file, "w") as f:
        f.write(
            qf.molecule.return_output_format()
            + qf.remain_texts
            + qf.opt2.return_output_format()
        )

def read_final_carti_if_converged(out_file: str) -> Optional[np.ndarray]:
    """Return final carti if OPT converged, else None."""
    if not os.path.exists(out_file):
        return None
    qof = qchem_out_file()
    qof.read_opt_from_file(out_file)
    if getattr(qof, "opt_converged", False):
        return qof.return_final_molecule_carti()
    return None

@dataclass
class ScanJob:
    row_idx: int
    row_dis: float
    inp_file: str
    out_file: str
    attempts: int = 0
    popen: Optional[subprocess.Popen] = None
    started: bool = False
    finished: bool = False
    converged: bool = False
    start_carti_from_row: Optional[int] = None  # which row's final carti used

def run_1d_scan_parallel(
    path: str = "/scratch/moriya/calculation/Fe/scan/N-O4/",
    prefix: str = "Int2-O12-O3",
    ref_filename: str = "Int2-O12-O3.inp",
    row_max: int = 20,
    row_start: float = 2.2,
    row_distance: float = -0.05,
    ncore: int = 32,
    njob: int = 2,
    scan_limit_init: int = 0,
    poll_interval: float = 5.0,
) -> None:
    """
    Parallel 1D scan with progressive scan_limit expansion.
    - At most `njob` jobs in parallel; each uses `ncore` threads.
    - scan_limit starts at `scan_limit_init`. We can start any job with row_idx <= scan_limit.
    - Each time a job finishes (success or fail), we increase scan_limit to max(scan_limit, row_idx+1).
    - A job's starting geometry prefers the highest-converged row < row_idx; otherwise uses ref.
    """

    os.makedirs(path, exist_ok=True)

    # 1) Load ref file once
    ref_qf = qchem_file()
    ref_qf.molecule.check = True
    ref_qf.opt2.check = True
    ref_qf.read_from_file(os.path.join(path, ref_filename))
    ref_carti = ref_qf.molecule.carti

    # 2) Pre-construct jobs
    jobs: List[ScanJob] = []
    for row in range(row_max):
        row_dis = round(row_start + row_distance * row, 3)
        scan_name = make_scan_name(prefix, row_dis, row, col_idx=1)
        inp_file = os.path.join(path, f"{scan_name}.inp")
        out_file = os.path.join(path, f"{scan_name}.inp.out")
        jobs.append(ScanJob(row_idx=row, row_dis=row_dis, inp_file=inp_file, out_file=out_file))

    # 3) Pre-scan existing outputs
    final_carti_by_row: Dict[int, np.ndarray] = {}
    finished_count = 0
    for job in jobs:
        carti = read_final_carti_if_converged(job.out_file)
        if carti is not None:
            job.finished = True
            job.converged = True
            final_carti_by_row[job.row_idx] = carti
            finished_count += 1
            print(f"🟩 Already converged: row={job.row_idx}  r={job.row_dis} Å")
        else:
            if os.path.exists(job.out_file):
                print(f"❌ Previously failed: row={job.row_idx}  r={job.row_dis} Å — will retry.")
            else:
                print(f"🔹 Not found: row={job.row_idx}  r={job.row_dis} Å — will run.")

    # 4) Initialize scan_limit considering already-finished rows
    #    We want scan_limit to be the highest (row+1) among completed jobs, but
    #    at least scan_limit_init.
    highest_done = max([-1] + [j for j, _ in final_carti_by_row.items()])
    scan_limit = max(scan_limit_init, highest_done + 1)

    # Helper: pick best starting carti for a row
    def pick_start_carti_for_row(r: int):
        # Highest converged j < r
        candidates = [j for j in final_carti_by_row.keys() if j < r]
        if candidates:
            jbest = max(candidates)
            return final_carti_by_row[jbest], jbest
        return ref_carti, None

    # 5) Main scheduler loop
    running: Dict[int, ScanJob] = {}  # row_idx -> job
    while finished_count < row_max:
        # (a) Poll running jobs
        to_remove = []
        for row_idx, job in running.items():
            if job.popen is None:
                continue
            ret = job.popen.poll()
            if ret is not None:
                # process finished
                job.finished = True
                finished_count += 1
                carti = read_final_carti_if_converged(job.out_file)
                if carti is not None:
                    job.converged = True
                    final_carti_by_row[row_idx] = carti
                    print(f"✅ Converged: row={row_idx}  r={job.row_dis:.3f} Å  "
                          f"(start from row={job.start_carti_from_row})")
                else:
                    job.converged = False
                    print(f"❌ Not converged: row={row_idx}  r={job.row_dis:.3f} Å  "
                          f"(start from row={job.start_carti_from_row})")
                # expand scan_limit based on this finished job
                new_limit = max(scan_limit, row_idx + 1)
                if new_limit != scan_limit:
                    print(f"📈 scan_limit {scan_limit} → {new_limit}")
                scan_limit = new_limit
                to_remove.append(row_idx)

        for row_idx in to_remove:
            running.pop(row_idx, None)

        # (b) Launch new jobs while we have slots
        can_start = [j for j in jobs
                     if (not j.started) and (not j.finished) and (j.row_idx <= scan_limit)]
        while len(running) < njob and can_start:
            job = can_start.pop(0)
            # Choose starting carti
            start_carti, from_row = pick_start_carti_for_row(job.row_idx)
            job.start_carti_from_row = from_row
            # (Re)write input
            write_input_from_carti(path, ref_filename, start_carti, job.row_dis, job.inp_file)
            # Launch
            job.attempts += 1
            job.popen = run_qchem_job_async(job.inp_file, job.out_file, nthreads=ncore, launcher="srun")
            job.started = True
            running[job.row_idx] = job
            print(f"▶️ Launch row={job.row_idx}  r={job.row_dis:.3f} Å  "
                  f"(attempt {job.attempts}, start from row={from_row}, "
                  f"threads={ncore})")

        if finished_count >= row_max:
            break

        # (c) Sleep before next poll
        time.sleep(poll_interval)

    # 6) Summary
    n_conv = sum(1 for j in jobs if j.converged)
    print(f"🎉 All done. Finished={finished_count}/{row_max}, Converged={n_conv}")

if __name__ == "__main__":
    run_1d_scan_parallel(
        path="/scratch/moriya/calculation/Fe/scan/N-O4/",
        prefix="Int2-O12-O3",
        ref_filename="Int2-O12-O3.inp",
        row_max=20,
        row_start=2.2,
        row_distance=-0.05,
        ncore=8,
        njob=8,
        scan_limit_init=0,
        poll_interval=8.0
    )

