# -*- coding: utf-8 -*-
import os
import time
import subprocess
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
from chemlab.util.file_system import qchem_file, qchem_out_opt

# ================== Cluster / Q-Chem env ==================
QCHEM_ENV_SETUP = """
module purge
export QC=/scratch/moriya/software/soc
export QCAUX=/scratch/zhengpei/q-chem/qcaux
source $QC/bin/qchem.setup.sh
export QCSCRATCH=/scratch/$USER
module load impi/2021.2.0
module load intel/2021.2.0
"""

# ================== Utils ==================
def make_scan_name(prefix: str, row_dis: float, row_idx: int, col_idx: int = 1) -> str:
    return f"{prefix}_{row_dis}_0.0_row{row_idx+1}_col{col_idx}"

def run_qchem_job_async(input_file: str, output_file: str, nthreads: int, launcher: str = "srun") -> subprocess.Popen:
    """
    Launch one Q-Chem task in background.
    We intentionally discard Q-Chem stdout/stderr; we only track state.
    """
    scratch_tag = os.path.splitext(os.path.basename(input_file))[0]

    if launcher == "srun":
        # Do NOT use --exclusive/-N1 to avoid serializing across nodes.
        cmd = f"""
{QCHEM_ENV_SETUP}
mkdir -p /scratch/$USER/$SLURM_JOB_ID/{scratch_tag}
export QCSCRATCH=/scratch/$USER/$SLURM_JOB_ID/{scratch_tag}
export OMP_NUM_THREADS={nthreads}
srun -n1 -c {nthreads} --cpu-bind=cores --hint=nomultithread qchem -nt {nthreads} {input_file} {output_file}
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

def write_input_from_carti(path: str, ref_filename: str, base_carti, row_dis: float, inp_file: str) -> None:
    """
    Build an input by cloning the ref .inp, replacing coordinates and r12 distance.
    Requires: ref file contains $opt2 with an r12 line.
    """
    qf = qchem_file()
    qf.molecule.check = True
    qf.opt2.check = True
    qf.read_from_file(os.path.join(path, ref_filename))
    if base_carti is not None:
        qf.molecule.carti = base_carti
    if qf.opt2.r12:
        qf.opt2.modify_r12(0, row_dis)  # modify first r12 line
    elif qf.opt2.r12mr34:
        qf.opt2.modify_r12mr34(0, row_dis)
    else:
        raise RuntimeError("Ref input has no $opt2 r12 or r12mr34 definition; cannot modify scan coordinate.")
    qf.generate_inp(inp_file)

def read_final_carti_if_converged(out_file: str) -> Tuple[Optional[list], bool]:
    """
    Parse an optimization .out; return (final_carti, converged).
    """
    if not os.path.exists(out_file):
        return None, False
    out = qchem_out_opt(out_file)
    out.read_file()  # will call parse() internally
    return out.final_geom, out.opt_converged

# ================== Status Panel ==================
def _fmt_job_list(jobs: List["ScanJob"], max_show: int = 16) -> str:
    """Format job list: idx(dis<-from)."""
    items = [f"{j.row_idx}({j.row_dis:.3f}" + (f"<-{j.start_from_row}" if j.start_from_row is not None else "") + ")"
             for j in jobs]
    if len(items) > max_show:
        return ", ".join(items[:max_show]) + f", ... (+{len(items)-max_show})"
    return ", ".join(items)

def print_status(jobs: List["ScanJob"], running: Dict[int, "ScanJob"], scan_limit: int) -> None:
    """Print real-time status panel (ASCII only)."""
    running_list = sorted(running.values(), key=lambda j: j.row_idx)
    ready_list   = sorted([j for j in jobs if (not j.started) and (not j.finished) and (j.row_idx <= scan_limit)],
                          key=lambda j: j.row_idx)
    done_ok      = sorted([j for j in jobs if j.finished and j.converged], key=lambda j: j.row_idx)
    done_fail    = sorted([j for j in jobs if j.finished and (not j.converged)], key=lambda j: j.row_idx)
    locked_list  = sorted([j for j in jobs if (not j.started) and (not j.finished) and (j.row_idx > scan_limit)],
                          key=lambda j: j.row_idx)

    total = len(jobs)
    print("\n" + "="*78, flush=True)
    print(f"SCAN STATUS  |  scan_limit={scan_limit}  |  total={total}  "
          f"|  RUN={len(running_list)} READY={len(ready_list)} "
          f"DONE_OK={len(done_ok)} DONE_FAIL={len(done_fail)} LOCK={len(locked_list)}", flush=True)
    print("-"*78, flush=True)

    # RUN with elapsed time
    if running_list:
        now = time.time()
        lines = []
        for j in running_list:
            elapsed = (now - j.start_time) if j.start_time else 0.0
            lines.append(f"{j.row_idx}({j.row_dis:.3f}" +
                         (f"<-{j.start_from_row}" if j.start_from_row is not None else "") +
                         f") t={elapsed/60:.1f}min try={j.attempts}")
        print("[RUN]   " + ", ".join(lines), flush=True)
    else:
        print("[RUN]   (none)", flush=True)

    print("[READY] " + (_fmt_job_list(ready_list) if ready_list else "(none)"), flush=True)
    print("[DONE_OK] " + (_fmt_job_list(done_ok) if done_ok else "(none)"), flush=True)
    print("[DONE_FAIL] " + (_fmt_job_list(done_fail) if done_fail else "(none)"), flush=True)
    print("[LOCK]  " + (_fmt_job_list(locked_list) if locked_list else "(none)"), flush=True)
    print("="*78 + "\n", flush=True)

# ================== Job Structures ==================
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
    start_from_row: Optional[int] = None  # which row's final carti used
    start_time: Optional[float] = None    # when it started

# ================== Main Runner ==================
def run_1d_scan_parallel(
    path: str,
    prefix: str,
    ref_filename: str,
    row_max: int,
    row_start: float,
    row_distance: float,
    ncore: int = 8,
    njob: int = 8,
    scan_limit_init: int = 0,
    scan_limit_progress = 4,
    poll_interval: float = 8.0,
    launcher: str = "srun",  # "srun" for multi-node, "local" for single node
) -> None:
    """
    Parallel 1D scan with strict scan_limit gating:
      - Up to `njob` concurrent Q-Chem processes, each with `ncore` threads.
      - Only rows with row_idx <= scan_limit are eligible to START.
      - Each time any job finishes (success/fail), promote scan_limit to unlock the next row.
      - Row i inherits start geometry from the highest converged row < i; otherwise from ref.
    """
    os.makedirs(path, exist_ok=True)

    # Load ref once
    ref_qf = qchem_file()
    ref_qf.molecule.check = True
    ref_qf.opt2.check = True
    ref_qf.read_from_file(os.path.join(path, ref_filename))
    ref_carti = ref_qf.molecule.carti

    # Build job list
    jobs: List[ScanJob] = []
    for r in range(row_max):
        row_dis = round(row_start + row_distance * r, 3)
        name = make_scan_name(prefix, row_dis, r, col_idx=1)
        inp_file = os.path.join(path, f"{name}.inp")
        out_file = os.path.join(path, f"{name}.inp.out")
        jobs.append(ScanJob(row_idx=r, row_dis=row_dis, inp_file=inp_file, out_file=out_file))

    # Pre-scan existing outputs
    final_carti_by_row: Dict[int, list] = {}
    finished = 0
    for job in jobs:
        carti, ok = read_final_carti_if_converged(job.out_file)
        if ok and carti:
            job.finished = True
            job.converged = True
            final_carti_by_row[job.row_idx] = carti
            finished += 1
            print(f"[FOUND_OK] row={job.row_idx}  r={job.row_dis} A  (already converged)", flush=True)
        else:
            if os.path.exists(job.out_file):
                print(f"[FOUND_FAIL] row={job.row_idx}  r={job.row_dis} A  (will retry)", flush=True)
            else:
                print(f"[NOT_FOUND] row={job.row_idx}  r={job.row_dis} A  (will run)", flush=True)

    highest_done = max([-1] + list(final_carti_by_row.keys()))
    scan_limit = max(scan_limit_init, highest_done + scan_limit_progress)

    def pick_start_carti(r: int) -> Tuple[list, Optional[int]]:
        """Pick the nearest converged row (lower or higher); fallback to ref."""
        if not final_carti_by_row:
            return ref_carti, None
        nearest = min(final_carti_by_row.keys(), key=lambda x: abs(x - r))
        return final_carti_by_row[nearest], nearest


    running: Dict[int, ScanJob] = {}

    # Initial panel
    print_status(jobs, running, scan_limit)

    while finished < row_max:
        # ---- Poll running jobs
        to_clear = []
        for ridx, job in list(running.items()):
            if job.popen is None:
                continue
            ret = job.popen.poll()
            if ret is not None:
                job.finished = True
                finished += 1
                carti, ok = read_final_carti_if_converged(job.out_file)
                if ok and carti:
                    job.converged = True
                    final_carti_by_row[ridx] = carti
                    print(f"[DONE_OK] row={ridx}  r={job.row_dis:.3f} A  (from={job.start_from_row})", flush=True)
                    # ✅ 只有成功时才推进 scan_limit
                    new_limit = ridx + scan_limit_init
                    if new_limit > scan_limit:
                        print(f"[PROMOTE] scan_limit {scan_limit} -> {new_limit}", flush=True)
                    scan_limit = max(scan_limit, new_limit)
                else:
                    job.converged = False
                    print(f"[DONE_FAIL] row={ridx}  r={job.row_dis:.3f} A  (from={job.start_from_row})", flush=True)
                    # 失败的点重新尝试，直到超过最大尝试次数
                    if job.attempts < 3:
                        job.started = False
                        job.finished = False
                        print(f"[RETRY] row={ridx}  r={job.row_dis:.3f} A  (attempt={job.attempts+1})", flush=True)
                    else:
                        print(f"[GIVEUP] row={ridx} after {job.attempts} attempts", flush=True)
                to_clear.append(ridx)

        for ridx in to_clear:
            running.pop(ridx, None)

        # Status after polling
        print_status(jobs, running, scan_limit)

        # ---- Launch new jobs while we have free slots
        ready = [j for j in jobs if (not j.started) and (not j.finished) and (j.row_idx <= scan_limit)]
        while len(running) < njob and ready:
            job = ready.pop(0)
            start_carti, from_row = pick_start_carti(job.row_idx)
            job.start_from_row = from_row
            write_input_from_carti(path, ref_filename, start_carti, job.row_dis, job.inp_file)
            job.attempts += 1
            job.popen = run_qchem_job_async(job.inp_file, job.out_file, nthreads=ncore, launcher=launcher)
            job.started = True
            job.start_time = time.time()
            running[job.row_idx] = job
            print(f"[LAUNCH] row={job.row_idx}  r={job.row_dis:.3f} A  "
                  f"(attempt={job.attempts}, from={from_row}, threads={ncore})", flush=True)

        # Status after launching
        print_status(jobs, running, scan_limit)

        if finished >= row_max:
            break
        time.sleep(poll_interval)


    n_conv = sum(1 for j in jobs if j.converged)
    print(f"[ALL_DONE] Finished={finished}/{row_max}, Converged={n_conv}", flush=True)


if __name__ == "__main__":
    run_1d_scan_parallel(
        path="/scratch/moriya/calculation/Fe/scan/TS2/",
        prefix="TS",
        ref_filename="ref.in",
        row_max=40,
        row_start=2.0,
        row_distance=-0.02,
        ncore=16,            # 8 cores per Q-Chem job
        njob=4,             # 2 nodes x (32/8) = 8 concurrent jobs
        scan_limit_init=5,
        scan_limit_progress=4,
        poll_interval=60.0,
        launcher="srun",    # use Slurm multi-node steps
    )
