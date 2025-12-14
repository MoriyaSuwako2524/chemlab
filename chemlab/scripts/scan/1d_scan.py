# -*- coding: utf-8 -*-
"""
改进的 1D Scan 脚本 - 集成了 ResultSaver
展示如何在现有脚本中使用ResultSaver自动保存结果和绘图

这个文件展示了如何修改现有的 chemlab/scripts/scan/1d_scan.py
"""
import os
import time
import subprocess
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
import numpy as np

from chemlab.scripts.base import Script
from chemlab.config.config_loader import ConfigBase
from chemlab.util.file_system import qchem_file, qchem_out_opt
from chemlab.config.config_loader import QchemEnvConfig
from chemlab.util.result_saver import create_result_saver_from_config  # 新增导入


# ======================
# Config (添加ResultSaver配置字段)
# ======================
class Scan1DConfig(ConfigBase):
    """
    1D Scan配置类

    新增的ResultSaver相关字段会自动从config.toml加载
    并自动生成命令行参数
    """
    section_name = "scan1d"


# ======================
# 其他辅助函数保持不变
# ======================
def make_scan_name(prefix: str, row_dis: float, row_idx: int, col_idx: int = 1) -> str:
    return f"{prefix}_{row_dis}_0.0_row{row_idx + 1}_col{col_idx}"


def run_qchem_job_async(input_file: str, output_file: str, nthreads: int,
                        env_script, launcher: str = "srun") -> subprocess.Popen:
    """Launch Q-Chem without blocking."""
    tag = os.path.splitext(os.path.basename(input_file))[0]

    if launcher == "srun":
        cmd = f"""
{env_script}
mkdir -p /scratch/$USER/$SLURM_JOB_ID/{tag}
export QCSCRATCH=/scratch/$USER/$SLURM_JOB_ID/{tag}
export OMP_NUM_THREADS={nthreads}
srun -n1 -c {nthreads} --cpu-bind=cores --hint=nomultithread qchem -nt {nthreads} {input_file} {output_file}
"""
    else:
        cmd = f"""
{env_script}
mkdir -p /scratch/$USER/{tag}
export QCSCRATCH=/scratch/$USER/{tag}
export OMP_NUM_THREADS={nthreads}
qchem -nt {nthreads} {input_file} {output_file}
"""
    return subprocess.Popen(
        cmd, shell=True, executable="/bin/bash",
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
    )


def write_input_from_carti(path: str, ref_filename: str, base_carti,
                           row_dis: float, inp_file: str) -> None:
    """Clone ref input and set coordinates + r12."""
    qf = qchem_file()
    qf.molecule.check = True
    qf.opt2.check = True
    qf.read_from_file(os.path.join(path, ref_filename))

    if base_carti is not None:
        qf.molecule.carti = base_carti

    if qf.opt2.r12:
        qf.opt2.modify_r12(0, row_dis)
    elif qf.opt2.r12mr34:
        qf.opt2.modify_r12mr34(0, row_dis)
    else:
        raise RuntimeError("Ref file missing r12/r12mr34 definition.")

    qf.generate_inp(inp_file)


def read_final_carti_if_converged(out_file: str) -> Tuple[Optional[list], bool]:
    if not os.path.exists(out_file):
        return None, False
    out = qchem_out_opt(out_file)
    out.read_file()
    return out.final_geom, out.opt_converged


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
    start_from_row: Optional[int] = None
    start_time: Optional[float] = None


def _fmt_job_list(jobs: List[ScanJob], max_show=16):
    items = [
        f"{j.row_idx}({j.row_dis:.3f}" + (f"<-{j.start_from_row}" if j.start_from_row is not None else "") + ")"
        for j in jobs
    ]
    if len(items) > max_show:
        return ", ".join(items[:max_show]) + f", ... (+{len(items) - max_show})"
    return ", ".join(items)


def print_status(jobs: List[ScanJob], running: Dict[int, ScanJob], scan_limit: int):
    running_list = sorted(running.values(), key=lambda j: j.row_idx)
    ready_list = [j for j in jobs if (not j.started) and (not j.finished) and (j.row_idx <= scan_limit)]
    done_ok = [j for j in jobs if j.finished and j.converged]
    done_fail = [j for j in jobs if j.finished and (not j.converged)]
    locked_list = [j for j in jobs if (not j.started) and (not j.finished) and (j.row_idx > scan_limit)]

    print("\n" + "=" * 78, flush=True)
    print(
        f"SCAN STATUS | scan_limit={scan_limit} "
        f"| RUN={len(running_list)} READY={len(ready_list)} "
        f"DONE_OK={len(done_ok)} DONE_FAIL={len(done_fail)} LOCK={len(locked_list)}",
        flush=True
    )
    print("-" * 78, flush=True)
    print("[RUN]      " + (_fmt_job_list(running_list) if running_list else "(none)"), flush=True)
    print("[READY]    " + (_fmt_job_list(ready_list) if ready_list else "(none)"), flush=True)
    print("[DONE_OK]  " + (_fmt_job_list(done_ok) if done_ok else "(none)"), flush=True)
    print("[DONE_FAIL]" + (_fmt_job_list(done_fail) if done_fail else "(none)"), flush=True)
    print("[LOCK]     " + (_fmt_job_list(locked_list) if locked_list else "(none)"), flush=True)
    print("=" * 78 + "\n", flush=True)


# ======================
# 主脚本 - 集成ResultSaver
# ======================
class Scan1D(Script):
    """
    Perform parallel 1D scan with Q-Chem.
    自动保存每个step的结果并绘图。
    """
    name = "scan1d"
    config = Scan1DConfig

    def run(self, cfg):
        # ========== 创建ResultSaver实例 ==========
        result_saver = create_result_saver_from_config(cfg)
        print(f"[Scan1D] ResultSaver enabled: {result_saver.enable}")

        # ========== 解包配置 ==========
        qenv = QchemEnvConfig()
        env_srcipt = qenv.env_script
        path = cfg.inp_path
        out_path = cfg.out_path
        prefix = cfg.prefix
        ref_filename = cfg.ref
        row_max = cfg.row_max
        row_start = cfg.row_start
        row_distance = cfg.row_distance
        ncore = cfg.ncore
        njob = cfg.njob
        scan_limit_init = cfg.scan_limit_init
        scan_limit_progress = cfg.scan_limit_progress
        poll_interval = cfg.poll_interval
        launcher = cfg.launcher

        os.makedirs(out_path, exist_ok=True)

        # ===== 加载参考输入文件 =====
        ref_qf = qchem_file()
        ref_qf.molecule.check = True
        ref_qf.opt2.check = True
        ref_qf.read_from_file(os.path.join(path, ref_filename))
        ref_carti = ref_qf.molecule.carti

        # ===== 构建任务列表 =====
        jobs = []
        for r in range(row_max):
            row_dis = round(row_start + row_distance * r, 3)
            name = make_scan_name(prefix, row_dis, r)
            inp_file = os.path.join(out_path, f"{name}.inp")
            out_file = os.path.join(out_path, f"{name}.inp.out")
            jobs.append(ScanJob(r, row_dis, inp_file, out_file))

        # ===== 预扫描已有输出 =====
        final_carti_by_row = {}
        finished = 0
        for job in jobs:
            carti, ok = read_final_carti_if_converged(job.out_file)
            if ok and carti:
                job.finished = True
                job.converged = True
                final_carti_by_row[job.row_idx] = carti
                finished += 1
                print(f"[FOUND_OK] row={job.row_idx}  r={job.row_dis}")

                # ========== 保存已完成的结果 ==========
                if result_saver.enable:
                    out = qchem_out_opt(job.out_file)
                    out.read_file()
                    result_saver.save_step(
                        step_idx=job.row_idx,
                        structure=np.array(carti).T,  # 转换为 (natom, 3)
                        energy=out.ene,
                        distance=job.row_dis,
                        converged=True,
                        row_idx=job.row_idx
                    )
            else:
                if os.path.exists(job.out_file):
                    print(f"[FOUND_FAIL] row={job.row_idx}")
                else:
                    print(f"[NOT_FOUND] row={job.row_idx}")

        highest_ok = max([-1] + list(final_carti_by_row.keys()))
        scan_limit = max(scan_limit_init, highest_ok + scan_limit_progress)

        running = {}

        def pick_start_carti(r: int):
            if not final_carti_by_row:
                return ref_carti, None
            nearest = min(final_carti_by_row.keys(), key=lambda x: abs(x - r))
            return final_carti_by_row[nearest], nearest

        # ===== 控制循环 =====
        print_status(jobs, running, scan_limit)

        while finished < row_max:
            # 轮询任务
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
                        final_carti_by_row[jidx := ridx] = carti
                        print(f"[DONE_OK] row={ridx}")

                        # ========== 保存新完成的结果 ==========
                        if result_saver.enable:
                            out = qchem_out_opt(job.out_file)
                            out.read_file()
                            result_saver.save_step(
                                step_idx=job.row_idx,
                                structure=np.array(carti).T,
                                energy=out.ene,
                                distance=job.row_dis,
                                converged=True,
                                row_idx=job.row_idx
                            )

                        # ========== 实时绘图（可选）==========
                        if result_saver.enable and result_saver.save_plot:
                            result_saver.plot_1d_scan(
                                title=f"1D Scan Progress (Step {finished}/{row_max})",
                                xlabel="Distance (Å)",
                                ylabel="Relative Energy (kcal/mol)"
                            )

                        new_limit = ridx + scan_limit_progress
                        if new_limit > scan_limit:
                            print(f"[PROMOTE] scan_limit {scan_limit} → {new_limit}")
                            scan_limit = max(scan_limit, new_limit)
                    else:
                        job.converged = False
                        print(f"[DONE_FAIL] row={ridx}")
                        if job.attempts < 3:
                            job.started = False
                            job.finished = False
                        else:
                            print(f"[GIVEUP] row={ridx}")

                    running.pop(ridx, None)

            print_status(jobs, running, scan_limit)

            # 启动新任务
            ready = [j for j in jobs if (not j.started) and (not j.finished) and (j.row_idx <= scan_limit)]
            while len(running) < njob and ready:
                job = ready.pop(0)
                base_carti, fr = pick_start_carti(job.row_idx)
                job.start_from_row = fr

                write_input_from_carti(path, ref_filename, base_carti, job.row_dis, job.inp_file)
                job.attempts += 1
                job.popen = run_qchem_job_async(job.inp_file, job.out_file, ncore, env_srcipt, launcher)
                job.started = True
                job.start_time = time.time()
                running[job.row_idx] = job

                print(f"[LAUNCH] row={job.row_idx}  from={fr} attempt={job.attempts}")

            print_status(jobs, running, scan_limit)
            time.sleep(poll_interval)

        # ===== 总结 =====
        n_conv = sum(j.converged for j in jobs)
        print(f"[ALL_DONE] Finished={finished}/{row_max}, Converged={n_conv}")

        # ========== 保存最终汇总 ==========
        if result_saver.enable:
            result_saver.save_summary(
                total_converged=n_conv,
                total_steps=row_max,
                scan_type="1d_scan",
                scan_range=[row_start, row_start + row_distance * (row_max - 1)],
                prefix=prefix
            )

            # 生成最终图片
            result_saver.plot_1d_scan(
                title=f"1D Scan Final Results ({n_conv}/{row_max} converged)",
                xlabel="Distance (Å)",
                ylabel="Relative Energy (kcal/mol)"
            )

            print("\n" + "=" * 78)
            print("[ResultSaver] All results saved to:")
            print(f"  NPZ files: {result_saver.npz_path}")
            print(f"  Plots:     {result_saver.plot_path}")
            print("=" * 78 + "\n")

        return jobs