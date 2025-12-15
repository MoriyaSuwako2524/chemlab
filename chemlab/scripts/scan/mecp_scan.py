# -*- coding: utf-8 -*-
"""
MECP Scan Script - 在不同距离约束下进行 MECP 优化
结合了 run_mecp.py 和 1d_scan.py 的功能
"""
import os
import time
import subprocess
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
import numpy as np
import matplotlib.pyplot as plt

from chemlab.scripts.base import Script
from chemlab.config.config_loader import ConfigBase, QchemEnvConfig
from chemlab.util.mecp import mecp, mecp_soc
from chemlab.util.file_system import qchem_file, Hartree_to_kcal
from chemlab.util.result_saver import create_result_saver_from_config


# ======================
# Config
# ======================
class MecpScanConfig(ConfigBase):
    """
    MECP Scan 配置类
    在不同距离约束下运行 MECP 优化
    """
    section_name = "mecp_scan"


# ======================
# 辅助函数
# ======================
@dataclass
class MecpScanJob:
    """单个 MECP scan 点的任务"""
    scan_idx: int  # scan 点索引
    distance: float  # 约束距离
    work_dir: str  # 工作目录
    started: bool = False
    finished: bool = False
    converged: bool = False
    attempts: int = 0
    mecp_obj: Optional[object] = None
    final_energy_1: Optional[float] = None
    final_energy_2: Optional[float] = None
    final_structure: Optional[np.ndarray] = None
    atom_i = None
    atom_j = None


def make_scan_dir_name(prefix: str, distance: float, scan_idx: int) -> str:
    """生成 scan 点的目录名"""
    return f"{prefix}_d{distance:.3f}_idx{scan_idx}"


def check_mecp_convergence(work_dir: str, prefix: str, job_num: int) -> Tuple[bool, Optional[float], Optional[float]]:
    """
    检查 MECP 优化是否收敛

    Returns:
        (converged, energy_1, energy_2)
    """
    log_file = os.path.join(work_dir, "mecp.log")
    if not os.path.exists(log_file):
        return False, None, None

    # 简单检查：查找 "Converged" 关键字
    with open(log_file, 'r') as f:
        content = f.read()
        if "Converged at step" in content:
            # 尝试读取最终能量（这里简化处理）
            return True, None, None  # 能量从 mecp 对象中获取更可靠

    return False, None, None


# ======================
# 主脚本
# ======================
class MecpScan(Script):
    """
    在不同距离约束下执行 MECP 优化

    使用场景：
    - 研究 MECP 点沿某个关键距离的变化
    - 生成 MECP 能量随距离的扫描曲线
    """

    name = "mecp_scan"
    config = MecpScanConfig

    def run(self, cfg):
        # ========== 加载环境配置 ==========
        qenv = QchemEnvConfig()
        env_script = qenv.env_script.strip()
        if not env_script:
            raise ValueError("[mecp_scan] No env_script found in [qchem_env].")

        print("=" * 78)
        print("[MECP Scan] Configuration:")
        print("=" * 78)
        print(f"Reference file: {cfg.ref_file}")
        print(f"Reference path: {cfg.path}")
        print(f"Output path: {cfg.out_path}")
        print(f"Prefix: {cfg.prefix}")
        print(f"Spin states: {cfg.spin1}, {cfg.spin2}")
        print(f"Restrain atoms: {cfg.restrain_atom_i} - {cfg.restrain_atom_j}")
        print(f"Distance range: {cfg.distance_start} to {cfg.distance_end}")
        print(f"Distance step: {cfg.distance_step}")
        print(f"Restrain force constant: {cfg.restrain_k}")
        print(f"MECP jobtype: {cfg.jobtype}")
        print(f"MECP max steps: {cfg.mecp_max_steps}")
        print(f"Threads per job: {cfg.nthreads}")
        print(f"Parallel jobs: {cfg.njob}")
        print("=" * 78 + "\n")

        os.makedirs(cfg.out_path, exist_ok=True)

        # ========== 创建 ResultSaver ==========
        result_saver = create_result_saver_from_config(cfg)
        print(f"[MecpScan] ResultSaver enabled: {result_saver.enable}\n")

        # ========== 构建任务列表 ==========
        jobs = []
        distances = np.arange(cfg.distance_start, cfg.distance_end + cfg.distance_step / 2, cfg.distance_step)

        for idx, dist in enumerate(distances):
            dist = round(dist, 3)
            dir_name = make_scan_dir_name(cfg.prefix, dist, idx)
            work_dir = os.path.join(cfg.out_path, dir_name)
            os.makedirs(work_dir, exist_ok=True)

            jobs.append(MecpScanJob(
                scan_idx=idx,
                distance=dist,
                work_dir=work_dir
            ))

        print(f"[INFO] Created {len(jobs)} scan jobs for distances: {distances}\n")

        # ========== 预扫描已完成的任务 ==========
        finished_count = 0
        for job in jobs:
            converged, e1, e2 = check_mecp_convergence(job.work_dir, cfg.prefix, 0)
            if converged:
                job.finished = True
                job.converged = True
                finished_count += 1
                print(f"[FOUND] Scan {job.scan_idx} (d={job.distance:.3f}) already converged")

        print(f"\n[INFO] Found {finished_count}/{len(jobs)} completed jobs\n")

        # ========== 运行 MECP scan ==========
        running: Dict[int, MecpScanJob] = {}

        def print_status():
            """打印当前状态"""
            n_running = len(running)
            n_ready = sum(1 for j in jobs if not j.started and not j.finished)
            n_done = sum(1 for j in jobs if j.finished and j.converged)
            n_fail = sum(1 for j in jobs if j.finished and not j.converged)

            print("\n" + "=" * 78)
            print(f"[STATUS] RUNNING={n_running} | READY={n_ready} | DONE={n_done} | FAIL={n_fail}")
            print("=" * 78 + "\n")

        print_status()

        while finished_count < len(jobs):

            # ===== 检查正在运行的任务 =====
            for scan_idx, job in list(running.items()):
                converged, e1, e2 = check_mecp_convergence(job.work_dir, cfg.prefix, cfg.mecp_max_steps - 1)

                if converged:
                    job.finished = True
                    job.converged = True
                    job.final_energy_1 = e1
                    job.final_energy_2 = e2
                    finished_count += 1

                    print(f"[DONE] Scan {job.scan_idx} (d={job.distance:.3f}) CONVERGED")

                    # 保存结果
                    if result_saver.enable and job.mecp_obj:
                        try:
                            structure = job.mecp_obj.state_1.inp.molecule.return_xyz_list().astype(float)
                            energy_avg = (job.mecp_obj.state_1.out.ene + job.mecp_obj.state_2.out.ene) / 2

                            result_saver.save_step(
                                step_idx=job.scan_idx,
                                structure=structure,
                                energy=energy_avg,
                                distance=job.distance,
                                converged=True,
                                scan_idx=job.scan_idx,
                                energy_state1=job.mecp_obj.state_1.out.ene,
                                energy_state2=job.mecp_obj.state_2.out.ene
                            )
                        except Exception as e:
                            print(f"[WARNING] Failed to save results for scan {job.scan_idx}: {e}")

                    running.pop(scan_idx)

                elif job.attempts >= 1:  # MECP 优化已运行完毕（无论是否收敛）
                    job.finished = True
                    job.converged = False
                    finished_count += 1
                    print(f"[FAIL] Scan {job.scan_idx} (d={job.distance:.3f}) did not converge")
                    running.pop(scan_idx)

            # ===== 启动新任务 =====
            ready_jobs = [j for j in jobs if not j.started and not j.finished]

            while len(running) < cfg.njob and ready_jobs:
                job = ready_jobs.pop(0)

                print(f"[START] Launching scan {job.scan_idx} (d={job.distance:.3f}) in {job.work_dir}")

                # 创建 MECP 对象
                if cfg.jobtype == "mecp":
                    mecp_obj = mecp()
                    mecp_obj.different_type = cfg.gradient_type
                elif cfg.jobtype == "mecp_soc":
                    mecp_obj = mecp_soc()
                else:
                    mecp_obj = mecp()

                # 配置 MECP
                mecp_obj.ref_path = cfg.path
                mecp_obj.ref_filename = cfg.ref_file
                mecp_obj.out_path = job.work_dir
                mecp_obj.prefix = cfg.prefix
                mecp_obj.step_size = cfg.step_size
                mecp_obj.max_stepsize = cfg.max_stepsize
                mecp_obj.state_1.spin = cfg.spin1
                mecp_obj.state_2.spin = cfg.spin2
                mecp_obj.converge_limit = cfg.mecp_conv

                # **关键：添加距离约束**
                mecp_obj.add_restrain(
                    cfg.restrain_atom_i,
                    cfg.restrain_atom_j,
                    job.distance,  # 当前 scan 点的目标距离
                    cfg.restrain_k
                )

                # 初始化
                mecp_obj.read_init_structure()
                print(f"[START] Generating Inp {job.scan_idx} (d={job.distance:.3f}) in {mecp_obj.out_path}")
                mecp_obj.generate_new_inp()
                mecp_obj.initialize_bfgs()

                # 保存 mecp 对象供后续使用
                job.mecp_obj = mecp_obj
                job.started = True
                job.attempts += 1
                running[job.scan_idx] = job

                # 启动 MECP 优化（后台运行）
                self._run_mecp_optimization_async(job, env_script, cfg)

            print_status()
            time.sleep(cfg.poll_interval)

        # ========== 总结 ==========
        n_converged = sum(j.converged for j in jobs)
        print("\n" + "=" * 78)
        print(f"[COMPLETE] MECP Scan finished: {n_converged}/{len(jobs)} converged")
        print("=" * 78 + "\n")

        # ========== 保存和绘图 ==========
        if result_saver.enable:
            result_saver.save_summary(
                total_converged=n_converged,
                total_steps=len(jobs),
                scan_type="mecp_scan",
                scan_range=[cfg.distance_start, cfg.distance_end],
                prefix=cfg.prefix
            )

            # 绘制 MECP 能量曲线
            result_saver.plot_1d_scan(
                title=f"MECP Scan ({n_converged}/{len(jobs)} converged)",
                xlabel="Distance (Å)",
                ylabel="MECP Energy (kcal/mol, relative)"
            )

            print(f"[ResultSaver] Results saved to: {result_saver.output_path}")

        return jobs

    def _run_mecp_optimization_async(self, job: MecpScanJob, env_script: str, cfg):
        """
        在后台运行 MECP 优化
        这里简化实现：直接在主进程中运行（因为 MECP 本身需要同步等待 Q-Chem）
        如果需要真正的并行，需要更复杂的进程管理
        """
        mecp_obj = job.mecp_obj
        log_path = os.path.join(job.work_dir, "mecp.log")

        with open(log_path, "w", buffering=1) as log:
            log.write(f"MECP Scan Point {job.scan_idx}\n")
            log.write(f"Distance constraint: {job.distance:.3f} Å\n")
            log.write(f"Restrain: atoms {cfg.restrain_atom_i}-{cfg.restrain_atom_j}\n")
            log.write("=" * 60 + "\n\n")

            # MECP 优化循环
            for step in range(cfg.mecp_max_steps):
                print(f"  [Scan {job.scan_idx}] MECP step {step}")
                log.write(f"\n>>> MECP iteration step {step}\n")

                mecp_obj.job_num = step
                mecp_obj.state_1.inp.molecule.modify_bond_length(cfg.restrain_atom_i,cfg.restrain_atom_j,job.distance)
                mecp_obj.generate_new_inp()
                # 运行两个态的 Q-Chem 计算

                processes, out_files = [], []
                if cfg.jobtype == "mecp":
                    for state in [mecp_obj.state_1, mecp_obj.state_2]:
                        inp = os.path.join(mecp_obj.out_path, state.job_name)
                        out = inp[:-4] + ".out"
                        out_files.append(out)

                        cmd = f"""{env_script}
    export QCSCRATCH=/scratch/$USER/mecp_scan_{job.scan_idx}_{step}
    qchem -nt {cfg.nthreads // 2} {inp} {out}
    """
                        p = subprocess.Popen(cmd, shell=True, executable="/bin/bash")
                        processes.append(p)
                elif cfg.jobtype == "mecp_soc":
                    inp = os.path.join(mecp_obj.out_path, mecp_obj.state_1.job_name)
                    out = inp[:-4] + ".out"
                    out_files.append(out)
                    cmd = f"""{env_script}
                srun -n1 -c {cfg.nthreads} --cpu-bind=cores --hint=nomultithread qchem -nt {cfg.nthreads} {inp} {out} > qc.log 2>&1
                """
                    p = subprocess.Popen(cmd, shell=True, executable="/bin/bash")
                    processes.append(p)

                # 等待完成
                for p in processes:
                    p.wait()

                # 读取结果
                mecp_obj.read_output()
                mecp_obj.calc_new_gradient()

                e1 = mecp_obj.state_1.out.ene
                e2 = mecp_obj.state_2.out.ene

                if e1 and e2:
                    gap = abs(e1 - e2) * Hartree_to_kcal
                    log.write(f"Energy gap: {gap:.4f} kcal/mol\n")

                # 更新结构
                mecp_obj.update_structure()

                # 检查收敛
                if mecp_obj.check_convergence():
                    log.write(f"\n>>> Converged at step {step}\n")
                    print(f"  [Scan {job.scan_idx}] MECP converged at step {step}")
                    break
            else:
                log.write(f"\n>>> Did not converge after {cfg.mecp_max_steps} steps\n")
                print(f"  [Scan {job.scan_idx}] MECP did not converge")