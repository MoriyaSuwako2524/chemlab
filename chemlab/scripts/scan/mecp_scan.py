import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import os
import time
import subprocess
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple
from enum import Enum
import numpy as np
import traceback

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



class JobStatus(Enum):
    PENDING = "pending"  # 等待启动
    RUNNING_QCHEM = "running"  # Q-Chem正在运行
    WAITING_READ = "waiting"  # 等待读取结果
    CONVERGED = "converged"  # MECP收敛
    FAILED = "failed"  # 失败（SCF失败或其他错误）
    GAVE_UP = "gave_up"  # 多次失败后放弃


# ======================
# 辅助数据类
# ======================
@dataclass
class MecpScanJob:
    """单个 MECP scan 点的任务"""
    scan_idx: int  # scan 点索引
    distance: float  # 约束距离
    work_dir: str  # 工作目录

    # 状态管理
    status: JobStatus = JobStatus.PENDING
    current_step: int = 0  # 当前MECP优化步数
    attempts: int = 0  # 尝试次数（用于重试）
    max_attempts: int = 2  # 最大尝试次数

    # MECP对象和进程
    mecp_obj: Optional[object] = None
    processes: List[subprocess.Popen] = field(default_factory=list)
    out_files: List[str] = field(default_factory=list)

    # 结果
    final_energy_1: Optional[float] = None
    final_energy_2: Optional[float] = None
    final_structure: Optional[np.ndarray] = None
    error_message: Optional[str] = None

    # 时间戳
    start_time: Optional[float] = None

    @property
    def is_active(self) -> bool:
        """任务是否活跃（可以被处理）"""
        return self.status in [JobStatus.PENDING, JobStatus.RUNNING_QCHEM, JobStatus.WAITING_READ]

    @property
    def is_finished(self) -> bool:
        """任务是否已完成（成功或失败）"""
        return self.status in [JobStatus.CONVERGED, JobStatus.FAILED, JobStatus.GAVE_UP]


def make_scan_dir_name(prefix: str, distance: float, scan_idx: int) -> str:
    """生成 scan 点的目录名"""
    return f"{prefix}_d{distance:.3f}_idx{scan_idx}"


def check_qchem_success(out_file: str) -> bool:
    """检查Q-Chem是否成功完成"""
    if not os.path.exists(out_file):
        return False
    try:
        with open(out_file, 'r') as f:
            content = f.read()
            return "Thank you very much for using Q-Chem" in content
    except Exception:
        return False


def check_qchem_scf_error(out_file: str) -> bool:
    """检查是否有SCF收敛错误"""
    if not os.path.exists(out_file):
        return False
    try:
        with open(out_file, 'r') as f:
            content = f.read()
            error_keywords = [
                "SCF failed to converge",
                "Maximum number of SCF cycles exceeded",
            ]
            return any(kw in content for kw in error_keywords)
    except Exception:
        return False


def check_mecp_log_convergence(work_dir: str) -> bool:
    """检查 MECP 日志是否显示收敛"""
    log_file = os.path.join(work_dir, "mecp.log")
    if not os.path.exists(log_file):
        return False
    try:
        with open(log_file, 'r') as f:
            return "Converged at step" in f.read()
    except Exception:
        return False


def _fmt_job_list(jobs: List[MecpScanJob], max_show: int = 12) -> str:
    """格式化任务列表用于显示"""
    items = [f"{j.scan_idx}(d={j.distance:.3f},step={j.current_step})" for j in jobs]
    if len(items) > max_show:
        return ", ".join(items[:max_show]) + f", ... (+{len(items) - max_show})"
    return ", ".join(items) if items else "(none)"


def print_status(jobs: List[MecpScanJob], running: Dict[int, MecpScanJob]):
    """打印当前任务状态"""
    running_list = [j for j in jobs if j.status == JobStatus.RUNNING_QCHEM]
    pending_list = [j for j in jobs if j.status == JobStatus.PENDING]
    converged_list = [j for j in jobs if j.status == JobStatus.CONVERGED]
    failed_list = [j for j in jobs if j.status in [JobStatus.FAILED, JobStatus.GAVE_UP]]

    print("\n" + "=" * 78, flush=True)
    print(f"[MECP SCAN STATUS] RUN={len(running_list)} PENDING={len(pending_list)} "
          f"CONVERGED={len(converged_list)} FAILED={len(failed_list)}", flush=True)
    print("-" * 78, flush=True)
    print(f"[RUNNING]   {_fmt_job_list(running_list)}", flush=True)
    print(f"[PENDING]   {_fmt_job_list(pending_list)}", flush=True)
    print(f"[CONVERGED] {_fmt_job_list(converged_list)}", flush=True)
    print(f"[FAILED]    {_fmt_job_list(failed_list)}", flush=True)
    print("=" * 78 + "\n", flush=True)


# ======================
# 主脚本
# ======================
class MecpScan(Script):
    """
    在不同距离约束下执行 MECP 优化 - 改进版

    特点：
    - 真正的并行执行：同时运行多个MECP点
    - 容错处理：SCF失败不会中断整个任务
    - 支持重试机制
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
        print("[MECP Scan - Improved Version] Configuration:")
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
        jobs: List[MecpScanJob] = []
        distances = np.arange(
            cfg.distance_start,
            cfg.distance_end + cfg.distance_step / 2,
            cfg.distance_step
        )

        for idx, dist in enumerate(distances):
            dist = round(dist, 3)
            dir_name = make_scan_dir_name(cfg.prefix, dist, idx)
            work_dir = os.path.join(cfg.out_path, dir_name)
            os.makedirs(work_dir, exist_ok=True)

            job = MecpScanJob(
                scan_idx=idx,
                distance=dist,
                work_dir=work_dir,
                max_attempts=getattr(cfg, 'max_attempts', 2)
            )
            jobs.append(job)

        print(f"[INFO] Created {len(jobs)} scan jobs for distances: "
              f"{[round(d, 3) for d in distances]}\n")

        # ========== 预扫描已完成的任务 ==========
        for job in jobs:
            if check_mecp_log_convergence(job.work_dir):
                job.status = JobStatus.CONVERGED
                print(f"[FOUND] Scan {job.scan_idx} (d={job.distance:.3f}) already converged")

        n_already_done = sum(1 for j in jobs if j.status == JobStatus.CONVERGED)
        print(f"\n[INFO] Found {n_already_done}/{len(jobs)} completed jobs\n")

        # ========== 主循环 ==========
        running: Dict[int, MecpScanJob] = {}

        print_status(jobs, running)

        while any(j.is_active for j in jobs):
            # ===== 1. 检查正在运行的Q-Chem进程 =====
            for scan_idx in list(running.keys()):
                job = running[scan_idx]

                if job.status != JobStatus.RUNNING_QCHEM:
                    continue

                # 检查所有进程是否完成
                all_done = all(p.poll() is not None for p in job.processes)

                if all_done:
                    job.status = JobStatus.WAITING_READ

                    # 检查是否有SCF错误
                    has_scf_error = any(check_qchem_scf_error(f) for f in job.out_files)
                    all_success = all(check_qchem_success(f) for f in job.out_files)

                    if has_scf_error or not all_success:
                        # SCF失败处理
                        job.error_message = "SCF convergence failed"
                        print(f"[SCF_FAIL] Scan {job.scan_idx} step {job.current_step} SCF failed")

                        if job.attempts < job.max_attempts:
                            # 重试：重置状态
                            print(f"[RETRY] Scan {job.scan_idx} will retry (attempt {job.attempts + 1})")
                            job.status = JobStatus.PENDING
                            job.current_step = 0
                            job.mecp_obj = None
                            running.pop(scan_idx)
                        else:
                            # 放弃
                            job.status = JobStatus.GAVE_UP
                            print(f"[GAVE_UP] Scan {job.scan_idx} gave up after {job.attempts} attempts")
                            running.pop(scan_idx)
                        continue

                    # 读取输出并更新
                    try:
                        self._process_qchem_output(job, cfg, result_saver)

                        if job.status == JobStatus.CONVERGED:
                            print(f"[CONVERGED] Scan {job.scan_idx} (d={job.distance:.3f}) "
                                  f"converged at step {job.current_step}")
                            running.pop(scan_idx)
                        elif job.status == JobStatus.FAILED:
                            running.pop(scan_idx)
                        else:
                            # 继续下一步MECP优化
                            job.current_step += 1
                            if job.current_step >= cfg.mecp_max_steps:
                                job.status = JobStatus.FAILED
                                job.error_message = "Max MECP steps exceeded"
                                print(f"[MAX_STEPS] Scan {job.scan_idx} exceeded max steps")
                                running.pop(scan_idx)
                            else:
                                # 启动下一步Q-Chem计算
                                self._launch_qchem_step(job, env_script, cfg)

                    except Exception as e:
                        job.status = JobStatus.FAILED
                        job.error_message = str(e)
                        print(f"[ERROR] Scan {job.scan_idx} error: {e}")
                        traceback.print_exc()
                        running.pop(scan_idx)

            # ===== 2. 启动新任务 =====
            pending_jobs = [j for j in jobs if j.status == JobStatus.PENDING]

            while len(running) < cfg.njob and pending_jobs:
                job = pending_jobs.pop(0)

                print(f"[START] Launching scan {job.scan_idx} (d={job.distance:.3f})")

                try:
                    # 初始化MECP对象
                    self._initialize_mecp_job(job, cfg)
                    job.attempts += 1
                    job.start_time = time.time()

                    # 启动第一步Q-Chem计算
                    self._launch_qchem_step(job, env_script, cfg)

                    running[job.scan_idx] = job

                except Exception as e:
                    job.status = JobStatus.FAILED
                    job.error_message = str(e)
                    print(f"[INIT_ERROR] Scan {job.scan_idx} initialization failed: {e}")
                    traceback.print_exc()

            print_status(jobs, running)
            time.sleep(cfg.poll_interval)

        # ========== 总结 ==========
        n_converged = sum(1 for j in jobs if j.status == JobStatus.CONVERGED)
        n_failed = sum(1 for j in jobs if j.status in [JobStatus.FAILED, JobStatus.GAVE_UP])

        print("\n" + "=" * 78)
        print(f"[COMPLETE] MECP Scan finished:")
        print(f"  - Converged: {n_converged}/{len(jobs)}")
        print(f"  - Failed:    {n_failed}/{len(jobs)}")
        print("=" * 78 + "\n")

        # 打印失败任务详情
        if n_failed > 0:
            print("[Failed Jobs Details]:")
            for job in jobs:
                if job.status in [JobStatus.FAILED, JobStatus.GAVE_UP]:
                    print(f"  Scan {job.scan_idx} (d={job.distance:.3f}): {job.error_message}")
            print()

        # ========== 保存和绘图 ==========
        if result_saver.enable:
            result_saver.save_summary(
                total_converged=n_converged,
                total_steps=len(jobs),
                scan_type="mecp_scan",
                scan_range=[cfg.distance_start, cfg.distance_end],
                prefix=cfg.prefix
            )

            result_saver.plot_1d_scan(
                title=f"MECP Scan ({n_converged}/{len(jobs)} converged)",
                xlabel="Distance (Å)",
                ylabel="MECP Energy (kcal/mol, relative)"
            )

            print(f"[ResultSaver] Results saved to: {result_saver.output_path}")

        return jobs

    def _initialize_mecp_job(self, job: MecpScanJob, cfg):
        """初始化单个MECP任务"""
        # 创建 MECP 对象
        if cfg.jobtype == "mecp":
            mecp_obj = mecp()
            mecp_obj.different_type = cfg.gradient_type
        elif cfg.jobtype == "mecp_soc":
            mecp_obj = mecp_soc()
        else:
            mecp_obj = mecp()
        print(f"Current mecp opject:{mecp_obj}")

        mecp_obj.ref_path = cfg.path
        mecp_obj.ref_filename = cfg.ref_file
        mecp_obj.out_path = job.work_dir
        mecp_obj.prefix = cfg.prefix
        mecp_obj.step_size = cfg.step_size
        mecp_obj.max_stepsize = cfg.max_stepsize
        mecp_obj.state_1.spin = cfg.spin1
        mecp_obj.state_2.spin = cfg.spin2
        mecp_obj.converge_limit = cfg.mecp_conv


        mecp_obj.add_restrain(
            cfg.restrain_atom_i,
            cfg.restrain_atom_j,
            job.distance,
            cfg.restrain_k
        )

        # 初始化
        mecp_obj.read_init_structure()

        # 修改初始结构的键长
        mecp_obj.state_1.inp.molecule.modify_bond_length(
            cfg.restrain_atom_i, cfg.restrain_atom_j, job.distance
        )
        mecp_obj.state_2.inp.molecule.modify_bond_length(
            cfg.restrain_atom_i, cfg.restrain_atom_j, job.distance
        )

        mecp_obj.initialize_bfgs()

        job.mecp_obj = mecp_obj
        job.current_step = 0

        # 创建日志文件
        log_path = os.path.join(job.work_dir, "mecp.log")
        with open(log_path, "w") as log:
            log.write(f"MECP Scan Point {job.scan_idx}\n")
            log.write(f"Distance constraint: {job.distance:.3f} Å\n")
            log.write(f"Restrain: atoms {cfg.restrain_atom_i}-{cfg.restrain_atom_j}\n")
            log.write("=" * 60 + "\n\n")

    def _launch_qchem_step(self, job: MecpScanJob, env_script: str, cfg):
        """启动单步Q-Chem计算（异步）"""
        mecp_obj = job.mecp_obj
        mecp_obj.job_num = job.current_step
        mecp_obj.generate_new_inp()

        job.processes = []
        job.out_files = []

        # 计算每个Q-Chem任务的线程数
        nthreads_per_job = cfg.nthreads // 2 if cfg.jobtype == "mecp" else cfg.nthreads

        if cfg.jobtype == "mecp":
            # 两个态分别运行
            for state in [mecp_obj.state_1, mecp_obj.state_2]:
                inp = os.path.join(mecp_obj.out_path, state.job_name)
                out = inp[:-4] + ".out"
                job.out_files.append(out)

                scratch_dir = f"mecp_scan_{job.scan_idx}_{job.current_step}_{state._spin}"
                cmd = f"""{env_script}
mkdir -p /scratch/$USER/cache/{scratch_dir}
export QCSCRATCH=/scratch/$USER/cache/{scratch_dir}
export OMP_NUM_THREADS={nthreads_per_job}
qchem -nt {nthreads_per_job} {inp} {out}
"""
                p = subprocess.Popen(
                    cmd, shell=True, executable="/bin/bash",
                    stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
                )
                job.processes.append(p)

        elif cfg.jobtype == "mecp_soc":
            inp = os.path.join(mecp_obj.out_path, mecp_obj.state_1.job_name)
            out = inp[:-4] + ".out"
            job.out_files.append(out)

            scratch_dir = f"mecp_scan_{job.scan_idx}_{job.current_step}"
            cmd = f"""{env_script}
mkdir -p /scratch/$USER/cache/{scratch_dir}
export QCSCRATCH=/scratch/$USER/cache/{scratch_dir}
export OMP_NUM_THREADS={nthreads_per_job}
srun -n1 -c {nthreads_per_job} --cpu-bind=cores --hint=nomultithread qchem -nt {nthreads_per_job} {inp} {out}
"""
            p = subprocess.Popen(
                cmd, shell=True, executable="/bin/bash",
                stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
            )
            job.processes.append(p)

        job.status = JobStatus.RUNNING_QCHEM

        # 记录日志
        log_path = os.path.join(job.work_dir, "mecp.log")
        with open(log_path, "a") as log:
            log.write(f"\n>>> MECP iteration step {job.current_step}\n")

    def _process_qchem_output(self, job: MecpScanJob, cfg, result_saver):
        mecp_obj = job.mecp_obj

        # 读取结果
        mecp_obj.read_output()
        mecp_obj.calc_new_gradient()

        e1 = mecp_obj.state_1.out.ene
        e2 = mecp_obj.state_2.out.ene

        # 记录能量
        log_path = os.path.join(job.work_dir, "mecp.log")
        with open(log_path, "a") as log:
            if e1 is not None and e2 is not None:
                gap = abs(e1 - e2) * Hartree_to_kcal
                log.write(f"E1 = {e1:.8f} Ha, E2 = {e2:.8f} Ha\n")
                log.write(f"Energy gap: {gap:.4f} kcal/mol\n")

        # 更新结构
        mecp_obj.update_structure()
        mecp_obj.plot_energy_progress()

        # 检查收敛
        if mecp_obj.check_convergence():
            job.status = JobStatus.CONVERGED
            job.final_energy_1 = e1
            job.final_energy_2 = e2

            # 记录收敛
            with open(log_path, "a") as log:
                log.write(f"\n>>> Converged at step {job.current_step}\n")

            # 保存结果
            if result_saver.enable:
                try:
                    structure = mecp_obj.state_1.inp.molecule.return_xyz_list().astype(float)
                    energy_avg = (e1 + e2) / 2 if e1 and e2 else None

                    result_saver.save_step(
                        step_idx=job.scan_idx,
                        structure=structure,
                        energy=energy_avg,
                        distance=job.distance,
                        converged=True,
                        scan_idx=job.scan_idx,
                        energy_state1=e1,
                        energy_state2=e2
                    )
                except Exception as e:
                    print(f"[WARNING] Failed to save results for scan {job.scan_idx}: {e}")
        else:
            # 继续优化
            job.status = JobStatus.WAITING_READ  # 会在主循环中进入下一步



    def _plot_convergence_progress(self, job: MecpScanJob, cfg):

        import matplotlib.pyplot as plt

        log_path = os.path.join(job.work_dir, "mecp.log")
        if not os.path.exists(log_path):
            return

        # 解析日志文件
        steps = []
        e1_list = []
        e2_list = []
        gap_list = []

        try:
            with open(log_path, 'r') as f:
                current_step = None
                for line in f:
                    if "MECP iteration step" in line:
                        try:
                            current_step = int(line.split("step")[-1].strip())
                        except:
                            pass
                    elif "E1 =" in line and "E2 =" in line:
                        # 格式: E1 = -xxx.xxx Ha, E2 = -xxx.xxx Ha
                        parts = line.split(',')
                        try:
                            e1 = float(parts[0].split('=')[1].strip().split()[0])
                            e2 = float(parts[1].split('=')[1].strip().split()[0])
                            if current_step is not None:
                                steps.append(current_step)
                                e1_list.append(e1)
                                e2_list.append(e2)
                        except:
                            pass
                    elif "Energy gap:" in line:
                        try:
                            gap = float(line.split(':')[1].strip().split()[0])
                            gap_list.append(gap)
                        except:
                            pass
        except Exception as e:
            print(f"[WARNING] Failed to parse log for scan {job.scan_idx}: {e}")
            return

        if not steps:
            return

        # 创建图表
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        # 子图1: 能量变化
        ax1 = axes[0]
        ax1.plot(steps, e1_list, 'o-', label=f'State 1 (spin={cfg.spin1})', linewidth=2, markersize=6)
        ax1.plot(steps, e2_list, 's-', label=f'State 2 (spin={cfg.spin2})', linewidth=2, markersize=6)
        ax1.set_xlabel('MECP Step', fontsize=12)
        ax1.set_ylabel('Energy (Hartree)', fontsize=12)
        ax1.set_title(f'MECP Point {job.scan_idx} - Distance = {job.distance:.3f} Å',
                      fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # 子图2: 能量差
        ax2 = axes[1]
        if gap_list:
            ax2.plot(steps, gap_list, 'o-', color='red', linewidth=2, markersize=6)
            ax2.axhline(y=cfg.mecp_conv * Hartree_to_kcal, color='green',
                        linestyle='--', label=f'Convergence threshold ({cfg.mecp_conv * Hartree_to_kcal:.4f} kcal/mol)')
            ax2.set_xlabel('MECP Step', fontsize=12)
            ax2.set_ylabel('Energy Gap (kcal/mol)', fontsize=12)
            ax2.set_title('Energy Gap Convergence', fontsize=12)
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)
            ax2.set_yscale('log')

        plt.tight_layout()

        # 保存图片
        plot_path = os.path.join(job.work_dir, f"convergence_progress.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"[PLOT] Saved convergence plot for scan {job.scan_idx}: {plot_path}", flush=True)

