"""
prepare_tddft_inp.py

从 XYZ 文件或 AIMD 输出文件生成 Q-Chem TDDFT 输入文件

支持两种输入模式:
1. xyz 模式: 直接从 xyz 文件（单个或目录）生成输入
2. aimd 模式: 从 Q-Chem AIMD 输出文件提取结构并生成输入

使用方法:
    chemlab ml_data prepare_tddft_inp --mode xyz --path ./structures/ --ref ref.in
    chemlab ml_data prepare_tddft_inp --mode aimd --path ./md/ --file traj.out --ref ref.in
"""

import os
import glob
from pathlib import Path
import numpy as np

from chemlab.scripts.base import Script
from chemlab.config.config_loader import ConfigBase
from chemlab.util.modify_inp import single_spin_job, qchem_out_aimd_multi
from chemlab.util.ml_data import MLData


class PrepareTddftConfig(ConfigBase):
    """
    TDDFT 输入文件生成配置类
    """
    section_name = "prepare_tddft"


class PrepareTddftInp(Script):
    """
    从 XYZ 文件或 AIMD 输出生成 Q-Chem TDDFT 输入文件
    """

    name = "prepare_tddft_inp"
    config = PrepareTddftConfig

    def run(self, cfg):
        """主执行函数"""

        mode = getattr(cfg, "mode", "auto")

        if mode == "auto":
            mode = self._detect_mode(cfg)
            print(f"[prepare_tddft_inp] 自动检测模式: {mode}")

        if mode == "xyz":
            self._run_xyz_mode(cfg)
        elif mode == "aimd":
            self._run_aimd_mode(cfg)
        else:
            raise ValueError(f"未知模式: {mode}. 支持: xyz, aimd, auto")

    def _detect_mode(self, cfg):
        """自动检测输入模式"""
        path = cfg.path
        file_arg = getattr(cfg, "file", "")

        if file_arg and ".out" in file_arg:
            return "aimd"

        if path.endswith(".xyz"):
            return "xyz"

        if os.path.isdir(path):
            xyz_files = glob.glob(os.path.join(path, "*.xyz"))
            if xyz_files:
                return "xyz"
            out_files = glob.glob(os.path.join(path, "*.out"))
            if out_files:
                return "aimd"

        return "xyz"

    def _run_xyz_mode(self, cfg):
        """XYZ 模式: 从 xyz 文件生成 Q-Chem 输入"""

        path = cfg.path
        out_dir = cfg.out
        ref_file = cfg.ref
        charge = cfg.charge
        spin = cfg.spin

        os.makedirs(out_dir, exist_ok=True)

        xyz_files = self._collect_xyz_files(path)

        if not xyz_files:
            raise FileNotFoundError(f"在 {path} 中未找到 xyz 文件")

        print(f"[prepare_tddft_inp] 找到 {len(xyz_files)} 个 xyz 文件")
        print(f"[prepare_tddft_inp] 参考文件: {ref_file}")
        print(f"[prepare_tddft_inp] 电荷: {charge}, 自旋多重度: {spin}")
        print(f"[prepare_tddft_inp] 输出目录: {out_dir}")

        generated_count = 0
        for xyz_file in xyz_files:
            try:
                self._generate_inp_from_xyz(
                    xyz_file=xyz_file,
                    ref_file=ref_file,
                    out_dir=out_dir,
                    charge=charge,
                    spin=spin
                )
                generated_count += 1
            except Exception as e:
                print(f"[prepare_tddft_inp] 警告: 处理 {xyz_file} 时出错: {e}")

        print(f"[prepare_tddft_inp] 完成! 生成了 {generated_count} 个输入文件")

    def _run_aimd_mode(self, cfg):
        """AIMD 模式: 从 Q-Chem AIMD 输出文件提取结构并生成输入"""

        path = cfg.path
        files = cfg.file.split(",")
        out_dir = os.path.join(path, cfg.out)
        ref_file = cfg.ref
        charge = cfg.charge
        spin = cfg.spin
        dataset_size = cfg.dataset_size
        energy_unit = cfg.energy_unit
        distance_unit = cfg.distance_unit
        force_unit = cfg.force_unit

        full_files = [os.path.join(path, f.strip()) for f in files]

        print(f"[prepare_tddft_inp] AIMD 模式")
        print(f"[prepare_tddft_inp] 输入文件: {full_files}")

        multi = qchem_out_aimd_multi()
        multi.read_files(full_files)

        os.makedirs(out_dir, exist_ok=True)

        tmp_prefix = os.path.join(out_dir, "tmp_")
        multi.export_numpy(
            prefix=tmp_prefix,
            energy_unit=energy_unit,
            distance_unit=distance_unit,
            force_unit=force_unit
        )

        dataset = MLData(prefix=tmp_prefix, files=["coord", "energy", "grad", "type"])
        dataset.save_split(n_train=dataset_size, n_val=0, n_test=0, prefix=out_dir)
        dataset.export_xyz_from_split(
            split_file=os.path.join(out_dir, "split.npz"),
            outdir=out_dir,
            prefix_map=None
        )

        print(f"[prepare_tddft_inp] 生成输入文件, charge={charge}, spin={spin}")

        ref_full_path = os.path.join(path, ref_file)
        for xyz_file in Path(out_dir).glob('*.xyz'):
            self._generate_inp_from_xyz(
                xyz_file=str(xyz_file),
                ref_file=ref_full_path,
                out_dir=out_dir,
                charge=charge,
                spin=spin
            )

        print("[prepare_tddft_inp] 完成!")

    def _collect_xyz_files(self, path):
        """收集 xyz 文件列表"""
        if os.path.isfile(path):
            return [path] if path.endswith(".xyz") else []
        elif os.path.isdir(path):
            return sorted(glob.glob(os.path.join(path, "*.xyz")))
        return []

    def _generate_inp_from_xyz(self, xyz_file, ref_file, out_dir, charge, spin):
        """从单个 xyz 文件生成 Q-Chem 输入文件"""

        job = single_spin_job()
        job.charge = charge
        job.spin = spin
        job.ref_name = ref_file
        job.xyz_name = xyz_file

        xyz_basename = os.path.basename(xyz_file)
        out_prefix = out_dir.rstrip("/") + "/"
        job.generate_outputs(new_file_name=xyz_basename, prefix=out_prefix)
