"""
prepare_tddft_inp.py

从 XYZ 文件或 AIMD 输出文件生成 Q-Chem TDDFT 输入文件

使用方法:
    chemlab ml_data prepare_tddft_inp --file mol.xyz --ref ref.in --out ./output/
    chemlab ml_data prepare_tddft_inp --file traj.out --ref ref.in --out ./output/
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

        input_file = cfg.file
        ref_file = cfg.ref
        out_dir = cfg.out
        charge = cfg.charge
        spin = cfg.spin

        if not input_file:
            raise ValueError("必须指定输入文件 --file")

        os.makedirs(out_dir, exist_ok=True)

        # 根据文件后缀自动判断模式
        if input_file.endswith(".xyz"):
            self._run_xyz_mode(input_file, ref_file, out_dir, charge, spin)
        elif input_file.endswith(".out"):
            self._run_aimd_mode(input_file, ref_file, out_dir, charge, spin, cfg)
        else:
            raise ValueError(f"不支持的文件类型: {input_file}. 支持 .xyz 或 .out")

    def _run_xyz_mode(self, xyz_file, ref_file, out_dir, charge, spin):
        """XYZ 模式: 从单个 xyz 文件生成 Q-Chem 输入"""

        print(f"[prepare_tddft_inp] XYZ 模式")
        print(f"[prepare_tddft_inp] 输入文件: {xyz_file}")
        print(f"[prepare_tddft_inp] 参考文件: {ref_file}")
        print(f"[prepare_tddft_inp] 电荷: {charge}, 自旋多重度: {spin}")
        print(f"[prepare_tddft_inp] 输出目录: {out_dir}")

        job = single_spin_job()
        job.charge = charge
        job.spin = spin
        job.ref_name = ref_file
        job.xyz_name = xyz_file

        xyz_basename = os.path.basename(xyz_file)
        out_prefix = out_dir.rstrip("/") + "/"
        job.generate_outputs(new_file_name=xyz_basename, prefix=out_prefix)

        print("[prepare_tddft_inp] 完成!")

    def _run_aimd_mode(self, out_file, ref_file, out_dir, charge, spin, cfg):
        """AIMD 模式: 从 Q-Chem AIMD 输出文件提取结构并生成输入"""

        dataset_size = cfg.dataset_size
        energy_unit = cfg.energy_unit
        distance_unit = cfg.distance_unit
        force_unit = cfg.force_unit

        print(f"[prepare_tddft_inp] AIMD 模式")
        print(f"[prepare_tddft_inp] 输入文件: {out_file}")
        print(f"[prepare_tddft_inp] 参考文件: {ref_file}")

        multi = qchem_out_aimd_multi()
        multi.read_files([out_file])

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

        for xyz_file in Path(out_dir).glob('*.xyz'):
            job = single_spin_job()
            job.charge = charge
            job.spin = spin
            job.ref_name = ref_file
            job.xyz_name = str(xyz_file)

            xyz_basename = os.path.basename(str(xyz_file))
            out_prefix = out_dir.rstrip("/") + "/"
            job.generate_outputs(new_file_name=xyz_basename, prefix=out_prefix)

        print("[prepare_tddft_inp] 完成!")