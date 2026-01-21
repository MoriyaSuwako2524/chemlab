"""
prepare_tddft_inp.py

从 XYZ 轨迹文件或 AIMD 输出文件生成 Q-Chem TDDFT 输入文件

使用方法:
    chemlab ml_data prepare_tddft_inp --file traj.xyz --ref ref.in --out ./output/ --dataset_size 2000
    chemlab ml_data prepare_tddft_inp --file traj.out --ref ref.in --out ./output/ --dataset_size 2000
"""

import os
from pathlib import Path
import numpy as np

from chemlab.scripts.base import Script
from chemlab.config.config_loader import ConfigBase
from chemlab.util.modify_inp import single_spin_job, qchem_out_aimd_multi
from chemlab.util.ml_data import MLData
from chemlab.util.file_system import atom_charge_dict


class PrepareTddftConfig(ConfigBase):
    """
    TDDFT 输入文件生成配置类
    """
    section_name = "prepare_tddft"


class PrepareTddftInp(Script):
    """
    从 XYZ 轨迹或 AIMD 输出生成 Q-Chem TDDFT 输入文件
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
        dataset_size = cfg.dataset_size

        if not input_file:
            raise ValueError("必须指定输入文件 --file")

        os.makedirs(out_dir, exist_ok=True)

        # 根据文件后缀自动判断模式
        if input_file.endswith(".xyz"):
            self._run_xyz_mode(input_file, ref_file, out_dir, charge, spin, dataset_size)
        elif input_file.endswith(".out"):
            self._run_aimd_mode(input_file, ref_file, out_dir, charge, spin, cfg)
        else:
            raise ValueError(f"不支持的文件类型: {input_file}. 支持 .xyz 或 .out")

    def _run_xyz_mode(self, xyz_file, ref_file, out_dir, charge, spin, dataset_size):
        """XYZ 模式: 从多帧 xyz 轨迹文件生成 Q-Chem 输入"""

        print(f"[prepare_tddft_inp] XYZ 轨迹模式")
        print(f"[prepare_tddft_inp] 输入文件: {xyz_file}")
        print(f"[prepare_tddft_inp] 参考文件: {ref_file}")
        print(f"[prepare_tddft_inp] 电荷: {charge}, 自旋多重度: {spin}")
        print(f"[prepare_tddft_inp] 输出目录: {out_dir}")

        # 读取多帧 xyz 文件
        frames, atom_types = self._read_xyz_trajectory(xyz_file)
        n_frames = len(frames)
        print(f"[prepare_tddft_inp] 读取到 {n_frames} 帧")

        # 选择 frame
        if dataset_size > 0 and dataset_size < n_frames:
            indices = np.random.choice(n_frames, size=dataset_size, replace=False)
            indices = np.sort(indices)
            print(f"[prepare_tddft_inp] 随机选择 {dataset_size} 帧")
            selected_frames = [frames[i] for i in indices]
        else:
            indices = np.arange(n_frames)
            selected_frames = frames
            print(f"[prepare_tddft_inp] 使用全部 {n_frames} 帧")

        # 保存到 numpy 文件（类似 AIMD 模式）
        tmp_prefix = os.path.join(out_dir, "tmp_")
        self._export_xyz_to_numpy(
            frames=selected_frames,
            atom_types=atom_types,
            prefix=tmp_prefix
        )

        # 使用 MLData 加载并导出
        dataset = MLData(prefix=tmp_prefix, files=["coord", "type"])
        dataset.save_split(n_train=len(selected_frames), n_val=0, n_test=0, prefix=out_dir)
        dataset.export_xyz_from_split(
            split_file=os.path.join(out_dir, "split.npz"),
            outdir=out_dir,
            prefix_map=None
        )

        print(f"[prepare_tddft_inp] 生成输入文件, charge={charge}, spin={spin}")

        # 生成 Q-Chem 输入文件
        for xyz_file_out in sorted(Path(out_dir).glob('*.xyz')):
            job = single_spin_job()
            job.charge = charge
            job.spin = spin
            job.ref_name = ref_file
            job.xyz_name = str(xyz_file_out)

            xyz_basename = os.path.basename(str(xyz_file_out))
            out_prefix = out_dir.rstrip("/") + "/"
            job.generate_outputs(new_file_name=xyz_basename, prefix=out_prefix)

        print(f"[prepare_tddft_inp] 完成! 生成了 {len(selected_frames)} 个输入文件")

    def _export_xyz_to_numpy(self, frames, atom_types, prefix):
        """将 xyz 坐标导出为 numpy 文件，与 MLData 格式兼容"""

        # 转换为 numpy 数组
        coords = np.array(frames)  # shape: (n_frames, n_atoms, 3)
        atom_types_arr = np.array(atom_types, dtype=np.int32)  # shape: (n_atoms,)

        # 保存为 MLData 格式
        np.save(f"{prefix}coord.npy", coords)
        np.save(f"{prefix}type.npy", atom_types_arr)

        print(f"[prepare_tddft_inp] 保存了 {len(frames)} 帧到 numpy 文件")

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

        for xyz_file in sorted(Path(out_dir).glob('*.xyz')):
            job = single_spin_job()
            job.charge = charge
            job.spin = spin
            job.ref_name = ref_file
            job.xyz_name = str(xyz_file)

            xyz_basename = os.path.basename(str(xyz_file))
            out_prefix = out_dir.rstrip("/") + "/"
            job.generate_outputs(new_file_name=xyz_basename, prefix=out_prefix)

        print("[prepare_tddft_inp] 完成!")

    def _read_xyz_trajectory(self, xyz_file):
        """读取多帧 xyz 轨迹文件"""
        frames = []
        atom_types = None

        with open(xyz_file, 'r') as f:
            content = f.read()

        lines = content.strip().split('\n')
        i = 0
        while i < len(lines):
            # 读取原子数
            natoms = int(lines[i].strip())
            i += 1

            # 跳过注释行
            i += 1

            # 读取坐标
            coords = []
            types = []
            for j in range(natoms):
                parts = lines[i].split()
                types.append(parts[0])
                coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
                i += 1

            frames.append(np.array(coords))
            if atom_types is None:
                atom_types = types

        return frames, atom_types

    def _generate_inp(self, coords, atom_types, ref_file, out_dir, charge, spin, frame_idx):
        """从坐标生成单个 Q-Chem 输入文件"""

        # 构建 carti 格式 (atom_type, x, y, z)
        carti = []
        for atom, (x, y, z) in zip(atom_types, coords):
            carti.append([atom, x, y, z])

        job = single_spin_job()
        job.charge = charge
        job.spin = spin
        job.ref_name = ref_file
        job._xyz.carti = carti

        out_prefix = out_dir.rstrip("/") + "/"
        out_name = f"frame_{frame_idx:04d}.xyz"
        job.generate_outputs(new_file_name=out_name, prefix=out_prefix)