# chemlab/util/result_saver.py
"""
通用的结果保存和可视化工具
支持各种计算任务(1d_scan, 2d_scan, MECP等)的结果自动保存和绘图
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime


class ResultSaver:
    """
    通用结果保存器，用于：
    1. 保存每个step的npz文件（结构、能量等）
    2. 自动绘图并保存
    3. 支持不同类型的计算任务

    使用方法：
        saver = ResultSaver(output_path="./results", prefix="scan")
        saver.save_step(step_idx=0, structure=xyz, energy=E, distance=d)
        saver.plot_1d_scan()
    """

    def __init__(
            self,
            output_path: str = "./results",
            prefix: str = "result",
            enable: bool = True,
            save_npz: bool = True,
            save_plot: bool = True,
            plot_format: str = "png",
            dpi: int = 300,
            auto_close_plots: bool = True
    ):
        """
        初始化ResultSaver

        Args:
            output_path: 输出路径
            prefix: 文件前缀
            enable: 是否启用（总开关）
            save_npz: 是否保存npz文件
            save_plot: 是否保存图片
            plot_format: 图片格式 (png, pdf, svg等)
            dpi: 图片分辨率
            auto_close_plots: 是否自动关闭图片（节省内存）
        """
        self.output_path = Path(output_path)
        self.prefix = prefix
        self.enable = enable
        self.save_npz = save_npz
        self.save_plot = save_plot
        self.plot_format = plot_format
        self.dpi = dpi
        self.auto_close_plots = auto_close_plots

        # 创建输出目录
        if self.enable:
            self.output_path.mkdir(parents=True, exist_ok=True)
            self.npz_path = self.output_path / "npz"
            self.plot_path = self.output_path / "plots"
            if self.save_npz:
                self.npz_path.mkdir(exist_ok=True)
            if self.save_plot:
                self.plot_path.mkdir(exist_ok=True)

        # 存储所有step的数据
        self.history = []

    def save_step(
            self,
            step_idx: int,
            structure: np.ndarray,
            energy: float,
            **kwargs
    ) -> Optional[str]:
        """
        保存单个step的结果

        Args:
            step_idx: step索引
            structure: 分子结构 (natom, 3) 或 (3, natom)
            energy: 能量值（Hartree）
            **kwargs: 其他数据，如:
                - distance: scan距离
                - gradient: 梯度
                - charge: 电荷
                - spin: 自旋
                - converged: 是否收敛
                - 任何其他需要保存的数据

        Returns:
            保存的npz文件路径，如果未启用则返回None
        """
        if not self.enable:
            return None

        # 整理数据
        data = {
            'step_idx': step_idx,
            'structure': np.array(structure),
            'energy': energy,
            'timestamp': datetime.now().isoformat(),
            **kwargs
        }

        # 添加到历史记录
        self.history.append(data)

        # 保存npz文件
        if self.save_npz:
            filename = f"{self.prefix}_step{step_idx:04d}.npz"
            filepath = self.npz_path / filename
            np.savez_compressed(filepath, **data)
            print(f"[ResultSaver] Saved: {filepath}")
            return str(filepath)

        return None

    def save_summary(self, **additional_data) -> Optional[str]:
        """
        保存所有step的汇总数据

        Args:
            **additional_data: 额外的汇总信息

        Returns:
            汇总文件路径
        """
        if not self.enable or not self.save_npz or not self.history:
            return None

        summary = {
            'total_steps': len(self.history),
            'history': self.history,
            **additional_data
        }

        filename = f"{self.prefix}_summary.npz"
        filepath = self.npz_path / filename
        np.savez_compressed(filepath, **summary)
        print(f"[ResultSaver] Summary saved: {filepath}")
        return str(filepath)

    def plot_1d_scan(
            self,
            title: str = "1D Potential Energy Scan",
            xlabel: str = "Distance (Å)",
            ylabel: str = "Relative Energy (kcal/mol)",
            show: bool = False
    ) -> Optional[str]:
        """
        绘制1D scan结果

        Args:
            title: 图片标题
            xlabel: x轴标签
            ylabel: y轴标签
            show: 是否显示图片

        Returns:
            保存的图片路径
        """
        if not self.enable or not self.save_plot or not self.history:
            return None

        # 提取数据
        distances = [h.get('distance', h.get('step_idx')) for h in self.history]
        energies = np.array([h['energy'] for h in self.history])

        # 转换为相对能量 (kcal/mol)
        hartree_to_kcal = 627.51
        rel_energies = (energies - energies.min()) * hartree_to_kcal

        # 绘图
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(distances, rel_energies, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3)

        # 标记最低点
        min_idx = np.argmin(rel_energies)
        ax.plot(distances[min_idx], rel_energies[min_idx], 'r*',
                markersize=15, label=f'Min at {distances[min_idx]:.2f}')
        ax.legend()

        plt.tight_layout()

        # 保存
        filename = f"{self.prefix}_1d_scan.{self.plot_format}"
        filepath = self.plot_path / filename
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        print(f"[ResultSaver] Plot saved: {filepath}")

        if show:
            plt.show()

        if self.auto_close_plots:
            plt.close(fig)

        return str(filepath)

    def plot_2d_scan(
            self,
            title: str = "2D Potential Energy Surface",
            xlabel: str = "Distance 1 (Å)",
            ylabel: str = "Distance 2 (Å)",
            cmap: str = "viridis",
            show: bool = False
    ) -> Optional[str]:
        """
        绘制2D scan结果

        Args:
            title: 图片标题
            xlabel: x轴标签
            ylabel: y轴标签
            cmap: 色图
            show: 是否显示图片

        Returns:
            保存的图片路径
        """
        if not self.enable or not self.save_plot or not self.history:
            return None

        # 提取数据 (假设history中有row_idx, col_idx)
        # 这需要根据实际的2d scan数据结构调整
        rows = [h.get('row_idx', 0) for h in self.history]
        cols = [h.get('col_idx', 0) for h in self.history]
        energies = np.array([h['energy'] for h in self.history])

        # 构建2D数组
        n_rows = max(rows) + 1
        n_cols = max(cols) + 1
        energy_grid = np.full((n_rows, n_cols), np.nan)

        for h in self.history:
            r = h.get('row_idx', 0)
            c = h.get('col_idx', 0)
            energy_grid[r, c] = h['energy']

        # 转换为相对能量
        hartree_to_kcal = 627.51
        rel_grid = (energy_grid - np.nanmin(energy_grid)) * hartree_to_kcal

        # 绘图
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(rel_grid, cmap=cmap, origin='lower', aspect='auto')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Relative Energy (kcal/mol)', fontsize=12)

        plt.tight_layout()

        # 保存
        filename = f"{self.prefix}_2d_scan.{self.plot_format}"
        filepath = self.plot_path / filename
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        print(f"[ResultSaver] Plot saved: {filepath}")

        if show:
            plt.show()

        if self.auto_close_plots:
            plt.close(fig)

        return str(filepath)

    def plot_convergence(
            self,
            title: str = "Convergence History",
            show: bool = False
    ) -> Optional[str]:
        """
        绘制收敛历史（如MECP优化）

        Args:
            title: 图片标题
            show: 是否显示图片

        Returns:
            保存的图片路径
        """
        if not self.enable or not self.save_plot or not self.history:
            return None

        steps = [h['step_idx'] for h in self.history]
        energies = np.array([h['energy'] for h in self.history])

        # 提取梯度范数（如果有）
        grad_norms = [h.get('gradient_norm', np.nan) for h in self.history]

        # 创建子图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

        # 能量
        ax1.plot(steps, energies, 'o-', linewidth=2)
        ax1.set_xlabel('Step', fontsize=12)
        ax1.set_ylabel('Energy (Hartree)', fontsize=12)
        ax1.set_title('Energy vs Step', fontsize=12)
        ax1.grid(True, alpha=0.3)

        # 梯度范数
        if not all(np.isnan(grad_norms)):
            ax2.semilogy(steps, grad_norms, 's-', linewidth=2, color='red')
            ax2.set_xlabel('Step', fontsize=12)
            ax2.set_ylabel('Gradient Norm', fontsize=12)
            ax2.set_title('Gradient Norm vs Step', fontsize=12)
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No gradient data',
                     ha='center', va='center', transform=ax2.transAxes)

        plt.suptitle(title, fontsize=14)
        plt.tight_layout()

        # 保存
        filename = f"{self.prefix}_convergence.{self.plot_format}"
        filepath = self.plot_path / filename
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        print(f"[ResultSaver] Plot saved: {filepath}")

        if show:
            plt.show()

        if self.auto_close_plots:
            plt.close(fig)

        return str(filepath)

    def plot_energy_gap(
            self,
            title: str = "Energy Gap History",
            show: bool = False
    ) -> Optional[str]:
        """
        绘制能量差历史（如MECP中两个态的能量差）

        Args:
            title: 图片标题
            show: 是否显示图片

        Returns:
            保存的图片路径
        """
        if not self.enable or not self.save_plot or not self.history:
            return None

        steps = [h['step_idx'] for h in self.history]

        # 提取两个态的能量
        e1 = np.array([h.get('energy_state1', np.nan) for h in self.history])
        e2 = np.array([h.get('energy_state2', np.nan) for h in self.history])

        if np.all(np.isnan(e1)) or np.all(np.isnan(e2)):
            print("[ResultSaver] No state energies found for gap plot")
            return None

        gap = np.abs(e1 - e2)

        # 绘图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))


        ax1.plot(steps, e1, 'o-', label='State 1', linewidth=2)
        ax1.plot(steps, e2, 's-', label='State 2', linewidth=2)
        ax1.set_xlabel('Step', fontsize=12)
        ax1.set_ylabel('Energy (Hartree)', fontsize=12)
        ax1.set_title('State Energies', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)


        ax2.plot(steps, gap * 627.51, 'd-', color='red', linewidth=2)
        ax2.set_xlabel('Step', fontsize=12)
        ax2.set_ylabel('Energy Gap (kcal/mol)', fontsize=12)
        ax2.set_title('Energy Gap', fontsize=12)
        ax2.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=14)
        plt.tight_layout()


        filename = f"{self.prefix}_energy_gap.{self.plot_format}"
        filepath = self.plot_path / filename
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        print(f"[ResultSaver] Plot saved: {filepath}")

        if show:
            plt.show()

        if self.auto_close_plots:
            plt.close(fig)

        return str(filepath)

    def load_step(self, step_idx: int) -> Optional[Dict[str, Any]]:

        if not self.enable or not self.save_npz:
            return None

        filename = f"{self.prefix}_step{step_idx:04d}.npz"
        filepath = self.npz_path / filename

        if not filepath.exists():
            print(f"[ResultSaver] File not found: {filepath}")
            return None

        with np.load(filepath, allow_pickle=True) as data:
            return {key: data[key] for key in data.files}

    def load_summary(self) -> Optional[Dict[str, Any]]:

        if not self.enable or not self.save_npz:
            return None

        filename = f"{self.prefix}_summary.npz"
        filepath = self.npz_path / filename

        if not filepath.exists():
            print(f"[ResultSaver] Summary file not found: {filepath}")
            return None

        with np.load(filepath, allow_pickle=True) as data:
            return {key: data[key] for key in data.files}


def create_result_saver_from_config(cfg) -> ResultSaver:

    return ResultSaver(
        output_path=getattr(cfg, 'result_saver_output_path', './results'),
        prefix=getattr(cfg, 'result_saver_prefix', 'result'),
        enable=getattr(cfg, 'result_saver_enable', True),
        save_npz=getattr(cfg, 'result_saver_save_npz', True),
        save_plot=getattr(cfg, 'result_saver_save_plot', True),
        plot_format=getattr(cfg, 'result_saver_plot_format', 'png'),
        dpi=getattr(cfg, 'result_saver_dpi', 300),
        auto_close_plots=getattr(cfg, 'result_saver_auto_close', True)
    )