# chemlab.scripts 文档

## 概述
:I Claude生成的，我自己实在懒得写document


`chemlab.scripts` 是 chemlab 项目的核心脚本模块，提供量子化学计算、机器学习数据处理和几何扫描的自动化工具。所有脚本通过 CLI 命令调用：

```bash
chemlab <module> <script> [options]
```

## 模块结构

| 模块 | 功能 |
|------|------|
| `base` | 基础类：Script、QchemBaseScript |
| `ml_data` | 机器学习数据处理 |
| `qchem` | Q-Chem 计算工具 |
| `scan` | 势能面扫描 |

---

## base 模块

### Script 基类

所有 CLI 脚本的父类，子类需定义：

- `name` - CLI 子命令名
- `config` - 配置类（可选）
- `run(self, cfg)` - 执行方法

### QchemBaseScript 基类

Q-Chem 脚本专用基类，提供：

- `check_qchem_success(out_file)` - 检查计算是否成功
- `wait_for_jobs(out_files, log, interval)` - 等待作业完成
- `run_jobs(jobs, cfg)` - 并行运行作业

---

## ml_data 模块

### export_numpy

将 TDDFT 输出导出为 numpy 数组。

```bash
chemlab ml_data export_numpy --data ./raw_data/ --out ./
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `data` | `./raw_data/` | 输入目录 |
| `out` | `./` | 输出目录 |
| `prefix` | `full_` | 文件前缀 |
| `state_idx` | `1` | 激发态索引 |
| `energy_unit` | `kcal/mol` | 能量单位 |
| `ex_energy_unit` | `ev` | 激发能单位 |

### traj_split

分割轨迹为 train/val/test 并导出 xyz。

```bash
chemlab ml_data traj_split --input_dir ./traj/ --train 1000 --val 200 --test 200
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `input_dir` | `""` | 输入轨迹目录 |
| `outdir` | `""` | 输出目录 |
| `split_file` | `split.npz` | 分割信息文件 |
| `train/val/test` | `0` | 各集样本数 |

### prepare_tddft_inp.py

从 MD 轨迹提取帧生成 TDDFT 输入文件（通过 argparse 调用）。

---

## qchem 模块

### convert_out_to_inp

优化输出转单点输入。

```bash
chemlab qchem convert_out_to_inp --file opt.out --ref ref.in
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `file` | `""` | 待转换 .out 文件 |
| `ref` | `ref.in` | 参考输入文件 |

### multi_states

生成多自旋态输入文件。

```bash
chemlab qchem multi_states --file mol.xyz --spins 1,3,5
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `file` | `""` | xyz 文件 |
| `spins` | `[1,3,5,7]` | 自旋多重度列表 |
| `charge` | `0` | 电荷 |
| `ref` | `ref.in` | 参考文件 |

### run_mecp

MECP（最小能量交叉点）优化。

```bash
chemlab qchem run_mecp --file ref.in --spin1 1 --spin2 3 --max_steps 80
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `path` | - | 参考文件路径 |
| `file` | `ref.in` | 参考输入 |
| `out` | `./mecp_out` | 输出目录 |
| `jobtype` | `mecp` | 作业类型 (mecp/soc) |
| `spin1/spin2` | `1/3` | 两态自旋 |
| `nthreads` | `16` | 线程数 |
| `max_steps` | `80` | 最大步数 |
| `conv` | `1e-5` | 收敛阈值 |

### fpfd / tpfd

有限差分计算数值梯度（五点/三点）。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `path` | `./` | 输入路径 |
| `outpath` | `./out/` | 输出路径 |
| `distance` | `0.001` | 差分步长 (Å) |
| `ncore` | `32` | 总核心数 |
| `njob` | `8` | 并行作业数 |

---

## scan 模块

### scan1d

一维约束优化扫描。

```bash
chemlab scan scan1d --ref ref.in --row_max 40 --row_start 2.0 --row_distance -0.02
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `inp_path` | `./` | 输入路径 |
| `out_path` | `""` | 输出路径 |
| `ref` | `ref.in` | 参考文件 |
| `row_max` | `40` | 扫描步数 |
| `row_start` | `2.0` | 起始距离 |
| `row_distance` | `-0.02` | 步长增量 |
| `ncore` | `16` | 核心数 |
| `njob` | `4` | 并行作业数 |
| `scan_limit_init` | `5` | 初始门控限制 |
| `poll_interval` | `60` | 轮询间隔(秒) |

特性：自动几何继承、断点续算、实时状态监控。

---

## 配置文件

默认配置位于 `chemlab/config/config.toml`，CLI 参数会覆盖默认值。

Q-Chem 环境配置示例：

```toml
[qchem_env]
use_defaults = "pete"

[defaults.pete]
env_script = """
module purge
export QC=/scratch/moriya/software/soc
source $QC/bin/qchem.setup.sh
"""
```

---

## 添加新脚本

1. 在 `chemlab/scripts/<module>/` 下创建 `.py` 文件
2. 定义 Script 子类，设置 `name` 和 `run()` 方法
3. (可选) 定义 ConfigBase 子类，在 config.toml 添加对应 section
4. 脚本自动被 CLI 发现注册
