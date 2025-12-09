# chemlab

:I 我懒得写document，交给claude了

## 功能特性

- **CLI 自动发现** - 脚本自动注册为命令行工具
- **Q-Chem 集成** - 输入/输出文件解析、作业管理、并行计算
- **ML 数据处理** - TDDFT 数据导出、轨迹分割、格式转换
- **势能面扫描** - 1D/2D 约束优化扫描
- **MECP 优化** - 最小能量交叉点搜索
- **有限差分** - 数值梯度计算 (3点/5点)

## 安装

```bash
git clone <repo_url>
cd chemlab
pip install -e .
```

## 项目结构

```
chemlab/
├── cli/                    # CLI 入口和命令注册
│   └── base.py            # CLICommand 基类、脚本自动发现
├── config/                 # 配置管理
│   ├── config.toml        # 默认配置文件
│   └── config_loader.py   # ConfigBase 类
├── scripts/               # 所有 CLI 脚本
│   ├── base.py           # Script / QchemBaseScript 基类
│   ├── ml_data/          # 机器学习数据处理
│   ├── qchem/            # Q-Chem 工具
│   └── scan/             # 势能面扫描
└── util/                  # 工具函数库
    ├── file_system.py    # Q-Chem 文件解析
    ├── modify_inp.py     # 输入文件生成/修改
    ├── ml_data.py        # MLData 数据集类
    ├── mecp.py           # MECP 优化器
    └── unit.py           # 单位转换
```

## 快速开始

### 基本用法

```bash
chemlab <module> <script> [options]
```

### 常用命令示例

```bash
# 导出 TDDFT 数据为 numpy 格式
chemlab ml_data export_numpy --data ./raw_data/ --out ./npy/

# 优化输出转单点输入
chemlab qchem convert_out_to_inp --file opt.out

# 生成多自旋态输入
chemlab qchem multi_states --file mol.xyz --spins 1,3,5

# 一维势能面扫描
chemlab scan scan1d --ref ref.in --row_max 40

# MECP 优化
chemlab qchem run_mecp --file ref.in --spin1 1 --spin2 3
```

## 模块说明

### ml_data - 机器学习数据

| 脚本 | 功能 |
|------|------|
| `export_numpy` | TDDFT 输出 → numpy 数组 |
| `traj_split` | 轨迹分割 train/val/test |
| `prepare_tddft_inp` | MD 轨迹 → TDDFT 输入 |

### qchem - Q-Chem 工具

| 脚本 | 功能 |
|------|------|
| `convert_out_to_inp` | 优化输出 → 单点输入 |
| `multi_states` | 生成多自旋态输入 |
| `run_mecp` | MECP 优化 |
| `fpfd` / `tpfd` | 有限差分梯度 |

### scan - 势能面扫描

| 脚本 | 功能 |
|------|------|
| `scan1d` | 一维约束优化扫描 |

## 配置系统

配置文件位于 `chemlab/config/config.toml`，支持：

- **默认值继承** - `use_defaults = "qchem"`
- **CLI 覆盖** - 命令行参数优先级最高
- **自动类型推断** - 根据默认值推断参数类型

### 配置示例

```toml
[export_numpy]
data = "./raw_data/"
out = "./"
state_idx = 1
energy_unit = "kcal/mol"

[qchem_env]
use_defaults = "pete"

[defaults.pete]
env_script = """
export QC=/path/to/qchem
source $QC/bin/qchem.setup.sh
"""
```

## 添加新脚本

1. 在 `chemlab/scripts/<module>/` 下创建 `.py` 文件

2. 定义 Script 子类：

```python
from chemlab.scripts.base import Script
from chemlab.config.config_loader import ConfigBase

class MyConfig(ConfigBase):
    section_name = "my_script"

class MyScript(Script):
    name = "my_script"
    config = MyConfig
    
    def run(self, cfg):
        print(f"Running with: {cfg.param}")
```

3. 在 `config.toml` 添加配置：

```toml
[my_script]
param = "default_value"
```

4. 使用：

```bash
chemlab <module> my_script --param value
```

## util 工具库

### file_system.py

- `qchem_file` - Q-Chem 输入文件解析/生成
- `qchem_out_opt` - 优化输出解析
- `qchem_out_force` - 力/梯度输出解析
- `molecule` - 分子结构类

### modify_inp.py

- `qchem_out_excite_multi` - 多帧 TDDFT 输出处理
- `qchem_out_aimd_multi` - 多帧 AIMD 输出处理
- `single_spin_job` / `multi_spin_job` - 输入文件生成器

### ml_data.py

- `MLData` - 机器学习数据集管理
- 支持 train/val/test 分割
- xyz 文件导入导出

### unit.py

- `ENERGY` - 能量单位转换
- `DISTANCE` - 距离单位转换
- `GRADIENT` - 梯度单位转换

## 依赖

- Python >= 3.11
- numpy
- matplotlib (MECP 可视化)

## License

MIT
