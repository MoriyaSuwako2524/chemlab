# chemlab

:I This readme is written by Claude

A quantum chemistry automation toolkit focused on Q-Chem workflow automation, machine learning data preparation, and potential energy surface scanning.

## Features

- **Auto CLI Discovery** - Scripts automatically registered as CLI commands
- **Q-Chem Integration** - Input/output file parsing, job management, parallel execution
- **ML Data Processing** - TDDFT data export, trajectory splitting, format conversion
- **PES Scanning** - 1D/2D constrained optimization scans
- **MECP Optimization** - Minimum Energy Crossing Point search
- **Finite Difference** - Numerical gradient calculation (3-point/5-point)

## Installation

```bash
git clone <repo_url>
cd chemlab
pip install -e .
```

## Project Structure

```
chemlab/
├── cli/                    # CLI entry and command registration
│   └── base.py            # CLICommand base class, script discovery
├── config/                 # Configuration management
│   ├── config.toml        # Default configuration file
│   └── config_loader.py   # ConfigBase class
├── scripts/               # All CLI scripts
│   ├── base.py           # Script / QchemBaseScript base classes
│   ├── ml_data/          # Machine learning data processing
│   ├── qchem/            # Q-Chem utilities
│   └── scan/             # PES scanning
└── util/                  # Utility library
    ├── file_system.py    # Q-Chem file parsing
    ├── modify_inp.py     # Input file generation/modification
    ├── ml_data.py        # MLData dataset class
    ├── mecp.py           # MECP optimizer
    └── unit.py           # Unit conversion
```

## Quick Start

### Basic Usage

```bash
chemlab <module> <script> [options]
```

### Common Examples

```bash
# Export TDDFT data to numpy format
chemlab ml_data export_numpy --data ./raw_data/ --out ./npy/

# Convert optimization output to single-point input
chemlab qchem convert_out_to_inp --file opt.out

# Generate multi-spin-state inputs
chemlab qchem multi_states --file mol.xyz --spins 1,3,5

# 1D potential energy surface scan
chemlab scan scan1d --ref ref.in --row_max 40

# MECP optimization
chemlab qchem run_mecp --file ref.in --spin1 1 --spin2 3
```

## Modules

### ml_data - Machine Learning Data

| Script | Function |
|--------|----------|
| `export_numpy` | TDDFT output → numpy arrays |
| `traj_split` | Split trajectory into train/val/test |
| `prepare_tddft_inp` | MD trajectory → TDDFT inputs |

### qchem - Q-Chem Utilities

| Script | Function |
|--------|----------|
| `convert_out_to_inp` | Optimization output → single-point input |
| `multi_states` | Generate multi-spin-state inputs |
| `run_mecp` | MECP optimization |
| `fpfd` / `tpfd` | Finite difference gradient |

### scan - PES Scanning

| Script | Function |
|--------|----------|
| `scan1d` | 1D constrained optimization scan |

## Configuration System

Configuration file located at `chemlab/config/config.toml`, supporting:

- **Default Inheritance** - `use_defaults = "qchem"`
- **CLI Override** - Command line arguments have highest priority
- **Auto Type Inference** - Parameter types inferred from defaults

### Configuration Example

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

## Adding New Scripts

1. Create a `.py` file under `chemlab/scripts/<module>/`

2. Define a Script subclass:

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

3. Add configuration to `config.toml`:

```toml
[my_script]
param = "default_value"
```

4. Use it:

```bash
chemlab <module> my_script --param value
```

## Utility Library

### file_system.py

- `qchem_file` - Q-Chem input file parsing/generation
- `qchem_out_opt` - Optimization output parsing
- `qchem_out_force` - Force/gradient output parsing
- `molecule` - Molecular structure class

### modify_inp.py

- `qchem_out_excite_multi` - Multi-frame TDDFT output processing
- `qchem_out_aimd_multi` - Multi-frame AIMD output processing
- `single_spin_job` / `multi_spin_job` - Input file generators

### ml_data.py

- `MLData` - Machine learning dataset management
- Supports train/val/test splitting
- xyz file import/export

### unit.py

- `ENERGY` - Energy unit conversion
- `DISTANCE` - Distance unit conversion
- `GRADIENT` - Gradient unit conversion

## Dependencies

- Python >= 3.11
- numpy
- matplotlib (for visualization)

## License

MIT
