# Chemlab

This is a private repository for storing my scripts, tools, and other code-related resources.

## How to Use

1. Clone the repository:
   ```bash
   git clone <repo_url>
   cd chemlab
2. Install in editable mode:
   ```bash
    pip install -e .
3. Once installed, you can access the provided scripts directly.
Example scripts can be found under:
   ```bash
    chemlab/scripts/examples

## Utility Modules

```bash
file_system
```


This module provides tools to **read, write, and analyze Q-Chem input/output files**.

#### Main Features

- **Unit conversion**
  - Classes for `ENERGY`, `DISTANCE`, `FORCE`, `DIPOLE`, etc.
  - Supports conversions between Hartree, kcal/mol, eV, Å, bohr, fs, and more.

- **Q-Chem input handling**
  - `qchem_file` reads or generates standard `.inp` files.
  - Includes `$molecule`, `$rem`, `$opt`, `$opt2`, and other sections.

- **Molecule operations**
  - Parse and modify geometries.
  - Compute bond lengths, angles, and dihedrals.

- **Q-Chem output parsing**
  - Classes like:
    - `qchem_out_opt` – geometry optimization
    - `qchem_out_scan` – PES scan
    - `qchem_out_force` – forces and gradients
    - `qchem_out_excite` – excited-state analysis
    - `qchem_out_aimd` – AIMD trajectory
  - Extracts energies, gradients, geometries, ESP charges, etc.

- **Batch & multi-job support**
  - `multiple_qchem_jobs` for multi-`@@@` inputs.
  - `qchem_out_multi` for reading and summarizing multiple outputs.
