#!/usr/bin/env python3
"""
Convert a cluster XYZ file to PDB format with atom names matching a MOL2 template.
Each group of N atoms is assigned to a separate residue.

Usage:
    python xyz2pdb.py H2OBPc.mol2 H2OBPc-3-3-3.xyz H2OBPc-3-3-3.pdb
"""

import sys


def read_mol2_atom_names(mol2_file):
    """Extract atom names from a MOL2 file."""
    atom_names = []
    in_atom_section = False
    with open(mol2_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('@<TRIPOS>ATOM'):
                in_atom_section = True
                continue
            if line.startswith('@<TRIPOS>') and in_atom_section:
                break
            if in_atom_section and line:
                parts = line.split()
                # MOL2 ATOM format: id name x y z type res_id res_name charge
                atom_names.append(parts[1])  # atom name
    return atom_names


def read_xyz(xyz_file):
    """Read an XYZ file and return list of (element, x, y, z)."""
    atoms = []
    with open(xyz_file, 'r') as f:
        natoms = int(f.readline().strip())
        f.readline()  # comment line
        for _ in range(natoms):
            parts = f.readline().split()
            element = parts[0]
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            atoms.append((element, x, y, z))
    print(f"Read {len(atoms)} atoms from {xyz_file}")
    return atoms


def write_pdb(atoms, atom_names_template, resname, output_file, atoms_per_mol):
    """Write a PDB file with proper residue assignments."""
    n_mol = len(atoms) // atoms_per_mol
    if len(atoms) % atoms_per_mol != 0:
        print(f"WARNING: {len(atoms)} atoms is not divisible by {atoms_per_mol}!")
        print(f"  Remainder: {len(atoms) % atoms_per_mol} atoms")
        sys.exit(1)

    print(f"Writing {n_mol} molecules, {atoms_per_mol} atoms each")

    # PDB residue name: max 4 characters
    resname_pdb = resname[:4]

    with open(output_file, 'w') as f:
        atom_serial = 0
        for mol_idx in range(n_mol):
            res_id = mol_idx + 1
            for atom_idx in range(atoms_per_mol):
                atom_serial += 1
                global_idx = mol_idx * atoms_per_mol + atom_idx
                elem, x, y, z = atoms[global_idx]
                aname = atom_names_template[atom_idx]

                # PDB atom name formatting (4 chars, left-justified if <=3 chars)
                if len(aname) < 4:
                    aname_fmt = f" {aname:<3s}"
                else:
                    aname_fmt = f"{aname:<4s}"

                # PDB ATOM record format
                f.write(
                    f"ATOM  {atom_serial:5d} {aname_fmt:4s} {resname_pdb:4s} {res_id:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {elem:>2s}\n"
                )
            f.write("TER\n")
        f.write("END\n")

    print(f"Wrote {output_file} with {atom_serial} atoms, {n_mol} residues")


def validate_elements(atoms, atom_names_template, atoms_per_mol):
    """Quick sanity check: compare elements between xyz and mol2."""
    # Simple element extraction from mol2 atom names
    element_map = {
        'C': 'C', 'N': 'N', 'O': 'O', 'H': 'H', 'S': 'S', 'P': 'P', 'F': 'F',
        'Cl': 'Cl', 'Br': 'Br'
    }

    mismatches = 0
    # Check first molecule
    for i in range(atoms_per_mol):
        xyz_elem = atoms[i][0]
        mol2_name = atom_names_template[i]
        # Extract element from mol2 atom name (first 1-2 chars that are letters)
        mol2_elem = ''
        for ch in mol2_name:
            if ch.isalpha():
                mol2_elem += ch
            else:
                break
        if not mol2_elem:
            mol2_elem = mol2_name[0]

        # Compare (case-insensitive first letter)
        if xyz_elem[0].upper() != mol2_elem[0].upper():
            mismatches += 1
            if mismatches <= 10:
                print(f"  MISMATCH at atom {i+1}: xyz={xyz_elem}, mol2_name={mol2_name}")

    if mismatches > 0:
        print(f"\nWARNING: {mismatches} element mismatches in first molecule!")
        print("The atom ordering in xyz may NOT match the mol2.")
        print("Please verify the atom ordering before proceeding.")
        return False
    else:
        print("Element check passed for first molecule - ordering looks consistent.")
        return True


if __name__ == '__main__':
    mol2_file = "H2OBPc.mol2"
    xyz_file = "H2OBPc-3-3-3.xyz"
    output_pdb = "H2OBPc-3-3-3.pdb"

    # Read template
    atom_names = read_mol2_atom_names(mol2_file)
    atoms_per_mol = len(atom_names)
    print(f"Template: {atoms_per_mol} atoms per molecule from {mol2_file}")
    print(f"First few atom names: {atom_names[:10]}")

    # Read cluster
    atoms = read_xyz(xyz_file)

    # Validate
    validate_elements(atoms, atom_names, atoms_per_mol)

    # Write PDB
    resname = "BPC"  # Max 4 chars for PDB
    write_pdb(atoms, atom_names, resname, output_pdb, atoms_per_mol)