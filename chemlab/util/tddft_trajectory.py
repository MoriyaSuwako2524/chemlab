"""Geometry-only readers for TDDFT preparation; all returned coordinates are Å."""
from pathlib import Path
import re
import warnings

import numpy as np

from .file_system import atom_charge_dict


def _symbols(types):
    inverse = {v: k for k, v in atom_charge_dict.items()}
    result = []
    for value in np.asarray(types):
        if isinstance(value, (int, np.integer)):
            symbol = inverse.get(int(value))
        else:
            symbol = str(value)
        if symbol not in atom_charge_dict:
            raise ValueError(f"Invalid atom type: {value!r}")
        result.append(symbol)
    return result


def _validate(coords, types):
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 3 or coords.shape[2] != 3 or not all(coords.shape):
        raise ValueError("Coordinates must have shape (n_frames, n_atoms, 3), with nonzero dimensions")
    if not np.isfinite(coords).all():
        raise ValueError("Coordinates contain NaN or infinity")
    if np.asarray(types).ndim != 1 or len(types) != coords.shape[1]:
        raise ValueError("Atom types must have shape (n_atoms,)")
    return coords, _symbols(types)


def read_xyz(path):
    frames, symbols = [], None
    with open(path) as stream:
        while True:
            line = stream.readline()
            if not line:
                break
            if not line.strip():
                continue
            count = int(line)
            if count <= 0 or not stream.readline():
                raise ValueError("Invalid or truncated XYZ header")
            types, xyz = [], []
            for _ in range(count):
                parts = stream.readline().split()
                if len(parts) < 4:
                    raise ValueError("Truncated XYZ coordinate block")
                types.append(parts[0])
                xyz.append([float(x.replace("D", "E")) for x in parts[1:4]])
            if symbols is not None and types != symbols:
                raise ValueError("Atom identities/order change between frames")
            symbols = types
            frames.append(xyz)
    return frames, symbols


def _orientation(block):
    """Require a closing separator so a partial last geometry is not kept."""
    marker = "Standard Nuclear Orientation"
    if marker not in block:
        raise ValueError("Missing Standard Nuclear Orientation")
    lines = block.split(marker)[-1].splitlines()[1:]
    atoms, coords = [], []
    for line in lines:
        parts = line.split()
        if parts and parts[0].isdigit():
            if len(parts) != 5 or int(parts[0]) != len(atoms) + 1:
                raise ValueError("Incomplete AIMD coordinate row")
            atoms.append(parts[1])
            coords.append([float(x.replace("D", "E").replace("d", "e")) for x in parts[2:]])
        elif atoms:
            if line.strip().startswith("---"):
                return coords, atoms
            raise ValueError("Unterminated AIMD coordinate table")
    raise ValueError("Incomplete AIMD geometry")


def read_aimd(path, allow_incomplete=False):
    text = Path(path).read_text()
    completed = ("Thank you very much for using Q-Chem" in text
                 and "Q-Chem fatal error" not in text)
    if not completed and not allow_incomplete:
        raise ValueError("AIMD output did not finish normally; use --allow_incomplete true to recover complete geometries")
    steps = list(re.finditer(r"^\s*TIME STEP #\s*(\d+)", text, re.M))
    frames, symbols, indices = [], None, []
    if steps and "Standard Nuclear Orientation" in text[:steps[0].start()]:
        _, symbols = _orientation(text[:steps[0].start()])
    for i, step in enumerate(steps):
        end = steps[i + 1].start() if i + 1 < len(steps) else len(text)
        try:
            coords, types = _orientation(text[step.start():end])
            if symbols is not None and types != symbols:
                raise ValueError("Atom identities/order change between AIMD frames")
        except ValueError:
            if not completed and allow_incomplete and i == len(steps) - 1:
                warnings.warn("Discarding incomplete final AIMD geometry", UserWarning)
                continue
            raise
        frames.append(coords)
        symbols = types
        indices.append(i)
    return frames, symbols, np.asarray(indices, dtype=int)


def read_trajectory(path, types="auto", allow_incomplete=False, input_distance_unit="ang"):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".xyz", ".traj"):
        coords, atoms = read_xyz(path)
        indices = np.arange(len(coords))
    elif suffix == ".out":
        if input_distance_unit != "ang":
            raise ValueError("Q-Chem Standard Nuclear Orientation is in angstrom")
        coords, atoms, indices = read_aimd(path, allow_incomplete)
    elif suffix == ".npy":
        coords = np.load(path, allow_pickle=False)
        if types == "auto":
            if not path.name.endswith("coord.npy"):
                raise ValueError("Specify --types for a coordinate NPY not named *coord.npy")
            prefix = path.name[:-len("coord.npy")]
            candidates = [path.with_name(prefix + name) for name in ("type.npy", "qm_type.npy")]
            found = [candidate for candidate in candidates if candidate.is_file()]
            if len(found) != 1:
                raise ValueError("Specify --types: expected exactly one matching type.npy or qm_type.npy")
            types = found[0]
        atoms = np.load(types, allow_pickle=False)
        indices = np.arange(len(coords)) if coords.ndim else np.array([], dtype=int)
    else:
        raise ValueError("Supported trajectory formats: .xyz, .traj, .out, .npy")
    coords, atoms = _validate(coords, atoms)
    factors = {"ang": 1.0, "bohr": 0.529177210903, "nm": 10.0}
    if input_distance_unit not in factors:
        raise ValueError("input_distance_unit must be ang, bohr or nm")
    return coords * factors[input_distance_unit], atoms, indices
