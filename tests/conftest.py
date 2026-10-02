from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def reference(tmp_path):
    path = tmp_path / "ref.in"
    path.write_text("$molecule\n0 1\nH 0 0 0\nH 0 0 0.7\n$end\n\n"
                    "$rem\njobtype sp\nmethod pbe0\nbasis 6-31g*\ncis_n_roots 2\n$end\n")
    return path


def orientation(x=0):
    return ("Standard Nuclear Orientation (Angstroms)\n"
            " I Atom X Y Z\n ----------------------\n"
            f" 1 H {x} 0.0 0.0\n 2 H {x} 0.0 0.7\n ----------------------\n")


@pytest.fixture
def trajectories(tmp_path):
    coords = np.zeros((8, 2, 3))
    coords[:, :, 0] = np.arange(8)[:, None]
    coords[:, 1, 2] = 0.7
    xyz = tmp_path / "trajectory.xyz"
    xyz.write_text("".join(f"2\nframe {i}\nH {i} 0 0\nH {i} 0 0.7\n" for i in range(8)))
    traj = tmp_path / "trajectory.traj"
    traj.write_text(xyz.read_text())
    npy = tmp_path / "full_coord.npy"
    np.save(npy, coords)
    np.save(tmp_path / "full_qm_type.npy", [1, 1])
    aimd = tmp_path / "aimd.out"
    aimd.write_text("".join(f"TIME STEP # {i} (t = {i} a.u. = {i} fs)\n" + orientation(i)
                             for i in range(8)) + "Thank you very much for using Q-Chem\n")
    return {"xyz": xyz, "traj": traj, "npy": npy, "out": aimd}


@pytest.fixture
def prep_config(tmp_path, reference):
    return SimpleNamespace(file="", ref=str(reference), out=str(tmp_path / "inputs"),
                           charge=0, spin=1, dataset_size=3, mode="custom", start=0,
                           seed=42, types="auto", allow_incomplete=False, input_distance_unit="ang")


@pytest.fixture
def tddft_text():
    return ("$molecule\n0 1\nH 0 0 0\nH 0 0 .7\n$end\n" + orientation() + "Charge = 0 Multiplicity = 1\n SCF   energy = -1.0\n"
            "Dipole Moment (Debye)\n X 2.54174623 Y 0.0 Z 0.0\n"
            "Excited state   1: excitation energy (eV) = 3.3943\n"
            " Total energy for state 1: -0.875260 a.u.\n"
            " Multiplicity: Singlet\n Strength: 0.0\n"
            " Trans. Mom.: 1.0 X 0.0 Y 0.0 Z\n\n"
            "Excited state   2: excitation energy (eV) = 4.2\n"
            " Total energy for state 2: -0.845652 a.u.\n"
            " Multiplicity: Singlet\n Strength: 0.1\n"
            " Trans. Mom.: 0.0 X 1.0 Y 0.0 Z\n\n"
            " CIS 1 State Energy\n Gradient of the state energy\n"
            " 1 2\n 1 0.1 0.2\n 2 0.3 0.4\n 3 0.5 0.6\nGradient time\n"
            "Thank you very much for using Q-Chem\n")


@pytest.fixture
def export_config(tmp_path):
    data = tmp_path / "raw"
    data.mkdir()
    return SimpleNamespace(data=str(data), out=str(tmp_path / "arrays"), prefix="full_",
                           state_idx=1, energy_unit="hartree", ex_energy_unit="ev",
                           distance_unit="ang", grad_unit=["hartree", "bohr"],
                           force_unit=["hartree", "bohr"], align_ref="none",
                           train_splits=[1], val_splits=0, test_splits=0)
