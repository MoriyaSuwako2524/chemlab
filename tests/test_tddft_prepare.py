import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from chemlab.scripts.ml_data.prepare_tddft_inp import PrepareTddftConfig, PrepareTddftInp
from chemlab.util.tddft_trajectory import read_trajectory
from conftest import orientation


@pytest.mark.parametrize("kind", ["xyz", "traj", "npy", "out"])
def test_sampling_and_source_mapping(kind, trajectories, prep_config):
    prep_config.file = str(trajectories[kind])
    PrepareTddftInp().run(prep_config)
    out = Path(prep_config.out)
    expected = np.sort(np.random.default_rng(42).choice(8, 3, replace=False))
    np.testing.assert_array_equal(np.load(out / "source_indices.npy"), expected)
    assert sorted(p.name for p in out.glob("*.inp")) == [f"train_{i:04d}.inp" for i in range(3)]
    manifest = json.loads((out / "frames.json").read_text())
    assert [f["source_frame"] for f in manifest["frames"]] == expected.tolist()
    for i, idx in enumerate(expected):
        coords, _, _ = read_trajectory(out / f"train_{i:04d}.xyz")
        assert coords[0, 0, 0] == idx
        inp = (out / f"train_{i:04d}.inp").read_text().lower()
        assert "cis_n_roots" in inp and "pbe0" in inp


@pytest.mark.parametrize("kind", ["xyz", "out", "npy"])
def test_all_and_start(kind, trajectories, prep_config):
    prep_config.file = str(trajectories[kind])
    prep_config.mode, prep_config.start = "all", 2
    PrepareTddftInp().run(prep_config)
    np.testing.assert_array_equal(np.load(Path(prep_config.out) / "source_indices.npy"), np.arange(2, 8))


@pytest.mark.parametrize("field,value", [("start", -1), ("start", 8), ("dataset_size", -2), ("mode", "invalid")])
def test_invalid_sampling(field, value, trajectories, prep_config):
    prep_config.file = str(trajectories["xyz"])
    setattr(prep_config, field, value)
    with pytest.raises(ValueError):
        PrepareTddftInp().run(prep_config)
    assert not Path(prep_config.out).exists()


def test_existing_jobs_not_overwritten(trajectories, prep_config):
    prep_config.file = str(trajectories["xyz"])
    out = Path(prep_config.out)
    out.mkdir()
    previous = out / "train_0000.out"
    previous.write_text("valuable previous result")
    with pytest.raises(FileExistsError):
        PrepareTddftInp().run(prep_config)
    assert previous.read_text() == "valuable previous result"


def test_unrelated_xyz_not_converted(trajectories, prep_config):
    prep_config.file = str(trajectories["xyz"])
    out = Path(prep_config.out)
    out.mkdir()
    (out / "unrelated.xyz").write_text("unrelated")
    PrepareTddftInp().run(prep_config)
    assert not (out / "unrelated.inp").exists()


def test_truncated_aimd_opt_in_and_recovery(tmp_path):
    path = tmp_path / "truncated.out"
    path.write_text("TIME STEP # 0\n" + orientation() + "TIME STEP # 1\n"
                    + orientation(1).rsplit(" 2 H", 1)[0])
    with pytest.raises(ValueError, match="allow_incomplete"):
        read_trajectory(path)
    with pytest.warns(UserWarning, match="incomplete final"):
        coords, atoms, indices = read_trajectory(path, allow_incomplete=True)
    assert coords.shape == (1, 2, 3) and atoms == ["H", "H"]
    assert indices.tolist() == [0]


def test_incomplete_interior_aimd_rejected(tmp_path):
    path = tmp_path / "broken.out"
    path.write_text("TIME STEP # 0\n" + orientation() + "TIME STEP # 1\n"
                    "no geometry\nTIME STEP # 2\n" + orientation())
    with pytest.raises(ValueError, match="Missing"):
        read_trajectory(path, allow_incomplete=True)


@pytest.mark.parametrize("content", ["", "2\ncomment\nH 0 0 0\n", "1\n\nXx 0 0 0\n",
                                   "1\n\nH nan 0 0\n", "1\n\nH 0 0 0\n1\n\nHe 0 0 0\n"])
def test_invalid_xyz(tmp_path, content):
    path = tmp_path / "bad.xyz"
    path.write_text(content)
    with pytest.raises(ValueError):
        read_trajectory(path)


def test_numpy_explicit_symbols_and_units(tmp_path):
    path = tmp_path / "positions.npy"
    types = tmp_path / "symbols.npy"
    np.save(path, np.ones((2, 1, 3)))
    np.save(types, ["H"])
    with pytest.raises(ValueError, match="types"):
        read_trajectory(path)
    coords, atoms, _ = read_trajectory(path, types=str(types), input_distance_unit="bohr")
    np.testing.assert_allclose(coords, 0.529177210903)
    assert atoms == ["H"]


@pytest.mark.parametrize("coords,types", [(np.ones((2, 3)), [1, 1]), (np.ones((2, 1, 3)), [1, 8]),
                                        (np.ones((2, 1, 3)), [999]), (np.full((2, 1, 3), np.nan), [1])])
def test_numpy_invalid_shapes_and_types(tmp_path, coords, types):
    np.save(tmp_path / "coord.npy", coords)
    np.save(tmp_path / "type.npy", types)
    with pytest.raises(ValueError):
        read_trajectory(tmp_path / "coord.npy")


@pytest.mark.parametrize("value,expected", [("true", True), ("false", False), ("0", False)])
def test_boolean_cli_override(value, expected):
    parser = argparse.ArgumentParser()
    PrepareTddftConfig.add_to_argparse(parser)
    args = parser.parse_args(["--file", "aimd.out", "--allow_incomplete", value])
    cfg = PrepareTddftConfig()
    cfg.apply_override(vars(args))
    assert cfg.allow_incomplete is expected


def test_prepare_cli(trajectories, prep_config):
    subprocess.run([sys.executable, "-m", "chemlab", "ml_data", "prepare_tddft_inp",
                    "--file", str(trajectories["npy"]), "--out", prep_config.out,
                    "--ref", prep_config.ref, "--dataset_size", "2", "--seed", "7"], check=True, timeout=60)
    assert len(list(Path(prep_config.out).glob("*.inp"))) == 2
