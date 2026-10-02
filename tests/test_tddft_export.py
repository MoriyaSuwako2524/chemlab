from pathlib import Path

import numpy as np
import pytest

from chemlab.util.file_system import qchem_out_multi
from chemlab.util.modify_inp import qchem_out_excite_multi
from chemlab.scripts.ml_data.export_numpy import ExportNumpy


@pytest.fixture
def parsed(tmp_path, tddft_text):
    path = tmp_path / "train_0000.out"
    path.write_text(tddft_text)
    multi = qchem_out_excite_multi()
    multi.read_files([str(path)])
    assert len(multi.tasks) == 1
    return multi


def test_excitation_units(parsed, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert parsed.export_ex_energy()[0] == pytest.approx(3.3943)
    assert parsed.export_ex_energy(energy_unit="hartree")[0] == pytest.approx(3.3943 / 27.2113863)
    assert parsed.export_gs_energy(energy_unit="hartree")[0] == -1.0
    assert not list(tmp_path.glob("*.npy"))


def test_permanent_vs_transition_dipole_units(parsed):
    np.testing.assert_allclose(parsed.export_dipolemom(), [[1, 0, 0]], rtol=1e-6)
    np.testing.assert_allclose(parsed.export_dipolemom(unit="Debye"), [[2.54174623, 0, 0]])
    np.testing.assert_allclose(parsed.export_transmom(), [[1, 0, 0]])


def test_gradient_force_sign_and_missing_states(parsed):
    gradient = parsed.export_gradients(grad_unit=("hartree", "bohr"))
    np.testing.assert_allclose(gradient[0], [[.1, .3, .5], [.2, .4, .6]])
    np.testing.assert_allclose(parsed.export_forces(grad_unit=("hartree", "bohr")), -gradient)
    assert np.isnan(parsed.export_gradients(state_idx=2)).all()
    assert np.isnan(parsed.export_forces(state_idx=2)).all()


def test_failed_files_do_not_shift_splits(export_config, tddft_text, monkeypatch, tmp_path):
    data = Path(export_config.data)
    (data / "train_0000.out").write_text(tddft_text)
    (data / "train_0001.out").write_text("Q-Chem fatal error")
    (data / "train_0002.out").write_text(tddft_text)
    (data / "val_0000.out").write_text(tddft_text)
    (data / "test_0000.out").write_text("truncated")
    monkeypatch.chdir(tmp_path)
    ExportNumpy().run(export_config)
    out = Path(export_config.out)
    with np.load(out / "full_tddft.npz") as arrays:
        assert arrays["coords"].shape == (3, 2, 3)
        assert arrays["idx_train"].tolist() == [0, 1]
        assert arrays["idx_val"].tolist() == [2]
        assert [Path(p).name for p in arrays["source_files"]] == ["train_0000.out", "train_0002.out", "val_0000.out"]
        assert np.isnan(arrays["gradients"][:, 1]).all()
        assert np.isnan(arrays["esp_charges_excited"]).all()
        np.testing.assert_allclose(arrays["excitation_energies"][:, 0], np.load(out / "full_ex_energy.npy"))
        np.testing.assert_allclose(arrays["gs_dipoles"] * .393430307, np.load(out / "full_dipolemom.npy"))
        assert str(arrays["excitation_energy_unit"]) == "eV"
    assert not list(tmp_path.glob("*.npy"))


def test_empty_or_failed_export_is_clear(export_config):
    with pytest.raises(ValueError, match="No valid"):
        ExportNumpy().run(export_config)
    (Path(export_config.data) / "train_0000.out").write_text("Q-Chem fatal error")
    with pytest.raises(ValueError, match="No valid"):
        ExportNumpy().run(export_config)


def test_inconsistent_states_rejected(export_config, tddft_text):
    data = Path(export_config.data)
    (data / "train_0000.out").write_text(tddft_text)
    (data / "train_0001.out").write_text(tddft_text.replace("Excited state   2:", "Excited state   3:"))
    with pytest.raises(ValueError, match="Inconsistent"):
        ExportNumpy().run(export_config)


def test_fatal_error_takes_precedence_over_footer(tmp_path):
    path = tmp_path / "bad.out"
    path.write_text("Thank you very much for using Q-Chem\nQ-Chem fatal error")
    assert qchem_out_multi.check_qchem_error(path) == 1


def test_small_export_skips_impossible_optional_splits(export_config, tddft_text):
    (Path(export_config.data) / "train_0000.out").write_text(tddft_text)
    export_config.val_splits = export_config.test_splits = 400
    with pytest.warns(UserWarning, match="exceed"):
        ExportNumpy().run(export_config)
    assert (Path(export_config.out) / "full_tddft.npz").exists()


def test_alignment_uses_valid_reference_and_preserves_nan():
    moment = np.array([[np.nan] * 3, [1, 0, 0], [-1, 0, 0]])
    density = np.array([[np.nan] * 2, [1, -1], [-1, 1]])
    m, d = ExportNumpy()._align_data(moment, density, "dipole")
    assert np.isnan(m[0]).all()
    np.testing.assert_allclose(m[1], m[2])
    np.testing.assert_allclose(d[1], d[2])
