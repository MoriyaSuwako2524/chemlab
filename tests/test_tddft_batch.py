import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from chemlab.util.tddft_batch import main, make_plan, pending_inputs, run_manifest


def inputs(tmp_path, names):
    paths = []
    for name in names:
        path = tmp_path / name
        path.write_text("input")
        paths.append(path)
    return paths


def test_actual_files_zero_frame_gaps_and_remainder(tmp_path):
    files = inputs(tmp_path, ["train_0000.inp", "train_0002.inp", "train_0010.inp", "val_0000.inp", "frame_0001.inp"])
    selected = pending_inputs(tmp_path)
    scripts = make_plan(selected, tmp_path / "plans", batch_size=2)
    manifests = [json.loads(p.with_suffix(".json").read_text())["inputs"] for p in scripts]
    assert list(map(len, manifests)) == [2, 2, 1]
    assert set(sum(manifests, [])) == {str(p) for p in files}
    assert "--cpus-per-task=32" in scripts[0].read_text()
    for script in scripts:
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_resume_and_failed_only(tmp_path):
    files = inputs(tmp_path, ["good.inp", "fatal.inp", "partial.inp", "missing.inp"])
    files[0].with_suffix(".out").write_text("Thank you very much for using Q-Chem")
    files[1].with_suffix(".out").write_text("Q-Chem fatal error\nThank you very much for using Q-Chem")
    files[2].with_suffix(".out").write_text("incomplete")
    assert {p.stem for p in pending_inputs(tmp_path)} == {"fatal", "partial", "missing"}
    assert {p.stem for p in pending_inputs(tmp_path, failed_only=True)} == {"fatal", "partial"}
    assert len(pending_inputs(tmp_path, force=True)) == 4
    # A regenerated input invalidates an older successful output.
    old = files[0].stat().st_mtime_ns - 1_000_000_000
    os.utime(files[0].with_suffix(".out"), ns=(old, old))
    assert files[0] in pending_inputs(tmp_path)


def test_plans_are_immutable_and_default_does_not_submit(tmp_path, monkeypatch):
    files = inputs(tmp_path, ["train_0000.inp"])
    def unexpected_submit(*args, **kwargs):
        pytest.fail("Planning must not invoke sbatch")
    monkeypatch.setattr(subprocess, "run", unexpected_submit)
    assert main(["--data", str(tmp_path)]) == 0
    before = list((tmp_path / "slurm_plans").glob("*/*.json"))[0]
    original = before.read_text()
    make_plan(files, tmp_path / "slurm_plans")
    assert before.read_text() == original
    assert len(list((tmp_path / "slurm_plans").glob("*/*.json"))) == 2


@pytest.mark.parametrize("setting,value", [("cores", 0), ("jobs", -1), ("batch_size", 0)])
def test_invalid_resources(tmp_path, setting, value):
    with pytest.raises(ValueError):
        make_plan([], tmp_path, **{setting: value})


def test_runner_uses_exit_code_footer_and_preserves_previous(tmp_path):
    files = inputs(tmp_path, ["good input.inp", "bad.inp", "no_footer.inp"])
    fake = tmp_path / "fake_qchem"
    fake.write_text(f"#!{sys.executable}\n"
                    "import sys\nfrom pathlib import Path\n"
                    "inp, out = map(Path, sys.argv[-2:])\n"
                    "out.write_text('partial' if inp.stem == 'no_footer' else 'Thank you very much for using Q-Chem')\n"
                    "sys.exit(1 if inp.stem == 'bad' else 0)\n")
    fake.chmod(0o755)
    previous = files[2].with_suffix(".out")
    previous.write_text("old output")
    script = make_plan(files, tmp_path / "plans", qchem=str(fake), jobs=2)[0]
    assert run_manifest(script.with_suffix(".json")) == 1
    assert next(tmp_path.glob("no_footer.out.previous-*")).read_text() == "old output"
    assert files[0].with_suffix(".out").read_text().startswith("Thank you")
    assert {p.stem for p in pending_inputs(tmp_path)} == {"bad", "no_footer"}


def test_changed_input_after_planning_rejected(tmp_path):
    files = inputs(tmp_path, ["train_0000.inp"])
    script = make_plan(files, tmp_path / "plans")[0]
    files[0].write_text("different geometry")
    with pytest.raises(ValueError, match="changed after planning"):
        run_manifest(script.with_suffix(".json"))
    assert not files[0].with_suffix(".out").exists()


def test_no_inputs_needs_no_slurm(tmp_path):
    assert main(["--data", str(tmp_path)]) == 0
    assert not (tmp_path / "slurm_plans").exists()
