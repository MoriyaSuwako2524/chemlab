import sys

import pytest

from chemlab.__main__ import main


@pytest.mark.parametrize("args", [["--help"], ["ml_data", "--help"],
                                 ["ml_data", "prepare_tddft_inp", "--help"],
                                 ["ml_data", "export_numpy", "--help"], ["crystal", "--help"]])
def test_help_does_not_import_unrelated_optional_backends(args):
    with pytest.raises(SystemExit) as result:
        main(args)
    assert result.value.code == 0
    assert "chemlab.scripts.ml_data.soap" not in sys.modules
    assert "chemlab.scripts.qmhub.qmmm_training_set_data" not in sys.modules
