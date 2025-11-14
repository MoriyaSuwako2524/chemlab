from argparse import ArgumentParser
from chemlab.scripts.ml_data.export_numpy import main as export_numpy_main
from chemlab.scripts.ml_data.prepare_tddft_inp  import main as prepare_tddft_main

def add_subcommands(parser: ArgumentParser):
    sub = parser.add_subparsers(dest="command")

    # export_numpy
    p_export = sub.add_parser("export_numpy")
    p_export.set_defaults(func=lambda args: export_numpy_main())

    # prepare_tddft
    p_prepare = sub.add_parser("prepare_tddft")
    p_prepare.set_defaults(func=lambda args: prepare_tddft_main())
