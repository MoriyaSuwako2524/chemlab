import argparse
from chemlab.cli import ml, scan

def main():
    parser = argparse.ArgumentParser(prog="chemlab")
    subparsers = parser.add_subparsers(dest="group")


    ml_parser = subparsers.add_parser("ml")
    ml.add_subcommands(ml_parser)

    scan_parser = subparsers.add_parser("scan")
    scan.add_subcommands(scan_parser)

    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
