import argparse
import chemlab.cli
from chemlab.cli.base import load_cli_commands


def main():
    parser = argparse.ArgumentParser(prog="chemlab")
    subparsers = parser.add_subparsers(dest="command")

    # auto discover and register all subclasses
    for cmd in load_cli_commands(chemlab.cli):
        cmd.register(subparsers)

    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
