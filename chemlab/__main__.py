# chemlab/__main__.py

import argparse
import pkgutil
import importlib
import inspect
import sys
import chemlab.cli
from chemlab.cli.base import CLICommand


def discover_cli_commands():
    """Discover all CLICommand subclasses under chemlab.cli."""
    commands = []

    for _, modname, _ in pkgutil.walk_packages(
        chemlab.cli.__path__, chemlab.cli.__name__ + "."
    ):
        module = importlib.import_module(modname)

        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, CLICommand) and obj is not CLICommand:
                commands.append(obj())

    return commands


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(prog="chemlab")
    subparsers = parser.add_subparsers(dest="command")

    # ===== auto discover CLICommand groups =====
    for cmd in discover_cli_commands():
        selected = argv[1] if len(argv) > 1 and argv[0] == cmd.name else None
        cmd.register(subparsers, selected_script=selected)

    args = parser.parse_args(argv)

    # ===== run script =====
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
