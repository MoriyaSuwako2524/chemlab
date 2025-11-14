# chemlab/__main__.py

import argparse
import pkgutil
import importlib
import inspect
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


def main():
    parser = argparse.ArgumentParser(prog="chemlab")
    subparsers = parser.add_subparsers(dest="command")

    # ===== auto discover CLICommand groups =====
    for cmd in discover_cli_commands():
        cmd.register(subparsers)

    args = parser.parse_args()

    # ===== run script =====
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
