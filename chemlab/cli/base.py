import argparse
import pkgutil
import importlib
import inspect
from types import ModuleType


class CLICommand:
    """Base class for all CLI command modules.

    Subclasses must define:
        name: str
        add_arguments(self, parser, subparsers)
    """
    name = None

    def add_arguments(self, parser, subparsers):
        raise NotImplementedError

    def register(self, top_subparsers):
        """Register this command as a first-level subcommand."""
        parser = top_subparsers.add_parser(self.name)
        subparsers = parser.add_subparsers(dest=f"{self.name}_cmd")
        self.add_arguments(parser, subparsers)


def is_valid_cli_class(obj):
    """Check whether obj is a valid CLICommand subclass."""
    return (
        inspect.isclass(obj)
        and issubclass(obj, CLICommand)
        and obj is not CLICommand       # exclude base class
        and obj.name is not None        # must define name
        and isinstance(obj.name, str)
        and obj.name.strip() != ""
    )


def load_module_recursive(package: ModuleType):
    """Recursively load all modules under package."""
    for finder, modname, ispkg in pkgutil.walk_packages(
        package.__path__, package.__name__ + "."
    ):
        yield importlib.import_module(modname)


def load_cli_commands(package):
    """Discover and instantiate CLICommand classes inside package."""
    commands = []
    seen = set()

    # Recursively load modules under chemlab.cli
    for module in load_module_recursive(package):

        # Scan each module for CLICommand subclasses
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if is_valid_cli_class(obj):
                if obj not in seen:     # avoid duplicates
                    seen.add(obj)
                    commands.append(obj())

    return commands
