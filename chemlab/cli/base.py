import argparse
import pkgutil
import importlib
import inspect


class CLICommand:
    """
    Base class for all CLI command modules.
    Subclasses must define: name = "cmdname"
    And implement: add_arguments(self, parser)
    """
    name = None  # must be overridden

    def add_arguments(self, parser):
        raise NotImplementedError

    def register(self, subparsers):
        cmd_parser = subparsers.add_parser(self.name)
        self.add_arguments(cmd_parser)


def load_cli_commands(package):
    """Automatically discover all CLICommand subclasses in package."""
    commands = []

    # Scan all modules in package
    for _, modname, _ in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
        module = importlib.import_module(modname)

        # Find classes inheriting CLICommand
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, CLICommand) and obj is not CLICommand:
                commands.append(obj())

    return commands
