import pkgutil
import importlib
import inspect
import ast
from pathlib import Path
from types import ModuleType

import chemlab.scripts
from chemlab.scripts.base import Script


class CLICommand:


    name = None   # e.g. "ml_data"

    def add_arguments(self, parser, subparsers, selected_script=None):
        # Discover names without importing every optional scientific backend.
        package_name = f"chemlab.scripts.{self.name}"
        if importlib.util.find_spec(package_name) is None:
            return
        package = importlib.import_module(package_name)
        for _, command, is_package in pkgutil.iter_modules(package.__path__):
            if is_package or command.startswith("_"):
                continue
            spec = importlib.util.find_spec(f"{package.__name__}.{command}")
            tree = ast.parse(Path(spec.origin).read_text(encoding="utf-8"))
            if not any(isinstance(node, ast.ClassDef) and any(
                isinstance(item, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "name"
                                                     for t in item.targets)
                for item in node.body) for node in tree.body):
                continue
            p = subparsers.add_parser(command)
            if command != selected_script:
                continue
            module = importlib.import_module(f"{package.__name__}.{command}")
            classes = [obj for _, obj in inspect.getmembers(module, inspect.isclass)
                       if issubclass(obj, Script) and obj is not Script
                       and obj.__module__ == module.__name__ and obj.name]
            if len(classes) != 1:
                raise ValueError(f"Expected one Script implementation in {module.__name__}")
            script_cls = classes[0]

            # Add config parameters if the script uses a config class
            cfg_class = getattr(script_cls, "config", None)
            if cfg_class:
                cfg_class.add_to_argparse(p)

            script_obj = script_cls()

            # Bind function
            p.set_defaults(
                func=lambda args,
                             script=script_obj,
                             cfg_class=cfg_class:
                    self.run_script(script, cfg_class, args)
            )

    def run_script(self, script, cfg_class, args):
        """Load cfg (if needed) and run the script."""
        if cfg_class:
            cfg = cfg_class()
            cfg.apply_override(vars(args))
        else:
            cfg = None

        return script.run(cfg)

    def register(self, top_subparsers, selected_script=None):
        """Register this CLI group under top-level CLI."""
        parser = top_subparsers.add_parser(self.name)
        subparsers = parser.add_subparsers(dest=f"{self.name}_cmd")
        self.add_arguments(parser, subparsers, selected_script=selected_script)



_DISCOVER_CACHE = None
def discover_scripts():
    """
    Automatically discover ALL Script subclasses under chemlab.scripts.*
    """
    global _DISCOVER_CACHE
    if _DISCOVER_CACHE is not None:
        return _DISCOVER_CACHE
    scripts = []

    for _, modname, _ in pkgutil.walk_packages(
        chemlab.scripts.__path__, chemlab.scripts.__name__ + "."
    ):
        module = importlib.import_module(modname)

        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, Script) and obj is not Script:
                scripts.append(obj)
    _DISCOVER_CACHE = scripts
    return scripts



def script_to_cli(script_cls):
    """
    Convert script module path to (group, command).

    E.g.
        chemlab.scripts.ml_data.export_numpy

    => group = "ml_data"
       command = "export_numpy"
    """
    module = script_cls.__module__
    parts = module.split(".")

    # ... scripts.<group>.<scriptname>
    group = parts[-2]
    command = parts[-1]

    return group, command
