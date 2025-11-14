import pkgutil
import importlib
import inspect
from types import ModuleType

import chemlab.scripts
from chemlab.scripts.base import Script


# ============================================================
#   Base CLICommand class
# ============================================================

class CLICommand:
    """
    Base class for ALL top-level CLI groups, e.g.:

        chemlab ml_data ...
        chemlab scan ...
        chemlab soc ...

    Subclasses only need to define:
        name = "ml_data"

    and Script 自动发现会注册所有子命令。
    """

    name = None   # e.g. "ml_data"

    def add_arguments(self, parser, subparsers):
        """
        自动发现 chemlab.scripts 下的 Script 子类，
        根据路径映射成二级 CLI 命令。
        """

        # Find scripts dynamically
        for script_cls in discover_scripts():

            group, command = script_to_cli(script_cls)

            # Only register scripts under this CLI group
            if group != self.name:
                continue

            # Create parser for the script
            p = subparsers.add_parser(command, help=script_cls.__doc__)

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

    def register(self, top_subparsers):
        """Register this CLI group under top-level CLI."""
        parser = top_subparsers.add_parser(self.name)
        subparsers = parser.add_subparsers(dest=f"{self.name}_cmd")
        self.add_arguments(parser, subparsers)


# ============================================================
#   Helper: Script discovery
# ============================================================
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


# ============================================================
#   Helper: script path → CLI command mapping
# ============================================================
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
