# chemlab/config/__init__.py

from .config_loader import (
    ConfigBase,
    MecpConfig,
    ExportNumpyConfig,
)


def get_mecp_config():
    return MecpConfig()

def get_export_numpy_config():
    return ExportNumpyConfig()

__all__ = [
    "ConfigBase",
    "MecpConfig",
    "ExportNumpyConfig",
    "get_mecp_config",
    "get_export_numpy_config",
]
