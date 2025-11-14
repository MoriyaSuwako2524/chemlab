# chemlab/config/__init__.py

from .config_loader import (
    ConfigBase,
    MecpConfig,
    ExportNumpyConfig,
    PrepareTDDFTConfig,
)


def get_mecp_config():
    return MecpConfig()

def get_export_numpy_config():
    return ExportNumpyConfig()
def get_prepare_tddft_config():
    return PrepareTDDFTConfig()
__all__ = [
    "ConfigBase",
    "MecpConfig",
    "ExportNumpyConfig",
    "PrepareTDDFTConfig",
    "get_mecp_config",
    "get_export_numpy_config",
    "get_prepare_tddft_config",
]
