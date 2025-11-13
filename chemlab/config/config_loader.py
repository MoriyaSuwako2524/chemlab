# chemlab/config/config_loader.py

import tomllib
from pathlib import Path
import argparse



class ConfigBase:
    _cache = None   # TOML cached data

    section_name = None   # Must be defined by subclass

    @classmethod
    def load_all(cls):
        if cls._cache is not None:
            return cls._cache

        config_path = Path(__file__).resolve().parent / "config.toml"
        with open(config_path, "rb") as f:
            cls._cache = tomllib.load(f)
        return cls._cache

    @classmethod
    def section_dict(cls):
        all_data = cls.load_all()
        if cls.section_name not in all_data:
            raise ValueError(f"[config] Section '{cls.section_name}' not found")
        return all_data[cls.section_name]

    def __init__(self):
        data = self.section_dict()

        for k, v in data.items():
            setattr(self, k, v)

    def apply_override(self, cli_dict):
        for k, v in cli_dict.items():
            if v is not None:
                setattr(self, k, v)

    @classmethod
    def add_to_argparse(cls, parser):
        data = cls.section_dict()
        for key, value in data.items():
            parser.add_argument(
                f"--{key}",
                default=None,
                help=f"(default in config: {value})"
            )
        return parser

class MecpConfig(ConfigBase):
    section_name = "mecp"

class ExportNumpyConfig(ConfigBase):
    section_name = "export_numpy"
