# chemlab/config/config_loader.py

import tomllib
from pathlib import Path


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
        data = all_data.get(cls.section_name, {})

        # Inheritance
        parent_key = data.get("use_defaults")
        if parent_key:
            parent_data = all_data.get("defaults", {}).get(parent_key, {})
            merged = {**parent_data, **data}  # child overrides parent
            return merged

        return data

    def __init__(self):
        data = self.section_dict()

        for k, v in data.items():
            setattr(self, k, v)

    def apply_override(self, overrides: dict):
        for key, val in overrides.items():
            if val is None or val == "":
                continue
            if not hasattr(self, key):
                continue

            current = getattr(self, key)

            # list
            if isinstance(current, list):
                if isinstance(val, str):
                    import ast
                    text = val.strip()
                    if "," in text and not text.startswith("["):
                        text = "[" + text + "]"
                    setattr(self, key, ast.literal_eval(text))
                else:
                    setattr(self, key, list(val))
                continue

            # int
            if isinstance(current, int):
                setattr(self, key, int(val))
                continue

            # float
            if isinstance(current, float):
                setattr(self, key, float(val))
                continue

            # bool
            if isinstance(current, bool):
                if isinstance(val, str):
                    setattr(self, key, val.lower() in ["1", "true", "yes"])
                else:
                    setattr(self, key, bool(val))
                continue

            # string
            setattr(self, key, val)

    @classmethod
    def add_to_argparse(cls, parser):
        """Automatically add all config fields as CLI arguments."""
        data = cls.section_dict()

        for key, default in data.items():
            arg = f"--{key}"

            # required rule:
            # default == "" → must be provided by CLI
            required = (default == "")

            # infer type
            if isinstance(default, bool):
                parser.add_argument(arg, default=None, type=str,
                                    help=f"(default: {default}) [bool]",
                                    required=required)
            elif isinstance(default, int):
                parser.add_argument(arg, default=None, type=int,
                                    help=f"(default: {default}) [int]",
                                    required=required)
            elif isinstance(default, float):
                parser.add_argument(arg, default=None, type=float,
                                    help=f"(default: {default}) [float]",
                                    required=required)
            elif isinstance(default, list):
                parser.add_argument(arg, default=None, type=str,
                                    help=f"(default: {default}) [list]",
                                    required=required)
            else:
                # string
                parser.add_argument(arg, default=None,
                                    help=f"(default: '{default}') [str]",
                                    required=required)

        return parser


# ---- User-defined config sections ----

class MecpConfig(ConfigBase):
    section_name = "mecp"

class ExportNumpyConfig(ConfigBase):
    section_name = "export_numpy"

class PrepareTDDFTConfig(ConfigBase):
    section_name = "prepare_tddft"

class ConvertOutToInpConfig(ConfigBase):
    section_name = "convert_out_to_inp"

class QchemEnvConfig(ConfigBase):
    section_name = "qchem_env"