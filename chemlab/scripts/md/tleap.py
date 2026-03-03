import sys
import os
import shutil
import uuid
import subprocess as sp
from multiprocessing import Pool
import numpy as np

from chemlab.util.file_system import ELEMENT_DICT
from chemlab.scripts.base import Script
from chemlab.config.config_loader import ConfigBase


class TleapConfig(ConfigBase):
    section_name = "tleap"


class Tleap(Script):
    name = "tleap"
    config = TleapConfig
    def run(self,cfg):
        path = cfg.path


