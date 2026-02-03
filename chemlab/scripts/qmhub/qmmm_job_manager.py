import sys
import os
import shutil
import uuid
import subprocess as sp
from multiprocessing import Pool
import numpy as np

from chemlab.util.file_system import ELEMENT_DICT
from chemlab.scripts.base import QchemBaseScript,QMJob,print_status
from chemlab.config.config_loader import ConfigBase


class QMMMJobManagerConfig(ConfigBase):
    section_name = "qmmm_job_manager"


class QMMMJobManager(QchemBaseScript):
    name = "qmmm_job_manager"
    config = QMMMJobManagerConfig
    def run(self,cfg):
        qmmmpath = cfg.qmmmpath
        window = "{:02d}".format(cfg.window)
        tem_path = f"{qmmmpath}/{window}/"
        jobs = []
        for root, dirs, files in os.walk(tem_path):
            for file in files:
                if file.endswith(".inp"):
                    inp_path = os.path.join(root, file)
                    out_path = inp_path.replace(".inp", ".out")
                    jobs.append(QMJob(inp_path,out_path))
        self.run_jobs(jobs, cfg, print_status_func=print_status)


