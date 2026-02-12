from dscribe.descriptors import SOAP
from ase import Atoms
import numpy as np
from chemlab.scripts.base import Script
from chemlab.config.config_loader import ConfigBase
from chemlab.util.file_system import NUM2ELEMENT
import os

class BuildSOAPConfig(ConfigBase):
    section_name = "build_soap"

class BuildSOAP(Script):
    name = "build_soap"
    config = BuildSOAPConfig
    def run(self,cfg):
        windows = cfg.windows
        npy_path = cfg.npy_path
        qm_type_file = cfg.qm_type
        qm_type = np.load(os.path.join(npy_path,qm_type_file))
        elements = [NUM2ELEMENT[int(i)] for i in qm_type]
        out_path=cfg.out_path
        r_cut = cfg.r_cut
        n_max = cfg.n_max
        l_max = cfg.l_max
        sigma = cfg.sigma
        soap = SOAP(
            species=elements,
            r_cut=6.0,
            n_max=8,
            l_max=6,
            sigma=0.5,
            periodic=False,
            average="outer"
        )

        for i in range(windows):
            coords = np.load(os.path.join(npy_path,f"qm_coord_w{i:02d}.npy"))
            soap_features = []
            for j, coord in enumerate(coords):
                soap_features.append(self.single_frame_soap(coord,qm_type,soap))
            soap_features = np.array(soap_features)  # (n_structures, n_features)
            np.save(f"{out_path}/soap_features_w{i:02d}.npy", soap_features)
    @staticmethod
    def single_frame_soap(coord,qm_type,soap):
        atoms = Atoms(numbers=qm_type, positions=coord)
        soap_vec = soap.create(atoms)
        return soap_vec


